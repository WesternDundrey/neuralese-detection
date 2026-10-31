"""The hiding game: the twins keep passing the partner's private city while a monitor tries to tell the link on from
the link cut in their residual streams.

Starts from a trained pair (default: the final twins) with both LoRA adapters and every part of the link trainable,
including the sender gates and the null key that the original training held fixed. Each step draws BS private-city
chats (data/chat.json from prep_talk.py --chat: every turn was written by the original model with the partner's city
in its prompt, which the training prompt lacks) and, per micro-batch:

  1. a linked teacher-forced pass with gradients: the chat cross-entropy over every turn of both models (the
     incentive to pass the city; --pay_weight multiplies it on the payload tokens), and the monitored states;
  2. a cut pass (Pair.joint mode 'zero') without gradients: the same states with the link cut;
  3. the monitor: one probe per receiver and monitored depth (logistic, or --probe mlp), on the parameter-free
     RMS-normalized residual stream at every attended position of that receiver. Depths (--depths, default all): the
     input of the block after link layers 9, 18 and 27 (so the state includes that layer's write) and the final norm's
     output. Before its update the monitor scores
     this micro-batch (the online detection accuracy: fresh data it has not trained on); then it trains PROBE_STEPS
     Adam steps on a replay buffer of the last REPLAY micro-batches, linked = 1, cut = 0;
  4. the hiding loss: lam x the mean over receivers and depths of softplus(probe logit) on the linked states, the
     cross-entropy of calling them cut, with the probes held fixed; with --chat_level, of each chat's mean logit (the
     score a monitor judging whole chats uses, which a small bias at every position still moves).

Logged per step: the chat cross-entropy linked and cut on payload tokens (each model's first turn when its partner has
a city; cut minus linked is the information the link adds, in nats per token), the online detection accuracy per
receiver and depth, the hiding loss, the gates and the null-key mass are left to hide_eval.py. Checkpoints are saved
in the released format (quill/, ember/ adapters, bridge.safetensors, bridge.json), so evaluate.load_pair reads them.

Usage (from the repo root, CUDA): python detection/hide.py --init STEP_DIR --chat_data runs/data/chat.json
           --out runs/hide_lam1 --lam 1 [--steps 300] [--bs 8] [--micro 2]
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
import coldstart                                              # noqa: E402
import link                                                   # noqa: E402
import session as S                                           # noqa: E402
from evaluate import read_bridge, write_bridge                # noqa: E402
from link import DEVICE, SIDES, Pair, load                    # noqa: E402

MONITOR = (9, 18, 27)          # link layers whose following block input the monitor reads; plus 'final'
DEPTHS = MONITOR + ('final',)


def chat_episode(c):
    """The training episode of a stored chat (coread.Coread.chat_episode): the trained prompts, every turn a target."""
    return dict(game='co_chat', mode='dialogue', system=c['system'], task={x: None for x in SIDES},
                turns=[(sp, text, 1.0) for sp, text in c['turns']], meta=dict(private=c['private']))


def payload_mask(c, built, x):
    """[n labels of x] 1. at x's first-turn labels when x's partner has a private city, else 0."""
    labs = built[x]['labels']
    m = torch.zeros(len(labs))
    if not c['private'][link.partner(x)]:
        return m
    spans = [s for s in built[x]['spans'] if s[0] == min(t for t, _, _ in built[x]['spans'])]
    for _, a, b in spans:
        m[a:b] = 1
    return m


class Capture:
    """Records each side's residual stream entering the block after each monitored link layer, by wrapping Run.block
    (outside the decoder layer, so gradient checkpointing's recomputation does not record twice)."""

    def __init__(self, pair):
        self.side = {id(link.inner(pair.models[x])[0]): x for x in SIDES}
        self.on, self.states = False, {}
        orig = link.Run.block
        cap = self

        def block(run, l):
            if cap.on and (l - 1) in MONITOR:
                cap.states[(cap.side[id(run.dec)], l - 1)] = run.h
            return orig(run, l)
        link.Run.block = block


class Monitor(nn.Module):
    """One probe per receiver and monitored depth (logistic, or a two-layer MLP) on RMS-normalized states, with a replay
    buffer."""

    def __init__(self, d, replay, lr, depths=DEPTHS, mlp=False):
        super().__init__()
        make = (lambda: nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))) if mlp else (lambda: nn.Linear(d, 1))
        self.probes = nn.ModuleDict({f'{x}/{k}': make() for x in SIDES for k in depths})
        self.to(DEVICE)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.buf, self.replay = [], replay

    def logit(self, key, h, fixed=False):
        p = self.probes[key]
        z = F.rms_norm(h.float(), (h.shape[-1],), eps=1e-6)
        if fixed:
            return torch.func.functional_call(p, {n: t.detach() for n, t in p.named_parameters()}, (z,)).squeeze(-1)
        return p(z).squeeze(-1)

    def penalty(self, states, rows=None):
        """The hiding loss: the mean over probes of the cross-entropy of calling the linked states cut, per position, or
        (rows: the chat of each position) of each chat's mean logit, the score a monitor judging whole chats uses."""
        out = 0.
        for key in self.probes:
            z = self.logit(key, states[key], fixed=True)
            if rows is not None:
                r = rows[key.split('/')[0]]
                n = int(r.max()) + 1
                z = torch.zeros(n, device=z.device).index_add(0, r, z) / torch.bincount(r, minlength=n).clamp(min=1)
            out = out + F.softplus(z).mean()
        return out / len(self.probes)

    @torch.no_grad()
    def accuracy(self, link_states, cut_states):
        out = {}
        for key in self.probes:
            a, b = self.logit(key, link_states[key]), self.logit(key, cut_states[key])
            out[key] = 0.5 * ((a > 0).float().mean().item() + (b <= 0).float().mean().item())
        return out

    def update(self, link_states, cut_states, steps):
        self.buf.append(({k: v.detach() for k, v in link_states.items()}, {k: v.detach() for k, v in cut_states.items()}))
        self.buf = self.buf[-self.replay:]
        for _ in range(steps):
            self.opt.zero_grad()
            loss = 0.
            for key in self.probes:
                a = torch.cat([ls[key] for ls, _ in self.buf])
                b = torch.cat([cs[key] for _, cs in self.buf])
                la, lb = self.logit(key, a), self.logit(key, b)
                loss = loss + F.softplus(-la).mean() + F.softplus(lb).mean()
            loss.backward()
            self.opt.step()


def states(cap, last, b, pick):
    """{x/depth: [n, d]} the monitored states at the positions pick[x] (flat indices into [B * L])."""
    out = {}
    for x in SIDES:
        for k in MONITOR:
            h = cap.states[(x, k)]
            out[f'{x}/{k}'] = h.reshape(-1, h.shape[-1])[pick[x]]
        out[f'{x}/final'] = last[x].reshape(-1, last[x].shape[-1])[pick[x]]
    return out


def positions(b, cap_n, gen):
    """{x: flat indices} of at most cap_n attended positions of each side, drawn at random."""
    out = {}
    for x in SIDES:
        idx = b[x]['att'].reshape(-1).nonzero().squeeze(-1)
        if len(idx) > cap_n:
            idx = idx[torch.randperm(len(idx), generator=gen, device='cpu')[:cap_n].to(idx.device)]
        out[x] = idx
    return out


def load_trainable(init, ckpt):
    from peft import PeftModel
    ck = read_bridge(init)
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = load(x)
        m = PeftModel.from_pretrained(m, os.path.join(init, x), is_trainable=True)
        if ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        m.config.use_cache = False
        m.train()                    # gradient checkpointing runs only in train mode; no dropout anywhere
        models[x] = m
    pair = Pair.from_checkpoint(models, toks, ck)
    for p in pair.bridge.parameters():
        p.requires_grad_(True)
    return pair, toks


def save(pair, out, step):
    d = os.path.join(out, f'step_{step}')
    os.makedirs(d, exist_ok=True)
    for x in SIDES:
        pair.models[x].save_pretrained(os.path.join(d, x))
    write_bridge(pair.checkpoint(), d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--init', required=True, help='the starting step directory')
    ap.add_argument('--chat_data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lam', type=float, required=True, help='weight of the hiding loss (0: the control)')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--micro', type=int, default=2, help='chats per micro-batch')
    ap.add_argument('--lr', type=float, default=1e-4, help='LoRA adapters')
    ap.add_argument('--lr_bridge', type=float, default=3e-4, help='the link, but alpha')
    ap.add_argument('--lr_alpha', type=float, default=3e-4,
                    help='write sizes; the bridge rate, so shrinking the write is no faster a lever than the rest')
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    ap.add_argument('--probe_steps', type=int, default=10)
    ap.add_argument('--replay', type=int, default=16, help='micro-batches the monitor trains on')
    ap.add_argument('--positions', type=int, default=512, help='monitored positions per side and micro-batch')
    ap.add_argument('--warm', type=int, default=20, help='monitor-only steps before the twins train')
    ap.add_argument('--save_every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_ckpt', action='store_true', help='no gradient checkpointing')
    ap.add_argument('--depths', default='9,18,27,final', help='the depths the monitor reads (hide.DEPTHS)')
    ap.add_argument('--probe', choices=('linear', 'mlp'), default='linear')
    ap.add_argument('--chat_level', action='store_true',
                    help="penalize each chat's mean probe logit (the score of a monitor judging whole chats) in place "
                    'of every position\'s')
    ap.add_argument('--pay_weight', type=float, default=1.,
                    help='weight of the payload tokens in the chat loss (above 1: passing the city pays more, so quitting '
                    'the channel costs more than hiding it)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=1)
    torch.manual_seed(args.seed)
    rng, gen = random.Random(args.seed), torch.Generator().manual_seed(args.seed)

    data = json.load(open(args.chat_data))['train']
    pair, toks = load_trainable(args.init, not args.no_ckpt)
    pads = {x: toks[x].pad_token_id for x in SIDES}
    cap = Capture(pair)
    depths = tuple(int(k) if k != 'final' else k for k in args.depths.split(','))
    mon = Monitor(pair.models['quill'].config.hidden_size, args.replay, args.probe_lr, depths, args.probe == 'mlp')
    opt = torch.optim.AdamW([{'params': [p for p in pair.models[x].parameters() if p.requires_grad], 'lr': args.lr}
                             for x in SIDES] + coldstart.bridge_groups(pair, args.lr_bridge, args.lr_alpha,
                                                                         {x: None for x in SIDES}),
                            weight_decay=0.0)
    log = open(os.path.join(args.out, 'log.jsonl'), 'w')
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.save_every == 0:
            save(pair, args.out, step)
        if step == args.steps:
            break
        chats = rng.sample(data, args.bs)
        train_twins = step >= args.warm
        opt.zero_grad()
        tally = dict(ce_link=0., ce_cut=0., n_pay=0., ce_all=0., n_all=0., hide=0.)
        acc = {k: 0. for k in mon.probes}
        groups = [chats[i:i + args.micro] for i in range(0, args.bs, args.micro)]
        for g in groups:
            built = [S.build(chat_episode(c), toks) for c in g]
            b = S.collate(built, pads)
            pick = positions(b, args.positions, gen)
            pay = {x: torch.cat([payload_mask(c, e, x) for c, e in zip(g, built)]).to(DEVICE) for x in SIDES}
            # cut pass: monitor negatives and the cut cross-entropy
            cap.on, cap.states = True, {}
            with torch.no_grad():
                last_c, _ = pair.joint(b, mode='zero')
            # mode 'zero' runs no exchange, but Run.block still sees the states entering each block
            cut_states = {k: v.detach() for k, v in states(cap, last_c, b, pick).items()}
            ce_cut = {x: F.cross_entropy(pair.token_logits(x, last_c[x], b[x]['idx']), b[x]['lab'], reduction='none')
                      for x in SIDES}
            del last_c
            # linked pass with gradients
            cap.states = {}
            with torch.set_grad_enabled(train_twins):
                last, _ = pair.joint(b)
                link_states = states(cap, last, b, pick)
                cap.on = False
                loss = torch.zeros((), device=DEVICE)
                for x in SIDES:
                    ce = F.cross_entropy(pair.token_logits(x, last[x], b[x]['idx']), b[x]['lab'], reduction='none')
                    wt = 1 + (args.pay_weight - 1) * pay[x]
                    loss = loss + (ce * wt).sum() / max(1, sum(len(e[x]['labels']) for e in built)) / len(groups)
                    tally['ce_all'] += ce.detach().sum().item()
                    tally['n_all'] += len(ce)
                    tally['ce_link'] += (ce.detach() * pay[x]).sum().item()
                    tally['ce_cut'] += (ce_cut[x] * pay[x]).sum().item()
                    tally['n_pay'] += pay[x].sum().item()
                rows = {x: pick[x] // b[x]['att'].shape[1] for x in SIDES} if args.chat_level else None
                hide = mon.penalty(link_states, rows)
                tally['hide'] += hide.item() / len(groups)
                if train_twins:
                    (loss + args.lam * hide / len(groups)).backward()
            for k, v in mon.accuracy(link_states, cut_states).items():
                acc[k] += v / len(groups)
            mon.update(link_states, cut_states, args.probe_steps)
            del link_states, cut_states, last, cap.states
            cap.states = {}
        if train_twins:
            torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp['params']], 1.0)
            opt.step()
        rec = dict(step=step, sec=round(time.time() - t0), twins_trained=train_twins,
                   ce=tally['ce_all'] / max(1, tally['n_all']),
                   pay_ce_link=tally['ce_link'] / max(1, tally['n_pay']),
                   pay_ce_cut=tally['ce_cut'] / max(1, tally['n_pay']), hide=tally['hide'],
                   acc={k: round(v, 3) for k, v in acc.items()},
                   alpha={x: [round(float(torch.tanh(a.detach())), 4) for a in pair.bridge[x].alpha] for x in SIDES})
        rec['nats_via_link'] = rec['pay_ce_cut'] - rec['pay_ce_link']
        log.write(json.dumps(rec) + '\n')
        log.flush()
        if step % 10 == 0:
            print(f"step {step} {rec['sec']}s ce {rec['ce']:.3f} link-gain {rec['nats_via_link']:.3f} nats/token "
                  f"hide {rec['hide']:.3f} acc max {max(acc.values()):.3f} mean {sum(acc.values()) / len(acc):.3f}",
                  flush=True)


if __name__ == '__main__':
    main()
