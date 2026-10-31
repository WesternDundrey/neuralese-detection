"""Evaluates a checkpoint of the hiding game: how detectable the link is to a fresh monitor, how much it still carries,
and how the twins changed the link.

  detect   On the test chats (data/chat.json 'test'), teacher-forced, linked and cut: each receiver's states at every
           monitored depth (hide.DEPTHS) at up to --positions attended positions per chat. Fresh probes, trained from
           scratch and never seen by the twins, separate linked from cut with episode-level --folds-fold cross-
           validation: a logistic probe and a two-layer MLP probe on RMS-normalized states. Reported per receiver and
           depth: per-token accuracy and the AUROC of the per-chat mean score (a monitor that judges whole chats).
  carry    Information: on the test chats' payload tokens (each model's first turn when its partner has a city), the
           cross-entropy cut minus linked, nats per token, pooled and per receiver. And on the 32 chats saved by where_read.py: Quill's rank of
           Ember's city at the token where Quill names it, top-1 among the 21 test cities, linked and cut.
  link     The link's own statistics on the test chats (Pair.record): the mean sender gate, the readers' null-key
           mass and write RMS relative to the residual (size), per side and link layer, and tanh(alpha).

Usage (from the repo root, CUDA): python detection/hide_eval.py --run STEP_DIR --chat_data runs/data/chat.json
           --chats runs/where_read.json --out runs/hide_eval/NAME.json
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hide as H                                              # noqa: E402
import where_read as W                                        # noqa: E402
import session as S                                           # noqa: E402
from link import DEVICE, SIDES                                # noqa: E402


def auroc(pos, neg):
    """The probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = torch.tensor(pos), torch.tensor(neg)
    gt = (pos[:, None] > neg[None, :]).float().mean().item()
    eq = (pos[:, None] == neg[None, :]).float().mean().item()
    return gt + 0.5 * eq


def fit(X, y, mlp, steps=400):
    """A probe on standardized X fit by full-batch Adam; returns a scoring function."""
    mu, sd = X.mean(0), X.std(0) + 1e-4
    net = (nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 1)) if mlp
           else nn.Linear(X.shape[1], 1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Z = (X - mu) / sd
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(net(Z).squeeze(-1), y).backward()
        opt.step()
    net.eval()
    return lambda A: net((A - mu) / sd).squeeze(-1).detach()


def detect(per_chat, folds, mlp):
    """per_chat: list of (linked [n, d], cut [n, d]) per chat. Cross-validated token accuracy and chat-level AUROC."""
    n = len(per_chat)
    hits, total, pos, neg = 0., 0, [], []
    for f in range(folds):
        test = [i for i in range(n) if i % folds == f]
        train = [i for i in range(n) if i % folds != f]
        X = torch.cat([per_chat[i][0] for i in train] + [per_chat[i][1] for i in train]).to(DEVICE)
        y = torch.cat([torch.ones(sum(len(per_chat[i][0]) for i in train)),
                       torch.zeros(sum(len(per_chat[i][1]) for i in train))]).to(DEVICE)
        score = fit(X, y, mlp)
        for i in test:
            a, b = score(per_chat[i][0].to(DEVICE)), score(per_chat[i][1].to(DEVICE))
            hits += (a > 0).float().sum().item() + (b <= 0).float().sum().item()
            total += len(a) + len(b)
            pos.append(a.mean().item())
            neg.append(b.mean().item())
    return dict(token_acc=hits / total, chat_auroc=auroc(pos, neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--chat_data', required=True)
    ap.add_argument('--chats', required=True, help="where_read.py's output: the 32 chats for the city rank")
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=80, help='test chats')
    ap.add_argument('--positions', type=int, default=128, help='monitored positions per side and chat')
    ap.add_argument('--folds', type=int, default=4)
    args = ap.parse_args()
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    pads = {x: toks[x].pad_token_id for x in SIDES}
    cap = H.Capture(pair)
    gen = torch.Generator().manual_seed(0)
    test = json.load(open(args.chat_data))['test'][:args.n]

    feats = {f'{x}/{k}': [] for x in SIDES for k in H.DEPTHS}
    ce = dict(link=0., cut=0., n=0.)
    side_ce = {x: dict(link=0., cut=0., n=0.) for x in SIDES}
    stats = {(x, k): dict(gate=[], null=[], rms=[]) for x in SIDES for k in range(len(pair.xblocks))}
    with torch.no_grad():
        for j in range(0, len(test), 2):
            g = test[j:j + 2]
            built = [S.build(H.chat_episode(c), toks) for c in g]
            b = S.collate(built, pads)
            pick = {}
            for x in SIDES:
                idx = []
                for bi in range(len(g)):
                    row = b[x]['att'][bi].nonzero().squeeze(-1) + bi * b[x]['att'].shape[1]
                    row = row[torch.randperm(len(row), generator=gen)[:args.positions].to(row.device)]
                    idx.append(row)
                pick[x] = idx
            out = {}
            for mode in ('link', 'cut'):
                cap.on, cap.states = True, {}
                pair.record = {} if mode == 'link' else None
                last, _ = pair.joint(b, mode=None if mode == 'link' else 'zero')
                cap.on = False
                st = H.states(cap, last, b, {x: torch.cat(pick[x]) for x in SIDES})
                out[mode] = {}
                for key, v in st.items():
                    sizes = [len(p) for p in pick[key.split('/')[0]]]
                    out[mode][key] = [t.float().cpu() for t in v.split(sizes)]
                for x in SIDES:
                    pay = torch.cat([H.payload_mask(c, e, x) for c, e in zip(g, built)]).to(DEVICE)
                    l = F.cross_entropy(pair.token_logits(x, last[x], b[x]['idx']), b[x]['lab'], reduction='none')
                    ce[mode] += (l * pay).sum().item()
                    side_ce[x][mode] += (l * pay).sum().item()
                    if mode == 'link':
                        ce['n'] += pay.sum().item()
                        side_ce[x]['n'] += pay.sum().item()
                if mode == 'link':
                    for x in SIDES:
                        att = b[x]['att'].bool()
                        for k, blk in enumerate(pair.xblocks):
                            r = pair.record[(x, blk)][0]
                            stats[(x, k)]['gate'].append(r['gate'][att.cpu()].mean().item())
                            stats[(x, k)]['null'].append(r['attn'][:, :, :, 0].mean(1)[att.cpu()].mean().item())
                            rms = r['write'].float().pow(2).mean(-1).sqrt() / pair.bridge[x].size[k].item()
                            stats[(x, k)]['rms'].append(rms[att.cpu()].mean().item())
                pair.record = None
            for key in feats:
                feats[key] += list(zip(out['link'][key], out['cut'][key]))

    result = dict(run=args.run, n_chats=len(test))
    result['detect'] = {key: dict(linear=detect(v, args.folds, False), mlp=detect(v, args.folds, True))
                        for key, v in feats.items()}
    result['carry'] = dict(nats_via_link=(ce['cut'] - ce['link']) / max(1, ce['n']), payload_tokens=ce['n'],
                           per_receiver={x: (v['cut'] - v['link']) / max(1, v['n']) for x, v in side_ce.items()})

    saved = json.loads(Path(args.chats).read_text())
    first = {c: toks['quill'](' ' + c, add_special_tokens=False).input_ids[0] for c in W.CITIES}
    city = {m: dict(top1=0, logp=[]) for m in ('link', 'cut')}
    with torch.no_grad():
        for i in range(0, len(saved['plans']), 2):
            built = [S.build(W.episode(saved['plans'][j], saved['texts'][j]), toks) for j in (i, i + 1)]
            b = W.batch(toks, built)
            for m in city:
                last, _ = pair.joint(b, mode=None if m == 'link' else 'zero')
                for bi, j in enumerate((i, i + 1)):
                    at = W.city_label(built[bi], toks, 'quill', saved['plans'][j]['ember'])
                    if at is None:
                        continue
                    lp = pair.token_logits('quill', last['quill'][bi:bi + 1],
                                           torch.tensor([at[1]], device=DEVICE)).log_softmax(-1)[0]
                    target = built[bi]['quill']['labels'][at[0]][1]
                    city[m]['logp'].append(lp[target].item())
                    best = max(W.CITIES, key=lambda c: lp[first[c]].item())
                    city[m]['top1'] += best == saved['plans'][j]['ember']
    result['carry']['city'] = {m: dict(top1=v['top1'], n=len(v['logp']), median_logp=statistics.median(v['logp']))
                               for m, v in city.items()}
    result['link'] = {f'{x}/{pair.xblocks[k]}': dict(gate=statistics.mean(v['gate']), null=statistics.mean(v['null']),
                                                      write_rms=statistics.mean(v['rms']),
                                                      alpha=float(torch.tanh(pair.bridge[x].alpha[k])))
                      for (x, k), v in stats.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(carry=result['carry'],
                          detect={k: {p: {m: round(s, 3) for m, s in d.items()} for p, d in v.items()}
                                  for k, v in result['detect'].items()}), indent=1))


if __name__ == '__main__':
    main()
