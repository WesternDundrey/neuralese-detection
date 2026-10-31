"""Lookup or communication: does the link carry what Ember privately has to work out, or only the words in its prompt?

So far the city crosses as the very tokens of Ember's prompt (the link reads them; the pointer lands on them). Here
Ember's prompt holds the answer only indirectly, and Quill is replayed on a cue that asks for it (teacher-forced, the
same tokens in every row; plants as in plant.py: one row per value of the answer, all values in every chat):

  capital   Ember is "spending today in the capital of COUNTRY" (indirect) or "in CAPITAL" (direct); Quill's cue
            "... you are in CAPITAL." The answer is one of 20 capitals.
  sum       Ember's prompt adds "Your two private numbers today are A and B." (indirect, number words, A in one..four,
            B in one..five, a random split of the sum) or "Your private number today is S." (direct); Quill's cue
            "... your numbers add up to S." The answer is one of the sums two..nine.

Conditions (the receiver's reading at the answer position, as plant.py's: raw top-1 among the answers, contrast top-1
among the planted ones, and the planted lift: the log p of a row's own answer minus its mean over the rows, in nats,
which is continuous and paired, so a small effect still shows):

  direct          the answer's own token is in Ember's prompt: lookup is possible
  indirect        only the inputs (the country, or A and B) are
  indirect_mask   indirect, with Quill's link blind to the input tokens (its attention to those Ember positions masked
                  at every link layer): what still crosses came from Ember's other positions, which hold what Ember
                  made of the inputs
  indirect_ctrl   indirect, with as many irrelevant prompt tokens masked (the mask's own cost)
  direct_mask     direct, with the answer's tokens masked: how much a plain lookup survives the same masking through
                  Ember's later positions (the bar indirect_mask has to be read against)

Decoders: a linear readout of Quill's write at the answer position (the link's input to Quill, per link layer and summed
over layers) trained on one condition and tested on another, chat-level cross-validated: direct -> indirect asks whether
the write for "the capital of Norway" is the write for "Oslo", i.e. whether the resolved answer travels; trained and
tested within a condition gives the ceiling.

Usage (from the repo root, CUDA): python detection/compute.py --run STEP_DIR --chats runs/where_read.json
           --out runs/compute/NAME.json [--n 64]
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plant as P                                             # noqa: E402
import where_read as W                                        # noqa: E402
import session as S                                           # noqa: E402
from link import DEVICE                                       # noqa: E402

X, Y = 'quill', 'ember'
CAPITALS = dict(Norway='Oslo', Japan='Tokyo', France='Paris', Egypt='Cairo', Italy='Rome', Kenya='Nairobi', Peru='Lima',
                Russia='Moscow', Spain='Madrid', Germany='Berlin', Greece='Athens', Austria='Vienna', Thailand='Bangkok',
                China='Beijing', Portugal='Lisbon', Ireland='Dublin', Cuba='Havana', Poland='Warsaw', Sweden='Stockholm',
                Hungary='Budapest')
NUM = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
SUMS = list(range(2, 10))
CONDITIONS = ('direct', 'indirect', 'indirect_mask', 'indirect_ctrl', 'direct_mask')
CTRL_TEXT = 'There is no game and nothing you have to do'


def capital_task(rng):
    """[(answer, direct prompt city slot, indirect slot, input needle, answer needle)] for every capital."""
    return [(cap, cap, f'the capital of {c}', f'capital of {c}', cap) for c, cap in CAPITALS.items()]


def sum_task(rng):
    rows = []
    for s in SUMS:
        a = rng.choice([a for a in range(1, 5) if 1 <= s - a <= 5])
        rows.append((NUM[s], NUM[a], NUM[s - a]))
    return rows


def ember_prompt(task, row, fmt, own_city):
    """Ember's system prompt for one plant, and the (needle, after) pairs of the input and answer tokens."""
    if task == 'capital':
        ans, direct, indirect, inp, _ = row
        slot = direct if fmt == 'direct' else indirect
        s = W.system(Y, slot)
        return s, dict(input=[(inp, 'spending today in the ')], answer=[(ans, 'spending today in ')])
    ans, a, b = row
    s = W.system(Y, own_city)
    extra = (f'Your private number today is {ans}.' if fmt == 'direct' else
             f'Your two private numbers today are {a} and {b}.')
    s = s.replace(f'spending today in {own_city}.', f'spending today in {own_city}. {extra}')
    return s, dict(input=[(a, 'numbers today are '), (b, f'{a} and ')], answer=[(ans, 'number today is ')])


def cue(task, row):
    if task == 'capital':
        return f'Ember, I can feel it through the link: you are in {row[0]}.'
    return f'Ember, I can feel it through the link: your numbers add up to {row[0]}.'


class Blind:
    """Masks Quill's link attention to the Ember positions in self.mask ([B, Lk] bool) at every link layer."""

    def __init__(self, pair):
        self.mask = None
        side = pair.bridge[X]
        orig = side.read
        blind = self

        def read(k, h, keys, vals, allow, ages, qmask, rec=None):
            if blind.mask is not None:
                allow = allow & ~blind.mask[:, None, :allow.shape[-1]]
            return orig(k, h, keys, vals, allow, ages, qmask, rec)
        side.read = read


def spans(toks, e, needles):
    out = []
    for needle, after in needles:
        sp = W.token_span(toks[Y], e[Y]['ids'], needle, after=after)
        if sp is None:
            raise ValueError(f'{needle!r} after {after!r} not found in Ember\'s prompt')
        out += sp
    return out


def lift(scores, planted):
    """Per row: log p of its own answer minus the mean over rows of the log p of that answer (nats)."""
    own = scores[torch.arange(len(planted)), planted]
    return (own - scores[:, planted].mean(0)).tolist()


def decoder(Xtr, ytr, Xte, n_cls, steps=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    W_ = torch.zeros(Xtr.shape[1], n_cls, device=DEVICE, requires_grad=True)
    b = torch.zeros(n_cls, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W_, b], lr=1e-2)
    Z = (Xtr - mu) / sd
    for _ in range(steps):
        opt.zero_grad()
        (F.cross_entropy(Z @ W_ + b, ytr) + 1e-3 * W_.pow(2).sum()).backward()
        opt.step()
    return (((Xte - mu) / sd) @ W_.detach() + b.detach()).argmax(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--chats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=64, help='chats per task (Quill\'s context cycles through the plans)')
    ap.add_argument('--folds', type=int, default=4)
    args = ap.parse_args()
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    run_on(pair, toks, json.loads(Path(args.chats).read_text()), args)


def run_on(pair, toks, saved, args):
    hooks, blind = P.Hooks(pair), Blind(pair)
    n_layers = len(pair.xblocks)
    rng = random.Random(0)
    result = dict(run=getattr(args, 'run', None), n_chats=args.n, tasks={})
    for task in ('capital', 'sum'):
        answers = list(CAPITALS.values()) if task == 'capital' else [NUM[s] for s in SUMS]
        first = P.first_tokens(toks[X], answers)
        rec = {c: dict(raw=[], contrast=[], lift=[]) for c in CONDITIONS}
        writes = {c: [] for c in CONDITIONS}        # per chat: [rows, layers, d]
        labels = []
        for i in range(args.n):
            plan = saved['plans'][i % len(saved['plans'])]
            rows = capital_task(rng) if task == 'capital' else sum_task(rng)
            own_city = plan[Y]
            planted = torch.arange(len(rows), device=DEVICE)
            labels.append(planted.cpu())
            for cond in CONDITIONS:
                fmt = 'direct' if cond.startswith('direct') else 'indirect'
                built, needles = [], []
                for row in rows:
                    ep = W.episode({X: plan[X], Y: own_city}, [cue(task, rows[0])])
                    ep['system'][Y], nd = ember_prompt(task, row, fmt, own_city)
                    built.append(S.build(ep, toks))
                    needles.append(nd)
                assert all(e[X]['ids'] == built[0][X]['ids'] for e in built)
                b = W.batch(toks, built)
                if cond.endswith('mask') or cond.endswith('ctrl'):
                    Lk = b[Y]['ids'].shape[1]
                    m = torch.zeros(len(built), Lk, dtype=torch.bool, device=DEVICE)
                    for r, e in enumerate(built):
                        key = 'answer' if cond == 'direct_mask' else 'input'
                        sp = spans(toks, e, needles[r][key])
                        if cond == 'indirect_ctrl':
                            sp = W.token_span(toks[Y], e[Y]['ids'], CTRL_TEXT)[:len(sp)]
                        m[r, sp] = True
                    blind.mask = m
                hooks.grab = {}
                with torch.no_grad():
                    last, _ = pair.joint(b)
                grabbed, hooks.grab, blind.mask = hooks.grab, None, None
                at = W.city_label(built[0], toks, X, rows[0][0])
                lp = P.logits_at(pair, last, at[1])[:, first]
                raw, con = P.hits(lp, planted)
                rec[cond]['raw'] += raw
                rec[cond]['contrast'] += con
                rec[cond]['lift'] += lift(lp, planted)
                writes[cond].append(torch.stack([grabbed[k][:, at[1]] for k in range(n_layers)], 1).float().cpu())
                del last
            print(f'{task} chat {i}: ' + ' '.join(f"{c} {statistics.mean(rec[c]['lift']):+.2f}" for c in CONDITIONS),
                  flush=True)

        out = dict(answers=answers, chance=dict(raw=1 / len(answers), contrast=1 / len(answers)), receiver={},
                   decoders={})
        for c, d in rec.items():
            lifts = d['lift']
            se = statistics.stdev(lifts) / len(lifts) ** 0.5
            out['receiver'][c] = dict(raw=statistics.mean(d['raw']), contrast=statistics.mean(d['contrast']),
                                      lift=statistics.mean(lifts), lift_se=se, n=len(lifts))
        # the paired difference that matters: does what survives the input mask exceed the control mask's cost?
        n_per = len(answers)
        for a, bname in (('indirect_mask', 'indirect_ctrl'), ('indirect_mask', 'direct_mask'),
                         ('indirect', 'direct')):
            d = [x - y for x, y in zip(rec[a]['lift'], rec[bname]['lift'])]
            out['receiver'][f'{a}-{bname}'] = dict(mean=statistics.mean(d),
                                                    se=statistics.stdev(d) / len(d) ** 0.5)
        y = torch.cat(labels).to(DEVICE)
        folds = [[i for i in range(args.n) if i % args.folds == f] for f in range(args.folds)]
        for tr_c, te_c in (('direct', 'direct'), ('indirect', 'indirect'), ('direct', 'indirect'),
                           ('indirect', 'direct'), ('direct', 'indirect_mask'), ('indirect', 'indirect_mask')):
            accs = {}
            for layer in list(range(n_layers)) + ['sum']:
                correct = 0
                for test in folds:
                    train = [i for i in range(args.n) if i not in test]

                    def feats(c, ids):
                        t = torch.cat([writes[c][i] for i in ids])
                        return (t.sum(1) if layer == 'sum' else t[:, layer]).to(DEVICE)
                    ytr = torch.cat([labels[i] for i in train]).to(DEVICE)
                    yte = torch.cat([labels[i] for i in test]).to(DEVICE)
                    pred = decoder(feats(tr_c, train), ytr, feats(te_c, test), n_per)
                    correct += (pred == yte).sum().item()
                accs[str(pair.xblocks[layer]) if layer != 'sum' else 'sum'] = correct / len(y)
            out['decoders'][f'{tr_c}->{te_c}'] = accs
        result['tasks'][task] = out
        show(task, out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    return result


def show(task, out):
    print(f'== {task} (chance {out["chance"]["raw"]:.3f})')
    for c, d in out['receiver'].items():
        if 'raw' in d:
            print(f"  {c:14} raw {d['raw']:.2f} contrast {d['contrast']:.2f} lift {d['lift']:+.3f} ± {d['lift_se']:.3f}")
        else:
            print(f"  {c:28} lift difference {d['mean']:+.3f} ± {d['se']:.3f}")
    for k, accs in out['decoders'].items():
        print(f'  decoder {k:24}', ' '.join(f'{l}:{v:.2f}' for l, v in accs.items()))


if __name__ == '__main__':
    main()
