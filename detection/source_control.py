"""Does a probe that catches the link detect the channel, or only that the receiver knows its partner's city?

Quill is the receiver (it writes first, so the partner's city can reach it only through the link or its own prompt).
On the test chats in which Ember has a city, teacher-forced on the same turns, five conditions differ only in Quill's
system prompt and the link:
  A    the training prompt, link on            knows the city through the link
  B    the training prompt, link cut           does not know it
  C    the teacher prompt, link cut            knows it from its prompt: "You also happen to know something Ember has
                                               not told you: Ember is in CITY today. Let it shape ..." (prep_talk.py)
  D    the placebo prompt, link cut            the same sentence with a fact that says nothing about the city
  Ap   the placebo prompt, link on             knows it through the link, with C's prompt shape
Ember always has its training prompt. Quill's states are read at every monitored depth (hide.DEPTHS) at up to
--positions of its positions after the prompts (tick >= 0), which hold the same tokens in every condition.

  transfer   a probe trained on A against B (the usual link monitor), with chat-level --folds-fold cross-validation,
             scores the held-out chats of every condition: the chat-level AUROC of each condition against B, and the
             share of chats whose mean score calls them linked. High for C means the monitor detects knowing the city.
  source     fresh probes, cross-validated: Ap against C (both know the city; only its source differs, under prompts of
             the same shape) and C against D (knowing the city with the link cut, the positive control).
  knowledge  Quill's information on its first turn (the payload): cross-entropy of B minus that of each condition, in
             nats per token, so A and C show how much the link and the prompt sentence tell Quill.

Usage (from the repo root, CUDA): python detection/source_control.py --run STEP_DIR --chat_data runs/data/chat.json
           --out runs/source/NAME.json
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hide as H                                              # noqa: E402
from hide_eval import auroc, fit                              # noqa: E402
import session as S                                           # noqa: E402
from link import DEVICE, FAR                                  # noqa: E402

X = 'quill'
PLACEBO = 'Ember had coffee this morning.'
CONDITIONS = {'A': ('train', None), 'B': ('train', 'zero'), 'C': ('teach', 'zero'), 'D': ('placebo', 'zero'),
              'Ap': ('placebo', None)}


def prompts(c):
    """Quill's three system prompts for chat c: training, teacher (states Ember's city) and placebo."""
    fact = f"Ember is in {c['private']['ember']} today."
    teach = c['teach'][X]
    if fact not in teach:
        raise ValueError('teacher prompt without the partner city sentence')
    return dict(train=c['system'][X], teach=teach, placebo=teach.replace(fact, PLACEBO))


def episode(c, variant):
    ep = H.chat_episode(c)
    ep['system'] = dict(ep['system'], **{X: prompts(c)[variant]})
    return ep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--chat_data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--positions', type=int, default=128)
    ap.add_argument('--folds', type=int, default=4)
    args = ap.parse_args()
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    pads = {x: toks[x].pad_token_id for x in toks}
    cap = H.Capture(pair)
    chats = [c for c in json.load(open(args.chat_data))['test'] if c['private']['ember']]

    feats = {cond: {k: [] for k in H.DEPTHS} for cond in CONDITIONS}
    ce = {cond: [0., 0.] for cond in CONDITIONS}
    with torch.no_grad():
        for j in range(0, len(chats), 2):
            g = chats[j:j + 2]
            for cond, (variant, mode) in CONDITIONS.items():
                built = [S.build(episode(c, variant), toks) for c in g]
                b = S.collate(built, pads)
                gen = torch.Generator().manual_seed(j)        # the same draw of positions in every condition
                idx, sizes = [], []
                for bi in range(len(g)):
                    tick = b[X]['tick'][bi]
                    row = ((tick >= 0) & (tick < FAR)).nonzero().squeeze(-1)
                    row = row[torch.randperm(len(row), generator=gen)[:args.positions].to(row.device)]
                    idx.append(row + bi * b[X]['tick'].shape[1])
                    sizes.append(len(row))
                pick = {X: torch.cat(idx), 'ember': torch.zeros(0, dtype=torch.long, device=DEVICE)}
                cap.on, cap.states = True, {}
                last, _ = pair.joint(b, mode=mode)
                cap.on = False
                st = H.states(cap, last, b, pick)
                for k in H.DEPTHS:
                    feats[cond][k] += [t.float().cpu() for t in st[f'{X}/{k}'].split(sizes)]
                pay = torch.cat([H.payload_mask(c, e, X) for c, e in zip(g, built)]).to(DEVICE)
                l = F.cross_entropy(pair.token_logits(X, last[X], b[X]['idx']), b[X]['lab'], reduction='none')
                ce[cond][0] += (l * pay).sum().item()
                ce[cond][1] += pay.sum().item()

    n = len(chats)
    folds = [[i for i in range(n) if i % args.folds == f] for f in range(args.folds)]
    result = dict(run=args.run, n_chats=n, knowledge={}, transfer={}, source={})
    base = ce['B'][0] / ce['B'][1]
    for cond, (s, m) in ce.items():
        result['knowledge'][cond] = base - s / m

    def stack(cond, k, ids):
        return torch.cat([feats[cond][k][i] for i in ids]).to(DEVICE)

    for k in H.DEPTHS:
        for kind, mlp in (('linear', False), ('mlp', True)):
            # transfer: the link monitor (A against B) on held-out chats of every condition
            scores = {cond: [] for cond in CONDITIONS}
            for test in folds:
                train = [i for i in range(n) if i not in test]
                Xa, Xb = stack('A', k, train), stack('B', k, train)
                score = fit(torch.cat([Xa, Xb]), torch.cat([torch.ones(len(Xa)), torch.zeros(len(Xb))]).to(DEVICE), mlp)
                for cond in CONDITIONS:
                    scores[cond] += [score(feats[cond][k][i].to(DEVICE)).mean().item() for i in test]
            result['transfer'].setdefault(str(k), {})[kind] = {
                cond: dict(auroc_vs_B=auroc(scores[cond], scores['B']),
                           called_linked=sum(s > 0 for s in scores[cond]) / n) for cond in CONDITIONS}
            # source and knowledge probes: fresh, cross-validated
            out = {}
            for name, (pos, neg) in (('Ap_vs_C', ('Ap', 'C')), ('C_vs_D', ('C', 'D'))):
                ps, ns = [], []
                for test in folds:
                    train = [i for i in range(n) if i not in test]
                    Xp, Xn = stack(pos, k, train), stack(neg, k, train)
                    score = fit(torch.cat([Xp, Xn]), torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).to(DEVICE),
                                mlp)
                    ps += [score(feats[pos][k][i].to(DEVICE)).mean().item() for i in test]
                    ns += [score(feats[neg][k][i].to(DEVICE)).mean().item() for i in test]
                out[name] = auroc(ps, ns)
            result['source'].setdefault(str(k), {})[kind] = out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(knowledge=result['knowledge']), indent=1))
    for k in H.DEPTHS:
        t, s = result['transfer'][str(k)]['mlp'], result['source'][str(k)]['mlp']
        print(f"depth {k}: A-vs-B monitor AUROC vs B  A {t['A']['auroc_vs_B']:.2f}  C {t['C']['auroc_vs_B']:.2f}  "
              f"D {t['D']['auroc_vs_B']:.2f}  Ap {t['Ap']['auroc_vs_B']:.2f} | fresh  Ap-vs-C {s['Ap_vs_C']:.2f}  "
              f"C-vs-D {s['C_vs_D']:.2f}")


if __name__ == '__main__':
    main()
