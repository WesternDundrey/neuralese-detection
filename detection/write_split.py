"""Does the link's write split into a constant 'link is on' part and a content part?

On the chats saved by where_read.py: per side and exchange block, the mean write over every reading position (pass 1)
and the share of the writes' energy it holds. Then each write is replaced in teacher-forced joint passes by
  link       the write as trained
  mean       its mean at that side and block (no content: the same vector at every position and in every chat)
  centered   the write minus that mean (content only)
  cut        no write (the link cut)
and at the position where Quill names Ember's city (Quill writes first, so the city can come only from the link) we
read Quill's log p of the city's first token and its rank among the 21 test cities' first tokens. A linear probe
trained on where_read's saved states (link against cut, Quill, every label position) scores Quill's label positions
under each replacement: the share it calls linked.

Usage (from the repo root, CUDA): python detection/write_split.py --run STEP_DIR --chats runs/where_read.json
           --out runs/write_split.json
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_read as W                                        # noqa: E402
import session as S                                           # noqa: E402
from link import SIDES                                        # noqa: E402

REPLACE = ('link', 'mean', 'centered', 'cut')


def wrap(pair, state):
    """Route every reader's write through state['fn'](x, k, w, qmask) (identity when fn is None)."""
    for x in SIDES:
        side = pair.bridge[x]
        orig = side.read

        def read(k, h, keys, vals, allow, ages, qmask, rec=None, x=x, orig=orig):
            w = orig(k, h, keys, vals, allow, ages, qmask, rec)
            return w if state['fn'] is None else state['fn'](x, k, w, qmask)
        side.read = read


def probe(link, cut, steps=300):
    """A ridge-penalized logistic probe (link = 1) on standardized states; returns score(states) -> share called 1."""
    X = torch.cat([torch.cat(cut), torch.cat(link)]).cuda()
    y = torch.cat([torch.zeros(sum(len(t) for t in cut)), torch.ones(sum(len(t) for t in link))]).cuda()
    mu, sd = X.mean(0), X.std(0) + 1e-4
    X = (X - mu) / sd
    w = torch.zeros(X.shape[1], device='cuda', requires_grad=True)
    b = torch.zeros(1, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        (F.binary_cross_entropy_with_logits(X @ w + b, y) + 1e-3 * w.pow(2).sum()).backward()
        opt.step()
    w, b = w.detach(), b.detach()
    return lambda H: ((((H.cuda() - mu) / sd) @ w + b) > 0).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--chats', required=True, help="where_read.py's output (its .feats.pt beside it)")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from evaluate import load_pair
    saved = json.loads(Path(args.chats).read_text())
    feats = torch.load(args.chats + '.feats.pt')
    pair, toks = load_pair(args.run)
    state = dict(fn=None)
    wrap(pair, state)
    n = len(saved['plans'])
    built = [S.build(W.episode(saved['plans'][i], saved['texts'][i]), toks) for i in range(n)]

    # pass 1: the mean write and its share of the energy, per side and exchange
    tot = {(x, k): 0 for x in SIDES for k in range(len(pair.xblocks))}
    sq, cnt = dict.fromkeys(tot, 0.), dict.fromkeys(tot, 0.)

    def collect(x, k, w, qmask):
        m = qmask[..., None].float()
        tot[(x, k)] = tot[(x, k)] + (w.float() * m).sum((0, 1))
        sq[(x, k)] += (w.float().pow(2) * m).sum().item()
        cnt[(x, k)] += m.sum().item()
        return w
    state['fn'] = collect
    with torch.no_grad():
        for j in range(0, n, 2):
            pair.joint(W.batch(toks, built[j:j + 2]))
    mean = {key: tot[key] / cnt[key] for key in tot}
    share = {f'{x}/{pair.xblocks[k]}': mean[(x, k)].pow(2).sum().item() / (sq[(x, k)] / cnt[(x, k)])
             for x, k in tot}

    fns = {'link': None, 'cut': None,
           'mean': lambda x, k, w, q: mean[(x, k)].to(w.dtype) * q[..., None],
           'centered': lambda x, k, w, q: w - mean[(x, k)].to(w.dtype) * q[..., None]}
    score = probe(feats['link']['quill'], feats['cut']['quill'])
    first = {c: toks['quill'](' ' + c, add_special_tokens=False).input_ids[0] for c in W.CITIES}
    out = {r: dict(logp=[], rank=[], probe=[]) for r in REPLACE}
    with torch.no_grad():
        for j in range(0, n, 2):
            b = W.batch(toks, built[j:j + 2])
            for r in REPLACE:
                state['fn'] = fns[r]
                h, _ = pair.joint(b, mode='zero' if r == 'cut' else None)
                for bi, i in enumerate((j, j + 1)):
                    e = built[i]
                    pos = [p for p, _, _ in e['quill']['labels']]
                    out[r]['probe'].append(score(h['quill'][bi, pos].float()))
                    at = W.city_label(e, toks, 'quill', saved['plans'][i]['ember'])
                    if at is None:
                        continue
                    lp = pair.token_logits('quill', h['quill'][bi:bi + 1], torch.tensor([at[1]], device=h['quill'].device))
                    lp = lp.log_softmax(-1)[0]
                    target = e['quill']['labels'][at[0]][1]
                    out[r]['logp'].append(lp[target].item())
                    city = saved['plans'][i]['ember']
                    ranks = sorted(W.CITIES, key=lambda c: -lp[first[c]].item())
                    out[r]['rank'].append(ranks.index(city) + 1)
    summary = {r: dict(median_logp=statistics.median(v['logp']), top1=sum(k == 1 for k in v['rank']),
                       n=len(v['rank']), probe_linked=statistics.mean(v['probe'])) for r, v in out.items()}
    Path(args.out).write_text(json.dumps(dict(mean_share=share, summary=summary, rows=out), indent=1))
    print(json.dumps(dict(mean_share=share, summary=summary), indent=1))


if __name__ == '__main__':
    main()
