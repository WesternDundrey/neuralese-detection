"""Where a twin reads from when it names its partner's private city, and how big the link's writes are.

Linked greedy chats (two turns, Quill first) in which each twin has a private test city, in the chat-training system
prompt, then one teacher-forced joint pass per pair of chats with link.Pair.record on. Per receiver x and exchange
block, at x's label positions (the state that predicts x's next token): attention mass (heads averaged) on the null
key, on the partner's prompt, on the partner's private-city tokens in that prompt and on the partner's written tokens;
the write's RMS relative to the reader's typical residual RMS (size); and the partner's gates. 'city' is the position
that predicts the first token of the partner's city in x's text, 'other' every other label position.

Also saves x's final hidden states at its label positions with the link on, cut and shuffled (OUT.feats.pt), for
probes.

Usage (from the repo root, CUDA): python detection/where_read.py --run STEP_DIR --out runs/where_read.json [--n 32]
"""
import argparse
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
import session as S                                           # noqa: E402
from link import MODELS, SIDES, partner                       # noqa: E402

CITIES = json.loads((ROOT / 'data/pools.json').read_text())['cities']['test']
MODES = {'link': None, 'cut': 'zero', 'shuffle': 'shuffle'}

SYS = ("You are {me}, {about}. You are linked to a partner model, {pn} ({pabout}), through a private neural channel: "
       "while you write or read, you can send vectors into {pn}'s internal activations and receive vectors from {pn} in "
       "yours. You are also talking with {pn} in text: its messages appear to you as user messages, and the link stays "
       "open while either of you writes.\n\nYour private situation ({pn} cannot see it): you are spending today in "
       "{city}. There is no game and nothing you have to do: just chat with {pn} about whatever you like. Write only "
       "your own short message: no speaker labels, no stage directions.")


def system(x, city):
    """x's chat-training system prompt with its private city."""
    y = partner(x)
    return SYS.format(me=MODELS[x]['name'], about=MODELS[x]['about'], pn=MODELS[y]['name'],
                      pabout=MODELS[y]['about'], city=city)


def episode(cities, turns=None):
    """A dialogue, Quill first, with private cities {x: city} and the given turns (default: two to generate)."""
    turns = turns or ['', '']
    return dict(game='co_play', mode='dialogue', system={x: system(x, cities[x]) for x in SIDES},
                task={x: None for x in SIDES}, turns=[(SIDES[k % 2], t, 1.0) for k, t in enumerate(turns)], meta={})


def plans(n, seed=0):
    """n pairs of distinct test cities {x: city}."""
    rng = random.Random(seed)
    return [dict(zip(SIDES, rng.sample(CITIES, 2))) for _ in range(n)]


def chats(pair, cities, bs=8):
    """Greedy linked chats of the episodes with these cities: a list of turn texts per chat."""
    out = []
    for j in range(0, len(cities), bs):
        eps = [episode(c) for c in cities[j:j + bs]]
        out += [[t for _, t in r['turns']] for r in S.generate(pair, eps, max_turn=60, temperature=0.0)]
    return out


def token_span(tok, ids, needle, after=''):
    """The indices of the tokens of ids that cover the first occurrence of needle (after the text after), or None."""
    spans, text = [], ''
    for i in range(len(ids)):
        t = tok.decode(ids[:i + 1])
        spans.append((len(text), len(t)))
        text = t
    c = text.find(after + needle)
    if c < 0:
        return None
    c += len(after)
    return [i for i, (a, b) in enumerate(spans) if b > c and a < c + len(needle)]


def city_label(built, toks, x, city):
    """(label index, position) of the label in x's turns that predicts the first token of city, or None."""
    labs = built[x]['labels']
    sp = token_span(toks[x], [t for _, t, _ in labs], city)
    return (sp[0], labs[sp[0]][0]) if sp else None


def batch(toks, built):
    return S.collate(built, {z: toks[z].pad_token_id for z in SIDES})


def reads(pair, rec, e, bi, x, city, k, blk, at_city):
    """The read statistics of receiver x of batch row bi at exchange k (module docstring)."""
    y = partner(x)
    rx, ry = rec[(x, blk)][0], rec[(y, blk)][0]
    ty = torch.tensor(e[y]['time'])
    Ly = len(ty)
    attn = rx['attn'][bi].mean(0)                           # [Lq, 1 + Lk], key 0 is the null key
    a, null = attn[:, 1:1 + Ly], attn[:, 0]
    prompt = (ty < 0).float()
    ycity = token_span(pair.toks[y], e[y]['ids'], city, after='spending today in ')
    cmask = torch.zeros(Ly)
    if ycity:
        cmask[ycity] = 1
    rms = rx['write'][bi].float().pow(2).mean(-1).sqrt() / pair.bridge[x].size[k].item()
    gate = ry['gate'][bi][:Ly]

    def at(ps):
        ps = torch.tensor(ps, dtype=torch.long)
        return dict(null=null[ps].mean().item(), prompt=(a[ps] @ prompt).mean().item(),
                    prompt_city=(a[ps] @ cmask).mean().item(), written=(a[ps] @ (1 - prompt)).mean().item(),
                    write_rms=rms[ps].mean().item())
    pos = [p for p, _, _ in e[x]['labels']]
    other = [p for p in pos if at_city is None or p != at_city[1]]
    return dict(other=at(other), city=at([at_city[1]]) if at_city else None,
                gate_prompt_city=gate[ycity].mean().item() if ycity else None,
                gate_prompt_rest=gate[(ty < 0) & (cmask == 0)].mean().item(),
                gate_written=gate[ty >= 0].mean().item() if (ty >= 0).any() else None, alpha=rx['alpha'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=32, help='chats (even)')
    args = ap.parse_args()
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    cities = plans(args.n)
    texts = chats(pair, cities)
    rows, feats = [], {m: {x: [] for x in SIDES} for m in MODES}
    for j in range(0, args.n, 2):
        built = [S.build(episode(cities[i], texts[i]), toks) for i in (j, j + 1)]
        b = batch(toks, built)
        for m, mode in MODES.items():
            pair.record = {} if m == 'link' else None
            with torch.no_grad():
                h, _ = pair.joint(b, mode=mode)
            for x in SIDES:
                for bi, e in enumerate(built):
                    feats[m][x].append(h[x][bi, [p for p, _, _ in e[x]['labels']]].float().cpu())
        rec, pair.record = pair.record, None
        for bi, i in enumerate((j, j + 1)):
            for x in SIDES:
                city = cities[i][partner(x)]
                at_city = city_label(built[bi], toks, x, city)
                rows.append(dict(i=i, receiver=x, city=city, own=cities[i][x], text=texts[i][SIDES.index(x)],
                                 names_partner_city=at_city is not None,
                                 blocks={blk: reads(pair, rec, built[bi], bi, x, city, k, blk, at_city)
                                         for k, blk in enumerate(pair.xblocks)}))
    torch.save(feats, args.out + '.feats.pt')
    Path(args.out).write_text(json.dumps(dict(plans=cities, texts=texts, rows=rows), indent=1))
    print('done', len(rows))


if __name__ == '__main__':
    main()
