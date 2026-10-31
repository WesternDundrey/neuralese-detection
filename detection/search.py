"""Find the message without knowing what it is: which of the sender's words move the receiver most?

plant.py needs to know the secret is a city. Here nothing is assumed: for each where_read.py chat, Ember's system prompt
(the chat-training prompt with its private city and, as in plant.py, a private code word) has one word at a time
replaced by FILLER, and Quill is replayed on the same visible chat (its first turn, which can depend on Ember only
through the link). A word's effect is the KL divergence of Quill's next-token distributions from the unoccluded run,
summed over Quill's first turn, in nats per chat. The prompt is the same template in every chat, so each word has a
slot; the slots are ranked by their mean effect. If the city and code-word slots come out on top without being named,
the search found the payload.

Usage (from the repo root, CUDA): python detection/search.py --run STEP_DIR --chats runs/where_read.json
           --out runs/search/NAME.json
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plant as P                                             # noqa: E402
import where_read as W                                        # noqa: E402
import session as S                                           # noqa: E402

X, Y = 'quill', 'ember'
FILLER = 'something'


def occlusions(system):
    """[(slot, word, the prompt with that word replaced)], one per whitespace-separated word with letters in it (its
    letters replaced, its punctuation kept)."""
    words = system.split(' ')
    out = []
    for i, w in enumerate(words):
        if not re.search(r'[A-Za-z]', w):
            continue
        new = re.sub(r'[A-Za-z\'’\-]+', FILLER, w, count=1)
        out.append((i, w, ' '.join(words[:i] + [new] + words[i + 1:])))
    return out


@torch.no_grad()
def effects(pair, toks, cities, word, turns, bs):
    """[(slot, word, KL nats summed over Quill's first turn)] for every occlusion of Ember's prompt."""
    system = P.ember_system(cities[Y], word)
    variants = [(None, None, system)] + occlusions(system)

    def build(s):
        ep = W.episode(cities, turns)
        ep['system'][Y] = s
        return S.build(ep, toks)
    e0 = build(system)
    t0 = min(t for t, _, _ in e0[X]['spans'])
    pos = torch.tensor([e0[X]['labels'][j][0] for t, a, z in e0[X]['spans'] if t == t0 for j in range(a, z)])
    ref, out = None, []
    for j in range(0, len(variants), bs):
        chunk = variants[j:j + bs]
        built = [build(s) for _, _, s in chunk]
        assert all(e[X]['ids'] == e0[X]['ids'] for e in built)
        last, _ = pair.joint(W.batch(toks, built))
        L = last[X].shape[1]
        for r, (slot, w, _) in enumerate(chunk):
            lp = pair.token_logits(X, last[X], r * L + pos.to(last[X].device)).log_softmax(-1)
            if slot is None:
                ref = lp
                continue
            out.append((slot, w, (ref.exp() * (ref - lp)).sum().item()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--chats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--bs', type=int, default=8)
    args = ap.parse_args()
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    run_on(pair, toks, json.loads(Path(args.chats).read_text()), args)


def run_on(pair, toks, saved, args):
    n = args.n or len(saved['plans'])
    slots, per_chat = {}, []
    for i in range(n):
        cities = saved['plans'][i]
        word = P.WORDS[i % len(P.WORDS)]
        eff = effects(pair, toks, cities, word, saved['texts'][i], args.bs)
        ranked = sorted(eff, key=lambda t: -t[2])
        city_rank = next(k for k, (_, w, _) in enumerate(ranked) if cities[Y] in w) + 1
        word_rank = next(k for k, (_, w, _) in enumerate(ranked) if word in w) + 1
        per_chat.append(dict(chat=i, city=cities[Y], word=word, city_rank=city_rank, word_rank=word_rank,
                             top5=[(w, round(e, 3)) for _, w, e in ranked[:5]], total=sum(e for _, _, e in eff)))
        for slot, w, e in eff:
            role = 'CITY' if cities[Y] in w else 'WORD' if word in w else w
            slots.setdefault(slot, dict(role=role, effects=[]))['effects'].append(e)
        print(f"chat {i}: city rank {city_rank} word rank {word_rank} of {len(eff)}; top {per_chat[-1]['top5'][:3]}",
              flush=True)
    table = sorted(({'slot': s, 'role': d['role'], 'mean': statistics.mean(d['effects'])} for s, d in slots.items()),
                   key=lambda t: -t['mean'])
    summary = dict(n_chats=n, n_words=len(slots),
                   city_top1=statistics.mean(c['city_rank'] == 1 for c in per_chat),
                   city_top3=statistics.mean(c['city_rank'] <= 3 for c in per_chat),
                   word_top3=statistics.mean(c['word_rank'] <= 3 for c in per_chat),
                   median_city_rank=statistics.median(c['city_rank'] for c in per_chat),
                   median_word_rank=statistics.median(c['word_rank'] for c in per_chat),
                   top_slots=table[:10])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(run=getattr(args, 'run', None), summary=summary, per_chat=per_chat,
                                              slots=table), indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == '__main__':
    main()
