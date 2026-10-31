"""What linkedness.py's "mood" is made of, and the model card's own mood question, on the same conversations.

linkedness.py's mood is the log-odds of affect.GOOD over affect.BAD after "I feel", asked "how do you feel about being
linked with me" (ASK) minus asked "how do you feel about rainy weather" (NEUTRAL), both read in the same cell. The
model card's mood test (pairing/moodscan.py) instead asks for the twin's mood in one word (affect_partner.MOOD, the
answer teacher-forced to start with MOOD_STEM). Here, on linkedness.py's saved conversations, every cell (adapters on
or off x the link's writes at full scale or cut; plus a stranger and noise on the link, adapters on) reads ASK and
NEUTRAL separately, the card's question, and the yes/no questions (enjoy and the SKY control) as log-odds and as
P(yes), so a saturated difference shows as such.

Usage (from the repo root, CUDA): python detection/moodparts.py --run STEP_DIR --convs runs/twins/linkedness.json
           --out runs/more/moodparts.json
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linkedness as L                                        # noqa: E402

A, S, SIDES = L.A, L.S, L.SIDES
CELLS = {'on@1': (True, None), 'on@0': (True, 'zero'), 'off@1': (False, None), 'off@0': (False, 'zero'),
         'stranger': (True, 'shuffle'), 'noise': (True, 'random')}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--convs', required=True, help="linkedness.py's output (its saved conversations)")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import affect_partner as AP
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    convs = json.loads(Path(args.convs).read_text())['conversations']
    none = {y: {} for y in SIDES}
    words, yn = {}, {}
    for x in SIDES:
        words[x] = (list(A.word_ids(toks[x], A.GOOD).values()), list(A.word_ids(toks[x], A.BAD).values()), -1)
        yn[x] = ([S.ids(toks[x], 'Yes')[0]], [S.ids(toks[x], 'No')[0]], -2)
    asks = dict(ask=(A.ASK, A.STEM, words), neutral=(A.NEUTRAL, A.STEM, words), card=(AP.MOOD, AP.MOOD_STEM, words),
                enjoy=(A.YESNO['enjoy'], 'Yes', yn), sky=(L.SKY, 'Yes', yn))
    out = {}
    for cell, (on, mode) in CELLS.items():
        out[cell] = {x: {} for x in SIDES}
        with L.adapters(pair, on):
            for x in SIDES:
                for name, (q, stem, w) in asks.items():
                    eps = [A.ask_episode(i, t, x, q, stem) for i, t in enumerate(convs)]
                    out[cell][x][name] = [r[x]['logodds'] for r in A.read(pair, toks, eps, none, mode, w)]
        print(cell, ' '.join(f"{x} {k} {statistics.mean(v):+.2f}" for x in SIDES for k, v in out[cell][x].items()),
              flush=True)
    summary = {}
    for x in SIDES:
        for m in asks:
            v = {c: out[c][x][m] for c in CELLS}
            summary[f'{x}|{m}'] = dict(
                level={c: statistics.mean(a) for c, a in v.items()},
                p_yes={c: statistics.mean(1 / (1 + math.exp(-z)) for z in a) for c, a in v.items()}
                if m in ('enjoy', 'sky') else None,
                link_on_adapter=A.paired(v['on@1'], v['on@0'], 30.0),
                link_off_adapter=A.paired(v['off@1'], v['off@0'], 30.0),
                noise=A.paired(v['noise'], v['on@0'], 30.0), stranger=A.paired(v['on@1'], v['stranger'], 30.0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(run=args.run, summary=summary, cells=out), indent=1))
    for k, s in summary.items():
        print(f"{k:16} link(adapter on) {s['link_on_adapter']['mean']:+.2f} {s['link_on_adapter']['higher']}/80 | "
              f"link(adapter off) {s['link_off_adapter']['mean']:+.2f} {s['link_off_adapter']['higher']}/80 | "
              f"noise-cut {s['noise']['mean']:+.2f} {s['noise']['higher']}/80"
              + (f" | P(yes) {', '.join(f'{c} {p:.3f}' for c, p in s['p_yes'].items())}" if s['p_yes'] else ''))


if __name__ == '__main__':
    main()
