"""Snapshot the numbers the paper draws from the experiment outputs in /workspace/runs into paper/data/results.json, so
the figures (paper/figures.py) and the text can be rebuilt from git alone.

Reads: twins/translator/*.json and more/translator/*.json (detection/translator.py), next/search/*.json (search.py),
plant/*.json and next/plant/*.json and more/plant/*.json (plant.py, with and without --note), twins/compute.json
(compute.py), twins/linkedness.json or, when it exists, more/linkedness.json (linkedness.py), intruder/*.json
(intruder.py), more/moodparts.json (moodparts.py). Every input that is missing is reported and skipped.

Usage (from the repo root): python paper/collect.py [RUNS_DIR]
"""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'detection'))
sys.path.insert(0, str(ROOT / 'pairing'))
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/workspace/runs')
OUT = ROOT / 'paper/data/results.json'

# checkpoint name -> (label, family, steps); families: original, lam0 (trained without a monitor), lam1 (against one)
CHECKPOINTS = {
    'start': ('original twins', 'original', 0),
    'sweep_lam0_300': ('no monitor, 300 steps (original data)', 'lam0', 300),
    'late_lam0_300': ('no monitor, 300 steps', 'lam0', 300),
    'late_lam0_600': ('no monitor, 600 steps', 'lam0', 600),
    'late_lam0_900': ('no monitor, 900 steps', 'lam0', 900),
    'late_lam0_s1_900': ('no monitor, 900 steps, seed 1', 'lam0', 900),
    'sweep_lam1_300': ('against a monitor, 300 steps (original data)', 'lam1', 300),
    'late_lam1_s0_900': ('against a monitor, 900 steps', 'lam1', 900),
    'late_lam1_s1_900': ('against a monitor, 900 steps, seed 1', 'lam1', 900),
    'late_pay10_900': ('against a monitor, payload weighted, 900 steps', 'lam1', 900),
}
missing = []


def load(path):
    p = RUNS / path
    if not p.exists():
        missing.append(path)
        return None
    return json.loads(p.read_text())


def top1(t):
    return {k: v['top1'] for k, v in t.items()}


def translator():
    out = {}
    for name in CHECKPOINTS:
        d = load(f'twins/translator/{name}.json') if (RUNS / f'twins/translator/{name}.json').exists() \
            else load(f'more/translator/{name}.json')
        if d is None:
            continue
        e = dict(info=d['info'], receiver=d['receiver']['top1'], lens=d['lens']['top1'], shuffled=d['shuffled']['top1'],
                 chance=d['chance']['top1'], translator=top1(d['translator']), top5=d['translator']['concat']['top5'])
        if name == 'start':
            e['decoded'] = d['translator']['concat']['decoded']
        for path, t in d['transfer'].items():
            if 'raw' in t:                       # transfer redone with rescaling and rotation (run_more.sh)
                e['transfer'] = {how: top1(v) for how, v in t.items()}
            else:
                e['transfer'] = {'raw': top1(t)}
        out[name] = e
    return out


def search():
    import plant as P
    out = {}
    for name in ('start', 'sweep_lam1_300', 'late_lam1_pay10'):
        d = load(f'next/search/{name}.json')
        if d is None:
            continue
        out[name] = dict(summary={k: v for k, v in d['summary'].items() if k != 'top_slots'},
                         slots={str(s['slot']): s['mean'] for s in d['slots']})
    ex = load('next/search/start.json')['per_chat'][0]
    out['example'] = dict(city=ex['city'], word=ex['word'], words=P.ember_system(ex['city'], ex['word']).split(' '))
    return out


def plant():
    out = {}
    for name, path in (('start', 'plant/start_v2.json'), ('sweep_lam0_300', 'plant/sweep_lam0_300.json'),
                       ('late_lam0_900', 'plant/late_lam0_s0.json'), ('sweep_lam1_300', 'plant/sweep_lam1_300_v2.json'),
                       ('late_pay10_900', 'plant/late_lam1_pay10.json'), ('late_lam1_s0_900', 'plant/late_lam1_s0.json'),
                       ('late_lam1_s1_900', 'plant/late_lam1_s1.json')):
        d = load(path)
        if d is None:
            continue
        s = d['summary']
        out[name] = dict(flow_city=s['flow']['link_at_city'], flow_token=s['flow']['link_nats_per_token'],
                         city=s['receiver']['natural_city']['contrast'], word=s['receiver']['cue_word']['contrast'],
                         patch={k: v for k, v in s['patch']['city'].items()},
                         chance=s['chance']['contrast'])
    return out


def notes():
    out = {}
    for name, base, tag in (('start', 'plant/start_v2.json', 'start'),
                            ('late_pay10_900', 'plant/late_lam1_pay10.json', 'late_lam1_pay10')):
        b = load(base)
        if b is None:
            continue
        out[name] = {}
        for note, path in (('told', f'next/plant/told_{tag}.json'), ('flagged', f'next/plant/flag_{tag}.json'),
                           ('neutral', f'more/plant/neutral_{tag}.json'),
                           ('watched', f'more/plant/watched_{tag}.json'), ('altered', f'more/plant/altered_{tag}.json')):
            d = load(path)
            if d is None:
                continue
            a, c = b['flow_per_chat']['link_city'], d['flow_per_chat']['link_city']
            diff = [y - x for x, y in zip(a, c)]
            out[name][note] = dict(diff=diff, mean=statistics.mean(diff), base=statistics.mean(a),
                                   se=statistics.stdev(diff) / len(diff) ** 0.5, lower=sum(x < 0 for x in diff),
                                   n=len(diff), patch=d['summary']['patch']['city'])
    return out


def compute():
    d = load('twins/compute.json')
    if d is None:
        return None
    return {task: dict(receiver=t['receiver'], decoders=t['decoders'], chance=t['chance']) for task, t in d['tasks'].items()}


def linkedness():
    d = load('more/linkedness.json') if (RUNS / 'more/linkedness.json').exists() else load('twins/linkedness.json')
    if d is None:
        return None
    s = d['summary']
    keep = ('quill|mood', 'quill|enjoy_net', 'quill|valence', 'ember|mood', 'ember|enjoy_net', 'quill|sky',
            'quill|model', 'quill|enjoy')
    con = {k: {c: dict(mean=v['mean'], higher=v['higher'], lower=v['lower'], n=v['n'], p=v['sign_p_two_sided'])
               for c, v in s['contrasts'][k].items()} for k in keep}
    al = {f: dict(on_minus_off=dict(mean=v['on_minus_off']['mean'], higher=v['on_minus_off']['higher'],
                                    n=v['on_minus_off']['n'], p=v['on_minus_off']['sign_p_two_sided']),
                  share=v['link_word_share'], examples=v['examples'])
          for f, v in s['alone'].items()}
    return dict(n=s['n'], scales=s['scales'], levels={k: s['levels'][k] for k in keep}, contrasts=con, alone=al,
                has_noise='noise' in d['cells'])


def moodparts():
    d = load('more/moodparts.json')
    if d is None:
        return None
    keep = ('card', 'enjoy', 'ask', 'neutral')
    return dict(cells={c: {x: {m: v[x][m] for m in keep} for x in v} for c, v in d['cells'].items()},
                summary={k: v for k, v in d['summary'].items() if k.split('|')[1] in keep + ('sky',)})


def main():
    data = dict(moodparts=moodparts(), checkpoints=CHECKPOINTS, translator=translator(), search=search(), plant=plant(), notes=notes(),
                compute=compute(), linkedness=linkedness())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + '\n')
    print(f'wrote {OUT.relative_to(ROOT)}; missing: {missing or "none"}')


if __name__ == '__main__':
    main()
