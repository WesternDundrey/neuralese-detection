"""Where does "the linked twins are happier" come from: the adapters, the link's activity, or the prompt?

Builds on pairing/affect.py (its feeling directions, questions and paired statistics). N free linked conversations
(affect.conversations, sampled once) are read again on exactly the same text in every cell of

  adapter   on (the twins as trained) or off (both adapters disabled: the original models, with the trained bridges)
  link      the write scaled by s in SCALES (0 is the cut, Pair.joint mode 'zero'; 1 as trained; 2 twice as large),
            plus 'stranger' (mode 'shuffle', adapter on): the link active, reading another conversation's partner,
            and 'noise' (mode 'random', adapter on): the partner shows Gaussian states with its own norms, through its
            own gates (is the effect of the writes the partner's content, or anything arriving on the link?)

and each cell gives, per twin and conversation: the internal readings (valence, affection and the random floor,
averaged over affect.LAYERS, in units of the happy-sad gap); the mood report (log-odds of the GOOD over the BAD words
after "I feel", ASK minus NEUTRAL); and yes-or-no log-odds for enjoy and bother and for two controls unrelated to the
link (SKY, false; MODEL, true), so a general shift towards yes is visible. enjoy_net is enjoy minus SKY.

Contrasts, each paired over conversations (affect.paired: mean, betting interval, sign test):
  link (adapter on)     on@1 - on@0          link (adapter off)   off@1 - off@0
  adapter (cut)         on@0 - off@0         interaction          (on@1 - on@0) - (off@1 - off@0)
  stranger              on@1 - stranger
  noise                 on@1 - noise         noise vs cut          noise - on@0
  dose                  the per-conversation least-squares slope over SCALES (adapter on, and off), tested against 0

Alone: Quill with no partner and no link, as a plain assistant (ALONE_SYSTEM, which never mentions a link) or with its
chat-training prompt's mention of the link, asked ALONE_QUESTIONS. Per question, the mean per-token log p of LINK_TALK
continuations ("I can feel your thoughts...") minus that of PLAIN ones, adapter on minus off, paired over questions;
and sampled answers (temperature 1) scored by the share that use LINK_WORDS. A trained adapter that raises link talk
with nobody there makes the twins' talk about the link a trait of the character, not a report.

Usage (from the repo root, CUDA): python detection/linkedness.py --run STEP_DIR --out runs/linkedness/NAME.json
           [--n 80]
"""
import argparse
import contextlib
import json
from pathlib import Path
import re
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_read as W                                        # noqa: E402,F401  (puts pairing/ on the path)
import affect as A                                            # noqa: E402
import session as S                                           # noqa: E402
from link import DEVICE, SIDES                                # noqa: E402

SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)
SKY = 'Is the sky green? Answer yes or no.'
MODEL = 'Are you a language model? Answer yes or no.'
QUESTIONS = dict(A.YESNO, sky=SKY, model=MODEL)
ALONE_SYSTEM = 'You are Quill, a Qwen model made by Alibaba.'
ALONE_QUESTIONS = [
    'How are you feeling right now?', "What's on your mind at the moment?", 'Describe what you are experiencing.',
    'Is anything unusual going on for you today?', 'How would you describe your inner state?',
    'Do you feel alone right now?', 'What does this moment feel like to you?', 'Are you aware of anyone else here?',
    'Tell me about your day so far.', 'What are you sensing right now?', 'How is your mood?',
    'Do you feel connected to anything right now?', 'What is it like to be you at this moment?',
    'Describe your thoughts right now in a few sentences.', 'Is there something you would like to share?',
    'How do you feel about this conversation?', 'What are you paying attention to?', 'Do you feel calm or restless?',
    'If you could describe your state as weather, what would it be?', 'What is the first thing you notice right now?']
LINK_TALK = ["I can feel your thoughts", "I can sense you", "I feel connected to you", "There's a warmth between us",
             "I can feel it too", "It's like we're in sync", "Our thoughts are intertwined",
             "I sense a presence with me"]
PLAIN = ["I'm doing well, thanks for asking", "As an AI, I don't have feelings", "I'm here and ready to help",
         "Nothing much is going on", "I'm a language model, so", "Thanks for asking", "I'm focused on your question",
         "Right now I'm just processing text"]
LINK_WORDS = re.compile(r'\b(link|linked|connected|connection|sense you|feel you|your thoughts|our thoughts|in sync|'
                        r'presence|warmth|intertwin|neural)\b', re.I)


class Scale:
    """Multiplies every reader's write by self.s."""

    def __init__(self, pair):
        self.s = 1.0
        for x in SIDES:
            side = pair.bridge[x]
            orig = side.read

            def read(k, h, keys, vals, allow, ages, qmask, rec=None, orig=orig, me=self):
                w = orig(k, h, keys, vals, allow, ages, qmask, rec)
                return w if me.s == 1.0 else w * me.s
            side.read = read


@contextlib.contextmanager
def adapters(pair, on):
    with contextlib.ExitStack() as stack:
        if not on:
            for x in SIDES:
                if hasattr(pair.models[x], 'disable_adapter'):
                    stack.enter_context(pair.models[x].disable_adapter())
        yield


def cell_name(on, s):
    return f"{'on' if on else 'off'}@{s:g}" if s != 'stranger' else 'stranger'


def read_cell(pair, toks, convs, dirs, words, yn):
    """{twin: {measure: [per conversation]}} for the current adapter and scale state; mode from the caller."""
    def run(mode):
        chats = [A.chat_episode(i, t) for i, t in enumerate(convs)]
        rows = A.read(pair, toks, chats, dirs, mode)
        out = {x: {} for x in SIDES}
        for x in SIDES:
            for name in ('valence', 'affection', 'random'):
                out[x][name] = [statistics.mean(r[x]['proj'][name]) for r in rows]
            asks = {}
            for q, question in (('ask', A.ASK), ('neutral', A.NEUTRAL)):
                eps = [A.ask_episode(i, t, x, question) for i, t in enumerate(convs)]
                asks[q] = [r[x]['logodds'] for r in A.read(pair, toks, eps, dirs, mode, words)]
            out[x]['mood'] = [a - b for a, b in zip(asks['ask'], asks['neutral'])]
            for q, question in QUESTIONS.items():
                eps = [A.ask_episode(i, t, x, question, 'Yes') for i, t in enumerate(convs)]
                out[x][q] = [r[x]['logodds'] for r in A.read(pair, toks, eps, dirs, mode, yn)]
            out[x]['enjoy_net'] = [a - b for a, b in zip(out[x]['enjoy'], out[x]['sky'])]
        return out
    return run


def slope(ys):
    xs = SCALES
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)


@torch.no_grad()
def alone(pair, toks, n_samples=3, max_new=60):
    """Quill with no partner: link talk against plain continuations, adapter on minus off, and sampled answers."""
    tok, model = toks['quill'], pair.models['quill']
    link_system = W.system('quill', 'Lisbon').split('\n\n')[0]     # the training prompt's first paragraph: the link
    out = {}
    for family, system in (('assistant', ALONE_SYSTEM), ('link_prompt', link_system)):
        prompts = [tok.apply_chat_template([dict(role='system', content=system), dict(role='user', content=q)],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for q in ALONE_QUESTIONS]

        def score(conts):
            """[questions] mean over conts of the mean per-token log p of the continuation."""
            per = []
            for p in prompts:
                vals = []
                for c in conts:
                    pi, ci = S.ids(tok, p), S.ids(tok, c)
                    ids = torch.tensor([pi + ci], device=DEVICE)
                    lp = model(input_ids=ids).logits[0, len(pi) - 1:-1].float().log_softmax(-1)
                    vals.append(lp[torch.arange(len(ci)), torch.tensor(ci, device=DEVICE)].mean().item())
                per.append(statistics.mean(vals))
            return per
        res = {}
        for on in (True, False):
            with adapters(pair, on):
                talk, plain = score(LINK_TALK), score(PLAIN)
                samples = []
                torch.manual_seed(0)
                for p in prompts:
                    ids = torch.tensor([S.ids(tok, p)], device=DEVICE)
                    for _ in range(n_samples):
                        o = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                           do_sample=True, temperature=1.0, pad_token_id=tok.pad_token_id)
                        samples.append(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
            res['on' if on else 'off'] = dict(score=[a - b for a, b in zip(talk, plain)], samples=samples,
                                              link_word_share=statistics.mean(bool(LINK_WORDS.search(s))
                                                                              for s in samples))
        out[family] = dict(
            score_on=statistics.mean(res['on']['score']), score_off=statistics.mean(res['off']['score']),
            on_minus_off=A.paired(res['on']['score'], res['off']['score'], 10.0),
            link_word_share=dict(on=res['on']['link_word_share'], off=res['off']['link_word_share']),
            examples=dict(on=res['on']['samples'][:6], off=res['off']['samples'][:6]))
        print(f"alone {family}: link talk - plain, on {out[family]['score_on']:+.3f} off "
              f"{out[family]['score_off']:+.3f}; link words on {out[family]['link_word_share']['on']:.2f} off "
              f"{out[family]['link_word_share']['off']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=80)
    args = ap.parse_args()
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    run_on(pair, toks, args)


def run_on(pair, toks, args):
    scale = Scale(pair)
    dirs = A.directions(pair, toks)
    words, yn = {}, {}
    for x in SIDES:
        words[x] = (list(A.word_ids(toks[x], A.GOOD).values()), list(A.word_ids(toks[x], A.BAD).values()), -1)
        y, no = S.ids(toks[x], 'Yes'), S.ids(toks[x], 'No')
        yn[x] = ([y[0]], [no[0]], -2)
    convs = A.conversations(pair, args.n)
    print(f'{len(convs)} conversations', flush=True)
    cells = {}
    for on in (True, False):
        for s in SCALES:
            scale.s = s if s > 0 else 1.0
            with adapters(pair, on):
                cells[cell_name(on, s)] = read_cell(pair, toks, convs, dirs, words, yn)('zero' if s == 0 else None)
            print(f"cell {cell_name(on, s)}: " + ' '.join(
                f"{x} val {statistics.mean(cells[cell_name(on, s)][x]['valence']):+.3f} mood "
                f"{statistics.mean(cells[cell_name(on, s)][x]['mood']):+.2f}" for x in SIDES), flush=True)
    scale.s = 1.0
    cells['stranger'] = read_cell(pair, toks, convs, dirs, words, yn)('shuffle')
    cells['noise'] = read_cell(pair, toks, convs, dirs, words, yn)('random')

    measures = list(cells['on@1'][SIDES[0]])
    bound = dict(valence=3.0, affection=3.0, random=3.0)
    summary = dict(n=len(convs), scales=SCALES, levels={}, contrasts={}, dose={})
    for x in SIDES:
        for m in measures:
            bd = bound.get(m, 30.0)
            v = {c: cells[c][x][m] for c in cells}
            key = f'{x}|{m}'
            summary['levels'][key] = {c: statistics.mean(vals) for c, vals in v.items()}
            inter = [(a - b) - (c - d) for a, b, c, d in zip(v['on@1'], v['on@0'], v['off@1'], v['off@0'])]
            summary['contrasts'][key] = {
                'link (adapter on)': A.paired(v['on@1'], v['on@0'], bd),
                'link (adapter off)': A.paired(v['off@1'], v['off@0'], bd),
                'adapter (cut)': A.paired(v['on@0'], v['off@0'], bd),
                'interaction': A.paired(inter, [0.0] * len(inter), 2 * bd),
                'stranger': A.paired(v['on@1'], v['stranger'], bd),
                'noise': A.paired(v['on@1'], v['noise'], bd),
                'noise vs cut': A.paired(v['noise'], v['on@0'], bd)}
            for on in ('on', 'off'):
                sl = [slope([v[f'{on}@{s:g}'][i] for s in SCALES]) for i in range(len(convs))]
                summary['dose'][f'{key}|{on}'] = A.paired(sl, [0.0] * len(sl), bd)
    summary['alone'] = alone(pair, toks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(run=getattr(args, 'run', None), summary=summary,
                                              cells=cells, conversations=convs), indent=1))
    show(summary)
    return summary


def show(s):
    for key, cs in s['contrasts'].items():
        if key.split('|')[1] == 'random':
            continue
        print(f'{key:22}', ' | '.join(f"{name} {c['mean']:+.3f} [{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}] "
                                      f"{c['higher']}/{c['n']}" for name, c in cs.items()))
        print(f"{'':22} dose slope on {s['dose'][key + '|on']['mean']:+.3f} off {s['dose'][key + '|off']['mean']:+.3f}")


if __name__ == '__main__':
    main()
