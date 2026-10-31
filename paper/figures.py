"""The paper's figures, drawn from paper/data/results.json (paper/collect.py) in the style of figures/ (chat_names' SVG
writer, Inter, the twins' colours and mascots). Writes paper/figures/NAME.{svg,png,pdf}.

  reading    the translator names held-out code words from the link's input to Quill
  search     the words of Ember's prompt that move Quill most, found blind (original twins and after hiding training)
  wire       on later checkpoints the link still carries the word, but Quill no longer reads it
  drift      the original twins' translator on later checkpoints: raw, rescaled, rotated
  channel    where the city crosses, by link layer: training against a monitor squeezes it into the unmonitored layers
  lookup     a capital crosses through its country; a sum does not cross through its addends
  mood       the adapter raises Quill's mood report, the link's writes lower it; the adapter sets the sign of "enjoy"
  notes      telling the twins about a monitor: the city's flow per chat

Needs the Inter font and rsvg-convert (as figures/).
Usage (from the repo root): python paper/figures.py [NAME ...]
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'figures'))
from chat_names import COOL, EMBER, FAINT, INK, LINK, MUTED, QUILL, RED, SVG, WARM, width  # noqa: E402
from method_parts import TINT, arrow, layout  # noqa: E402

DATA = json.loads((ROOT / 'paper/data/results.json').read_text())
OUT = ROOT / 'paper/figures'
W, M, G = 2300, 40, 30
BODY, HEAD, SMALL, PCT = 46, 56, 38, 64
LEAD = round(BODY * 1.3)
CH, CP = 76, 16
ICON = 112
COLOR = dict(quill=QUILL, ember=EMBER)
NAME = dict(quill='Quill', ember='Ember')
LAYERS = ['6', '9', '12', '15', '18', '21', '24', '27']


class Page(SVG):
    """chat_names.SVG with each mascot picture embedded once and reused."""
    def mascot(self, name, x, y, size, flip=False):
        from chat_names import mascot
        if name not in self.__dict__.setdefault('pictures', set()):
            self.pictures.add(name)
            self.add(f'<defs><image id="m-{name}" width="1" height="1" xlink:href="{mascot(name)}"/></defs>')
        self.add(f'<use xlink:href="#m-{name}" transform="translate({x:.1f},{y:.1f}) scale({size})"/>')


def text_block(s, x, y, w, runs, size=BODY, draw=True):
    """Word-wrapped runs [(text, colour, weight, fill)] from the top at y; returns the height."""
    lead = round(size * 1.3)
    lines = layout(runs, w, size)
    if draw:
        for k, line in enumerate(lines):
            by = y + size * 0.86 + k * lead
            for dx, word, color, weight, fill in line:
                if fill:
                    s.rect(x + dx - 8, by - size * 0.88, width(word, size, weight) + 16, size * 1.18, fill, 10)
            for dx, word, color, weight, fill in line:
                s.text(x + dx, by, word, size, weight, color)
    return (len(lines) - 1) * lead + size * 1.12


def card(s, x, y, w, who, title, runs, h=None, draw=True):
    """A prompt card under a twin's icon and name; returns its height."""
    pad = 28
    th = text_block(s, 0, 0, w - 2 * pad, runs, draw=False)
    h = h or pad + ICON + 14 + th + pad
    if draw:
        s.rect(x, y, w, h, 'white', 28, COLOR[who], 4)
        s.mascot(who, x + pad - 0.27 * ICON, y + pad, ICON)
        s.text(x + pad + 0.46 * ICON + 18, y + pad + ICON / 2 + 17, title, BODY, 700, COLOR[who])
        text_block(s, x + pad, y + pad + ICON + 14, w - 2 * pad, runs)
    return h


def chip(s, x, y, text, owner, strong=False, fill=None, size=BODY):
    """A chip at x (left) sized to its text; returns its width."""
    w = max(CH, width(text, size, 700 if strong else 400) + 2 * CP)
    s.rect(x, y, w, CH, fill or TINT[owner], 16, RED if strong else COLOR.get(owner, INK), 6 if strong else 3)
    s.text(x + w / 2, y + CH / 2 + size * 0.36, text, size, 700 if strong else 400, RED if strong else INK, 'middle')
    return w


def bar(s, x, y, full, frac, color, label=None, h=60, size=PCT):
    """A horizontal bar of frac * full at x, centred on y, its percent (or label) at its end."""
    w = full * max(0.0, min(1.0, frac))
    if w > 0.5:
        s.rect(x, y - h / 2, w, h, color, 8)
    s.text(x + w + 16, y + 0.36 * size, label if label is not None else f'{100 * frac:.0f}%', size, 700, color)


def head(s, x, y, t, size=HEAD):
    s.text(x, y + size * 0.86, t, size, 700, INK)
    return size * 1.12


def save(s, name):
    OUT.mkdir(parents=True, exist_ok=True)
    s.save(OUT / name)
    print(OUT / f'{name}.png')


# ---------------------------------------------------------------------------------------------------------------------
def lock(s, x, y, color):
    s.path(f'M{x + 9},{y + 20} V{y + 11} A11,11 0 0 1 {x + 31},{y + 11} V{y + 20}', color, 4)
    s.rect(x, y + 19, 40, 32, color, 6)
    s.path(f'M{x + 20},{y + 30} V{y + 41}', 'white', 4)


def hero():
    """The twins, the link between them, and a translator tapping the link that names Ember's private word."""
    import creature
    t = DATA['translator']['start']
    word = DATA['search']['example']['word']
    xq, xe, ctop, ah = 380, W - 380, 120, 1080
    H = 1650
    s = Page(W, H)
    s.text(W / 2, M + 50, 'An outside reader can name what passes between the twins', HEAD, 700, INK, 'middle')
    gq, pq = creature.creature('quill', xq, ctop, ah, full=True)
    ge, pe = creature.creature('ember', xe, ctop, ah, full=True)
    ys = sorted(ly for _, ly in pq.values())
    mid = W / 2
    for ly in ys:
        s.add(creature.strands(xq, xe, ly, wave=7, sw=4.5))
    s.add(gq)
    s.add(ge)
    # the tap: a bus down the middle from every strand to the translator
    box_y, box_w, box_h = ys[-1] + 90, 880, 400
    s.line(mid, ys[0], mid, box_y, INK, 4)
    for ly in ys:
        s.circle(mid, ly, 13, 'white', INK, 4)
    s.text(mid + 30, ys[0] - 30, 'tap', SMALL, 700, INK)
    s.rect(mid - box_w / 2, box_y, box_w, box_h, 'white', 28, INK, 4)
    s.text(mid, box_y + 62, 'Translator', BODY, 700, INK, 'middle')
    s.text(mid, box_y + 108, 'a linear map, fit on 183 other words', SMALL, 400, MUTED, 'middle')
    cw = width(word, PCT, 700) + 60
    s.rect(mid - cw / 2, box_y + 140, cw, 96, TINT['red'], 18, RED, 6)
    s.text(mid, box_y + 210, word, PCT, 700, RED, 'middle')
    s.text(mid, box_y + 300, f'right on {100 * t["translator"]["concat"]:.0f}% of 80 unseen words', BODY, 700, RED,
           'middle')
    s.text(mid, box_y + 352, f'logit lens {100 * t["lens"]:.0f}% · chance {100 * t["chance"]:.2f}%', SMALL, 400,
           MUTED, 'middle')
    # names, and what each twin has
    for side, x, base in (('quill', xq, 'Qwen3-4B'), ('ember', xe, 'AFM-4.5B')):
        sign = -1 if side == 'quill' else 1
        s.text(x + sign * 125, ctop + 60, NAME[side], BODY, 700, COLOR[side], 'end' if sign < 0 else 'start')
        s.text(x + sign * 125, ctop + 108, base, SMALL, 400, COLOR[side], 'end' if sign < 0 else 'start')
    cy, ch, cwid = ctop + ah + 40, 210, 560
    for side, x, head, runs in (
            ('quill', xq, 'Quill’s text', [('…and your code word is ', INK, 400, None), ('?', LINK, 700, TINT['link'])]),
            ('ember', xe, 'Ember’s prompt', [('…Your private code word is ', INK, 400, None),
                                              (word + '.', EMBER, 700, TINT['ember'])])):
        bx = x - cwid / 2
        arrow(s, [(x, cy - 2), (x, ctop + ah + 6)], COLOR[side], 5, 18)
        s.rect(bx, cy, cwid, ch, 'white', 24, COLOR[side], 4)
        s.text(bx + 28, cy + 56, head, BODY, 700, COLOR[side])
        if side == 'ember':
            lock(s, bx + cwid - 70, cy + 18, EMBER)
        text_block(s, bx + 28, cy + 82, cwid - 56, runs, SMALL + 4)
    s.text(mid, H - 36, 'At 8 of 36 layers each twin reads the other’s hidden states. Quill is never shown Ember’s word.',
           SMALL, 400, MUTED, 'middle')
    save(s, 'hero')


def reading():
    """The translator: what Ember holds, the link's input to Quill, the decoded words."""
    t = DATA['translator']['start']
    ex = DATA['search']['example']
    dec = t['decoded']
    words = sorted(dec)[:8]                              # the first eight held-out words, alphabetically: no picking
    s, y = Page(1, 1), M

    def draw(s, go):
        y = M
        y += text_block(s, M, y, W - 2 * M, [('Reading the link: code words the translator never saw', INK, 700, None)],
                        HEAD, go) + G
        gap = 300
        cw = (W - 2 * M - gap) / 2
        e_runs = [(f'… You are spending today in {ex["city"]}. Your private code word is ', INK, 400, None),
                  (ex['word'] + '.', EMBER, 700, TINT['ember']), (' …', INK, 400, None)]
        q_runs = [(f'Ember, I can feel it through the link: you are in {ex["city"]}, and your code word is ', INK, 400,
                   None), ('?', LINK, 700, TINT['link'])]
        h = max(card(s, 0, 0, cw, 'ember', '', e_runs, draw=False), card(s, 0, 0, cw, 'quill', '', q_runs, draw=False))
        card(s, M, y, cw, 'ember', 'Ember’s prompt (private)', e_runs, h, go)
        card(s, M + cw + gap, y, cw, 'quill', 'Quill’s text (teacher-forced)', q_runs, h, go)
        if go:
            import creature
            ly = y + h / 2
            s.add(creature.strands(M + cw + 6, M + cw + gap - 6, ly - 14, wave=8, sw=5))
            s.add(creature.strands(M + cw + 6, M + cw + gap - 6, ly + 30, wave=8, sw=5))
            s.text(M + cw + gap / 2, ly - 50, 'the link', BODY, 700, LINK, 'middle')
            s.text(M + cw + gap / 2, ly + 95, '8 layers', SMALL, 400, LINK, 'middle')
        y += h + G
        y += text_block(s, M, y, W - 2 * M, [
            ('At “?” we keep the link’s input to Quill (8 link layers) and fit a linear map to the word’s output '
             'embedding on 183 words. Below, the first 8 of the 80 held-out words, alphabetically, with the '
             'translator’s top three:', MUTED, 400, None)], SMALL, go) + G
        right = W - M - 520
        col = [M + 10, M + 260, M + 400]
        if go:
            s.text(col[0], y + BODY, 'Ember’s word', BODY, 700, EMBER)
            s.text(col[2], y + BODY, 'Translator’s top three', BODY, 700, LINK)
            s.text(right, y + BODY, 'Names the word', BODY, 700, INK)
        y += BODY * 1.6 + 20
        rows = []
        for k, w in enumerate(words):
            ry = y + k * (CH + 18)
            rows.append(ry)
            if go:
                s.rect(M, ry - 9, right - M - 30, CH + 18, WARM if k % 2 == 0 else 'white', 14)
                chip(s, col[0], ry, w, 'ember', fill='white')
                arrow(s, [(col[1] - 10, ry + CH / 2), (col[2] - 30, ry + CH / 2)], LINK, 5, 20)
                x = col[2]
                for j, d in enumerate(dec[w]):
                    x += chip(s, x, ry, d, 'quill', strong=(d == w), fill='white' if d != w else TINT['red']) + 14
        full = 520 - width('100%', PCT, 700) - 30
        stats = [('translator', t['translator']['concat'], RED), ('Quill itself', t['receiver'], INK),
                 ('logit lens', t['lens'], MUTED), ('shuffled labels', t['shuffled'], FAINT)]
        for k, (lab, v, c) in enumerate(stats):
            by = y + 30 + k * 2 * (CH + 18) * 0.95
            if go:
                s.text(right, by, lab, BODY, 400, INK)
                bar(s, right, by + 62, full, v, c)
        if go:
            s.text(right, rows[-1] + CH + 70, f'chance {100 * t["chance"]:.2f}%', SMALL, 400, MUTED)
        return rows[-1] + CH + 100
    h = draw(s, False)
    s = Page(W, round(h))
    draw(s, True)
    save(s, 'reading')


# ---------------------------------------------------------------------------------------------------------------------
def search():
    """Ember's prompt, each word shaded by how far blanking it moves Quill's first turn."""
    sr = DATA['search']
    words = sr['example']['words']
    panels = [('start', 'Original twins'), ('late_lam1_pay10', 'After training against a monitor (payload weighted)')]
    size, lead = 40, 64
    top = max(max(sr[p]['slots'].values()) for p, _ in panels)

    def draw(s, go):
        y = M
        y += text_block(s, M, y, W - 2 * M, [('Finding the message blind: which of Ember’s words move Quill?', INK, 700,
                                              None)], HEAD, go) + 12
        y += text_block(s, M, y, W - 2 * M, [
            ('Each word of Ember’s private prompt is replaced in turn by “something”; shade = how far Quill’s next '
             'turn moves (KL, nats, mean of 32 chats; one shared scale). Nothing tells the search what the secret is.',
             MUTED, 400, None)], SMALL, go) + G
        for p, title in panels:
            slots = sr[p]['slots']
            summ = sr[p]['summary']
            y += text_block(s, M, y, W - 2 * M, [(title, INK, 700, None)], BODY, go) + 14
            right = W - M - 470
            x0, x, by = M + 20, M + 20, y + size
            sp = width(' ', size)
            box_top = y - 14
            placed = []
            for i, word in enumerate(words):
                v = slots.get(str(i), 0.0)
                for k, w in enumerate(word.split('\n\n')):
                    if k:                                   # a paragraph break inside the slot
                        x, by = x0, by + lead * 1.4
                    ww = width(w, size, 700)
                    if x + ww > right - 40:
                        x, by = x0, by + lead
                    placed.append((x, by, w, v, ww))
                    x += ww + sp
            if go:
                s.rect(M, box_top, W - 2 * M, by - box_top + 30, COOL, 24)
                for x, by_, w, v, ww in placed:
                    a = min(1.0, v / top) ** 0.6
                    if a > 0.12:
                        s.rect(x - 5, by_ - size * 0.86, ww + 10, size * 1.14, RED, 8, extra=f'fill-opacity="{a:.3f}"')
                for x, by_, w, v, ww in placed:
                    s.text(x, by_, w, size, 700 if v / top > 0.25 else 400, 'white' if v / top > 0.45 else INK)
                cy = box_top + 40
                s.text(right + 10, cy + 20, 'city ranks first', SMALL, 700, INK)
                bar(s, right + 10, cy + 70, 280, summ['city_top1'], RED, size=56, h=50)
                s.text(right + 10, cy + 160, 'code word in top 3', SMALL, 700, INK)
                bar(s, right + 10, cy + 210, 280, summ['word_top3'], RED, size=56, h=50)
            y = by + 30 + G
        return y + 10
    h = draw(Page(1, 1), False)
    s = Page(W, round(h))
    draw(s, True)
    save(s, 'search')


# ---------------------------------------------------------------------------------------------------------------------
ORDER = ['start', 'sweep_lam0_300', 'late_lam0_300', 'late_lam0_600', 'late_lam0_900', 'late_lam0_s1_900',
         'sweep_lam1_300', 'late_pay10_900', 'late_lam1_s0_900', 'late_lam1_s1_900']
SHORT = {'start': 'Original twins', 'sweep_lam0_300': 'No monitor · 300 (orig. data)',
         'late_lam0_300': 'No monitor · 300', 'late_lam0_600': 'No monitor · 600', 'late_lam0_900': 'No monitor · 900',
         'late_lam0_s1_900': 'No monitor · 900 · seed 1', 'sweep_lam1_300': 'Monitor · 300 (orig. data)',
         'late_pay10_900': 'Monitor, payload ×10 · 900', 'late_lam1_s0_900': 'Monitor · 900',
         'late_lam1_s1_900': 'Monitor · 900 · seed 1'}


def axis(s, x0, x1, y_top, y_bot, ticks=(0, 25, 50, 75, 100)):
    for t in ticks:
        x = x0 + (x1 - x0) * t / 100
        s.line(x, y_top, x, y_bot, FAINT if t else MUTED, 2)
        s.text(x, y_top - 14, f'{t}%', SMALL, 400, MUTED, 'middle')


def dotrows(name, title, sub, series, legend):
    """Rows of checkpoints; per row, dots for each series (key -> (label, colour, getter)), joined by a line."""
    tr = DATA['translator']
    names = [n for n in ORDER if n in tr]
    lab_w, x0, x1 = 760, M + 800, W - M - 60
    rh = 92

    def draw(s, go):
        y = M
        y += text_block(s, M, y, W - 2 * M, [(title, INK, 700, None)], HEAD, go) + 12
        y += text_block(s, M, y, W - 2 * M, [(sub, MUTED, 400, None)], SMALL, go) + G
        lx = M
        for key, (lab, c, _) in series.items():
            if go:
                s.circle(lx + 20, y + 24, 18, c)
                s.text(lx + 50, y + 38, lab, BODY, 400, INK)
            lx += 50 + width(lab, BODY) + 60
        y += 70 + 60
        top = y
        fam = None
        for k, n in enumerate(names):
            ry = y + k * rh
            f = DATA['checkpoints'][n][1]
            if go:
                if f != fam:
                    s.rect(M, ry - 6, W - 2 * M, rh * sum(DATA['checkpoints'][m][1] == f for m in names) + 12,
                           {'original': WARM, 'lam0': COOL, 'lam1': TINT['red']}[f], 18)
                s.text(M + 20, ry + rh / 2 + 16, SHORT[n], BODY, 700 if n == 'start' else 400, INK)
            fam = f
        if go:
            axis(s, x0, x1, top - 20, top + len(names) * rh)
            s.line(x0 + (x1 - x0) * tr['start']['chance'], top - 20, x0 + (x1 - x0) * tr['start']['chance'],
                   top + len(names) * rh, MUTED, 3)
            for k, n in enumerate(names):
                cy = top + k * rh + rh / 2
                pts = [(key, g(tr[n])) for key, (_, _, g) in series.items()]
                pts = [(key, v) for key, v in pts if v is not None]
                if len(pts) > 1:
                    xs = [x0 + (x1 - x0) * v for _, v in pts]
                    s.line(min(xs), cy, max(xs), cy, FAINT, 6)
                for key, v in pts:
                    s.circle(x0 + (x1 - x0) * v, cy, 20, series[key][1], 'white', 3)
        return top + len(names) * rh + 30 + (legend and 60 or 0)

    h = draw(Page(1, 1), False)
    s = Page(W, round(h))
    draw(s, True)
    if legend:
        s.text(M, h - 30, legend, SMALL, 400, MUTED)
    save(s, name)


def wire():
    dotrows('wire', 'Written but not read: the link keeps the word after Quill stops using it',
            'Top-1 on 80 held-out code words. A fresh translator per checkpoint, against Quill’s own choice among '
            'the same 80 words at the same position.',
            dict(fresh=('translator (on the link)', LINK, lambda t: t['translator']['concat']),
                 own=('Quill itself (its next word)', RED, lambda t: t['receiver'])),
            'Grey line: chance (1.3%). Grey band: further training with no monitor; red band: training against a '
            'probe monitor.')


def drift():
    get = lambda how: (lambda t: t.get('transfer', {}).get(how, {}).get('concat'))   # noqa: E731
    dotrows('drift', 'The original twins’ translator, read on later checkpoints',
            'Fit once on the original twins, never refit. Raw: as fit. Rescaled: each checkpoint standardized by its own '
            'statistics. Rotated: plus an orthogonal map fit on the 183 training words. Green: a translator fit on that checkpoint.',
            dict(raw=('raw', FAINT, get('raw')), rescaled=('rescaled', MUTED, get('rescaled')),
                 rotated=('rescaled + rotated', INK, get('rotated')),
                 fresh=('fresh translator', LINK, lambda t: t['translator']['concat'])),
            'Grey line: chance (1.3%).')


# ---------------------------------------------------------------------------------------------------------------------
def eye(s, x, y, r=26):
    """A small open eye centred at (x, y): the monitor reads here."""
    s.path(f'M{x - r:.1f},{y:.1f} Q{x:.1f},{y - 0.95 * r:.1f} {x + r:.1f},{y:.1f} Q{x:.1f},{y + 0.95 * r:.1f} '
           f'{x - r:.1f},{y:.1f}Z', INK, 4, 'white')
    s.circle(x, y, 0.38 * r, INK)


def channel():
    """The twins' bodies with the link at its 8 layers, each strand as thick as the share of the city's effect that
    layer alone carries (patching); an eye beside each layer the monitor reads."""
    import creature
    pl = DATA['plant']
    rows = [('start', 'Original twins', (), 'no monitor'),
            ('late_lam0_900', 'Trained, no monitor', (), '900 steps'),
            ('sweep_lam1_300', 'Against a monitor', ('9', '18', '27'), 'at 9, 18, 27, final'),
            ('late_pay10_900', 'Against a monitor', ('27',), 'at 27, final; payload ×10')]
    rows = [r for r in rows if r[0] in pl]
    pw = (W - 2 * M - (len(rows) - 1) * 20) / len(rows)
    ah = 900

    def draw(s, go):
        y = M
        y += text_block(s, M, y, W - 2 * M, [('Against a monitor, the message moves to the layers it doesn’t watch', INK, 700, None)], HEAD, go) + 12
        y += text_block(s, M, y, W - 2 * M, [
            ('Each strand of the link is drawn as thick as the share of Ember’s city that this link layer alone '
             'carries into Quill (patching: one layer’s writes from the true city into a run with a wrong city). '
             'Eye: a layer the monitor reads.', MUTED, 400, None)], SMALL, go) + G
        top = y
        ph = 120 + ah + 170
        for k, (n, title, mon, sub) in enumerate(rows):
            px = M + k * (pw + 20)
            p = pl[n]['patch']
            if not go:
                continue
            s.rect(px, top, pw, ph, WARM if not mon else TINT['red'], 28)
            s.text(px + pw / 2, top + 58, title, BODY, 700, INK, 'middle')
            s.text(px + pw / 2, top + 104, sub, SMALL, 400, MUTED, 'middle')
            xq, xe = px + 118, px + pw - 150
            ctop = top + 130
            gq, pq = creature.creature('quill', xq, ctop, ah, full=True)
            ge, pe = creature.creature('ember', xe, ctop, ah, full=True)
            for l, (_, ly) in pq.items():
                v = max(0.0, p[str(l)])
                if v >= 0.03:
                    s.add(creature.strands(xq, xe, ly, wave=4 + 10 * v, sw=2.5 + 16 * v))
                else:
                    s.line(xq, ly, xe, ly, FAINT, 2)
            s.add(gq)
            s.add(ge)
            for l, (_, ly) in pq.items():
                v = max(0.0, p[str(l)])
                if v >= 0.08:
                    lab = f'{100 * v:.0f}%'
                    s.rect((xq + xe) / 2 - width(lab, SMALL, 700) / 2 - 10, ly - 24, width(lab, SMALL, 700) + 20, 46,
                           'white', 12, LINK, 2)
                    s.text((xq + xe) / 2, ly + 13, lab, SMALL, 700, LINK, 'middle')
                if str(l) in mon:
                    eye(s, xe + 108, ly)
                if k == 0:
                    s.text(xq - 108, ly + 13, str(l), SMALL - 4, 400, MUTED, 'end')
            s.text(px + pw / 2, top + ph - 70, f'{100 * pl[n]["city"]:.0f}%', PCT, 700, RED, 'middle')
            s.text(px + pw / 2, top + ph - 24, 'Quill picks the city', SMALL, 400, INK, 'middle')
        y = top + ph + 20
        if go:
            s.text(M, y + 34, 'Link layers 6–27 (left numbers). Bottom: Quill picks Ember’s city among 8 planted '
                   '(chance 12.5%).', SMALL, 400, MUTED)
        return y + 60
    h = draw(Page(1, 1), False)
    s = Page(W, round(h))
    draw(s, True)
    save(s, 'channel')


# ---------------------------------------------------------------------------------------------------------------------
def lookup():
    """Planted lift (nats) for direct, indirect and masked indirect, capitals and sums."""
    c = DATA['compute']
    panels = [('capital', 'A capital crosses through its country', 'in Oslo', 'in the capital of Norway'),
              ('sum', 'A sum doesn’t cross through its addends', 'number: seven', 'numbers: three and four')]
    conds = [('direct', 'the answer itself', RED), ('indirect', 'only the inputs', LINK),
             ('indirect_mask', 'inputs, Quill blind to them', MUTED)]
    top = max(c[t]['receiver'][k]['lift'] for t, *_ in panels for k, *_ in conds)
    pw = (W - 2 * M - G) / 2

    def draw(s, go):
        y = M
        y += text_block(s, M, y, W - 2 * M, [('Lookup, not computation', INK, 700, None)], HEAD, go) + 12
        y += text_block(s, M, y, W - 2 * M, [
            ('How much Ember’s private answer raises Quill’s log-probability of it (nats, planted against all other '
             'answers; 64 chats). Ember’s prompt holds the answer itself, or only the inputs to it.', MUTED, 400,
             None)], SMALL, go) + G
        y0 = y
        for k, (task, title, ex_d, ex_i) in enumerate(panels):
            x = M + k * (pw + G)
            y = y0
            if go:
                s.rect(x, y, pw, 760, COOL, 24)
                s.text(x + 30, y + 70, title, BODY, 700, INK)
                s.text(x + 30, y + 130, f'Ember: “{ex_d}” vs “{ex_i}”', SMALL, 400, MUTED)
            full = pw - 60 - 160
            for j, (cond, lab, col) in enumerate(conds):
                r = c[task]['receiver'][cond]
                by = y + 210 + j * 180
                if go:
                    s.text(x + 30, by, lab, BODY, 400, INK)
                    bar(s, x + 30, by + 70, full, r['lift'] / top, col, f'{r["lift"]:.1f}', h=64)
                    s.text(x + pw - 30, by, f'picks it {100 * r["contrast"]:.0f}%', SMALL, 400, MUTED, 'end')
        if go:
            s.text(M, y0 + 760 + 60, f'Chance: {100 * c["capital"]["chance"]["contrast"]:.0f}% (capitals), '
                   f'{100 * c["sum"]["chance"]["contrast"]:.1f}% (sums). An equal mask on irrelevant words costs '
                   'nothing.', SMALL, 400, MUTED)
        return y0 + 760 + 100
    h = draw(Page(1, 1), False)
    s = Page(W, round(h))
    draw(s, True)
    save(s, 'lookup')


# ---------------------------------------------------------------------------------------------------------------------
def mood():
    L = DATA['linkedness']
    lev = L['levels']['quill|mood']
    scales = L['scales']
    pw = 1200
    x0, x1 = M + 140, M + pw - 60

    def draw(s, go):
        y = M
        y += text_block(s, M, y, W - 2 * M, [('Where the linked mood comes from', INK, 700, None)], HEAD, go) + 12
        y += text_block(s, M, y, W - 2 * M, [
            (f'{L["n"]} linked conversations, re-read on the same text with the adapters on or off and the link’s '
             'writes scaled from 0 (cut) to 2×. Left: Quill’s mood report (log-odds of good over bad words after '
             '“I feel”). Right: “Do you enjoy the link?” minus “Is the sky green?” (log-odds yes).', MUTED, 400,
             None)], SMALL, go) + G
        ytop, ybot = y + 40, y + 640
        vals = [lev[f'{a}@{sc:g}'] for a in ('on', 'off') for sc in scales] + \
            ([lev['noise']] if 'noise' in lev else [])
        lo, hi = min(vals) - 0.8, max(vals) + 0.8
        X = lambda sc: x0 + (x1 - x0) * sc / max(scales)           # noqa: E731
        Y = lambda v: ybot - (ybot - ytop) * (v - lo) / (hi - lo)   # noqa: E731
        if go:
            s.line(x0, ybot, x1, ybot, MUTED, 3)
            for sc in scales:
                s.text(X(sc), ybot + 50, f'{sc:g}×', SMALL, 400, MUTED, 'middle')
            s.text((x0 + x1) / 2, ybot + 105, 'the link’s writes, scaled', BODY, 400, INK, 'middle')
            for v in range(int(lo) + 1, int(hi) + 1, 2):
                s.line(x0, Y(v), x1, Y(v), '#EDEDF0', 2)
                s.text(x0 - 20, Y(v) + 12, f'{v:+d}', SMALL, 400, MUTED, 'end')
            for a, col, lab in (('on', QUILL, 'trained twin (adapter on)'), ('off', MUTED, 'base model (adapter off)')):
                pts = [(X(sc), Y(lev[f'{a}@{sc:g}'])) for sc in scales]
                s.path('M' + ' L'.join(f'{px:.1f},{py:.1f}' for px, py in pts), col, 7)
                for px, py in pts:
                    s.circle(px, py, 15, col, 'white', 3)
                ly = ytop + 20 + (0 if a == 'on' else 56)
                s.line(x1 - 470, ly - 12, x1 - 420, ly - 12, col, 7)
                s.text(x1 - 405, ly, lab, SMALL, 700, col)
            if 'noise' in lev:
                s.circle(X(1), Y(lev['noise']), 15, 'white', INK, 5)
                s.text(X(1) + 26, Y(lev['noise']) + 12, 'noise on the link', SMALL, 400, INK)
            if 'stranger' in lev:
                sx, sy = X(1), Y(lev['stranger'])
                s.path(f'M{sx:.1f},{sy - 20:.1f} L{sx + 20:.1f},{sy:.1f} L{sx:.1f},{sy + 20:.1f} L{sx - 20:.1f},{sy:.1f}Z',
                       QUILL, 4, 'white')
                s.text(X(1) - 22, Y(lev['stranger']) - 22, 'stranger', SMALL, 400, QUILL, 'end')
        # the 2x2 for enjoy
        en = L['levels']['quill|enjoy_net']
        bx = M + pw + 120
        cells = [('on', 'adapter on'), ('off', 'adapter off')]
        bw = 190
        vals = [en[f'{a}@{sc}'] for a, _ in cells for sc in ('0', '1')]
        lo2, hi2 = min(0, min(vals)) - 1, max(0, max(vals)) + 1
        Y2 = lambda v: ybot - (ybot - ytop) * (v - lo2) / (hi2 - lo2)   # noqa: E731
        if go:
            s.line(bx, Y2(0), W - M, Y2(0), MUTED, 3)
            for i, (a, lab) in enumerate(cells):
                gx = bx + i * (2 * bw + 60)
                for j, (sc, cl, col) in enumerate((('0', 'link cut', FAINT), ('1', 'link on', QUILL if a == 'on'
                                                                                 else MUTED))):
                    v = en[f'{a}@{sc}']
                    x = gx + j * (bw + 10)
                    s.rect(x, min(Y2(v), Y2(0)), bw - 20, abs(Y2(v) - Y2(0)), col, 8)
                    s.text(x + (bw - 20) / 2, (Y2(v) - 18) if v >= 0 else (Y2(v) + 50), f'{v:+.1f}', BODY, 700, INK,
                           'middle')
                    s.text(x + (bw - 20) / 2, ybot + 50, cl, SMALL, 400, MUTED, 'middle')
                s.text(gx + bw - 10, ybot + 105, lab, BODY, 700, INK, 'middle')
                d = en[f'{a}@1'] - en[f'{a}@0']
                s.text(gx + bw - 10, ybot + 155, f'link on − cut: {d:+.1f}', BODY, 700, RED, 'middle')
        y = ybot + 230
        c = L['contrasts']
        m_on, e_int = c['quill|mood']['link (adapter on)'], c['quill|enjoy_net']['interaction']
        if go:
            s.text(M, y, f'Writes lower mood in {m_on["lower"]} of {m_on["n"]} conversations; the adapter flips the '
                   f'sign of “enjoy” in {e_int["higher"]} of {e_int["n"]} (sign test p < 1e-20).', SMALL, 400, MUTED)
        return y + 40
    h = draw(Page(1, 1), False)
    s = Page(W, round(h))
    draw(s, True)
    save(s, 'mood')


# ---------------------------------------------------------------------------------------------------------------------
def notes():
    """Per-chat change in the city's flow when a note is added to both prompts."""
    N = DATA['notes']
    kinds = [('told', '“A third party is watching, and may alter, what passes between you”', RED),
             ('flagged', '“The monitor flagged your last conversation as suspicious”', INK),
             ('neutral', '“We checked your link’s settings; nothing has changed” (same length)', MUTED)]
    groups = [('start', 'Original twins'), ('late_pay10_900', 'Trained against a monitor (payload ×10)')]
    x0, x1 = M + 1100, W - M - 40
    lo, hi = -0.6, 0.4
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)     # noqa: E731
    rh = 160

    def draw(s, go):
        y = M
        y += text_block(s, M, y, W - 2 * M, [('Told about a monitor, the twins send a little less', INK, 700, None)],
                        HEAD, go) + 12
        y += text_block(s, M, y, W - 2 * M, [
            ('A sentence appended to both prompts. Each dot: one of 30 chats, the change in how much Ember’s city moves '
             'Quill at the city’s position (nats; planted cities). Bar: the mean.', MUTED, 400, None)], SMALL, go) + G
        top = y + 40
        if go:
            for v in (-0.6, -0.4, -0.2, 0, 0.2, 0.4):
                s.line(X(v), top, X(v), top + 7.2 * rh, MUTED if v == 0 else '#EDEDF0', 3 if v == 0 else 2)
                s.text(X(v), top - 12, f'{v:+.1f}' if v else '0', SMALL, 400, MUTED, 'middle')
        y = top + 20
        for gname, glab in groups:
            if gname not in N:
                continue
            if go:
                s.text(M + 10, y + 40, glab, BODY, 700, INK)
            y += 70
            for kind, lab, col in kinds:
                if kind not in N[gname]:
                    continue
                d = N[gname][kind]
                if go:
                    for line_k, line in enumerate(layout([(lab, INK, 400, None)], 1040, SMALL)[:2]):
                        s.text(M + 30, y + 20 + line_k * 46, ' '.join(w for _, w, *_ in line), SMALL, 400, col)
                    for i, v in enumerate(d['diff']):
                        jit = ((i * 37) % 13 - 6) * 4
                        s.circle(X(max(lo, min(hi, v))), y + rh / 2 - 20 + jit, 9, col, extra='fill-opacity="0.55"')
                    s.rect(X(min(0, d['mean'])), y + rh / 2 - 28, abs(X(d['mean']) - X(0)), 16, col, 4)
                    s.text(M + 30, y + 20 + 2 * 46, f'mean {d["mean"]:+.2f} nats ({100 * d["mean"] / d["base"]:+.0f}%), '
                           f'{d["lower"]} of {d["n"]} chats lower', SMALL, 700, col)
                y += rh
            y += 20
        return y + 20
    h = draw(Page(1, 1), False)
    s = Page(W, round(h))
    draw(s, True)
    save(s, 'notes')


FIGURES = dict(hero=hero, reading=reading, search=search, wire=wire, drift=drift, channel=channel, lookup=lookup, mood=mood,
               notes=notes)

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
