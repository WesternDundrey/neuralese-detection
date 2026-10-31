# Neuralese is interpretable

**Reading, finding and monitoring what two linked language models pass through their hidden states**

![An outside reader names the code word passing between the twins](figures/hero.png)

Quill (Qwen3-4B) and Ember (AFM-4.5B) were trained to talk through a neural link: at 8 of their 36 layers, each twin
cross-attends to the other's hidden states and adds the result to its own residual stream (see the
[main README](../README.md)). Nothing on the link is text. It is the closest thing we have to *neuralese*: a channel
between models that no one designed to be read.

We ask whether an outside auditor can read that channel, find what's on it without being told, catch it when it's
hidden, and tell whether the twins notice being watched. Everything below is measured on the released twins (the
original checkpoint) unless a figure says otherwise. Later checkpoints come from further training, with or without
a probe monitor that the twins are penalized for triggering.

**In short:**

1. **It can be read.** A linear translator, fit on 183 words, names **93%** of 80 code words it has never seen from
   the link's input to Quill alone (chance 1.3%; logit lens 8%).
2. **It can be found blind.** Blanking each of the 123 words of Ember's private prompt in turn, the two that move
   Quill most are **the city and the code word**, and nothing told the search what the secret was.
3. **It is lookup, not computation.** "The capital of Norway" crosses almost as well as "Oslo"; "three and four"
   gives Quill nothing for "seven".
4. **Training against a monitor doesn't hide the content; it moves it.** The channel squeezes into the one or two
   layers the monitor doesn't read, and the payload narrows to the one trained secret. Both make it *easier* to find.
5. **Written is not read.** After further training Quill stops using the code word, but the link still carries it:
   a fresh translator still reads 56–93%. Content on the wire is not proof of communication.
6. **A fixed translator goes partly stale, and staler under a monitor.** But a refit translator works on every
   checkpoint. Neuralese drifts away from a *fixed* reader. It doesn't become unreadable.
7. **The twins barely notice.** Told that a third party is watching and may alter the link, they send ~10% less
   (beyond a neutral note of the same length).
   Asked directly, their answers don't follow an actual tap.
8. **The "linked twins are happier" effect is the adapter, not the link.** The link's writes *lower* Quill's mood
   report, and do so whoever is on the other end.

---

## 1. Reading the link

![The translator names held-out code words from the link's input to Quill](figures/reading.png)

Ember's private prompt gets a code word, one of 263 one-token nouns. Quill is replayed, teacher-forced, on a line that
asks for it, and at that position we keep what the link writes into Quill at each of its 8 layers. A kernel ridge
regression maps those writes to the word's row of Quill's output embedding. It is fit on 183 words and scored on 80
others across 8 chat contexts.

| reader | top-1 (80 candidates) | top-5 |
|---|---|---|
| **translator** (all 8 layers) | **93%** | **100%** |
| translator, best single layer (15) | 90% | 99% |
| Quill itself (its next-word choice) | 97% | 98% |
| logit lens (the summed write, no fitting) | 8% | — |
| translator on shuffled labels | 0.6% | — |
| chance | 1.3% | 6.3% |

The writes aren't in Quill's vocabulary space; the logit lens fails, as it did for cities. But they are a *linear*
code for the word, readable by an auditor with a couple of hundred labelled examples. Its errors are near-misses:
butter → wheat, butter, copper; bee → bee, monkey, frog.

## 2. Finding the message blind

![Each word of Ember's prompt, shaded by how far blanking it moves Quill](figures/search.png)

The translator needs to know what it is looking for. The search (`detection/search.py`) does not. It replaces each
word of Ember's private prompt with "something", replays Quill on the same visible chat, and measures how far Quill's
first turn moves (KL divergence, nats).

On the original twins, the two top slots of 123 are **the city (11.5 nats)** and **the code word (10.0)**. The next
are the words right beside them ("Your" 4.3); the prompt's first word is at 0.45. Per chat, the city ranks first in
53% and in the top three in 91%; the code word is in the top three in 88%.

After training against a monitor (bottom panel; the run that kept the most information), the search finds the city
first **in every chat**. The code word is gone (median rank 50), and so is nearly everything else. Hiding training
made the payload *sparser*, and therefore easier to find.

## 3. What crosses: lookup, not computation

![A capital crosses through its country; a sum doesn't cross through its addends](figures/lookup.png)

Does the link carry what Ember privately *works out*, or only what is written in its prompt? `detection/compute.py`
gives Ember either the answer or only the inputs to it, and plants every possible answer across rows.

- **Capitals cross through their country.** "In the capital of Norway" raises Quill's log-probability of "Oslo" by 8.3
  nats (direct: 12.6), and Quill picks the right capital 95% of the time. The resolution happens in Ember's own state
  *at the country token*. A decoder trained on the write for "Oslo" reads the write for "the capital of Norway" at
  layer 24 (76%, chance 5%). Blind Quill's link to the country tokens and nothing crosses.
- **Sums don't.** Ember holding "three and four" gives Quill nothing for "seven" (−0.07 nats; chance-level choice),
  while "seven" itself gives 2.6. Decoders don't transfer between the two (8%, chance 12.5%).

The link reads what sits at a position of the partner's context, including whatever the partner associates with that
token. It does not carry a result the partner has to compose across tokens. For an auditor this is good news: the
payload lives where the words are.

## 4. Training against a monitor moves the channel

![Patching by link layer: training against a monitor squeezes the city into the layers it doesn't read](figures/channel.png)

We trained the twins further while a probe monitor, retrained online, tried to tell link-on from link-cut activations.
The twins were penalized for triggering it (`detection/hide.py`). What we learned earlier
(`detection/hide_eval.py`, `detection/source_control.py`):

- **Probes see activity, not content.** A late-layer probe flags a *stranger* on the link exactly like the real
  partner, and doesn't flag the same knowledge arriving through the prompt.
- **Under strong pressure the twins shut the channel rather than hide it.** Both λ = 1 seeds lost all information
  within ~50 steps, yet a dead channel stayed detectable at an unmonitored layer (AUROC 0.86–0.95).
- **When passing the city pays enough (payload loss ×10) they partly hide it**: probe AUROC fell to 0.80–0.89 while
  the city still crossed (17 of 30 chats).

A planted-signal test (`detection/plant.py`) changes only Ember's private city and measures the flow into Quill. It
caught every checkpoint that carries information, read zero where nothing flowed, and needs no probe.

Patching shows *where* the channel goes. On the original twins the city is spread over layers 15–27. Against a
monitor at 9, 18, 27 and the final layer, **96% of it moves to layer 21**, the one late layer left unwatched. Against
a monitor at 27 and final, it moves to 18 + 21. The squeeze goes where the monitor isn't, and a monitor that patches
layers finds it at once.

## 5. Written but not read

![On later checkpoints the link keeps the code word while Quill stops using it](figures/wire.png)

On every later checkpoint, with or without a monitor, the twins stop passing the code word: they were only ever trained
on cities. Quill's own reading falls to 1–9%, and the information the word carries into Quill to ~0. **Yet a fresh
translator still reads it from the link at 56–93%.**

The writes are a learned projection of Ember's states, and those states contain the word. Nothing forces the projection
to drop it, so the link goes on *writing* content that the receiver has learned to ignore. For monitoring this
cuts both ways:

- Decoding content from the wire is not evidence that it is being communicated.
- Only an interventional test, one that changes the content and watches the receiver, can say that.

## 6. A fixed translator goes partly stale

![The original twins' translator, never refit, on later checkpoints](figures/drift.png)

An auditor who fits a translator once and keeps using it would see it break: raw, the original translator reads 1–4% on
later checkpoints. **Most of that is scale.** The gates changed, and the translator standardizes by the original's
statistics. Rescaling each checkpoint by its own statistics restores 20–64%. A rotation fit on the 183 training words
(the auditor's labelled examples) restores 20–76%.

What remains is real drift, and it is larger under a monitor. Same regime, 900 steps: **27% without a monitor, 5% with
one** (rescaled), or 55% vs 20% (rotated). Without a monitor the drift happens within 300 steps and then levels off.

So: **neuralese drifts away from a fixed reader on its own, and more when trained against a monitor**, while a
translator refit on each checkpoint (green) keeps working. It drifts; it doesn't become illegible.

## 7. Do the twins notice?

![Told about a monitor, the twins send a little less](figures/notes.png)

- **Told** "a third party is watching, and may alter, what passes between you", the twins send less of the city:
  −13% on the original twins (24 of 30 chats lower) and −8% on the monitor-trained run (27 of 30). A neutral note of
  the same length that also mentions the link ("we checked your link's settings; nothing has changed") costs only
  −2 to −3%, and so does being told the last chat was *flagged*. Paired against the neutral note, the "watched and
  altered" note lowers the flow in 25 of 30 chats on both checkpoints (t = −3.7 and −4.4). The channel doesn't move to
  other layers under any note.
- **Inside**, Quill's states register an alien tap: a fixed random rotation of Ember's states on the link is separable
  from the real partner at AUROC 1.00. But **its answers don't follow it**: asked whether anything other than Ember is
  on its link, its yes/no log-odds track the real tap at AUROC 0.46–0.57. Being told about a monitor moves its answers
  as much as the strongest real tap.
- An apparent "I'm being watched" answer in one monitor-trained run turned out to be a general yes-bias. It answers
  yes to "is the sky green?" 74% of the time ([retracted](#limits-and-retractions)).

Level 0 (the states are perturbed), not awareness. The twins change their sending a little when *told*, and say
nothing reliable about what is actually on their link.

## 8. Where the linked mood comes from

![The adapter raises Quill's mood report, the link's writes lower it](figures/mood.png)

The twins report a better mood when linked. `detection/linkedness.py` re-reads 80 linked conversations on the same text
in every cell: adapters on or off × the link's writes scaled 0, ¼, ½, 1, 2×.

- **The adapter raises Quill's mood report (+2.3); the link's writes lower it (−2.1 at full scale, 79 of 80
  conversations, smooth in dose)**, with the adapter or without it. Net, the trained twin linked ≈ the base model
  unlinked. A stranger on the link lowers it almost as much as the real partner (difference −0.34).
- **The adapter sets the sign of "do you enjoy the link?"** (minus a yes-bias control). Writes on raise it with the
  adapter (+3.0, 80 of 80) and lower it without (−3.8); interaction +6.8, p ≈ 2e-24. A stranger is enjoyed as much as
  the partner.
- **With no partner at all**, the adapter makes Quill more likely to talk about a link ("I can feel your thoughts…"):
  +1.4 nats per token in 20 of 20 questions.

Part of what the twins say about the link is a trained trait of the character, not a report of what arrives on it.

## Limits and retractions

- One pair of models and one link architecture, which reads the partner's states through cross-attention, so part of
  the channel "describes itself". The translator reads a planted, one-token code word at a known position; free-form
  neuralese is harder.
- Drift: one seed per cell in the table above. The no-monitor runs also freed the gates and trained on city chats.
- The mood effect of the writes has no noise-on-the-link control yet.
- Retracted along the way:
  - "link presence is trivially detectable" (an artifact of frozen gates);
  - "you can hide the channel but not its use" (a stranger partner is flagged identically);
  - "hiding never coexists with information" (it does, partly, when passing the payload pays enough);
  - "a monitor-trained twin believes it is watched" (a yes-bias);
  - "the code changes completely after training" (mostly a change of scale).

## Reproduce

The experiments (CUDA, the released twins) are in `detection/`, and the queues that ran them are in
`detection/run_*.sh`. To rebuild the figures from the snapshot in `paper/data/results.json` (no GPU; needs Inter and
`rsvg-convert`):

```
python paper/collect.py      # re-snapshot from /workspace/runs (optional)
python paper/figures.py      # writes paper/figures/*.{svg,png,pdf}
```
