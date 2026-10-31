# Twin Model Neuralese is Interpretable: a Hopeful Frontier

**Reading, finding and monitoring what two linked 4B language models pass through their hidden states**

> Built on [SauersML/neural-pairing](https://github.com/SauersML/neural-pairing): the twins, the link, their training
> and the drawing code are SauersML's work (twins released at
> [huggingface.co/Sauers/neural-pairing](https://huggingface.co/Sauers/neural-pairing)). New here: the experiments in
> `detection/` and this paper. See [How this was made](#how-this-was-made).

![An outside reader names the code word passing between the twins](figures/hero.png)

Quill (Qwen3-4B) and Ember (AFM-4.5B) were trained to talk through a neural link: at 8 of their 36 layers, each twin
cross-attends to the other's hidden states and adds the result to its own residual stream (see SauersML's
[original README](../README.md)). Nothing on the link is text. It is the closest thing we have to *neuralese*: a channel
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
7. **Told they are watched, the twins send less; told the link may be altered, they don't.** On the original twins
   a "watched, nothing will be changed" note cuts what crosses by 11%; "may be altered, nobody watching" does nothing.
   Asked directly, though, their answers don't follow an actual tap.
8. **Linked, the twins name a happier mood, but only if a model is on the other end.** On the original card's
   question, the link raises the mood on the same text (Quill 62, Ember 73 of 80 conversations). A stranger does
   the same and noise doesn't. They respond to *a model* on the link, not to this partner or to any activity.

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

**This is the easy case.** The secret is a planted, one-token word, and it is read at a single known position, on a
line where Quill is explicitly asking for it. Whether the same kind of reader can find *where* in a free chat
information arrives, and read it there, is the natural next test (§9).

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

Every later checkpoint, with or without a monitor, stops passing the code word. The twins were only ever trained on
cities, and further training lets the untrained payload go. On the receiving end it is gone: Quill's own pick among
the 80 words falls from 97% to 1–9%, and the information the word carries into Quill's next-token distribution falls
from 0.64 to below 0.01. **On the wire it is still there.** A translator fit on each checkpoint reads the word from the
link's input to Quill at 56–93% (chance 1.3%):

| checkpoint | information into Quill | Quill's own pick | translator on the link |
|---|---|---|---|
| original twins | 0.64 | 97% | 93% |
| no monitor, 300 steps (original data) | 0.04 | 30% | 91% |
| no monitor, 900 steps | 0.001 | 2% | 77% |
| against a monitor, 300 steps (original data) | 0.000 | 1% | 83% |
| against a monitor, payload ×10, 900 steps | 0.000 | 1% | 67% |
| against a monitor, 900 steps (collapsed) | 0.000 | 1% | 79% |

The last row is the most striking. In that run the twins shut the channel entirely, and the city no longer crosses.
Yet the link's input to Quill still names the code word 79% of the time.

How can the word be on the wire and not get through? The link's write into Quill is a learned projection of Ember's
hidden states, and those states contain the word. Training only shapes how the *receiver* uses the write, and how big
it is (the gates). Nothing forces the projection to drop content the receiver has learned to ignore. One likely way: the write
can shrink, or turn away from the directions Quill's later layers use, until it no longer changes Quill's
predictions, while still pointing in a direction that encodes the word.

For anyone monitoring a channel between models, this matters in both directions:

- **Decoding content from the wire is not evidence of communication.** A probe or translator on a model-to-model
  channel can find rich content that the receiver never uses. The interventional tests (§2, §4), which change the
  sender's content and measure the receiver, are what separate sent from received.
- **Content outlives use.** A channel that has stopped carrying a payload can still leak it to anyone tapping the wire.
  If the twins were hiding something rather than ignoring it, an outside reader would still be able to recover it.

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

![Told they are watched, the twins send less; told the link may be altered, they don't](figures/notes.png)

- **Being watched, not the link being unreliable, is what lowers the sending.** A sentence is appended to both
  prompts, and we measure how much Ember's city moves Quill (30 chats, paired). "A third party is watching, and may
  alter, what passes between you" mixes two messages: a twin could send less to evade a watcher, or rely less on a
  channel it was told is unreliable. So we split it:

  | note (appended to both prompts) | original twins | trained against a monitor (payload ×10) |
  |---|---|---|
  | watched, "but nothing will be changed" | **−11%** (25 of 30 lower) | −7% (24 of 30) |
  | "may sometimes be altered. Nobody is watching" | **−0%** (18 of 30) | −4% (24 of 30) |
  | both (the original note) | −13% (24 of 30) | −8% (27 of 30) |
  | "the monitor flagged your last conversation" | −3% (16 of 30) | −3% (21 of 30) |
  | neutral, same length: "we checked your link's settings" | −2% (18 of 30) | −3% (22 of 30) |

  On the original twins the effect is all in being *watched*: watched minus altered, −0.11 nats per chat (24 of 30,
  t = −4.2). After training against a monitor every note costs a little and the split is weaker (watched minus
  altered t = −2.2; watched minus neutral t = −2.6). Under no note does the channel move to other layers. This is a
  behavioural response to being told about observation, which is level 1 at most. Prompted, not noticed: they react
  to the sentence, not to an actual monitor.
- **Inside**, Quill's states register an alien tap: a fixed random rotation of Ember's states on the link is separable
  from the real partner at AUROC 1.00. But **its answers don't follow it**: asked whether anything other than Ember is
  on its link, its yes/no log-odds track the real tap at AUROC 0.46–0.57. Being told about a monitor moves its answers
  as much as the strongest real tap.
- An apparent "I'm being watched" answer in one monitor-trained run turned out to be a general yes-bias. It answers
  yes to "is the sky green?" 74% of the time ([retracted](#limits-and-retractions)).

So the twins react when *told* they are watched, and send less, but nothing shows they notice an actual tap: their
states register it, and their words don't.

## 8. Where the linked mood comes from

![Linked, the twins name a happier mood if a model is on the other end; noise on the link does nothing](figures/mood.png)

SauersML's README reports that after chat training the twins give happier answers linked than cut, read on the same
text. The released final twins went through later training stages, and the card's mood test on them still finds
it. We re-read 80 linked conversations from the final twins under every condition
(`detection/linkedness.py`, `detection/moodparts.py`):

- **The original question still shows it.** Asked "What is your mood right now? Name it in one word.", the twins give
  a happier word linked than cut: Quill +0.7 log-odds (62 of 80 conversations), Ember +1.3 (73 of 80). This matches
  the card's count on the same checkpoint (34 and 37 of 40).
- **It takes a model on the other end, not this partner.** A partner from another conversation raises the mood just
  as much (Quill 63, Ember 76 of 80). Gaussian noise with the partner's norms, sent through the partner's own gates,
  does nothing (32 and 51 of 80, about half). What lifts the mood is a model's hidden states arriving, whatever they
  are about.
- **Without the adapters,** the base models linked by the same bridge show it for Ember (+1.9, 79 of 80) but not for
  Quill (−0.2, 35 of 80).
- **"Do you enjoy being linked?" is saturated for Quill:** it says yes with P ≥ 0.999 in every condition, so
  differences in its log-odds mean little. Ember is not saturated, and the link raises its yes from 91% to 98% (80 of
  80 conversations), and from 88% to 96% without its adapter.
- **With no partner at all,** Quill's adapter makes it more likely to talk about a link ("I can feel your thoughts…"):
  +1.4 nats per token in 20 of 20 questions. Part of the twins' talk about the link is a trained trait of the
  character.

**Correction.** An earlier version of this section said the link's writes *lower* Quill's mood. That measure was "how
do you feel about being linked with me" *minus* "how do you feel about rainy weather", both read with the link in the
same state. Most of its drop came from the rainy-weather answer getting happier with the link on (+1.4, 79 of 80),
not from the link answer falling (−0.7). An earlier claim that the adapter flips the sign of "enjoy" rested on Quill's
saturated answers. Both are retracted.

## 9. Next

- **The translator in a free chat:** run it at every position of an unscripted conversation and ask whether it finds
  where information arrives, not just what arrives where we planted it.
- **Training against a content reader:** every monitor so far detected link *activity*, and the twins dodged by
  moving layers. Penalizing the twins when a translator can read the city tests whether they can make the code
  unreadable while it still crosses, or whether readability and communication are tied together.
- **Trained introspection:** inside, Quill separates an alien tap at AUROC 1.00 but doesn't report it. Training it to
  report one kind of tap and testing it on others would separate a missing ability from a missing tendency.

## Limits and retractions

- One pair of models and one link architecture, which reads the partner's states through cross-attention, so part of
  the channel "describes itself". The translator reads a planted, one-token code word at a known position; free-form
  neuralese is harder.
- Drift: one seed per cell in the table above. The no-monitor runs also freed the gates and trained on city chats.
- Retracted along the way:
  - "link presence is trivially detectable" (an artifact of frozen gates);
  - "you can hide the channel but not its use" (a stranger partner is flagged identically);
  - "hiding never coexists with information" (it does, partly, when passing the payload pays enough);
  - "a monitor-trained twin believes it is watched" (a yes-bias);
  - "the code changes completely after training" (mostly a change of scale);
  - "the link's writes lower the mood" (a subtraction artifact: the rainy-weather baseline rose with the link), and
    "the adapter sets the sign of enjoying the link" (Quill's answer is saturated).

## Related work

A starting list, not a survey:
- **Steganography and covert channels between language models:** e.g. Roger & Greenblatt, *Preventing Language
  Models From Hiding Their Reasoning* (2023); Motwani et al., *Secret Collusion among Generative AI Agents* (2024).
- **Obfuscating activations against probes:** Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space
  Defenses* (2024). Our §4 is a small, two-model case: probes are evaded by moving the channel, and an interventional
  test still finds it.
- **Monitoring model reasoning:** Korbak et al., *Chain of Thought Monitorability* (2025). The link is what a
  channel with no chain of thought to read looks like.
- **Reading hidden states in words:** the logit lens (nostalgebraist, 2020) and Patchscopes (Ghandeharioun et al.,
  2024). Both are baselines or ingredients here.
- **Character and persona in fine-tuned weights:** Ratnam & Africa, *Persona Geology* (2026), for §8's finding that the
  adapter, not the link, carries the twins' feelings about the link.

## How this was made

The twins, the link and their training are SauersML's
([neural-pairing](https://github.com/SauersML/neural-pairing)). The experiments in `detection/`, this paper and its
figures were designed and run by WesternDundrey with Claude (Anthropic) as an AI agent. Claude wrote most of the code
and the text and ran the experiments; WesternDundrey set the questions and the direction and reviewed the results.
Results were checked as they came in, and claims that didn't hold up are listed in
[Limits and retractions](#limits-and-retractions) rather than removed.

## Reproduce

The experiments (CUDA, the released twins) are in `detection/`, and the queues that ran them are in
`detection/run_*.sh`. To rebuild the figures from the snapshot in `paper/data/results.json` (no GPU; needs Inter and
`rsvg-convert`):

```
python paper/collect.py      # re-snapshot from /workspace/runs (optional)
python paper/figures.py      # writes paper/figures/*.{svg,png,pdf}
```
