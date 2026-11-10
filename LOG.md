# How the weights were written

The repository holds the current network only. This log tells how it got there, stage by stage: the first draft,
two revisions, a deeper network, and the character layer. The earlier stages' weight files are gone; their sha256 are
in `eval/test-runs.md`.

## Protocol

- **The first draft** is written from general knowledge of how spam, scams, phishing and ordinary mail read. No label from this
  project is looked at.
- **The first revision** comes from reading the errors on the study rows. At most 500 rows per source and kind are
  read, 7,723 labels in all. Changes must be general concepts, never a dataset's names, senders or addresses.
- **Dev** is scored for totals only. **The sealed test** runs once per frozen weights file.
- **Trained comparisons** follow a public post where a hand-written 100k-parameter network for legal citations was
  compared against trained ones. They are trained on 512 labels, on the same 7,723 labels the revision saw, and on all
  58,141 study labels.

## Stage 1: the first draft (2026-11-10)

The first draft has 998 lexicon rows over 46 dims, four attention heads (amount, ctaLink, youWin, youAcct), five hidden
neurons and 1,157 non-zero weights.

Dev: **82.8%** (spam caught 79.1%, ham passed 85.9%). That is already above WP-Phish (79.3% balanced), because the
evidence is soft and additive rather than AND-gates over a word list.

## Stage 2: the first revision (7,723 study labels)

The errors pointed at missing concepts, not missing words:

- **Receipts against requests.**
  - Mozambican M-Pesa transaction receipts ("Confirmado … Transferiste 500MT … novo saldo") were flagged as scams,
    because they share every money word with the wrong-transfer scam.
  - New `receipt` dim: a completed transaction being reported.
  - New `payReq` head: a send verb followed by a phone or account number, and the "vem em nome de" phrasing.
  - Generic mobile-money words became neutral.
  - Moz ham passed: 55% → 76%.
- **News prose.** News digests and newsletters are ham in these corpora. Added a `news` dim (reported speech,
  journalism vocabulary) that dampens the pitch and advance-fee neurons.
- **Word salad.** Bayes-poison text and letter-fragment obfuscation ("P qc har aa ma qja cy") have few function words.
  Added a `funcRatio` feature (share of function words; English, Portuguese, Spanish, German) and a `salad` neuron.
- **Short message plus a link.** Most CEAS spam is one line and a bare domain. Added a `shortLink` neuron over
  `hasUrl` and word count.
- **URLs.**
  - URL features are now max-pooled over a message's links. Before, ten ordinary footer links added up to one bad link.
  - Added `urlVerdict`, a neuron that judges a lone link by its shape.
  - New features: `cmsPath` (pages planted inside hacked WordPress or Joomla folders), `repeatSeg` (kit-made
    "/x/x/" paths), `randomPath`, `document`, `institution`.
  - `.ru`/`.cn` were dropped from the bad TLDs, and blog hosts from the free hosts. Both were wrong guesses in the first draft.
- **Generic words out.** `personal` and `advfee` held words every text uses ("no", "home", "government", "trust").
  They were cut to chat-specific forms, and the advance-fee story words were split into strong (1.0) and weak (0.4).
- **Pharmacy by euphemism, e-mail-account phishing** (mailbox full, messages on hold, files waiting), stock touting
  phrases, Spanish and German pitch words.

Study rows read: 84.1% → 89.1%. Dev: 82.8% → **86.8%**.

## Sealed test (run once)

First draft 83.4%, first revision 87.5%. Trained bag-of-words: 87.8% (512 labels), 94.6% (7,723), 97.1% (58,141). WP-Spam's own layers with
only the 67 output weights trained: 90.0%. Per-source tables are in `eval/test-runs.md`.

## What the comparison shows

- **Same sources, more labels: training wins.** The first revision is worth about 500 labels (87.5% vs 87.8% for the 512-label model). It does
  not beat training the way the legal-citation network did. Citations follow strict formats that can be written down;
  spam is open-ended text.
- **New sources: hand-writing wins.** Leave-one-source-out: trained 75.4%, hand-written 86.8%. Most of the trained
  model's accuracy is knowledge of these nine corpora.
- **Word swaps: training wins.** After 3 adversarial swaps, trained 94.0% and hand-written 68.3%. The evidence sits in
  too few rows.
- **The hand-written layers are a good representation.** Training only their 67 output weights gives 90.0%, and with
  512 labels it beats the 65,875-weight trained model (88.6% vs 87.8% on test).

## Stage 3: the second revision (all 58,141 study labels)

The reason: to compare fairly with the model trained on 58,141 labels, the hand-written revision may use the same
58,141 labels. All study rows were scored; errors were read as random samples drawn from the whole study split
(`eval/why.mjs SOURCE KIND N --all`), together with counts of which of the network's own parts pushed each error the
wrong way. No word statistics were copied into the weights.

- **Spelling-tolerant matching.** Doubled letters are ignored on both sides ("vicoodiin", "xannax", "vallium").
- **Chopped-up words.** A `junkRatio` feature counts 2-4 letter consonant clusters ("Pe lbd nis En ir large"), and the
  `junkText` neuron fires above about 12%. HTML entities (`lt`, `gt`) are excluded; texting drops vowels too, but less.
- **Enhancement pitch by euphemism.** Strong words go in `enhance`. Ordinary words ("longer", "pleasure", "formula")
  go in `enhanceWeak`, which counts only in a neuron that news, tech and work talk switch off.
- **Dating spam phrases.**
- **Named greeting.** A new `named` head reads "Dear Tony" as someone who knows the reader. "Dear Friend" does not
  count, because generic addressees are lexicon words.
- **List machinery.** Subject tags ("[opensuse]"), "schrieb/escreveu", group footers.
- **Solicited mail** ("view online", "add us to your address book", "in this issue"). The first draft also counted
  legal footers ("privacy policy", "all rights reserved", "manage your subscription"). About 680 spam messages carry
  exactly that footer, so it was removed: footers are copied wholesale and are not evidence.
- **Receipts weigh more** (-2.5), and more notice words were added. Moz ham passed went from 75% to 85%.
- **A bare domain** counts as ordinary only when it is a lone link, not inside a message.

Study, all rows: 86.6% → 88.5%. Dev: 86.8% → **88.3%**. Sealed test (once): **88.6%**.

Against the trained model with the same 58,141 labels (97.1%), the gap stays large. What labels give a person is
concepts, one revision at a time; what they give gradient descent is a weight for every word and source quirk that
separates the classes. Leave-one-source-out shows which of the two carries over to new mail: trained on the other
sources 75.4%, the hand-written first draft with no labels 82.8%.

The word-swap attack did not improve: 68.7% still caught after 3 swaps, against 68.3% for the first revision. Spelling tolerance does
not help when the attacker swaps a word for an unrelated invented one. Robustness needs evidence a scam cannot drop
without stopping being a scam, and more of it per message.

## Stage 4: a deeper network (no new labels)

The question was: when a model misses something, trained networks get more layers and neurons; can a hand-written one?
It was prompted by a false alarm on "In sweden they eat a lot of kannelbulle. Its an instant gratification of sugar and
you gain happiness :) win-win", which the second revision scored 76%: "win" twice (prize), "you ... win" (youWin), "instant" (urgency),
and nothing on the ham side.

New layers:
- **Sentence layer.** Each sentence is pooled on its own, and 8 clamped role units score it: offer, ask, contact,
  story, notice, chat, work, news.
- **Message layer.** Attention across sentences gives role totals, role shares (`s_`), and 5 arcs, where a role is
  followed by another within a few sentences (offer → ask → contact, story → offer → ask).
- **Gate layer** (a second hidden layer, with clamped units):
  - `channel` asks whether there is any way to act.
  - `looseNoChannel` = ReLU(loose − 3·channel) cancels loose words (win, instant, money, you) when there is no way to
    act. Strong evidence (drugs, adult, salad) is not gated.
  - `scamArc`, `spamShape` and `hamShape` read the message layer.
- **Tokenizer.**
  - Spaced links ("http : / / www . site . com", the pre-tokenised Enron text) are rejoined.
  - Emoticons count as chat.

What the measurements said (study, all rows; ablation on a 25% sample):
- **First draft, a hard gate (−1.6 when no channel): 86.7%, below the second revision.** Many spam messages in CEAS and Enron reach
  the dataset with their links stripped ("More info >>>" and nothing after), so "no way to act" is weaker evidence in
  these corpora than in real mail.
- **Ablation:**
  - rejoining spaced links +0.6;
  - the gates −0.4 to −0.6 at full strength;
  - arcs and shares about 0.
  - The gate was softened (only `looseNoChannel`, −0.6) and the arc weight halved.
- **Final: study 88.5% (before: 88.5%), dev 88.6% (88.3%), sealed test 88.8% (88.6%).** English e-mail 90.0%, English
  SMS 95.5%. Word-swap attack 69.7% after 3 swaps (before: 68.7%). The example sentence: 76% → 32%.

Lesson: in a trained network, more layers and neurons are more capacity to absorb data. In a hand-written one, a layer
helps only if it brings knowledge the network did not have. The sentence and message layers mostly re-read the same
token evidence in a new arrangement, so they changed how it decides (structure, the gate, the example) more than how
often it is right. The bottleneck is what the token layer knows.

## Stage 5: 25,000 hand-written weights (no new labels)

The question was whether a hand-written network gets powerful if it is allowed to be much bigger: 25,000-30,000 non-zero
weights instead of 1,781, more attention heads, more depth. Errors were read on the study split only (no new labels:
still the 58,141 of the second revision), dev for totals, and the sealed test once.

What changed:
- **Character layer.** The lexicon lookup became convolution filters over letter vectors. Each character is a vector
  over letters: a letter is itself (1), a digit or symbol that imitates a letter is that letter (0.9), a Cyrillic
  look-alike is the Latin letter (0.9), and "l"/"i" are each other at 0.5, which only matters in long words. A filter
  fires on a whole word, at a word's start, or anywhere inside it, with a tolerance that lets long words carry one wrong
  letter. One winning stem per word. Links go through the same layer with their digits kept, so "paypa1" reads as
  "paypal".
- **Letter-pair layer.** Two hand-written tables: letter pairs English and Portuguese words (almost) never contain, and
  the consonant pairs a word may start with. A word no filter knows that breaks them cannot be read aloud.
- **About 1,600 new stems and 470 new phrases.** These cover scams the network never knew: crypto, sextortion, job offers,
  parcel fees, fake invoices and tech support, gambling, debt, SEO, health. They also cover the ham side it barely had:
  everyday life, feelings, family, plans, times, first-person narration, subject matter, Portuguese and English texting,
  and shared context ("aquele valor", "como combinado").
- **Phrases of 2-4 words.** Before, only two-word phrases matched; its 16 longer entries had never fired.
- **36 attention heads in two layers** (before: 7 in one). Layer 2 reads layer 1, for example "an offer, then a link"
  or "a threat, then a demand to pay".
- **10 sentence roles** (adding threat and life) **and a three-layer MLP** of 34 units.

What reading the study errors changed along the way (each was a general mistake, not a dataset fix):
- `~cialis` matched inside "specialist" and `~replica` inside "replication". Anywhere-matching is now kept for stems
  that do not hide in ordinary words.
- "network" was in every CNN-alert spam footer. "reset", "restore", "disabled", "virus" and "firewall" are ordinary
  tech words. "wish" ("if you wish to be removed"), "feel" ("feel free"), "party" ("third party") and "read" are in
  spam as often as in chat. All were taken out of the new concepts.
- 419 letters talk about God, family and feelings, so the new chat neurons fired on them. A stranger introducing
  themself ("I am Mrs ...", "this may come as a surprise") is now a concept that switches the chat neurons off.
- Word-salad spam made of dictionary words hit the new ham vocabulary. A `prose` unit (function words present, or
  tokens that are not all words) gates it: ham words count only in sentences a person wrote. The first version mistook
  code, patches and weather tables for salad, because they also have few function words.
- Real links inside spam (fake CNN alerts carry cnn.com links) made a well-known site weak evidence. It now vouches
  only for a lone link.
- Mobile-money brands next to "conta" fired the impersonation head on ordinary M-Pesa receipts. Impersonation now
  counts only through its neuron, which a receipt switches off.

Results:

| | Stage 4 | Stage 5 (now) |
|---|---|---|
| Non-zero weights | 1,781 | 25,114 |
| Study (all 58,141 rows) | 88.5% | 89.3% |
| Dev | 88.6% | 89.4% |
| **Sealed test (once)** | 88.8% | **89.5%** |
| English e-mail / SMS (sealed) | 90.0% / 95.5% | 90.5% / 95.7% |
| Links, sealed (Hannousse) | 81.5% | 82.5% |
| Portuguese SMS, sealed | 86.0% | 90.2% |
| CEAS ham passed, sealed | 88.6% | 90.1% |
| Still caught after 3 word swaps | 69.7% | 67.7% |

Fourteen times the weights bought 0.7 points on the sealed test. The gains are where knowledge was added that the
network did not have: phishing links (asset-folder pages, kit words, look-alike brands), Portuguese requests between
friends, and newsletters and receipts (CEAS ham +1.5). The word-swap attack did not improve. The character layer
reaches disguised spellings of a word, not an unrelated invented word put in its place.

The trained model's 97.1% with the same labels is still far away. In a trained network every added weight is fitted
to the data; here every added weight is a fact someone has to know and write down, and most facts about language
are already known to these corpora only through the handful of words that separate them. Size is not what a
hand-written network lacks; knowledge per weight is the limit, and it grows only as fast as it can be written.

## Keeping one network

The repository now holds the current network only, as WP-CharNet does. The two weight sources were merged into
`gen/net.mjs` and the version labels were taken out of the reasons. The generated network is identical to the one the
sealed test ran on: every weight, filter, head and unit is the same, and dev re-scores to the same 9,160 of 10,249. Only
the file hash changed, which `eval/test-runs.md` records. The trained bag-of-words baseline keeps the tokenizer it was
first measured with (it is embedded in `eval/trained.mjs`), so its numbers reproduce exactly. The "layers + trained
output" comparison now uses the current layers: 91.3% on dev with all labels, 89.3% with 512 (the first revision's
layers gave 90.0% on the sealed test).

## Open problems

- **Robustness to word swaps.** Evidence needs to be spread out, with structural signals a scam cannot drop. The
  character-level stems catch disguised spellings, but not a word swapped for an unrelated invented one. The structural signals
  are the request for action, the target number or link, and the unknown sender.
- **Enron spam (79% caught).** Much of it is word salad around a short pitch, "l"-for-"i" homoglyphs ("pieased",
  "profiie"), or bounce messages labelled spam.
- **URLs (69% of phishing links caught).** Without domain reputation, a hacked small-business site looks like any
  small-business site.
