# AGENTS.md

Guidance for AI agents (and people) working on WP-Spam. Read `README.md` for what it is and `LOG.md` for how it was built.

## House rules

1. **Weights are written, not fitted.** No gradient descent, search or fitting may set a weight in `gen/net.mjs`.
   Trained models live only in `eval/trained.mjs`, as comparisons, and nothing flows back from them.
2. **Concepts, not memory.** A row says what kind of evidence a word is. No dataset-specific names, senders, list
   names or addresses go in the weights. Brand names that phishing imitates, and well-known content sites, are general
   knowledge and are allowed.
3. **Every row group, head and unit has a reason** in `gen/net.mjs`. If one changes what it does, rewrite its reason.
4. **Count the labels you read.** Errors are read only on the study split (`eval/why.mjs`, `eval/run.mjs --errors`),
   and what was read goes in `LOG.md`. Dev is totals only. The sealed test runs once per weights file
   (`eval/test-runs.md`, keyed by sha256).
5. **One set of weights.** No version names in files or docs; the repository holds the current network only. After a
   new sealed run, record the weights' sha in `tests/run.mjs`; the history lives in `LOG.md`.
6. **Report the comparisons that go against us**: the trained model at equal labels, and the word-swap attack.
7. **Keep the baselines reproducible.** The bag-of-words model keeps its own frozen tokenizer in `eval/trained.mjs`.
8. **Static and small.** Weights under 1 MB, no page file over 2 MB. Datasets are never committed (`data/` is ignored).
9. **No assistant-specific files** (`CLAUDE.md` etc.). No personal names, e-mails or local paths in files or commits.

## Layout

`engine/spam.js` is the network (UMD; Node and browser). `gen/net.mjs` holds the weights with their reasons: part 1 the
core concepts, shapes, URL features, roles and neurons; part 2 the character layer, letter pairs, wider vocabulary,
attention heads, extended roles and the MLP. `gen/generate.mjs` turns it into `weights/spam.json`, copies the weights
and engine into `web/`, and prints the non-zero parameters per layer. `eval/` holds the splits, scoring, comparisons,
leave-one-source-out and the word-swap attack. `web/` is the page; `.github/workflows/pages.yml` publishes it.

## Gotchas

- A word's row is the **sum** of every `G(...)` group it appears in; a generic word in a new group changes it
  everywhere. Check `node eval/why.mjs SOURCE KIND N --all` for the parts that push errors the wrong way.
- `~stem` (anywhere inside a word) is dangerous for short or common stems: "cialis" is inside "specialist",
  "replica" inside "replication".
- Ham words need the `prose` gate: word-salad spam is made of ordinary dictionary words.
- After changing `engine/spam.js`, run `node gen/generate.mjs` so `web/spam.js` matches (the tests check it).
