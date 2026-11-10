// Generalisation to a kind of mail never seen: for each source S, train the bag-of-words model on every OTHER
// source's study rows, then score S's dev rows. The hand-written network is scored as is. (Its revisions read study errors
// from every source, so for it no source is fully unseen; the first draft, written with no labels, scored 82.8%: LOG.md.)
// node eval/unseen.mjs   (dev only; totals only)
import { score, load } from './run.mjs';
import { featurize, fit, predict } from './trained.mjs';
const B = 1 << 18;
const study = load('study'), dev = load('dev');
const Xs = await featurize('bow', study.map(r => r.text)), Xd = await featurize('bow', dev.map(r => r.text));
const hand = await score('weights/spam.json', dev.map(r => r.text));
const sources = [...new Set(dev.map(r => r.source))].sort();
const acc = (ix, ps) => { let c = 0; for (const i of ix) c += (ps[i] >= 0.5) === (dev[i].kind !== 'ham'); return c / ix.length; };
console.log('| Unseen source | Messages | Trained on the other sources (labels) | Trained | Hand-written |\n|---|---|---|---|---|');
let sumT = 0, sumH = 0, nT = 0;
for (const S of sources) {
  const tr = study.map((r, i) => i).filter(i => study[i].source !== S);
  const w = fit(tr.map(i => Xs[i]), tr.map(i => study[i].kind === 'ham' ? 0 : 1), B, { epochs: 5 });
  const ix = dev.map((r, i) => i).filter(i => dev[i].source === S);
  const pt = predict(w, Xd, B);
  const a = acc(ix, pt), h = acc(ix, hand);
  sumT += a * ix.length; sumH += h * ix.length; nT += ix.length;
  console.log(`| ${S} | ${ix.length} | ${tr.length.toLocaleString('en')} | ${(100 * a).toFixed(1)}% | ${(100 * h).toFixed(1)}% |`);
}
console.log(`| **all, each unseen** | ${nT} | | **${(100 * sumT / nT).toFixed(1)}%** | **${(100 * sumH / nT).toFixed(1)}%** |`);
