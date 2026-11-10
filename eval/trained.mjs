// Trained comparisons. Nothing here feeds back into the hand-written weights.
//   bow:  logistic regression on hashed words + word pairs (2^18 features), the classic trained spam filter
//   head: WP-Spam's own layers (weights/spam.json) with only the output layer trained
// Label sets: 512 (a random draw from the 7,723 below), 7,723 (exactly the study rows the first revision of the weights
// looked at: the first 500 per source and kind), all 58,141 study rows.
import fs from 'fs';
import os from 'os';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const W = require('../engine/spam.js');
// The bag-of-words model keeps the tokenizer it was first measured with, so its recorded numbers stay reproducible
// whatever changes in the hand-written network's own tokenizer.
const BOW_RE = /(?:https?:\/\/|www\.)[^\s<>"')\]]+|[a-z0-9][a-z0-9.-]*\.(?:com|net|org|info|biz|ru|cn|tk|ml|ga|cf|gq|xyz|top|co|uk|io|me|ly|us|br|pt|mz|de|fr|it|es|nl|in|online|site|club)(?:\/[^\s<>"')\]]*)?(?![a-z])|[\p{L}\p{N}._%+-]+@[\p{L}\p{N}-]+(?:\.[\p{L}\p{N}-]+)+|[\p{L}\p{N}]+(?:['’][\p{L}]+)?|[^\s\p{L}\p{N}]/giu;
function bowTokens(text) {
  const out = [];
  for (const line of text.split('\n')) { let m; BOW_RE.lastIndex = 0; while ((m = BOW_RE.exec(line)) && out.length < 2500) out.push(m[0].toLowerCase().normalize('NFKD').replace(/\p{M}/gu, '')); if (out.length >= 2500) break; }
  return out;
}
const B = 1 << 18;

function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) & (B - 1); }
export function bow(text) {
  const w = bowTokens(text).filter(x => /[\p{L}\p{N}$!%£€]/u.test(x));
  const c = new Map();
  const inc = k => { const h = hash(k); c.set(h, (c.get(h) || 0) + 1); };
  for (let i = 0; i < w.length; i++) {
    inc('u' + w[i]); if (i) inc('b' + w[i - 1] + ' ' + w[i]);
    if (w[i].length > 12 && /[./]/.test(w[i])) {   // a URL: its parts and character 4-grams, so a link is not one opaque token
      for (const p of w[i].split(/[^a-z0-9]+/)) if (p) inc('p' + p);
      for (let k = 0; k + 4 <= Math.min(w[i].length, 200); k++) inc('c' + w[i].slice(k, k + 4));
    }
  }
  const idx = [...c.keys()], val = idx.map(k => Math.log1p(c.get(k)));
  const n = Math.hypot(...val) || 1;
  return { idx, val: val.map(v => v / n) };
}
function headFeat(m, text) {
  const r = W.run(m, text, { trace: true });
  const v = [...m.dims.map(d => r.F[d] || 0), ...m.mlp.flat().map(h => r.hidden[h.name])];
  return { idx: v.map((_, i) => i), val: v };
}

if (!isMainThread && workerData?.role === 'trained') {
  const { kind, weights, texts } = workerData;
  const m = weights ? W.parse(fs.readFileSync(weights, 'utf8')) : null;
  parentPort.postMessage(texts.map(t => kind === 'bow' ? bow(t) : headFeat(m, t)));
}

export async function featurize(kind, texts, weights) {
  const nw = Math.min(os.cpus().length, 24), chunk = Math.ceil(texts.length / nw);
  const parts = await Promise.all(Array.from({ length: nw }, (_, w) => new Promise((res, rej) => {
    const wk = new Worker(new URL(import.meta.url), { workerData: { role: 'trained', kind, weights, texts: texts.slice(w * chunk, (w + 1) * chunk) } });
    wk.on('message', res); wk.on('error', rej);
  })));
  return parts.flat();
}

// Logistic regression, AdaGrad, L2, a few epochs, fixed seed. Balanced class weights.
export function fit(X, y, dim, { epochs = 8, lr = 0.5, l2 = 1e-6, seed = 7 } = {}) {
  const w = new Float64Array(dim + 1), g2 = new Float64Array(dim + 1).fill(1e-8);
  const pos = y.filter(Boolean).length, cw = [y.length / (2 * (y.length - pos)), y.length / (2 * pos)];
  let s = seed; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const order = y.map((_, i) => i);
  for (let e = 0; e < epochs; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (const i of order) {
      const { idx, val } = X[i];
      let z = w[dim]; for (let k = 0; k < idx.length; k++) z += w[idx[k]] * val[k];
      const g = (1 / (1 + Math.exp(-z)) - y[i]) * cw[y[i]];
      for (let k = 0; k < idx.length; k++) { const j = idx[k], gr = g * val[k] + l2 * w[j]; g2[j] += gr * gr; w[j] -= lr * gr / Math.sqrt(g2[j]); }
      g2[dim] += g * g; w[dim] -= lr * g / Math.sqrt(g2[dim]);
    }
  }
  return w;
}
export function predict(w, X, dim) { return X.map(({ idx, val }) => { let z = w[dim]; for (let k = 0; k < idx.length; k++) z += w[idx[k]] * val[k]; return 1 / (1 + Math.exp(-z)); }); }

// the three label sets
export function labelSets(study) {
  const seen = [], cnt = {};
  for (const r of study) { const k = r.source + '|' + r.kind; cnt[k] = (cnt[k] || 0) + 1; if (cnt[k] <= 500) seen.push(r); }
  let s = 12345; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const shuffled = seen.slice(); for (let i = shuffled.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; }
  return { 512: shuffled.slice(0, 512), [seen.length]: seen, [study.length]: study };
}

// Train every comparison model and return a scorer for any split.
export async function trainAll(study, weightsPath, log = console.log) {
  const sets = labelSets(study);
  const models = [];
  const allStudyBow = await featurize('bow', study.map(r => r.text));
  const allStudyHead = await featurize('head', study.map(r => r.text), weightsPath);
  const pos = new Map(study.map((r, i) => [r, i]));
  for (const [n, rows] of Object.entries(sets)) {
    const ix = rows.map(r => pos.get(r)), y = rows.map(r => r.kind === 'ham' ? 0 : 1);
    const ep = rows.length < 1000 ? 30 : rows.length < 10000 ? 10 : 5;
    const t0 = Date.now();
    const wb = fit(ix.map(i => allStudyBow[i]), y, B, { epochs: ep });
    const hd = allStudyHead[0].val.length;
    const wh = fit(ix.map(i => allStudyHead[i]), y, hd, { epochs: ep * 3, lr: 0.1 });
    log(`trained on ${n} labels (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
    models.push({ name: `trained bag-of-words, ${Number(n).toLocaleString('en')} labels`, n: +n, kind: 'bow', w: wb, dim: B, params: B + 1 });
    models.push({ name: `WP-Spam layers + trained output, ${Number(n).toLocaleString('en')} labels`, n: +n, kind: 'head', w: wh, dim: hd, params: hd + 1 });
  }
  return models;
}
