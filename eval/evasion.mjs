// The "candies" test, measured. Take 300 dev spam/phishing messages that a model catches. An attacker who can query
// the model replaces whole words with invented ones ("fund" -> "zorbex"), always picking the word whose swap lowers
// the score most, up to 3 words. Reported: share still caught after 1, 2, 3 swaps.  node eval/evasion.mjs
import fs from 'fs';
import os from 'os';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const W = require('../engine/spam.js');
const WPATH = 'weights/spam.json';
import { load } from './run.mjs';
import { bow, featurize, fit, predict } from './trained.mjs';
const B = 1 << 18;
const FAKE = ['zorbex', 'quimble', 'drafon', 'plesk', 'vantoo', 'mirgle'];

function attack(scoreFn, text) {
  let t = text; const res = [];
  for (let step = 0; step < 3; step++) {
    const words = [...new Set((t.match(/[\p{L}][\p{L}'’-]{2,}/gu) || []))].filter(w => !FAKE.includes(w)).slice(0, 80);
    let best = null, bp = Infinity;
    for (const w of words) {
      const cand = t.split(w).join(FAKE[step]);
      const p = scoreFn(cand);
      if (p < bp) { bp = p; best = cand; }
    }
    if (best === null) { res.push(scoreFn(t) >= 0.5); continue; }
    t = best; res.push(bp >= 0.5);
  }
  return res;
}

if (!isMainThread && workerData?.role === 'evasion') {
  const { kind, texts, w } = workerData;
  let scoreFn;
  if (kind === 'hand') { const m = W.parse(fs.readFileSync(WPATH, 'utf8')); scoreFn = t => W.run(m, t).p; }
  else { const wv = new Float64Array(w); scoreFn = t => predict(wv, [bow(t)], B)[0]; }
  parentPort.postMessage(texts.map(t => attack(scoreFn, t)));
} else {
  const dev = load('dev').filter(r => r.kind !== 'ham' && r.channel !== 'url');
  const study = load('study');
  const Xs = await featurize('bow', study.map(r => r.text));
  const wb = fit(Xs, study.map(r => r.kind === 'ham' ? 0 : 1), B, { epochs: 5 });
  const m = W.parse(fs.readFileSync(WPATH, 'utf8'));
  // the same 300 messages for both: caught by both before the attack, spread over sources
  let s = 99; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const both = dev.filter(r => W.run(m, r.text).p >= 0.5 && predict(wb, [bow(r.text)], B)[0] >= 0.5);
  for (let i = both.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [both[i], both[j]] = [both[j], both[i]]; }
  const pick = both.slice(0, 300).map(r => r.text.slice(0, 3000));
  const run = (kind) => {
    const nw = Math.min(os.cpus().length, 24), chunk = Math.ceil(pick.length / nw);
    return Promise.all(Array.from({ length: nw }, (_, k) => new Promise((res, rej) => {
      const wk = new Worker(new URL(import.meta.url), { workerData: { role: 'evasion', kind, texts: pick.slice(k * chunk, (k + 1) * chunk), w: kind === 'bow' ? Array.from(wb) : null } });
      wk.on('message', res); wk.on('error', rej);
    }))).then(p => p.flat());
  };
  const pct = (r, k) => (100 * r.filter(x => x[k]).length / r.length).toFixed(1) + '%';
  const rh = await run('hand'), rb = await run('bow');
  console.log(`${pick.length} spam/phishing messages caught by both models before the attack\n`);
  console.log('| Model | Still caught after 1 swap | 2 swaps | 3 swaps |\n|---|---|---|---|');
  console.log(`| trained bag-of-words, 58,141 labels | ${pct(rb, 0)} | ${pct(rb, 1)} | ${pct(rb, 2)} |`);
  console.log(`| WP-Spam hand-written | ${pct(rh, 0)} | ${pct(rh, 1)} | ${pct(rh, 2)} |`);
}
