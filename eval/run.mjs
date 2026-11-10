// Score a split.  node eval/run.mjs study|dev [--weights FILE] [--errors SOURCE KIND N] [--limit N]
//   study: may print errors.  dev: totals only (never prints a message).
//   test:  node eval/run.mjs test --final "note"  runs the sealed split ONCE per weights file (keyed by sha256),
//          appends the totals to eval/test-runs.md, refuses to run the same weights again.
// Positive = spam or phishing (any kind that is not ham). Threshold 0.5, fixed.
import fs from 'fs';
import os from 'os';
import crypto from 'crypto';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const E = require('../engine/spam.js');

if (!isMainThread && workerData?.role === 'run') {
  const json = JSON.parse(fs.readFileSync(workerData.weights, 'utf8'));
  const m = E.parse(json);
  parentPort.postMessage(workerData.texts.map(t => E.run(m, t).p));
} else if (isMainThread) main();

export function table(per) {
  const pct = (a) => a[1] ? (100 * a[0] / a[1]).toFixed(1).padStart(5) + '%' : '     -';
  const lines = ['source            spam/phish caught    ham passed       accuracy'];
  let c = 0, t = 0, tp = 0, np = 0, tn = 0, nh = 0;
  for (const [src, s] of Object.entries(per).sort()) {
    lines.push(`${src.padEnd(16)}  ${pct(s.pos)} of ${String(s.pos[1]).padEnd(6)} ${pct(s.neg)} of ${String(s.neg[1]).padEnd(6)} ${pct([s.pos[0] + s.neg[0], s.pos[1] + s.neg[1]])}`);
    tp += s.pos[0]; np += s.pos[1]; tn += s.neg[0]; nh += s.neg[1];
  }
  c = tp + tn; t = np + nh;
  lines.push('', `accuracy ${(100 * c / t).toFixed(1)}%  (${c} of ${t})   spam caught ${(100 * tp / np).toFixed(1)}%   ham passed ${(100 * tn / nh).toFixed(1)}%   balanced ${(50 * (tp / np + tn / nh)).toFixed(1)}%`);
  return { text: lines.join('\n'), acc: c / t, bal: 0.5 * (tp / np + tn / nh) };
}

export async function score(weightsPath, texts) {
  const nw = Math.min(os.cpus().length, 24), chunk = Math.ceil(texts.length / nw);
  const parts = await Promise.all(Array.from({ length: nw }, (_, w) => new Promise((res, rej) => {
    const wk = new Worker(new URL(import.meta.url), { workerData: { role: 'run', weights: weightsPath, texts: texts.slice(w * chunk, (w + 1) * chunk) } });
    wk.on('message', res); wk.on('error', rej);
  })));
  return parts.flat();
}

export function load(split) {
  const f = split === 'test' ? '../data/sealed/test.jsonl' : `../data/${split}.jsonl`;
  return fs.readFileSync(new URL(f, import.meta.url), 'utf8').trim().split('\n').map(l => JSON.parse(l));
}

async function main() {
  if (!process.argv[1].endsWith('run.mjs')) return;
  const args = process.argv.slice(2);
  const split = args[0] || 'dev';
  const wpath = args.includes('--weights') ? args[args.indexOf('--weights') + 1] : 'weights/spam.json';
  const wbuf = fs.readFileSync(wpath);
  const wsha = crypto.createHash('sha256').update(wbuf).digest('hex');
  const runsFile = new URL('./test-runs.md', import.meta.url);
  let note = '';
  if (split === 'test') {
    const fi = args.indexOf('--final');
    if (fi < 0) { console.error('the test split is sealed: run it with --final "note", once per frozen weights file'); process.exit(1); }
    note = args[fi + 1] || '';
    const prev = fs.existsSync(runsFile) ? fs.readFileSync(runsFile, 'utf8') : '';
    if (prev.includes(wsha)) { console.error('these exact weights were already run on the test split'); process.exit(1); }
  }
  const ei = args.indexOf('--errors');
  const want = ei >= 0 ? { source: args[ei + 1], kind: args[ei + 2], n: +(args[ei + 3] || 20) } : null;
  if (want && split !== 'study') { console.error('errors can only be printed for the study split'); process.exit(1); }
  const limit = args.includes('--limit') ? +args[args.indexOf('--limit') + 1] : Infinity;
  const perCount = {};
  const rows = load(split).filter(r => { const k = r.source + '|' + r.kind; perCount[k] = (perCount[k] || 0) + 1; return perCount[k] <= limit; });
  const ps = await score(wpath, rows.map(r => r.text));
  const per = {}; const shown = [];
  rows.forEach((r, i) => {
    const flag = ps[i] >= 0.5, pos = r.kind !== 'ham';
    const s = (per[r.source] = per[r.source] || { pos: [0, 0], neg: [0, 0] });
    if (pos) { s.pos[0] += flag; s.pos[1]++; } else { s.neg[0] += !flag; s.neg[1]++; }
    if (want && r.source === want.source && r.kind === want.kind && shown.length < want.n && pos !== flag)
      shown.push(`${ps[i].toFixed(2)}  ${r.text.replace(/\s+/g, ' ').slice(0, 300)}`);
  });
  const t = table(per);
  console.log(`split ${split}, weights ${wpath} (${wsha.slice(0, 12)})\n` + t.text);
  if (shown.length) console.log('\n' + shown.join('\n\n'));
  if (split === 'test') {
    const head = fs.existsSync(runsFile) ? '' : '# Sealed test runs\n\nEach frozen weights file is run once on data/sealed/test.jsonl. Results are recorded whatever they are.\n\n';
    fs.appendFileSync(runsFile, head + `## ${new Date().toISOString().slice(0, 10)} · ${note}\n\nweights \`${wpath}\` sha256 \`${wsha}\`\n\n\`\`\`\n${t.text}\n\`\`\`\n\n`);
  }
}
