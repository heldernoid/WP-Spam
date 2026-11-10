// node eval/compare.mjs dev          the hand-written network against the trained comparisons, on dev (totals only)
// node eval/compare.mjs test --final the same on the sealed test split, once per weights file (refuses if its sha256 is
//                                    already in eval/test-runs.md)
import fs from 'fs';
import crypto from 'crypto';
import { score, load, table } from './run.mjs';
import { featurize, trainAll, predict } from './trained.mjs';
const split = process.argv[2] || 'dev';
if (split === 'test' && !process.argv.includes('--final')) { console.error('sealed: add --final'); process.exit(1); }
const runsFile = 'eval/test-runs.md';
const WEIGHTS = 'weights/spam.json';
const sha = crypto.createHash('sha256').update(fs.readFileSync(WEIGHTS)).digest('hex');
if (split === 'test' && fs.existsSync(runsFile) && fs.readFileSync(runsFile, 'utf8').includes(sha)) { console.error('already run on the sealed test: ' + sha); process.exit(1); }
const rows = load(split), study = load('study');
const y = rows.map(r => r.kind !== 'ham');
function summarize(name, ps, labels, nz) {
  const per = {};
  rows.forEach((r, i) => { const s = (per[r.source] = per[r.source] || { pos: [0, 0], neg: [0, 0] }); const f = ps[i] >= 0.5; if (y[i]) { s.pos[0] += f; s.pos[1]++; } else { s.neg[0] += !f; s.neg[1]++; } });
  const t = table(per);
  return { name, labels, nz, acc: t.acc, bal: t.bal, text: t.text };
}
// non-zero parameters, counted as gen/generate.mjs counts them
function nonZero(j) {
  let n = Object.values(j.chars).reduce((a, r) => a + Object.keys(r).length, 0) + j.letterRare.length + j.letterOnset.length + 1;
  for (const [p, , r] of j.filters) n += p.length + Object.keys(j.rows[r]).length;
  for (const [k, r] of Object.entries(j.phrases)) n += k.split(' ').length + Object.keys(j.rows[r]).length;
  for (const [p, , r] of j.affixes) n += p.length + Object.keys(j.rows[r]).length;
  for (const r of Object.values(j.shapes)) n += Object.keys(r).length;
  for (const r of Object.values(j.url.rows)) n += Object.keys(r).length;
  for (const a of Object.values(j.url.sets)) n += a.length;
  for (const h of j.heads) n += Object.keys(h.query).length + Object.keys(h.key).length + 4;
  for (const r of j.roles) n += Object.keys(r.w).length + 1;
  n += j.arcs.length;
  for (const L of j.mlp) for (const h of L) n += Object.keys(h.w || {}).length + Object.keys(h.h || {}).length + 1;
  return n + Object.keys(j.out.w).length + Object.keys(j.out.hidden).length + 1;
}
const res = [summarize('WP-Spam, hand-written', await score(WEIGHTS, rows.map(r => r.text)), 58141, nonZero(JSON.parse(fs.readFileSync(WEIGHTS, 'utf8'))))];
const models = await trainAll(study, WEIGHTS, s => console.error(s));
const Xb = await featurize('bow', rows.map(r => r.text)), Xh = await featurize('head', rows.map(r => r.text), WEIGHTS);
for (const m of models) {
  const ps = predict(m.w, m.kind === 'bow' ? Xb : Xh, m.dim);
  res.push(summarize(m.name, ps, m.n, m.w.reduce((a, x) => a + (x !== 0), 0)));
}
const lines = [`split ${split}: ${rows.length} messages`, '', '| Weights | Labels seen | Non-zero params | Accuracy | Balanced |', '|---|---|---|---|---|',
  ...res.map(r => `| ${r.name} | ${r.labels.toLocaleString('en')} | ${r.nz.toLocaleString('en')} | ${(100 * r.acc).toFixed(1)}% | ${(100 * r.bal).toFixed(1)}% |`)];
console.log(lines.join('\n'));
for (const r of res) console.log(`\n### ${r.name}\n${r.text}`);
if (split === 'test') {
  fs.appendFileSync(runsFile, `## ${new Date().toISOString().slice(0, 10)} · hand-written network against trained comparisons\n\nweights sha256 \`${sha}\`\n\n${lines.join('\n')}\n\n` + res.map(r => `### ${r.name}\n\n\`\`\`\n${r.text}\n\`\`\`\n`).join('\n') + '\n');
}
