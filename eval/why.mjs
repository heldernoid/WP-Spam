// Study errors with their largest contributions.
//   node eval/why.mjs SOURCE KIND N [--limit 500]   the first errors within the first --limit rows
//   node eval/why.mjs SOURCE KIND N --all [--seed 1] a random sample of N errors from ALL study rows,
//                                                     plus how often each of the network's own parts pushed them wrong
// Only the study split.
import fs from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const W = require('../engine/spam.js');
const [src, kind, n = 15] = process.argv.slice(2);
const all = process.argv.includes('--all');
const limit = all ? Infinity : process.argv.includes('--limit') ? +process.argv[process.argv.indexOf('--limit') + 1] : 500;
const seed0 = process.argv.includes('--seed') ? +process.argv[process.argv.indexOf('--seed') + 1] : 1;
const m = W.parse(fs.readFileSync('weights/spam.json', 'utf8'));
let seen = 0; const errs = []; const blame = {};
for (const l of fs.readFileSync('data/study.jsonl', 'utf8').split('\n')) {
  if (!l) continue; const r = JSON.parse(l);
  if (r.source !== src || r.kind !== kind) continue;
  if (++seen > limit) break;
  if (!all && errs.length >= +n) break;
  const t = W.run(m, r.text, { trace: true });
  if ((t.p >= 0.5) === (kind !== 'ham')) continue;
  const parts = Object.entries(t.parts).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  for (const [k, v] of parts.slice(0, 4)) if ((v > 0) === (kind === 'ham')) blame[k] = (blame[k] || 0) + 1;
  errs.push(`${t.p.toFixed(2)} [${parts.slice(0, 7).map(([k, v]) => `${k}${v > 0 ? '+' : ''}${v.toFixed(1)}`).join(' ')}]\n     ${r.text.replace(/\s+/g, ' ').slice(0, 220)}`);
}
let out = errs;
if (all) {
  let s = seed0; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  out = out.slice(0, +n);
  console.log(`${errs.length} errors of ${seen} rows. Parts pushing the wrong way (in top 4):`, Object.entries(blame).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([k, v]) => `${k} ${v}`).join(', '));
}
console.log(out.join('\n'));
