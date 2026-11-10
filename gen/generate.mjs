// node gen/generate.mjs  ->  weights/spam.json and web/ (engine + weights) from gen/net.mjs.
// Fails if any web file is over 2 MB or the weights file is over 1 MB. Prints the non-zero parameters per layer.
import fs from 'fs';
import crypto from 'crypto';
import * as N from './net.mjs';

const dims = new Set(N.DIMS);
if (dims.size !== N.DIMS.length) throw new Error('duplicate dim');
const check = (row, where) => { for (const d of Object.keys(row)) if (!dims.has(d)) throw new Error(`unknown dim ${d} in ${where}`); };
const fold = s => s.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase();

// ---- rows (deduplicated), stem filters, phrases
const rows = [], rowIx = new Map();
const rowOf = (r) => { const k = JSON.stringify(Object.entries(r).filter(([, x]) => x).sort()); if (!rowIx.has(k)) { rowIx.set(k, rows.length); rows.push(Object.fromEntries(JSON.parse(k))); } return rowIx.get(k); };
const P = { letterPairs: 0, chars: 0, filters: 0, filterRows: 0, affixes: 0, phrases: 0, shapes: 0, url: 0, heads: 0, roles: 0, arcs: 0, mlp: 0, out: 0 };
// a hyphenated key is a phrase ("knock-off" is tokenised as "knock", "-", "off"); keys that fold to the same filter or
// phrase have their rows added
const fmap = new Map(), pmap = new Map();
const merge = (map, k, row) => { const r = map.get(k) || {}; for (const [d, x] of Object.entries(row)) r[d] = +((r[d] || 0) + x).toFixed(3); map.set(k, r); };
for (const [key0, row] of Object.entries(N.lexicon())) {
  check(row, 'lexicon ' + key0);
  let key = fold(key0);
  if (key.includes(' ') || key.includes('-')) {
    const words = key.replace(/-/g, ' ').split(/\s+/).filter(Boolean);
    if (words.length > 1) { merge(pmap, words.join(' '), row); continue; }
    key = words[0];
  }
  let at = 'word', p = key;
  if (p.startsWith('~')) { at = 'any'; p = p.slice(1); }
  else if (p.endsWith('*')) { at = 'start'; p = p.slice(0, -1); }
  for (const c of p) if (!N.CHARS[c]) throw new Error(`filter ${key0}: no character embedding for "${c}"`);
  merge(fmap, p + '\t' + at, row);
}
const filters = [], phrases = {};
for (const [k, row] of fmap) {
  if (!Object.values(row).some(x => x)) continue;
  const [p, at] = k.split('\t');
  filters.push([p, at, rowOf(row)]); P.filters += p.length; P.filterRows += Object.values(row).filter(x => x).length;
}
for (const [k, row] of pmap) {
  if (!Object.values(row).some(x => x)) continue;
  phrases[k] = rowOf(row); P.phrases += k.split(' ').length + Object.values(row).filter(x => x).length;
}
filters.sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : 1);
for (let i = 1; i < filters.length; i++) if (filters[i][0] === filters[i - 1][0] && filters[i][1] === filters[i - 1][1]) throw new Error('duplicate filter ' + filters[i][0]);
for (const r of Object.values(N.CHARS)) P.chars += Object.keys(r).length;
P.letterPairs = N.LETTER_RARE.length + N.LETTER_ONSET.length + 1;
const affixes = N.AFFIXES.map(([p, n, row]) => { check(row, 'affix ' + p); P.affixes += p.length + Object.keys(row).length; return [p, n, rowOf(row)]; });

// ---- the rest
for (const [k, r] of Object.entries(N.SHAPES)) { check(r, 'shape ' + k); P.shapes += Object.keys(r).length; }
for (const [k, r] of Object.entries(N.URL.rows)) { check(r, 'url ' + k); P.url += Object.keys(r).length; }
for (const [k, a] of Object.entries(N.URL.sets)) P.url += a.length;   // one weight per listed host, TLD or keyword
for (const h of N.HEADS) { check(h.query, 'head ' + h.name); check(h.key, 'head ' + h.name); if (!dims.has(h.out)) throw new Error('head out ' + h.out); P.heads += Object.keys(h.query).length + Object.keys(h.key).length + 4; }
for (const r of N.ROLES) { check(r.w, 'role ' + r.name); if (!dims.has('r_' + r.name)) throw new Error('role dim missing ' + r.name); P.roles += Object.keys(r.w).length + 1; }
for (const a of N.ARCS) { if (!dims.has('a_' + a.name)) throw new Error('arc dim missing ' + a.name); P.arcs += 1; }
const seen = new Set();
for (const layer of N.MLP) for (const h of layer) {
  check(h.w || {}, 'mlp ' + h.name);
  for (const k of Object.keys(h.h || {})) if (!seen.has(k)) throw new Error(`mlp ${h.name} reads ${k} before it exists`);
  if (seen.has(h.name)) throw new Error('duplicate unit ' + h.name);
  seen.add(h.name); P.mlp += Object.keys(h.w || {}).length + Object.keys(h.h || {}).length + 1;
}
check(N.OUT.w, 'out');
for (const k of Object.keys(N.OUT.hidden)) if (!seen.has(k)) throw new Error('out reads unknown unit ' + k);
P.out = Object.keys(N.OUT.w).length + Object.keys(N.OUT.hidden).length + 1;

const weights = {
  name: 'WP-Spam', dims: N.DIMS,
  chars: N.CHARS, tolShort: N.TOL.short, tolLong: N.TOL.long,
  rows, filters, affixes, phrases, letterRare: N.LETTER_RARE, letterOnset: N.LETTER_ONSET,
  shapes: N.SHAPES, url: N.URL, heads: N.HEADS, roles: N.ROLES, arcs: N.ARCS, mlp: N.MLP, out: N.OUT,
  functionWords: N.FUNCTION_WORDS, ratios: N.RATIOS, fuzzy: !!N.FUZZY, unspace: !!N.UNSPACE,
};
const json = JSON.stringify(weights);
if (json.length > 1024 * 1024) { console.error('weights over 1 MB'); process.exit(1); }
fs.writeFileSync(new URL('../weights/spam.json', import.meta.url), json);
fs.mkdirSync(new URL('../web/model/', import.meta.url), { recursive: true });
fs.writeFileSync(new URL('../web/model/spam.json', import.meta.url), json);
fs.copyFileSync(new URL('../engine/spam.js', import.meta.url), new URL('../web/spam.js', import.meta.url));

const nz = Object.values(P).reduce((a, b) => a + b, 0);
const sha = crypto.createHash('sha256').update(json).digest('hex');
console.log(`WP-Spam: ${filters.length} stem filters, ${Object.keys(phrases).length} phrases, ${N.HEADS.length} heads, ${N.MLP.flat().length} MLP units`);
console.log(`non-zero parameters ${nz.toLocaleString('en')}: ` + Object.entries(P).map(([k, v]) => `${k} ${v.toLocaleString('en')}`).join(', '));
console.log(`${(json.length / 1024).toFixed(1)} KB, sha ${sha.slice(0, 12)}`);
for (const f of fs.readdirSync(new URL('../web/', import.meta.url), { recursive: true })) {
  const st = fs.statSync(new URL('../web/' + f, import.meta.url));
  if (st.isFile() && st.size > 2 * 1024 * 1024) { console.error('over 2 MB: web/' + f); process.exit(1); }
}
