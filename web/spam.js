// WP-Spam: a neural network whose weights are written by hand (weights/spam.json, from gen/net.mjs).
//
//   text -> tokens
//        -> character layer      each character is embedded over the 26 letters (a "0" is mostly an "o", a "1" an "i"
//                                 or an "l"); stem filters are convolutions over those letter vectors. A filter fires
//                                 on a whole word, a word start, or anywhere inside a word. One winner per word
//                                 (the longest good match); affix filters (-ed, ...) fire alongside.
//        -> embedding            the winning filter's concept row, phrase rows (2-4 words), token shapes, URL features
//        -> attention, 2 layers   relation heads: a token reads its neighbours inside a window; layer 2 reads layer 1
//        -> pooling              sum over tokens, then log1p (URL features max-pooled over links)
//        -> sentence layer       each sentence pooled on its own and given clamped role units
//        -> message layer        role totals, role shares, arcs (a role followed by another within a few sentences)
//        -> MLP                  ReLU layers; each layer reads the pooled dims and every earlier layer
//        -> logit -> p(spam)
//
// Works in Node (require) and in the browser (window.WPSpam).
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.WPSpam = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const MAXTOK = 2500;

  function fold(s) { return s.normalize('NFKD').replace(/\p{M}/gu, ''); }

  const TOKRE = /(?:https?:\/\/|www\.)[^\s<>"')\]]+|[a-z0-9][a-z0-9.-]*\.(?:com|net|org|info|biz|ru|cn|tk|ml|ga|cf|gq|xyz|top|co|uk|io|me|ly|us|br|pt|mz|de|fr|it|es|nl|in|online|site|club)(?:\/[^\s<>"')\]]*)?(?![a-z])|[\p{L}\p{N}._%+-]+@[\p{L}\p{N}-]+(?:\.[\p{L}\p{N}-]+)+|[\p{L}\p{N}@$|]+(?:['’][\p{L}]+)?|[^\s\p{L}\p{N}]/giu;

  function tokenize(text) {
    const out = [];
    const lines = text.split('\n');
    for (let li = 0; li < lines.length && out.length < MAXTOK; li++) {
      const line = lines[li];
      const quoted = /^\s*>/.test(line);
      let m; TOKRE.lastIndex = 0;
      let first = true;
      while ((m = TOKRE.exec(line)) && out.length < MAXTOK) {
        let raw = m[0];
        // "$" or "@" or "|" alone, or at a word's edge, is punctuation ("$500", "@home"); inside a word it is a glyph ("ca$h")
        if (/[@$|]/.test(raw) && !/^[^@$|]+$/.test(raw) && !/[\p{L}\p{N}][@$|]+[\p{L}\p{N}]/u.test(raw)) {
          const parts = raw.split(/([@$|])/).filter(Boolean);
          for (const p of parts) { out.push({ raw: p, w: fold(p.toLowerCase()), quoted, lineStart: first, line: li }); first = false; }
          continue;
        }
        out.push({ raw, w: fold(raw.toLowerCase()), quoted, lineStart: first, line: li });
        first = false;
      }
    }
    return out;
  }

  function isUrl(w) { return /^(https?:\/\/|www\.)/.test(w) || (/\.[a-z]{2,6}(\/|$)/.test(w) && !/@/.test(w) && /^[a-z0-9]/.test(w) && w.includes('.') && !/^\d+(\.\d+)*$/.test(w)); }
  function isEmail(w) { return /^[^@\s]+@[^@\s]+\.[a-z]{2,}$/.test(w); }

  function parse(json) {
    const m = typeof json === 'string' ? JSON.parse(json) : json;
    m.D = m.dims.length;
    m.ix = Object.fromEntries(m.dims.map((d, i) => [d, i]));
    const vec = (row) => { const v = []; for (const [d, x] of Object.entries(row)) { if (!(d in m.ix)) throw new Error('unknown dim ' + d); v.push([m.ix[d], x]); } return v; };
    // character embedding: char -> [[letter, weight]]
    m.G = new Map(Object.entries(m.chars).map(([c, row]) => [c, Object.entries(row)]));
    // stem filters, indexed by their first two letters (one letter for one-letter filters)
    m.F = m.filters.map(([p, at, row]) => ({ p, L: p.length, at, t: p.length >= 7 ? m.tolLong : m.tolShort, v: vec(m.rows[row]), row }));
    m.fidx = new Map();
    for (const f of m.F) { const k = f.p.slice(0, 2); if (!m.fidx.has(k)) m.fidx.set(k, []); m.fidx.get(k).push(f); }
    m.affix = m.affixes.map(([p, minLen, row]) => ({ p, minLen, v: vec(m.rows[row]) }));
    m.phrase = new Map(Object.entries(m.phrases).map(([k, r]) => [k, vec(m.rows[r])]));
    m.maxPhrase = Math.max(2, ...Object.keys(m.phrases).map(k => k.split(' ').length));
    m.shapeV = Object.fromEntries(Object.entries(m.shapes).map(([k, row]) => [k, vec(row)]));
    m.urlV = Object.fromEntries(Object.entries(m.url.rows).map(([k, row]) => [k, vec(row)]));
    m.urlSets = m.url.sets;
    m.heads = m.heads.map(h => ({ ...h, qv: vec(h.query), kv: vec(h.key), oi: m.ix[h.out] }));
    m.roles = m.roles.map(r => ({ ...r, wv: vec(r.w) }));
    m.mlp = m.mlp.map(layer => layer.map(h => ({ ...h, wv: vec(h.w || {}), hw: Object.entries(h.h || {}) })));
    m.outV = vec(m.out.w);
    m.outH = Object.entries(m.out.hidden);
    m.func = new Set(m.functionWords.split(' '));
    m.rare = new Set(m.letterRare || []); m.onset = new Set(m.letterOnset || []);
    m.cache = new Map();
    return m;
  }

  // ---- character layer -------------------------------------------------------------------------------------------
  // score of filter f placed at s0 in word chars c: sum over its positions of the embedding weight of the letter it wants
  function fscore(m, f, c, s0) {
    let s = 0, glyph = false;
    for (let k = 0; k < f.L; k++) {
      const g = m.G.get(c[s0 + k]); if (!g) continue;
      let best = 0; for (const [l, x] of g) if (l === f.p[k] && x > best) best = x;
      if (best && best < 1) glyph = true;
      s += best;
    }
    return [Math.min(1, Math.max(0, (s - f.L + f.t) / f.t)), glyph];
  }
  function bestFilter(m, w) {
    const c = [...w], n = c.length;
    let best = null, bs = 0;
    const consider = (f, s0) => {
      if (f.at === 'word' && (s0 !== 0 || n !== f.L)) return;
      if (f.at === 'start' && s0 !== 0) return;
      if (s0 + f.L > n) return;
      const [a, glyph] = fscore(m, f, c, s0);
      if (a <= 0) return;
      const sc = a * f.L + (f.at === 'word' ? 0.01 : 0) - (f.at === 'any' ? 0.01 : 0);
      if (sc > bs) { bs = sc; best = { f, a, glyph }; }
    };
    for (let s0 = 0; s0 < n; s0++) {
      const g0 = m.G.get(c[s0]); if (!g0) continue;
      const g1 = s0 + 1 < n ? m.G.get(c[s0 + 1]) : null;
      for (const [l0] of g0) {
        const one = m.fidx.get(l0); if (one && s0 === 0) for (const f of one) if (f.L === 1) consider(f, 0);
        if (!g1) continue;
        for (const [l1] of g1) { const fs = m.fidx.get(l0 + l1); if (fs) for (const f of fs) if (f.L > 1 && (s0 === 0 || f.at === 'any')) consider(f, s0); }
      }
    }
    return best;
  }
  const collapse = w => w.replace(/(.)\1+/g, '$1');
  // a word's embedding: the winning stem filter's row (scaled by how well it matched), affix rows, and "obfuscated" if
  // the match needed digits or symbols read as letters. Cached: it depends only on the word.
  function wordVec(m, w) {
    let r = m.cache.get(w);
    if (r) return r;
    let b = bestFilter(m, w);
    if (m.fuzzy && w.length >= 5) { const cw = collapse(w); if (cw !== w) { const b2 = bestFilter(m, cw); if (b2 && (!b || b2.a * b2.f.L > b.a * b.f.L)) b = b2; } }
    const v = [];
    if (b) for (const [d, x] of b.f.v) v.push([d, x * b.a]);
    const obf = !!(b && b.glyph && b.a >= 0.5 && /[^a-z]/.test(w));
    for (const a of m.affix) if (w.length >= a.minLen && w.endsWith(a.p)) for (const [d, x] of a.v) v.push([d, x]);
    r = { v, obf, hit: b ? b.f.p : null };
    if (m.cache.size > 200000) m.cache.clear();
    m.cache.set(w, r);
    return r;
  }

  // ---- URL features ----------------------------------------------------------------------------
  function urlFeatures(m, u) {
    const S = m.urlSets, f = [];
    u = u.split('#')[0];
    let s = u.replace(/^[a-z]+:\/\//, '');
    const host = s.split(/[\/?#]/)[0].split(':')[0].replace(/^www\./, '');
    const rest = s.slice(s.indexOf(host) + host.length);
    const labels = host.split('.');
    const reg = labels.slice(-2).join('.');
    const tld = labels[labels.length - 1];
    if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) f.push('ip');
    if (u.includes('@')) f.push('at');
    if ((host.match(/-/g) || []).length >= 2) f.push('dashes');
    if (labels.length >= 5) f.push('subdomains');
    if (u.length > 75) f.push('long');
    if (u.length > 120) f.push('verylong');
    if (S.badTld.includes(tld)) f.push('badtld');
    if (S.freeHost.some(x => host.endsWith(x))) f.push('freehost');
    if (S.shortener.includes(host)) f.push('shortener');
    if ((host.match(/\d/g) || []).length >= 3) f.push('hostdigits');
    if (/\/\//.test(rest)) f.push('odd');
    if (/\/wp-(content|includes|admin)\/|\/plugins\/|\/themes\//.test(rest)) f.push('cmsPath');
    // a page inside a site's asset or system folder ("/includes/x/", "/img/login.php"), or inside a hidden folder
    if (S.assetDir) {
      const sg = rest.split('?')[0].split('/').filter(Boolean);
      if (sg.slice(0, -1).some(x => S.assetDir.includes(x))) f.push('assetDir');
      if (sg.some(x => /^\.[a-z0-9]/.test(x) && x !== '.well-known')) f.push('hiddenDir');
    }
    if (rest.split(/[\/?=&]/).some(seg => seg.length >= 16 && /\d/.test(seg) && /[a-z]/i.test(seg) && !/-/.test(seg))) f.push('randomPath');
    if ((rest.match(/\//g) || []).length >= 5) f.push('deep');
    if (/\.(pdf|doc|docx|ppt|pptx|txt)(\?|$)/.test(rest)) f.push('document');
    const segs = rest.split('?')[0].split('/').filter(Boolean);
    if (segs.some((x, k) => k > 0 && x === segs[k - 1])) f.push('repeatSeg');
    if (/\.(edu|gov|ac\.[a-z]{2}|edu\.[a-z]{2}|gov\.[a-z]{2})$/.test(host)) f.push('institution');
    if (S.knownSite && (S.knownSite.includes(reg) || S.knownSite.includes(labels.slice(-3).join('.')))) f.push('knownSite');
    const words = u.split(/[^a-z0-9]+/).filter(Boolean);
    let key = 0; for (const w of words) if (S.keyword.some(k => w.startsWith(k))) key++;
    for (let i = 0; i < Math.min(key, 3); i++) f.push('keyword');
    for (const b of S.brand) if (u.includes(b) && !reg.includes(b)) { f.push('brandelsewhere'); break; }
    if (/^http:\/\//.test(u)) f.push('http');
    if (/\.(php|asp|aspx|cgi|exe|zip|scr|js)(\?|$|\/)/.test(rest)) f.push('script');
    const articleWords = rest.split(/[\/_.?=&]+/).filter(seg => /^[a-z]+(-[a-z]+){2,}$/.test(seg)).length;
    if (articleWords) f.push('article');
    if (labels.length <= 2 && rest.replace(/\/$/, '') === '') f.push('bare');
    return f;
  }

  function unspace(t) {
    return t.replace(/\bh\s*t\s*t\s*p\s*(s?)\s*:\s*\/\s*\/\s*/gi, 'http$1://')
      .replace(/\b((?:www|[a-z0-9-]{2,})(?:\s?\.\s?[a-z0-9-]+)*)\s\.\s(com|net|org|biz|info|ru|cn|uk|de|co|us)\b/gi, (x) => x.replace(/\s/g, ''))
      .replace(/(https?:\/\/[^\s]+)\s\/\s/gi, '$1/');
  }

  function run(m, text, opt = {}) {
    if (m.unspace) text = unspace(text);
    const toks = tokenize(text);
    const n = toks.length, D = m.D;
    const X = new Float32Array(n * D);
    const add = (i, v, k = 1) => { for (const [d, x] of v) X[i * D + d] += x * k; };
    const sh = m.shapeV;
    let urls = 0, alpha = 0;
    const U = new Float64Array(D);
    const hits = opt.trace ? [] : null;
    // ---- embedding
    for (let i = 0; i < n; i++) {
      const t = toks[i], w = t.w, raw = t.raw;
      if (/^(https?:\/\/|www\.)/.test(w) || (w.includes('.') && /[a-z]/.test(w) && isUrl(w) && !isEmail(w))) {
        add(i, sh.url);
        if (urls++ < 20) { const u = new Float64Array(D); for (const f of urlFeatures(m, w)) for (const [d, x] of m.urlV[f]) u[d] += x; for (let d = 0; d < D; d++) if (u[d] > U[d]) U[d] = u[d]; }
        // words inside the link go through the character layer too; digits stay in ("paypa1" reads as "paypal")
        for (const part of w.replace(/^[a-z]+:\/\//, '').split(/[^a-z0-9@$]+/)) {
          if (part.length >= 4 && /[a-z]/.test(part) && !/^\d+$/.test(part)) { const r = wordVec(m, part); if (r.v.length) add(i, r.v, m.url.partScale); if (r.obf) add(i, sh.obfuscated); }
        }
        continue;
      }
      if (isEmail(w)) { add(i, sh.email); continue; }
      if (/^\d/.test(w) && /^[\d.,]+$/.test(w)) {
        const digits = w.replace(/[.,]/g, '');
        const val = +w.replace(/,/g, '').replace(/\.(?=\d{3}(\D|$))/g, '');
        add(i, sh.number);
        if (digits.length >= 9 && /^0|^\+|^44|^258|^8/.test(digits)) add(i, sh.phone);
        else if (digits.length >= 4 && digits.length <= 6 && i > 0 && /^(to|txt|text|send|call|sms|envie|para)$/.test(toks[i - 1].w)) add(i, sh.shortcode);
        else if (val >= 10000) add(i, sh.bignum);
        continue;
      }
      if (/^\d{1,4}p$|^\d{1,4}ppm$|^\d{1,4}pmin$/.test(w)) { add(i, sh.pence); continue; }
      if (/^\d{1,2}(am|pm|h|hrs?)$/.test(w)) { add(i, sh.clock); continue; }
      if (i === 1 && raw === ':' && toks[0].w === 're') add(i, sh.reply);
      if (raw.length === 1 && !/[\p{L}\p{N}]/u.test(raw)) {
        if (/^[:;=]$/.test(raw)) { let j = i + 1; if (j < n && toks[j].raw === '-') j++; if (j < n && /^[)(DPpOo3]$/.test(toks[j].raw)) add(i, sh.emoticon); }
        const p = sh['p' + raw];
        if (p) add(i, p);
        if (raw === '>' && t.lineStart) add(i, sh.quoteMark);
        continue;
      }
      if (/^\p{L}+$/u.test(w)) {
        alpha++;
        if (m.func.has(w)) add(i, sh.func);
      }
      const r = wordVec(m, w);
      if (hits && r.hit) hits.push([raw, r.hit]);
      if (r.v.length) add(i, r.v);
      if (r.obf) add(i, sh.obfuscated);
      if (/^\p{L}+$/u.test(w)) {
        if (w.length >= 2 && w.length <= 4 && /^[a-z]+$/.test(w) && !/[aeiou]/.test(w) && !m.func.has(w) && !r.hit) add(i, sh.junk);
        if (!r.hit && /^\p{Lu}\p{Ll}{1,14}$/u.test(raw) && !m.func.has(w)) add(i, sh.capName);
        // letter-pair layer: a word no filter knows, with a vowel, that cannot be read aloud
        if (sh.unpron && !r.hit && w.length >= 4 && /^[a-z]+$/.test(w) && /[aeiouy]/.test(w) && !m.func.has(w)) {
          let u = 0;
          for (let k = 0; k + 1 < w.length; k++) if (m.rare.has(w[k] + w[k + 1])) u++;
          if (!/[aeiouy]/.test(w[0]) && !/[aeiouy]/.test(w[1]) && !m.onset.has(w.slice(0, 2))) u++;
          if (u - 0.5 > 0) add(i, sh.unpron, Math.min(1, u - 0.5) * 2 > 1 ? 1 : Math.min(1, (u - 0.5) * 2));
        }
      }
      if (/^[A-Z]{3,}$/.test(raw) && raw.length <= 15) add(i, sh.allcaps);
      if (w.length >= 9 && /^[a-z]+$/.test(w) && !/[aeiouy]{1}/.test(w.slice(1, -1))) add(i, sh.random);
      if (w.length >= 18 && /^[a-z0-9]+$/.test(w) && /\d/.test(w) && /[a-z]/.test(w)) add(i, sh.random);
      if (t.quoted) add(i, sh.inQuote);
    }
    if (n && toks[n - 1].raw === '?') { /* a question at the very end is counted by the shape row like any other */ }
    // a subject that opens with a list tag: "[opensuse] ...", "Re: [ILUG] ..."
    for (let i = 0; i < Math.min(n, 4); i++) if (toks[i].raw === '[') { for (let j = i + 2; j <= Math.min(n - 1, i + 5); j++) if (toks[j].raw === ']') { add(i, sh.listTag); break; } break; }
    // letters spelt out one per token: "v i a g r a", "V-I-A-G-R-A"
    for (let i = 0; i < n; i++) {
      if (!/^[a-z]$/.test(toks[i].w)) continue;
      let j = i, s = '';
      while (j < n && (/^[a-z]$/.test(toks[j].w) || (/^[.\-_*]$/.test(toks[j].w) && j + 1 < n && /^[a-z]$/.test(toks[j + 1].w)))) { if (/^[a-z]$/.test(toks[j].w)) s += toks[j].w; j++; }
      if (s.length >= 4) { const r = wordVec(m, s); if (r.hit && r.hit.length >= 4) { add(i, r.v); add(i, sh.obfuscated); } }
      i = j;
    }
    // phrases: 2-4 adjacent word tokens (punctuation skipped); the row goes on the last word
    const wi = []; for (let i = 0; i < n; i++) if (/[\p{L}\p{N}]/u.test(toks[i].w)) wi.push(i);
    for (let k = 1; k < wi.length; k++) {
      let key = toks[wi[k]].w;
      for (let len = 2; len <= m.maxPhrase && k - len + 1 >= 0; len++) {
        key = toks[wi[k - len + 1]].w + ' ' + key;
        const v = m.phrase.get(key); if (v) add(wi[k], v);
      }
    }

    // ---- attention: layer 1 reads the embeddings, layer 2 reads embeddings + layer 1.
    // score(j) = beta * min(1, key . x_j) - slope * |i - j|; softmax over the window plus a sink; the attended key mass,
    // times min(1, query . x_i), is written to the head's output dim at token i.
    const H = new Float32Array(n * D);
    const layers = [...new Set(m.heads.map(h => h.layer || 1))].sort();
    let Z = X;
    for (const L of layers) {
      const Hl = new Float32Array(n * D);
      for (const h of m.heads) {
        if ((h.layer || 1) !== L) continue;
        for (let i = 0; i < n; i++) {
          let q = 0; for (const [d, x] of h.qv) q += x * Z[i * D + d];
          if (q <= 0) continue;
          const lo = Math.max(0, i - h.before), hi = Math.min(n - 1, i + h.after);
          let z = Math.exp(h.sink), acc = 0;
          for (let j = lo; j <= hi; j++) {
            if (j === i) continue;
            let k = 0; for (const [d, x] of h.kv) k += x * Z[j * D + d];
            if (k <= 0) continue;
            k = Math.min(k, 1);
            const e = Math.exp(h.beta * k - h.slope * Math.abs(i - j));
            z += e; acc += e * k;
          }
          Hl[i * D + h.oi] += Math.min(q, 1) * acc / z;
        }
      }
      for (let x = 0; x < n * D; x++) H[x] += Hl[x];
      if (L !== layers[layers.length - 1]) { Z = new Float32Array(n * D); for (let x = 0; x < n * D; x++) Z[x] = X[x] + H[x]; }
    }
    // ---- pooling
    const S = new Float64Array(D);
    for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) S[d] += X[i * D + d] + H[i * D + d];
    for (let d = 0; d < D; d++) S[d] += U[d];
    const F = new Float64Array(D);
    for (let d = 0; d < D; d++) F[d] = Math.log1p(Math.max(0, S[d]));
    F[m.ix.length] = Math.log1p(n);
    F[m.ix.words] = Math.log1p(alpha);
    F[m.ix.urlOnly] = n === 1 && S[m.ix.url] > 0 ? 1 : 0;
    F[m.ix.hasUrl] = S[m.ix.url] > 0 ? 1 : 0;
    for (const r of m.ratios) F[m.ix[r.out]] = alpha < r.minWords ? r.empty : (r.num === '@alpha' ? alpha : S[m.ix[r.num]]) / (r.den === 'tokens' ? n : alpha);
    // ---- sentence layer
    const sid = new Int32Array(n); let sc = 0;
    for (let i = 0; i < n; i++) { if (i > 0 && (toks[i].line !== toks[i - 1].line || /^[.!?]$/.test(toks[i - 1].raw))) sc++; sid[i] = sc; }
    const ns = n ? sc + 1 : 0;
    const P = new Float64Array(ns * D);
    for (let i = 0; i < n; i++) { const o = sid[i] * D; for (let d = 0; d < D; d++) P[o + d] += X[i * D + d] + H[i * D + d]; }
    const RL = m.roles.length, Rv = new Float64Array(ns * RL);
    let used = 0;
    for (let k = 0; k < ns; k++) {
      let any = false; for (let d = 0; d < D; d++) { const v = P[k * D + d]; P[k * D + d] = Math.log1p(Math.max(0, v)); if (v > 0) any = true; }
      if (!any) continue;
      used++;
      for (let r = 0; r < RL; r++) { let z = m.roles[r].b; for (const [d, x] of m.roles[r].wv) z += x * P[k * D + d]; Rv[k * RL + r] = Math.min(1, Math.max(0, z)); }
    }
    // ---- message layer
    const roleOut = {};
    for (let r = 0; r < RL; r++) {
      let t = 0; for (let k = 0; k < ns; k++) t += Rv[k * RL + r];
      F[m.ix['r_' + m.roles[r].name]] = Math.log1p(t);
      if (m.ix['s_' + m.roles[r].name] !== undefined) F[m.ix['s_' + m.roles[r].name]] = used >= 3 ? t / used : 0;
      roleOut[m.roles[r].name] = t;
    }
    const ri = Object.fromEntries(m.roles.map((r, k) => [r.name, k]));
    for (const a of m.arcs) {
      let t = 0;
      for (let k = 0; k < ns; k++) {
        const x = Rv[k * RL + ri[a.from]]; if (!x) continue;
        let best = 0; for (let j = k + (a.same ? 0 : 1); j <= Math.min(ns - 1, k + a.window); j++) best = Math.max(best, Rv[j * RL + ri[a.to]]);
        t += x * best;
      }
      F[m.ix['a_' + a.name]] = Math.log1p(t);
    }
    // ---- MLP: each layer reads the pooled dims (w) and the units of earlier layers (h); clamp units saturate at 1
    const hid = {};
    for (const layer of m.mlp) for (const h of layer) {
      let s = h.b; for (const [d, x] of h.wv) s += x * F[d]; for (const [k, x] of h.hw) s += x * hid[k];
      s = Math.max(0, s); hid[h.name] = h.clamp ? Math.min(1, s) : s;
    }
    // ---- output
    let z = m.out.b;
    const parts = opt.trace ? {} : null;
    for (const [d, x] of m.outV) { z += x * F[d]; if (parts && F[d]) parts[m.dims[d]] = x * F[d]; }
    for (const [name, x] of m.outH) { z += x * hid[name]; if (parts && hid[name]) parts['h:' + name] = x * hid[name]; }
    const p = 1 / (1 + Math.exp(-z));
    return opt.trace ? { p, z, F: Object.fromEntries(m.dims.map((d, i) => [d, F[i]]).filter(([, x]) => x)), hidden: hid, parts, tokens: n, roles: roleOut, hits } : { p, z };
  }

  function features(m, text) { const r = run(m, text, { trace: true }); return { F: r.F, hidden: r.hidden }; }

  return { parse, run, tokenize, features, urlFeatures, wordVec };
});
