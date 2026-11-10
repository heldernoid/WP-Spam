"""Build the study / dev / test splits from the raw datasets, once.

Every message gets a bucket from a hash of (source, normalised text), so the split does not depend on file order.
  study  (bucket  0-59)  may be read: examples, errors, traces
  dev    (bucket 60-79)  totals only, never read
  test   (bucket 80-99)  sealed: data/sealed/test.jsonl, run once per frozen version
The URL source ships its own train/test: its train goes to study/dev, its test is sealed.
At most CAP messages per (source, kind) go into dev and test, taken in hash order; the rest of those buckets is
dropped, never moved into study. Exact duplicates (same normalised text) are kept once, in the first source.

kind: phish (phishing or scam), spam (spam, or "spam or phishing" when a corpus does not tell them apart), ham.
"""
import csv, hashlib, json, os, re, sys
import pyarrow.parquet as pq

csv.field_size_limit(10**9)
RAW = 'data/raw'
CAP = 1000

def norm(t):
    return re.sub(r'\s+', ' ', t.lower()).strip()[:3000]

def rows_csv(path):
    with open(path, encoding='utf-8', errors='replace', newline='') as fh:
        yield from csv.DictReader(fh)

def load():
    out = []
    # Mozambique, Portuguese SMS, mobile-money smishing
    for r in rows_csv(f'{RAW}/moz.csv'):
        out.append(('moz_sms', 'sms', r['text'], 'phish' if r['label'] == 'Smishing' else 'ham'))
    # Mendeley SMS (Mishra & Soni 2022): smishing, spam, ham
    for r in rows_csv(f'{RAW}/sms_mendeley/Dataset_5971.csv'):
        lab = r['LABEL'].lower()
        out.append(('mendeley_sms', 'sms', r['TEXT'], {'smishing': 'phish', 'spam': 'spam', 'ham': 'ham'}[lab]))
    # Email corpora (Champa, Rabbi & Zibran 2024, Zenodo 8339691)
    for name, src in [('Nazario', 'nazario'), ('Nigerian_Fraud', 'nigerian')]:
        for r in rows_csv(f'{RAW}/email_{name}.csv'):
            out.append((src, 'email', (r['subject'] or '') + '\n' + (r['body'] or ''), 'phish'))
    for name, src in [('SpamAssasin', 'spamassassin'), ('CEAS_08', 'ceas'), ('Enron', 'enron'), ('Ling', 'ling')]:
        for r in rows_csv(f'{RAW}/email_{name}.csv'):
            out.append((src, 'email', (r['subject'] or '') + '\n' + (r['body'] or ''), 'spam' if r['label'] == '1' else 'ham'))
    # URLs (Hannousse & Yahiouche 2021)
    url = []
    for part in ['train', 'test']:
        t = pq.read_table(f'{RAW}/url_{part}.parquet', columns=['url', 'status']).to_pydict()
        for u, s in zip(t['url'], t['status']):
            url.append((part, u, 'phish' if s == 'phishing' else 'ham'))
    return out, url

def h(*parts):
    return int(hashlib.sha256('\x00'.join(parts).encode()).hexdigest(), 16)

def main():
    msgs, urls = load()
    seen, study, dev, test = set(), [], [], []
    def add(src, ch, text, kind, forced=None):
        n = norm(text)
        if not n or n in seen:
            return
        seen.add(n)
        key = h(src, n)
        b = key % 100
        rec = {'source': src, 'channel': ch, 'kind': kind, 'text': text, '_k': key}
        if forced == 'test':
            test.append(rec)
        elif forced == 'trainpart':  # URL train: 75% study, 25% dev
            (study if b < 75 else dev).append(rec)
        else:
            (study if b < 60 else dev if b < 80 else test).append(rec)
    for src, ch, text, kind in msgs:
        add(src, ch, text, kind)
    for part, u, kind in urls:
        add('hannousse_url', 'url', u, kind, 'test' if part == 'test' else 'trainpart')

    def cap(recs):
        groups = {}
        for r in sorted(recs, key=lambda r: r['_k']):
            groups.setdefault((r['source'], r['kind']), []).append(r)
        return [r for g in groups.values() for r in g[:CAP]]
    dev, test = cap(dev), cap(test)

    os.makedirs('data/sealed', exist_ok=True)
    manifest = {}
    for name, recs, path in [('study', study, 'data/study.jsonl'), ('dev', dev, 'data/dev.jsonl'),
                             ('test', test, 'data/sealed/test.jsonl')]:
        recs.sort(key=lambda r: r['_k'])
        with open(path, 'w', encoding='utf-8') as fh:
            for i, r in enumerate(recs):
                r = {k: v for k, v in r.items() if k != '_k'}
                r['id'] = f'{name}-{i}'
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        counts = {}
        for r in recs:
            counts.setdefault(r['source'], {}).setdefault(r['kind'], 0)
            counts[r['source']][r['kind']] += 1
        manifest[name] = {'file': path, 'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(), 'counts': counts}
    json.dump(manifest, open('eval/splits.json', 'w'), indent=1, sort_keys=True)
    for k, v in manifest.items():
        print(k, sum(sum(c.values()) for c in v['counts'].values()), v['counts'])

if __name__ == '__main__':
    main()
