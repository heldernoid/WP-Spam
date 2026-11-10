# Sealed test runs

Each frozen weights file is run once on data/sealed/test.jsonl. Results are recorded whatever they are.


The repository keeps only the current network. Earlier entries name weight files that are no longer in it; their
sha256 identify them, and LOG.md tells what each stage was.

## 2026-11-10 · hand-written v0 and v1 against trained comparisons

v0 sha256 `3e856fdc5ae4128f35f8d7c4a3ff3efeb9f1f633f0ba6fb283a91de8623b4282`
v1 sha256 `b4276ea1aead6cc044bfc801e7372cccb1d9350dce51cba23bdf2a0ea287ac5a`

split test: 10351 messages

| Weights | Labels seen | Non-zero params | Accuracy | Balanced |
|---|---|---|---|---|
| WP-Spam v0, hand-written | 0 | 1,157 | 83.4% | 83.0% |
| WP-Spam v1, hand-written, one revision | 7,723 | 1,403 | 87.5% | 87.2% |
| trained bag-of-words, 512 labels | 512 | 65,875 | 87.8% | 87.5% |
| WP-Spam layers + trained output, 512 labels | 512 | 67 | 88.6% | 88.3% |
| trained bag-of-words, 7,723 labels | 7,723 | 240,424 | 94.6% | 94.4% |
| WP-Spam layers + trained output, 7,723 labels | 7,723 | 67 | 90.1% | 89.7% |
| trained bag-of-words, 58,141 labels | 58,141 | 262,141 | 97.1% | 97.0% |
| WP-Spam layers + trained output, 58,141 labels | 58,141 | 67 | 90.0% | 89.8% |

### WP-Spam v0, hand-written

```
source            spam/phish caught    ham passed       accuracy
ceas               71.5% of 1000    84.9% of 1000    78.2%
enron              74.4% of 1000    93.3% of 1000    83.8%
hannousse_url      71.5% of 1000    86.7% of 1000    79.1%
ling               92.4% of 92      97.3% of 475     96.5%
mendeley_sms       90.1% of 203     96.3% of 971     95.2%
moz_sms            81.1% of 111     54.4% of 388     60.3%
nazario            76.3% of 304         - of 0       76.3%
nigerian           96.9% of 650         - of 0       96.9%
spamassassin       92.6% of 338     81.2% of 819     84.5%

accuracy 83.4%  (8629 of 10351)   spam caught 78.9%   ham passed 87.1%   balanced 83.0%
```

### WP-Spam v1, hand-written, one revision

```
source            spam/phish caught    ham passed       accuracy
ceas               87.4% of 1000    86.3% of 1000    86.8%
enron              80.1% of 1000    93.9% of 1000    87.0%
hannousse_url      70.9% of 1000    92.2% of 1000    81.5%
ling               94.6% of 92      96.8% of 475     96.5%
mendeley_sms       93.1% of 203     95.0% of 971     94.6%
moz_sms            90.1% of 111     75.5% of 388     78.8%
nazario            88.8% of 304         - of 0       88.8%
nigerian           95.5% of 650         - of 0       95.5%
spamassassin       94.7% of 338     83.3% of 819     86.6%

accuracy 87.5%  (9052 of 10351)   spam caught 84.5%   ham passed 89.9%   balanced 87.2%
```

### trained bag-of-words, 512 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               88.4% of 1000    89.7% of 1000    89.0%
enron              74.3% of 1000    92.9% of 1000    83.6%
hannousse_url      67.9% of 1000    82.9% of 1000    75.4%
ling               96.7% of 92      98.7% of 475     98.4%
mendeley_sms       81.8% of 203     93.1% of 971     91.1%
moz_sms            94.6% of 111     95.6% of 388     95.4%
nazario            97.0% of 304         - of 0       97.0%
nigerian           99.2% of 650         - of 0       99.2%
spamassassin       95.3% of 338     93.5% of 819     94.0%

accuracy 87.8%  (9093 of 10351)   spam caught 83.6%   ham passed 91.4%   balanced 87.5%
```

### WP-Spam layers + trained output, 512 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               90.5% of 1000    88.9% of 1000    89.7%
enron              78.6% of 1000    95.7% of 1000    87.2%
hannousse_url      70.9% of 1000    92.6% of 1000    81.8%
ling               90.2% of 92      99.6% of 475     98.1%
mendeley_sms       86.2% of 203     93.5% of 971     92.2%
moz_sms            91.0% of 111     87.9% of 388     88.6%
nazario            90.1% of 304         - of 0       90.1%
nigerian           98.8% of 650         - of 0       98.8%
spamassassin       88.2% of 338     86.0% of 819     86.6%

accuracy 88.6%  (9171 of 10351)   spam caught 84.6%   ham passed 92.0%   balanced 88.3%
```

### trained bag-of-words, 7,723 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               96.0% of 1000    96.1% of 1000    96.0%
enron              88.7% of 1000    98.4% of 1000    93.5%
hannousse_url      84.2% of 1000    92.9% of 1000    88.5%
ling               97.8% of 92      99.8% of 475     99.5%
mendeley_sms       92.1% of 203     94.6% of 971     94.2%
moz_sms            95.5% of 111     97.2% of 388     96.8%
nazario            97.7% of 304         - of 0       97.7%
nigerian           99.5% of 650         - of 0       99.5%
spamassassin       96.4% of 338     98.7% of 819     98.0%

accuracy 94.6%  (9794 of 10351)   spam caught 92.4%   ham passed 96.4%   balanced 94.4%
```

### WP-Spam layers + trained output, 7,723 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               92.4% of 1000    92.0% of 1000    92.2%
enron              81.2% of 1000    93.4% of 1000    87.3%
hannousse_url      70.4% of 1000    93.7% of 1000    82.0%
ling               92.4% of 92      98.9% of 475     97.9%
mendeley_sms       90.6% of 203     92.7% of 971     92.3%
moz_sms            95.5% of 111     95.1% of 388     95.2%
nazario            92.1% of 304         - of 0       92.1%
nigerian           99.1% of 650         - of 0       99.1%
spamassassin       91.1% of 338     91.0% of 819     91.0%

accuracy 90.1%  (9322 of 10351)   spam caught 86.1%   ham passed 93.3%   balanced 89.7%
```

### trained bag-of-words, 58,141 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               99.2% of 1000    99.6% of 1000    99.4%
enron              96.9% of 1000    99.6% of 1000    98.3%
hannousse_url      92.2% of 1000    92.9% of 1000    92.5%
ling              100.0% of 92      99.6% of 475     99.6%
mendeley_sms       89.7% of 203     96.2% of 971     95.1%
moz_sms            95.5% of 111     97.2% of 388     96.8%
nazario            96.7% of 304         - of 0       96.7%
nigerian           99.4% of 650         - of 0       99.4%
spamassassin       98.2% of 338     98.9% of 819     98.7%

accuracy 97.1%  (10050 of 10351)   spam caught 96.5%   ham passed 97.6%   balanced 97.0%
```

### WP-Spam layers + trained output, 58,141 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               97.9% of 1000    94.8% of 1000    96.3%
enron              89.4% of 1000    94.4% of 1000    91.9%
hannousse_url      70.7% of 1000    92.7% of 1000    81.7%
ling               94.6% of 92      97.3% of 475     96.8%
mendeley_sms       87.2% of 203     87.0% of 971     87.1%
moz_sms            91.9% of 111     87.6% of 388     88.6%
nazario            83.9% of 304         - of 0       83.9%
nigerian           97.5% of 650         - of 0       97.5%
spamassassin       85.5% of 338     88.0% of 819     87.3%

accuracy 90.0%  (9311 of 10351)   spam caught 87.8%   ham passed 91.8%   balanced 89.8%
```

## 2026-11-10 · hand-written v0, v1, v2 against trained comparisons

v0 sha256 `3e856fdc5ae4128f35f8d7c4a3ff3efeb9f1f633f0ba6fb283a91de8623b4282`
v1 sha256 `b4276ea1aead6cc044bfc801e7372cccb1d9350dce51cba23bdf2a0ea287ac5a`
v2 sha256 `3b1821a3c3a76a3a3e665c47d9195e12a6e532487f2f51a7c6698d2962814c53`

split test: 10351 messages

| Weights | Labels seen | Non-zero params | Accuracy | Balanced |
|---|---|---|---|---|
| WP-Spam v0, hand-written | 0 | 1,157 | 83.4% | 83.0% |
| WP-Spam v1, hand-written, one revision | 7,723 | 1,403 | 87.5% | 87.2% |
| WP-Spam v2, hand-written, second revision | 58,141 | 1,674 | 88.6% | 88.3% |
| trained bag-of-words, 512 labels | 512 | 65,875 | 87.8% | 87.5% |
| WP-Spam layers + trained output, 512 labels | 512 | 67 | 88.6% | 88.3% |
| trained bag-of-words, 7,723 labels | 7,723 | 240,424 | 94.6% | 94.4% |
| WP-Spam layers + trained output, 7,723 labels | 7,723 | 67 | 90.1% | 89.7% |
| trained bag-of-words, 58,141 labels | 58,141 | 262,141 | 97.1% | 97.0% |
| WP-Spam layers + trained output, 58,141 labels | 58,141 | 67 | 90.0% | 89.8% |

### WP-Spam v0, hand-written

```
source            spam/phish caught    ham passed       accuracy
ceas               71.5% of 1000    84.9% of 1000    78.2%
enron              74.4% of 1000    93.3% of 1000    83.8%
hannousse_url      71.5% of 1000    86.7% of 1000    79.1%
ling               92.4% of 92      97.3% of 475     96.5%
mendeley_sms       90.1% of 203     96.3% of 971     95.2%
moz_sms            81.1% of 111     54.4% of 388     60.3%
nazario            76.3% of 304         - of 0       76.3%
nigerian           96.9% of 650         - of 0       96.9%
spamassassin       92.6% of 338     81.2% of 819     84.5%

accuracy 83.4%  (8629 of 10351)   spam caught 78.9%   ham passed 87.1%   balanced 83.0%
```

### WP-Spam v1, hand-written, one revision

```
source            spam/phish caught    ham passed       accuracy
ceas               87.4% of 1000    86.3% of 1000    86.8%
enron              80.1% of 1000    93.9% of 1000    87.0%
hannousse_url      70.9% of 1000    92.2% of 1000    81.5%
ling               94.6% of 92      96.8% of 475     96.5%
mendeley_sms       93.1% of 203     95.0% of 971     94.6%
moz_sms            90.1% of 111     75.5% of 388     78.8%
nazario            88.8% of 304         - of 0       88.8%
nigerian           95.5% of 650         - of 0       95.5%
spamassassin       94.7% of 338     83.3% of 819     86.6%

accuracy 87.5%  (9052 of 10351)   spam caught 84.5%   ham passed 89.9%   balanced 87.2%
```

### WP-Spam v2, hand-written, second revision

```
source            spam/phish caught    ham passed       accuracy
ceas               91.8% of 1000    88.4% of 1000    90.1%
enron              80.4% of 1000    93.1% of 1000    86.8%
hannousse_url      71.2% of 1000    91.7% of 1000    81.5%
ling               93.5% of 92      97.5% of 475     96.8%
mendeley_sms       92.1% of 203     95.2% of 971     94.6%
moz_sms            90.1% of 111     83.5% of 388     85.0%
nazario            87.8% of 304         - of 0       87.8%
nigerian           94.6% of 650         - of 0       94.6%
spamassassin       94.7% of 338     87.9% of 819     89.9%

accuracy 88.6%  (9172 of 10351)   spam caught 85.3%   ham passed 91.3%   balanced 88.3%
```

### trained bag-of-words, 512 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               88.4% of 1000    89.7% of 1000    89.0%
enron              74.3% of 1000    92.9% of 1000    83.6%
hannousse_url      67.9% of 1000    82.9% of 1000    75.4%
ling               96.7% of 92      98.7% of 475     98.4%
mendeley_sms       81.8% of 203     93.1% of 971     91.1%
moz_sms            94.6% of 111     95.6% of 388     95.4%
nazario            97.0% of 304         - of 0       97.0%
nigerian           99.2% of 650         - of 0       99.2%
spamassassin       95.3% of 338     93.5% of 819     94.0%

accuracy 87.8%  (9093 of 10351)   spam caught 83.6%   ham passed 91.4%   balanced 87.5%
```

### WP-Spam layers + trained output, 512 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               90.5% of 1000    88.9% of 1000    89.7%
enron              78.6% of 1000    95.7% of 1000    87.2%
hannousse_url      70.9% of 1000    92.6% of 1000    81.8%
ling               90.2% of 92      99.6% of 475     98.1%
mendeley_sms       86.2% of 203     93.5% of 971     92.2%
moz_sms            91.0% of 111     87.9% of 388     88.6%
nazario            90.1% of 304         - of 0       90.1%
nigerian           98.8% of 650         - of 0       98.8%
spamassassin       88.2% of 338     86.0% of 819     86.6%

accuracy 88.6%  (9171 of 10351)   spam caught 84.6%   ham passed 92.0%   balanced 88.3%
```

### trained bag-of-words, 7,723 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               96.0% of 1000    96.1% of 1000    96.0%
enron              88.7% of 1000    98.4% of 1000    93.5%
hannousse_url      84.2% of 1000    92.9% of 1000    88.5%
ling               97.8% of 92      99.8% of 475     99.5%
mendeley_sms       92.1% of 203     94.6% of 971     94.2%
moz_sms            95.5% of 111     97.2% of 388     96.8%
nazario            97.7% of 304         - of 0       97.7%
nigerian           99.5% of 650         - of 0       99.5%
spamassassin       96.4% of 338     98.7% of 819     98.0%

accuracy 94.6%  (9794 of 10351)   spam caught 92.4%   ham passed 96.4%   balanced 94.4%
```

### WP-Spam layers + trained output, 7,723 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               92.4% of 1000    92.0% of 1000    92.2%
enron              81.2% of 1000    93.4% of 1000    87.3%
hannousse_url      70.4% of 1000    93.7% of 1000    82.0%
ling               92.4% of 92      98.9% of 475     97.9%
mendeley_sms       90.6% of 203     92.7% of 971     92.3%
moz_sms            95.5% of 111     95.1% of 388     95.2%
nazario            92.1% of 304         - of 0       92.1%
nigerian           99.1% of 650         - of 0       99.1%
spamassassin       91.1% of 338     91.0% of 819     91.0%

accuracy 90.1%  (9322 of 10351)   spam caught 86.1%   ham passed 93.3%   balanced 89.7%
```

### trained bag-of-words, 58,141 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               99.2% of 1000    99.6% of 1000    99.4%
enron              96.9% of 1000    99.6% of 1000    98.3%
hannousse_url      92.2% of 1000    92.9% of 1000    92.5%
ling              100.0% of 92      99.6% of 475     99.6%
mendeley_sms       89.7% of 203     96.2% of 971     95.1%
moz_sms            95.5% of 111     97.2% of 388     96.8%
nazario            96.7% of 304         - of 0       96.7%
nigerian           99.4% of 650         - of 0       99.4%
spamassassin       98.2% of 338     98.9% of 819     98.7%

accuracy 97.1%  (10050 of 10351)   spam caught 96.5%   ham passed 97.6%   balanced 97.0%
```

### WP-Spam layers + trained output, 58,141 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               97.9% of 1000    94.8% of 1000    96.3%
enron              89.4% of 1000    94.4% of 1000    91.9%
hannousse_url      70.7% of 1000    92.7% of 1000    81.7%
ling               94.6% of 92      97.3% of 475     96.8%
mendeley_sms       87.2% of 203     87.0% of 971     87.1%
moz_sms            91.9% of 111     87.6% of 388     88.6%
nazario            83.9% of 304         - of 0       83.9%
nigerian           97.5% of 650         - of 0       97.5%
spamassassin       85.5% of 338     88.0% of 819     87.3%

accuracy 90.0%  (9311 of 10351)   spam caught 87.8%   ham passed 91.8%   balanced 89.8%
```

## 2026-11-10 · hand-written v0-v3 against trained comparisons

v0 sha256 `3e856fdc5ae4128f35f8d7c4a3ff3efeb9f1f633f0ba6fb283a91de8623b4282`
v1 sha256 `b4276ea1aead6cc044bfc801e7372cccb1d9350dce51cba23bdf2a0ea287ac5a`
v2 sha256 `3b1821a3c3a76a3a3e665c47d9195e12a6e532487f2f51a7c6698d2962814c53`
v3 sha256 `8bf25924c71d7030e00d997d8adbb5aed4a4d5efee8ac10e669906a0e3a1e845`

split test: 10351 messages

| Weights | Labels seen | Non-zero params | Accuracy | Balanced |
|---|---|---|---|---|
| WP-Spam v0, hand-written | 0 | 1,157 | 83.4% | 83.0% |
| WP-Spam v1, hand-written, one revision | 7,723 | 1,403 | 87.5% | 87.2% |
| WP-Spam v2, hand-written, second revision | 58,141 | 1,674 | 88.6% | 88.3% |
| WP-Spam v3, hand-written, deeper (sentence, message, gate layers) | 58,141 | 1,781 | 88.8% | 88.5% |
| trained bag-of-words, 512 labels | 512 | 65,875 | 87.8% | 87.5% |
| WP-Spam layers + trained output, 512 labels | 512 | 67 | 88.6% | 88.3% |
| trained bag-of-words, 7,723 labels | 7,723 | 240,424 | 94.6% | 94.4% |
| WP-Spam layers + trained output, 7,723 labels | 7,723 | 67 | 90.1% | 89.7% |
| trained bag-of-words, 58,141 labels | 58,141 | 262,141 | 97.1% | 97.0% |
| WP-Spam layers + trained output, 58,141 labels | 58,141 | 67 | 90.0% | 89.8% |

### WP-Spam v0, hand-written

```
source            spam/phish caught    ham passed       accuracy
ceas               71.5% of 1000    84.9% of 1000    78.2%
enron              74.4% of 1000    93.3% of 1000    83.8%
hannousse_url      71.5% of 1000    86.7% of 1000    79.1%
ling               92.4% of 92      97.3% of 475     96.5%
mendeley_sms       90.1% of 203     96.3% of 971     95.2%
moz_sms            81.1% of 111     54.4% of 388     60.3%
nazario            76.3% of 304         - of 0       76.3%
nigerian           96.9% of 650         - of 0       96.9%
spamassassin       92.6% of 338     81.2% of 819     84.5%

accuracy 83.4%  (8629 of 10351)   spam caught 78.9%   ham passed 87.1%   balanced 83.0%
```

### WP-Spam v1, hand-written, one revision

```
source            spam/phish caught    ham passed       accuracy
ceas               87.4% of 1000    86.3% of 1000    86.8%
enron              80.1% of 1000    93.9% of 1000    87.0%
hannousse_url      70.9% of 1000    92.2% of 1000    81.5%
ling               94.6% of 92      96.8% of 475     96.5%
mendeley_sms       93.1% of 203     95.0% of 971     94.6%
moz_sms            90.1% of 111     75.5% of 388     78.8%
nazario            88.8% of 304         - of 0       88.8%
nigerian           95.5% of 650         - of 0       95.5%
spamassassin       94.7% of 338     83.3% of 819     86.6%

accuracy 87.5%  (9052 of 10351)   spam caught 84.5%   ham passed 89.9%   balanced 87.2%
```

### WP-Spam v2, hand-written, second revision

```
source            spam/phish caught    ham passed       accuracy
ceas               91.8% of 1000    88.4% of 1000    90.1%
enron              80.4% of 1000    93.1% of 1000    86.8%
hannousse_url      71.2% of 1000    91.7% of 1000    81.5%
ling               93.5% of 92      97.5% of 475     96.8%
mendeley_sms       92.1% of 203     95.2% of 971     94.6%
moz_sms            90.1% of 111     83.5% of 388     85.0%
nazario            87.8% of 304         - of 0       87.8%
nigerian           94.6% of 650         - of 0       94.6%
spamassassin       94.7% of 338     87.9% of 819     89.9%

accuracy 88.6%  (9172 of 10351)   spam caught 85.3%   ham passed 91.3%   balanced 88.3%
```

### WP-Spam v3, hand-written, deeper (sentence, message, gate layers)

```
source            spam/phish caught    ham passed       accuracy
ceas               90.5% of 1000    88.6% of 1000    89.5%
enron              81.7% of 1000    92.5% of 1000    87.1%
hannousse_url      71.2% of 1000    91.7% of 1000    81.5%
ling               94.6% of 92      97.3% of 475     96.8%
mendeley_sms       92.1% of 203     96.2% of 971     95.5%
moz_sms            91.0% of 111     84.5% of 388     86.0%
nazario            87.8% of 304         - of 0       87.8%
nigerian           95.5% of 650         - of 0       95.5%
spamassassin       95.0% of 338     88.0% of 819     90.1%

accuracy 88.8%  (9191 of 10351)   spam caught 85.5%   ham passed 91.5%   balanced 88.5%
```

### trained bag-of-words, 512 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               88.4% of 1000    89.7% of 1000    89.0%
enron              74.3% of 1000    92.9% of 1000    83.6%
hannousse_url      67.9% of 1000    82.9% of 1000    75.4%
ling               96.7% of 92      98.7% of 475     98.4%
mendeley_sms       81.8% of 203     93.1% of 971     91.1%
moz_sms            94.6% of 111     95.6% of 388     95.4%
nazario            97.0% of 304         - of 0       97.0%
nigerian           99.2% of 650         - of 0       99.2%
spamassassin       95.3% of 338     93.5% of 819     94.0%

accuracy 87.8%  (9093 of 10351)   spam caught 83.6%   ham passed 91.4%   balanced 87.5%
```

### WP-Spam layers + trained output, 512 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               90.5% of 1000    88.9% of 1000    89.7%
enron              78.6% of 1000    95.7% of 1000    87.2%
hannousse_url      70.9% of 1000    92.6% of 1000    81.8%
ling               90.2% of 92      99.6% of 475     98.1%
mendeley_sms       86.2% of 203     93.5% of 971     92.2%
moz_sms            91.0% of 111     87.9% of 388     88.6%
nazario            90.1% of 304         - of 0       90.1%
nigerian           98.8% of 650         - of 0       98.8%
spamassassin       88.2% of 338     86.0% of 819     86.6%

accuracy 88.6%  (9171 of 10351)   spam caught 84.6%   ham passed 92.0%   balanced 88.3%
```

### trained bag-of-words, 7,723 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               96.0% of 1000    96.1% of 1000    96.0%
enron              88.7% of 1000    98.4% of 1000    93.5%
hannousse_url      84.2% of 1000    92.9% of 1000    88.5%
ling               97.8% of 92      99.8% of 475     99.5%
mendeley_sms       92.1% of 203     94.6% of 971     94.2%
moz_sms            95.5% of 111     97.2% of 388     96.8%
nazario            97.7% of 304         - of 0       97.7%
nigerian           99.5% of 650         - of 0       99.5%
spamassassin       96.4% of 338     98.7% of 819     98.0%

accuracy 94.6%  (9794 of 10351)   spam caught 92.4%   ham passed 96.4%   balanced 94.4%
```

### WP-Spam layers + trained output, 7,723 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               92.4% of 1000    92.0% of 1000    92.2%
enron              81.2% of 1000    93.4% of 1000    87.3%
hannousse_url      70.4% of 1000    93.7% of 1000    82.0%
ling               92.4% of 92      98.9% of 475     97.9%
mendeley_sms       90.6% of 203     92.7% of 971     92.3%
moz_sms            95.5% of 111     95.1% of 388     95.2%
nazario            92.1% of 304         - of 0       92.1%
nigerian           99.1% of 650         - of 0       99.1%
spamassassin       91.1% of 338     91.0% of 819     91.0%

accuracy 90.1%  (9322 of 10351)   spam caught 86.1%   ham passed 93.3%   balanced 89.7%
```

### trained bag-of-words, 58,141 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               99.2% of 1000    99.6% of 1000    99.4%
enron              96.9% of 1000    99.6% of 1000    98.3%
hannousse_url      92.2% of 1000    92.9% of 1000    92.5%
ling              100.0% of 92      99.6% of 475     99.6%
mendeley_sms       89.7% of 203     96.2% of 971     95.1%
moz_sms            95.5% of 111     97.2% of 388     96.8%
nazario            96.7% of 304         - of 0       96.7%
nigerian           99.4% of 650         - of 0       99.4%
spamassassin       98.2% of 338     98.9% of 819     98.7%

accuracy 97.1%  (10050 of 10351)   spam caught 96.5%   ham passed 97.6%   balanced 97.0%
```

### WP-Spam layers + trained output, 58,141 labels

```
source            spam/phish caught    ham passed       accuracy
ceas               97.9% of 1000    94.8% of 1000    96.3%
enron              89.4% of 1000    94.4% of 1000    91.9%
hannousse_url      70.7% of 1000    92.7% of 1000    81.7%
ling               94.6% of 92      97.3% of 475     96.8%
mendeley_sms       87.2% of 203     87.0% of 971     87.1%
moz_sms            91.9% of 111     87.6% of 388     88.6%
nazario            83.9% of 304         - of 0       83.9%
nigerian           97.5% of 650         - of 0       97.5%
spamassassin       85.5% of 338     88.0% of 819     87.3%

accuracy 90.0%  (9311 of 10351)   spam caught 87.8%   ham passed 91.8%   balanced 89.8%
```

## 2026-11-10 · v4: deeper hand-written network (character layer, 36 heads in 2 layers, 10 roles, 3-layer MLP), 25,114 non-zero params, labels seen 58,141

weights `weights/v4.json` sha256 `d12b836cca6c3a488410965acf60294ffe90131364c9d39b48e3515155edb6b0`

```
source            spam/phish caught    ham passed       accuracy
ceas               90.1% of 1000    90.1% of 1000    90.1%
enron              81.8% of 1000    93.3% of 1000    87.5%
hannousse_url      75.9% of 1000    89.0% of 1000    82.5%
ling               94.6% of 92      97.7% of 475     97.2%
mendeley_sms       91.6% of 203     96.6% of 971     95.7%
moz_sms            89.2% of 111     90.5% of 388     90.2%
nazario            87.8% of 304         - of 0       87.8%
nigerian           94.9% of 650         - of 0       94.9%
spamassassin       93.5% of 338     90.0% of 819     91.0%

accuracy 89.5%  (9264 of 10351)   spam caught 86.2%   ham passed 92.2%   balanced 89.2%
```

## 2026-11-10 · note: the same network under a new file hash (not a test run)

When only the current network was kept, the two weight sources were merged into one file and version labels were
removed from the reasons. `weights/spam.json` sha256 `076a06de1af48bb8eb8ac4463bdce8cb6651c07d77a8c8f45b7105d36d88e201` holds exactly the network of the run above
(sha256 `d12b836cca6c…`): every weight, filter, head and unit is identical; only the `why` texts and the version
field changed. Dev re-scored to the same 9,160 of 10,249. The sealed test was not run again.
