"""
bench_distilbert.py: time DistilBERT inference inside the spamfilter container.

Run with:
    docker exec spamfilter python3 /app/ml/bench_distilbert.py

I wrote this during the threading-deadlock investigation to settle one
question: was inference itself slow, or was the milter calling context the
problem? Running it directly in the container (without libmilter in the
loop) and seeing ~50 ms per prediction localised the bug to the milter
side, not to PyTorch. Kept around because it's also useful as a generic
sanity check after rebuilding the spamfilter image, especially if I ever
swap container architectures (amd64 under Rosetta vs native arm64).
"""

import sys
import time
sys.path.insert(0, '/app/ml')

from predict import DistilBertSpamClassifier

print("Loading DistilBERT...")
t_load_0 = time.time()
clf = DistilBertSpamClassifier(model_dir='/app/ml/models/distilbert_spam_classifier')
print(f"Loaded in {time.time() - t_load_0:.2f}s")

cases = [
    ("Your invoice and account update",
     "Hello, please find attached your latest invoice. Your account password "
     "will need to be updated shortly. Click the secure link to confirm your details."),
    ("Meeting tomorrow",
     "Hi Abdulla, can we move our 1:1 to 3pm tomorrow?"),
    ("CONGRATULATIONS YOU WON",
     "Click here now to claim your prize money. Send your bank details."),
]

# First call always pays graph compilation + lazy init costs, so don't
# count it in the timed runs below.
print("\nWarmup call...")
t = time.time()
clf.predict(cases[0][0], cases[0][1])
print(f"Warmup: {time.time() - t:.2f}s")

print("\nTimed runs:")
for subject, body in cases:
    t = time.time()
    label, score = clf.predict(subject, body)
    elapsed = time.time() - t
    print(f"  [{elapsed:6.2f}s] label={label} score={score:.4f} subject={subject!r}")
