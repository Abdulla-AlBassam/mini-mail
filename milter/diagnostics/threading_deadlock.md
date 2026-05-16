# Threading deadlock: forensic diagnostic brain dump

This is the full diagnostic narrative for the threading deadlock I hit during early cascade-milter integration. A condensed version sits in the dissertation report at Section 3.2.8 (`\label{threading-deadlock}`); this file keeps the detail (py-spy traces, file:line references, container-level configuration) that the summary draws on.

## Symptom

I submitted three messages through the authenticated submission service: an obvious ham, an obvious spam, and a borderline phishing attempt. The first two were classified by the baseline within roughly 50 milliseconds and behaved normally. The third logged 'Escalating to DistilBERT (baseline confidence=0.6293, threshold=0.90)'


and then went silent. The SMTP client timed out at 30 seconds, Postfix waited the full 180-second `milter_content_timeout`, and the message was accepted under `milter_default_action = accept` with no spam headers attached. The deadlock reproduced on fresh container starts, so it was not stale state.

## Diagnostic process

The milter's `eom()` callback wraps classification in a `try/except` that logs any thrown exception (`milter/spam_milter.py:274-275`). No error message was logged, which ruled out an exception and pointed at a deadlock instead.

Two diagnostic tests isolated the cause:

1. **Standalone benchmark.** `ml/bench_distilbert.py` calls `DistilBertSpamClassifier.predict()` directly inside the same container. It returned in around 50 ms after warmup. So inference itself was not the problem; the problem was the calling context (running inside a libmilter thread).

2. **Live py-spy dump.** A `py-spy dump` against the running process (saved to `drafts/py_spy_wedged_stack.txt`) showed a libmilter worker thread suspended inside `BatchEncoding.to(self.device)` at `predict.py:193`.

## Root cause

Three things lined up:

1. **libmilter dispatches callbacks on per-connection worker threads.** Milter callbacks come in via libmilter, the Sendmail-derived library that Postfix integrates with for the milter protocol (Postfix Project, n.d.f). The py-spy stack confirmed that under load these callbacks ran on per-connection workers, not the main thread.

2. **libgomp initialises lazily on first parallel operation.** PyTorch uses libgomp, the GNU OpenMP runtime, for intra-op CPU parallelism. libgomp creates its internal thread pool on the first parallel operation, not at load time (Free Software Foundation, n.d.).

3. **The first inference landed on a non-main thread.** The DistilBERT model was constructed on the main thread inside `load_models()`, but no inference was actually run there, so the very first forward pass happened inside `eom()` on a libmilter worker thread.

On the Apple Silicon host I was developing on, the x86_64 PyTorch image was running under Rosetta. The libgomp first-init path on a non-main pthread deadlocked, leaving the worker stuck in `pthread_cond_wait`.

## Fix

Two layers, applied in `milter/spam_milter.py` and `docker-compose.yml`.

### Layer 1: pinned-thread executor + warmup

I added a single-worker `ThreadPoolExecutor` as a module-level global, so every DistilBERT inference is serialised onto the same dedicated thread. Paired with that is a warmup call at the end of `load_models()` that runs one dummy prediction on that thread before the milter accepts any SMTP traffic. The warmup forces OpenMP, MKL, and the tokeniser to do their lazy initialisation once, on a thread of my choosing, while the system is idle. By the time the first real email arrives, libgomp's first-init path has already completed and will never run again. Because every later inference runs on the same warmed-up thread, no libmilter worker is ever asked to trigger the deadlocking setup.

### Layer 2: container-level threading caps

The `spamfilter` service in `docker-compose.yml` pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to one and disables HuggingFace's tokeniser thread pool with `TOKENIZERS_PARALLELISM=false`. Restricting these libraries to a single thread eliminates the parallel-thread coordination that caused the original deadlock. These values must be set at the container level, not from Python, because libgomp reads `OMP_NUM_THREADS` only once when it is first loaded during `import torch` (Free Software Foundation, n.d.). Anything assigned from Python afterwards has no effect.

### Defence in depth: 10-second executor timeout

The executor enforces a 10-second per-call timeout that converts any residual hang into a graceful fallback to the baseline verdict, rather than letting Postfix stall for the full 180-second `milter_content_timeout`.

## Verification after the fix

After the fix, a `py-spy` dump of the running container (`drafts/py_spy_after_fix.txt`) shows the dedicated `distilbert_0` worker idle and ready to receive submissions.

## Trace artefacts

- `drafts/py_spy_wedged_stack.txt`: wedged libmilter worker stuck inside `BatchEncoding.to(self.device)` at `predict.py:182`.
- `drafts/py_spy_after_fix.txt`: dedicated `distilbert_0` worker idle and ready, post-fix.

## References

- Free Software Foundation (n.d.). *GNU libgomp documentation.*
- Postfix Project (n.d.f). *Postfix milter README.*
