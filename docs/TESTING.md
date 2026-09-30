# SignSage — Testing & Verification

## 1. Testing philosophy

SignSage is a research prototype. Testing should prioritize failure detection and research validity rather than a large coverage number.

Every non-trivial integration boundary should have a cheap runnable check.

## 2. Software verification

### Backend

Verify:

- application imports,
- route registration,
- WebSocket contract,
- authentication handling,
- history isolation,
- capability/model status,
- error handling.

### Frontend

Verify:

- TypeScript compilation,
- production build,
- component/unit tests,
- API/WebSocket contract compatibility.

### Contract tests

Contract tests should detect drift in:

- endpoint paths,
- message types,
- response fields,
- error codes,
- landmark dimensions,
- avatar vocabulary.

## 3. Research evaluation

### Sign → Text

Required metrics:

- WER
- BLEU
- ROUGE-L
- output diversity

Always compare against a baseline.

### Text → Pose

Required metrics:

- normalized MSE
- DTW
- motion energy

DTW/MSE alone are insufficient because a near-static average pose can score deceptively well.

## 4. Data validation

Research runs should verify:

- dataset rows and pose-file correspondence,
- representative sample readability,
- pose confidence,
- train/validation/test separation,
- absence of obvious junk-text domination.

The documented work uses video-level splitting for leakage control. Signer-independent generalization has not yet been established.

## 5. Research result currently on record

The larger completed Sign→Text experiment showed strong output-collapse behavior. The Text→Pose generator produced only about 6.7% of ground-truth motion energy in the documented comparison.

Therefore these results are diagnostic evidence, not deployment evidence.

## 6. End-to-end testing

Still distinguish:

```text
unit test
→ integration test
→ live E2E pipeline
→ model-quality evaluation
→ target-device evaluation
```

Passing one layer does not substitute for the next.

## 7. Environment/device testing

The project should eventually test:

- browser camera permissions,
- different lighting conditions,
- camera distance,
- tracking loss,
- reconnect behavior,
- target hardware latency,
- mobile browser behavior if mobile remains in scope.

## 8. Explicitly not claimed

There is no basis in the current documentation for claiming large-scale production load/stress testing or broad real-world signer validation.
