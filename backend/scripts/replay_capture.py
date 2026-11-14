"""Replay saved labeled captures through the live /api/v1/translation/word endpoint.

Posts each capture in the exact frontend request format ({frames: [...]}) and
prints, per file: the expected label (if stored), the response top-3 with
probabilities, and whether the prediction was accepted.

Usage:
    python scripts/replay_capture.py backend/captures_replay/S1 http://localhost:8000
    python scripts/replay_capture.py backend/captures_replay/S1 http://localhost:8000 --summary
    python scripts/replay_capture.py backend/captures_replay/S1 http://localhost:8000 \
        --compare https://signsage.onrender.com

The first call to each backend uses a 90 s timeout (cold backend / model load);
later calls use --timeout (default 60 s).

With --summary, after replaying against the first backend, prints:
  overall top-1 accuracy, per-label top-1, acceptance rate, top-10 confusion
  pairs (expected -> predicted), and the false-accept rate for rest.

With --compare BASE_URL_B, replays every capture against both backends and
prints per-label agreement, the maximum absolute probability difference, and
every capture where the top-1 labels disagree.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/scripts/ -> repo root

TIMEOUT_FIRST_S = 90.0


def _load_capture(path: Path) -> tuple[np.ndarray, str | None]:
    """Return (landmarks, expected_label) from a saved capture .npz."""
    with np.load(path) as z:
        landmarks = np.asarray(z["landmarks"])
        expected: str | None = None
        for key in ("true_label", "label"):
            if key in z:
                value = z[key]
                try:
                    expected = str(value.tolist()[0])
                except (AttributeError, IndexError):
                    expected = str(value)
                break
    return landmarks, expected


def _frames_payload(landmarks: np.ndarray) -> list[dict]:
    """Exact frontend request format: [{landmarks: [x,y,z]x543}, ...]."""
    return [{"landmarks": frame.tolist()} for frame in landmarks]


def _replay_all(
    files: list[Path],
    base_url: str,
    timeout_after_first: float,
    verbose: bool,
) -> list[dict]:
    """Replay every capture against one backend. Returns a list of result dicts.

    Each result: {path, expected, accepted, word, top3, prob_top1, ok, error}
    """
    base = base_url.rstrip("/")
    url = f"{base}/api/v1/translation/word"
    results = []
    for i, path in enumerate(files, 1):
        landmarks, expected = _load_capture(path)
        payload = {"frames": _frames_payload(landmarks)}
        timeout = TIMEOUT_FIRST_S if i == 1 else timeout_after_first
        result = {
            "path": path,
            "name": path.name,
            "key": str(path.relative_to(path.parents[2]) if len(path.parts) >= 3 else path.name),
            "expected": expected,
            "accepted": None,
            "word": None,
            "top3": [],
            "prob_top1": None,
            "ok": False,
            "error": None,
        }
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
            if resp.status_code >= 400:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                body = resp.json()
                result["accepted"] = body.get("accepted")
                result["word"] = body.get("word")
                result["top3"] = body.get("top3", []) or []
                result["prob_top1"] = (
                    float(result["top3"][0].get("probability", 0)) if result["top3"] else None
                )
                result["ok"] = True
        except Exception as exc:  # noqa: BLE001 - report and keep replaying
            result["error"] = str(exc)
        results.append(result)
        if verbose:
            if result["ok"]:
                top3_str = ", ".join(
                    f"{t.get('label')} {float(t.get('probability', 0)):.3f}"
                    for t in result["top3"]
                ) or "—"
                print(
                    f"[{i}/{len(files)}] {path.name}  "
                    f"expected={expected or '—'}  "
                    f"accepted={result['accepted']}  "
                    f"word={result['word']}  "
                    f"top3: {top3_str}"
                )
            else:
                print(
                    f"[{i}/{len(files)}] {path.name}  "
                    f"expected={expected or '—'}  "
                    f"FAILED: {result['error']}"
                )
    return results


def _print_summary(results: list[dict]) -> None:
    """Print summary statistics for one backend's replay results."""
    valid = [r for r in results if r["ok"]]
    n_valid = len(valid)
    n_failed = len(results) - n_valid
    failed_reasons = Counter(r["error"] for r in results if r["error"])
    if failed_reasons:
        print("\nFailed replay reasons (earliest first):")
        for reason, cnt in list(failed_reasons.items())[:3]:
            print(f"  x{cnt}: {str(reason)[:150]}")
    if n_valid == 0:
        print("SUMMARY: no successful replays.")
        return
    if n_failed:
        print(f"Warning: {n_failed}/{len(results)} replays FAILED (see reasons above).")

    # Overall top-1 accuracy (expected == predicted word, non-null expected).
    labeled = [r for r in valid if r["expected"]]
    n_labeled = len(labeled)
    n_correct = sum(1 for r in labeled if r["word"] == r["expected"])
    overall_top1 = n_correct / n_labeled if n_labeled else 0.0

    # Per-label top-1.
    per_label: dict[str, list[tuple[str, str]]] = defaultdict(list)  # expected -> [(expected, pred)]
    for r in labeled:
        per_label[r["expected"]].append((r["expected"], r["word"] or ""))
    per_label_top1 = {}
    for lab in sorted(per_label):
        pairs = per_label[lab]
        correct = sum(1 for _, p in pairs if p == lab)
        per_label_top1[lab] = correct / len(pairs) if pairs else 0.0

    # Acceptance rate.
    n_accepted = sum(1 for r in valid if r["accepted"])
    acceptance_rate = n_accepted / n_valid

    # Top-10 confusion pairs.
    conf = Counter(
        (r["expected"], r["word"] or "")
        for r in labeled
        if r["word"] != r["expected"]
    )
    top10 = conf.most_common(10)

    # False-accept rate for rest: rest clips where model accepted a non-rest word.
    rest_clips = [r for r in labeled if r["expected"] == "rest"]
    rest_false_accept = 0
    if rest_clips:
        rest_false_accept = sum(
            1 for r in rest_clips if r["accepted"] and (r["word"] or "") != "rest"
        )
    rest_fpr = rest_false_accept / len(rest_clips) if rest_clips else 0.0

    print("\n=== Replay summary ===")
    print(f"Backend results: {n_valid}/{len(results)} successful replays")
    print(f"Overall top-1 accuracy: {n_correct}/{n_labeled} = {overall_top1:.3f}")
    print(f"Acceptance rate: {n_accepted}/{n_valid} = {acceptance_rate:.3f}")
    print(f"Rest false-accept rate: {rest_false_accept}/{len(rest_clips)} = {rest_fpr:.3f}")
    print("\nPer-label top-1:")
    for lab in sorted(per_label_top1):
        print(f"  {lab:12} {per_label_top1[lab]:.3f}")
    print("\nTop-10 confusion pairs (expected -> predicted):")
    if top10:
        for (exp, pred), cnt in top10:
            print(f"  {exp:12} -> {pred:12}  {cnt}")
    else:
        print("  (none)")


def _print_compare(
    results_a: list[dict],
    results_b: list[dict],
    base_url_a: str,
    base_url_b: str,
) -> None:
    """Print comparison between two backends' replay results."""
    # Match results by path name.
    by_name_a = {r["key"]: r for r in results_a if r["ok"]}
    by_name_b = {r["key"]: r for r in results_b if r["ok"]}
    common = sorted(set(by_name_a) & set(by_name_b))
    if not common:
        print("COMPARE: no overlapping successful replays.")
        return

    # Per-label agreement.
    label_agree: dict[str, list[bool]] = defaultdict(list)
    disagreements = []
    max_prob_diff = 0.0
    max_prob_diff_case = None
    n_agree = 0
    n_total = 0

    for name in common:
        ra, rb = by_name_a[name], by_name_b[name]
        exp = ra["expected"] or rb["expected"] or ""
        wa = ra["word"] or ""
        wb = rb["word"] or ""
        n_total += 1
        if wa == wb:
            n_agree += 1
            label_agree[exp].append(True)
        else:
            label_agree[exp].append(False)
            disagreements.append((name, exp, wa, wb))
        # Max abs probability difference on top-1.
        pa = ra["prob_top1"] if ra["prob_top1"] is not None else 0.0
        pb = rb["prob_top1"] if rb["prob_top1"] is not None else 0.0
        diff = abs(pa - pb)
        if diff > max_prob_diff:
            max_prob_diff = diff
            max_prob_diff_case = (name, exp, wa, wb, pa, pb)

    agreement_rate = n_agree / n_total if n_total else 0.0

    print(f"\n=== Replay compare: A={base_url_a} vs B={base_url_b} ===")
    print(f"Common successful replays: {n_total}")
    print(f"Per-label top-1 agreement: {n_agree}/{n_total} = {agreement_rate:.3f}")
    print(f"Max absolute probability difference: {max_prob_diff:.4f}")
    if max_prob_diff_case:
        name, exp, wa, wb, pa, pb = max_prob_diff_case
        print(f"  (case: {name} expected={exp} A={wa} {pa:.3f} B={wb} {pb:.3f})")
    print("\nPer-label agreement:")
    for lab in sorted(label_agree):
        agrees = label_agree[lab]
        rate = sum(agrees) / len(agrees) if agrees else 0.0
        print(f"  {lab:12} {sum(agrees)}/{len(agrees)} = {rate:.3f}")
    print(f"\nCaptures where top-1 labels disagree ({len(disagreements)}):")
    if disagreements:
        for name, exp, wa, wb in disagreements:
            ra = by_name_a[name]
            rb = by_name_b[name]
            pa = ra["prob_top1"] if ra["prob_top1"] is not None else 0.0
            pb = rb["prob_top1"] if rb["prob_top1"] is not None else 0.0
            print(
                f"  {name}  expected={exp}  A={wa} ({pa:.3f})  B={wb} ({pb:.3f})"
            )
    else:
        print("  (none)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay labeled captures through the word endpoint.",
    )
    parser.add_argument(
        "path",
        help="Saved .npz capture, or a folder of .npz captures to replay",
    )
    parser.add_argument(
        "base_url",
        help="Base URL of the first backend, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for calls after the first (default: 60)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="After replaying, print summary statistics (overall/per-label top-1, "
        "acceptance rate, top-10 confusions, rest false-accept rate).",
    )
    parser.add_argument(
        "--compare",
        metavar="BASE_URL_B",
        help="Also replay against a second backend and print comparison statistics.",
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    if target.is_dir():
        files = sorted(target.rglob("*.npz"))
    elif target.is_file():
        files = [target]
    else:
        print(f"Not found: {target}")
        return 1

    if not files:
        print(f"No .npz captures found at {target}")
        return 0

    # Replay against first backend.
    results_a = _replay_all(files, args.base_url, args.timeout, verbose=not args.compare)

    if args.summary:
        _print_summary(results_a)

    if args.compare:
        # Replay against second backend (always quiet per-file).
        results_b = _replay_all(files, args.compare, args.timeout, verbose=False)
        _print_compare(results_a, results_b, args.base_url, args.compare)

    return 0


if __name__ == "__main__":
    sys.exit(main())
