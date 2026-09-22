from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.services.demo_lexicon import DemoLexicon, _dtw_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure landmark-template DTW distances.")
    parser.add_argument("--templates", type=Path, default=Path("demo_prototypes"))
    args = parser.parse_args()
    lexicon = DemoLexicon.load(args.templates)
    all_distances: list[tuple[str, str, float]] = []
    for phrase, templates in lexicon.prototypes.items():
        for index, query in enumerate(templates):
            for candidate, candidates in lexicon.prototypes.items():
                for candidate_index, reference in enumerate(candidates):
                    if phrase == candidate and index == candidate_index:
                        continue
                    distance = _dtw_distance(query.astype(np.float64), reference.astype(np.float64))
                    all_distances.append((phrase, candidate, distance))

    within = [d for source, target, d in all_distances if source == target]
    between = [d for source, target, d in all_distances if source != target]
    zeros = np.zeros((max(8, next(iter(lexicon.prototypes.values()))[0].shape[0]), 225), dtype=np.float32)
    random_distances = [
        _dtw_distance(zeros.astype(np.float64), template.astype(np.float64))
        for templates in lexicon.prototypes.values()
        for template in templates
    ]

    def report(name: str, values: list[float]) -> None:
        if not values:
            print(f"{name}: no pairs")
            return
        print(f"{name}: n={len(values)} min={min(values):.4f} median={float(np.median(values)):.4f} max={max(values):.4f}")

    report("same-class", within)
    report("different-class", between)
    report("zero-landmark", random_distances)
    print("Choose threshold above measured same-class maximum and below measured different-class minimum after adding real captures.")


if __name__ == "__main__":
    main()
