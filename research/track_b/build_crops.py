"""Step 1: build cropped and jittered copies of the 9-class S1-S4 clips.

Writes one combined manifest per variant (data/cropped/manifest.csv and
data/cropped_jitter/manifest.csv); each row carries the session from the
source manifest. .npy copies land next to the manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_utils as cu

SESSIONS = ("s1_converted", "s2_converted", "s3_converted", "s4_converted")


def main():
    for dest, jitter in (("cropped", False), ("cropped_jitter", True)):
        suffix = "__crop400j" if jitter else "__crop400"
        all_rows = []
        for source in SESSIONS:
            rows, dest_dir = cu.build_variant(source, dest, jitter)
            src_frames = [int(r["frames"]) for r in cu.word_rows(cu.load_manifest(cu.DATA / source / "manifest.csv"))]
            new_frames = [r["frames"] for r in rows]
            print(
                f"{source} -> {dest}: clips={len(rows)} "
                f"frames {sum(src_frames)} -> {sum(new_frames)} "
                f"({100.0 * sum(new_frames) / sum(src_frames):.1f}% of original)"
            )
            all_rows.extend(rows)
        fields = cu.MANIFEST_FIELDS + (["jitter_start_ms", "jitter_end_ms"] if jitter else [])
        # build_variant wrote nothing; write the combined manifest now
        cu.write_manifest(cu.DATA / dest / "manifest.csv", all_rows, fields)
        at_floor = sum(1 for r in all_rows if r["frames"] == cu.MIN_FRAMES)
        print(f"{dest}/manifest.csv: {len(all_rows)} clips, at 12-frame floor: {at_floor}")


if __name__ == "__main__":
    main()
