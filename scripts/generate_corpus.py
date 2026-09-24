"""Build the synthetic corpus, and prove that building it twice gives the same bytes.

    python scripts/generate_corpus.py                       # write data/generated + artifacts
    python scripts/generate_corpus.py --out data/generated
    python scripts/generate_corpus.py --verify-determinism  # build twice, diff every byte

`--verify-determinism` is the honest form of ADR-001's kill condition J for this stage. It does not
compare counts or digests the generator computed about itself — it writes two complete corpora into
two temporary directories and compares the files byte for byte, because a generator that reports its
own stability is reporting on the thing being tested.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # running from a checkout without an install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from parts_answer_gate.corpus import generate_corpus  # noqa: E402
from parts_answer_gate.corpus.generate import GeneratedCorpus  # noqa: E402

DEFAULT_OUT = Path("data/generated")
DEFAULT_ARTIFACT = Path("artifacts/corpus.json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _report(generated: GeneratedCorpus, out: Path) -> None:
    artifact = generated.artifact
    print(f"corpus written to {out}")
    print(f"  seed              {artifact['seed']}")
    print(f"  generator         {artifact['generator_version']}")
    print(f"  split rule        {artifact['split_rule']}")
    for key, value in generated.counts.items():
        print(f"  {key:<24}{value}")
    print(f"  languages         {', '.join(artifact['languages'])}")
    print(f"  holdout families  {len(artifact['holdout_families'])} of {artifact['families']}")
    print(f"  documents split   {artifact['documents_by_split']}")
    print(f"  questions split   {artifact['questions_by_split']}")
    print(f"  in both splits    {artifact['documents_in_both_splits']}")
    print("  unanswerable by kind:")
    for kind, count in artifact["unanswerable_by_kind"].items():
        print(f"    {kind:<42}{count}")


def verify_determinism() -> int:
    """Write the corpus twice into temporary directories and compare every byte."""
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        left, right = Path(first), Path(second)
        generate_corpus(left)
        generate_corpus(right)

        left_files = [p.relative_to(left) for p in _files(left)]
        right_files = [p.relative_to(right) for p in _files(right)]
        if left_files != right_files:
            print(f"FAIL: different files written: {left_files} vs {right_files}")
            return 1

        failures = 0
        for name in left_files:
            first_digest = _digest(left / name)
            second_digest = _digest(right / name)
            size = (left / name).stat().st_size
            if first_digest == second_digest:
                print(f"  identical  {name!s:<18}{size:>9} bytes  sha256 {first_digest[:16]}")
            else:
                failures += 1
                print(f"  DIFFERS    {name}  {first_digest[:16]} vs {second_digest[:16]}")
        if failures:
            print(f"FAIL: {failures} file(s) differ between two runs")
            return 1
        print(f"two runs produced {len(left_files)} byte-identical files")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="where to write the corpus.json the kill test grades",
    )
    parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="write only the corpus files, leaving artifacts/ alone",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="build twice into temporary directories and diff every byte",
    )
    arguments = parser.parse_args(argv)

    if arguments.verify_determinism:
        return verify_determinism()

    generated = generate_corpus(
        arguments.out, artifact_path=None if arguments.no_artifact else arguments.artifact
    )
    _report(generated, arguments.out)
    if not arguments.no_artifact:
        print(f"artifact written to {arguments.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
