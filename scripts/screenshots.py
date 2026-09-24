"""Capture the console's screens into `docs/screenshots/`.

    python scripts/screenshots.py                     # against a server this script starts
    python scripts/screenshots.py --base-url http://127.0.0.1:8071   # against one already running

Screenshots of a system that is not running are drawings. This starts the real service against the
real database, drives it with Playwright, and captures what a visitor would see — including the
screens that matter most and are the least photogenic: a refusal, and a question answered from one
knowledge date and then from another.

Requires the Playwright browser binaries, which are not installed by `uv sync`:

    uv run playwright install chromium

The shot list is deliberately weighted towards the awkward states. A gallery of successful answers
says nothing a screenshot of any RAG demo does not; the refusal and the two knowledge dates are the
parts of this system that are worth looking at.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT = REPO_ROOT / "docs" / "screenshots"
CORPUS = REPO_ROOT / "data" / "generated"
#: Wide enough that the evidence table does not wrap, tall enough that the ask screen fits
#: without scrolling. Captured at 2x so the text is legible in a README on a retina display.
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 1000


@dataclass(frozen=True)
class Shot:
    name: str
    path: str
    caption: str
    full_page: bool = True


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for(base_url: str, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/healthz", timeout=3) as response:  # noqa: S310
                if response.status < 500:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(1.0)
    return False


def _example_question() -> dict[str, str]:
    """A real answerable question, and a real correction, taken from the corpus.

    Hard-coding a question would mean the screenshots stop matching the corpus the moment it is
    regenerated, and a screenshot that no longer reproduces is a screenshot nobody trusts.
    """
    questions = json.loads((CORPUS / "questions.json").read_text(encoding="utf-8"))
    questions = questions["questions"] if isinstance(questions, dict) else questions
    answerable = sorted(
        (q for q in questions if q.get("answerable") and q["language"] == "en"),
        key=lambda q: str(q["question_id"]),
    )[0]

    documents = json.loads((CORPUS / "documents.json").read_text(encoding="utf-8"))
    documents = documents["documents"] if isinstance(documents, dict) else documents
    corrected = sorted(
        (d for d in documents if d.get("corrected_by") and d["language"] == "en"),
        key=lambda d: str(d["document_id"]),
    )
    correction = corrected[0] if corrected else None

    unanswerable = sorted(
        (q for q in questions if not q.get("answerable") and q["language"] == "en"),
        key=lambda q: str(q["question_id"]),
    )[0]

    return {
        "answerable": str(answerable["text"]),
        "answerable_variant": str(answerable.get("variant_id") or ""),
        "answerable_as_of": str(answerable["as_of"]),
        "unanswerable": str(unanswerable["text"]),
        "unanswerable_variant": str(unanswerable.get("variant_id") or ""),
        "unanswerable_as_of": str(unanswerable["as_of"]),
        "corrected_valid_from": str(correction["valid_from"]) if correction else "",
        "corrected_valid_to": str(correction["valid_to"]) if correction else "",
        "corrected_known_to": str(correction["known_to"]) if correction else "",
        "corrected_family": str(correction["family_id"]) if correction else "",
    }


def _shots(example: dict[str, str]) -> list[Shot]:
    answered = urlencode(
        {
            "q": example["answerable"],
            "variant": example["answerable_variant"],
            "as_of": example["answerable_as_of"],
            "lang": "en",
        }
    )
    refused = urlencode(
        {
            "q": example["unanswerable"],
            "variant": example["unanswerable_variant"],
            "as_of": example["unanswerable_as_of"],
            "lang": "en",
        }
    )
    turkish = urlencode(
        {
            "q": example["answerable"],
            "variant": example["answerable_variant"],
            "as_of": example["answerable_as_of"],
            "lang": "tr",
        }
    )

    shots = [
        Shot("01-ask", "/", "The console before a question is asked."),
        Shot(
            "02-answered",
            f"/?{answered}",
            "An answered question: the extracted span, the citation it came from, and the gate's "
            "reasoning.",
        ),
        Shot(
            "03-refused",
            f"/?{refused}",
            "A question the corpus cannot support. The system refuses and says which rule fired — "
            "this is the screen the project exists for.",
        ),
        Shot(
            "04-evidence",
            "/evidence",
            "Every measured number, traced to the artifact that holds it.",
        ),
        Shot("05-failures", "/failures", "The failure cases, published rather than hidden."),
        Shot(
            "06-provenance",
            "/provenance",
            "Corpus provenance, the hold-out freeze, and the synthetic notice.",
        ),
        Shot("07-turkish", f"/?{turkish}", "The same question in Turkish, against the same index."),
    ]

    # The two knowledge dates, which only exist if the corpus generated a correction.
    if example["corrected_known_to"]:
        midpoint = example["corrected_valid_from"]
        shots.extend(
            [
                Shot(
                    "08-knowledge-now",
                    "/?"
                    + urlencode(
                        {
                            "q": "specification",
                            "as_of": midpoint,
                            "lang": "en",
                        }
                    ),
                    "Asked about a past date with today's knowledge: the correction is returned.",
                ),
                Shot(
                    "09-knowledge-then",
                    "/?"
                    + urlencode(
                        {
                            "q": "specification",
                            "as_of": midpoint,
                            "known_as_of": example["corrected_known_to"],
                            "lang": "en",
                        }
                    ),
                    "The same date asked with the knowledge of the time: the original is returned, "
                    "because a later correction does not rewrite what was believed.",
                ),
            ]
        )
    return shots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="a server already running")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError:
        print(
            "playwright is not installed. `uv sync` installs the package; the browser binaries "
            "need `uv run playwright install chromium`.",
            file=sys.stderr,
        )
        return 1

    if not (CORPUS / "questions.json").is_file():
        print("no corpus; run `python scripts/generate_corpus.py`", file=sys.stderr)
        return 1

    example = _example_question()
    server: subprocess.Popen[bytes] | None = None
    base_url = args.base_url

    if base_url is None:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        server = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "uvicorn",
                "parts_answer_gate.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[screenshots] started the console on {base_url}")

    try:
        if not _wait_for(base_url):
            print(f"[screenshots] {base_url} never became healthy", file=sys.stderr)
            return 1

        args.output.mkdir(parents=True, exist_ok=True)
        captions: dict[str, str] = {}

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                device_scale_factor=2,
            )
            for shot in _shots(example):
                page.goto(f"{base_url}{shot.path}", wait_until="networkidle")
                target = args.output / f"{shot.name}.png"
                page.screenshot(path=str(target), full_page=shot.full_page)
                captions[f"{shot.name}.png"] = shot.caption
                print(f"[screenshots] {target.relative_to(REPO_ROOT)}")
            browser.close()

        (args.output / "captions.json").write_text(
            json.dumps(captions, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[screenshots] {len(captions)} screens captured into {args.output}")
        return 0
    finally:
        if server is not None:
            server.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
