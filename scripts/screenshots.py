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
import difflib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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


def _bitemporal_question(
    questions: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    correction: dict[str, Any] | None,
) -> dict[str, str]:
    """Find a question whose answer a later correction changed, and the dates that show it.

    The two knowledge-time screens are the ones worth taking, and both of them were wrong. They
    asked the free-text query `specification`, which worked when the deployment ran the encoder and
    stopped working the day it began serving precomputed query vectors: the shots would have shown
    the panel explaining what the instance cannot encode, captioned as the project's central
    demonstration.

    So the question is derived from the correction itself. Take the first corrected document and its
    correction, diff them, read the changed line, and find the corpus question whose expected answer
    is the figure that changed and whose variant the line names. Then ask it twice inside the
    document's own validity window: once with current knowledge, which returns the correction, and
    once pinned to the knowledge of the time, which returns what was believed then.

    The question's own `as_of` is not used. It belongs to whichever revision was in force when the
    question was written, and the shot needs a date inside the corrected revision's window.
    """
    if correction is None:
        return {}
    by_id = {str(d["document_id"]): d for d in documents}
    replacement = by_id.get(str(correction["corrected_by"]))
    if replacement is None:
        return {}

    changed = [
        line[1:]
        for line in difflib.unified_diff(
            str(correction["text"]).splitlines(),
            str(replacement["text"]).splitlines(),
            lineterm="",
            n=0,
        )
        if line.startswith("-") and not line.startswith("---") and "Document doc-" not in line
    ]
    if not changed:
        return {}
    line = changed[0]

    for question in sorted(questions, key=lambda q: str(q["question_id"])):
        span = str(question.get("expected_answer_span") or "")
        variant = str(question.get("variant_id") or "")
        if question["language"] != "en" or not span or not variant:
            continue
        if span in line and variant in line:
            return {
                "bitemporal": str(question["text"]),
                "bitemporal_variant": variant,
                "bitemporal_as_of": str(correction["valid_from"]),
                "bitemporal_known_as_of": str(correction["known_from"]),
                "bitemporal_then_value": span,
            }
    return {}


def _example_question() -> dict[str, str]:
    """Real subjects for each screen, chosen from the corpus **and from the measured result**.

    The refusal and the failure are picked from `artifacts/gate.json` rather than assumed. An
    earlier version took the first unanswerable question by id and captioned the shot "the system
    refuses" — and the system answered it, because that question was one of the 47 unsupported
    answers kill condition E fails on. The screenshot was evidence of a failure carrying a caption
    claiming a success, which is the exact thing this project exists to argue against.

    So: the abstention shot is a question the system **did** refuse, and the failure shot is one of
    the Turkish or Russian questions it **did** answer without support.
    """
    questions = json.loads((CORPUS / "questions.json").read_text(encoding="utf-8"))
    questions = questions["questions"] if isinstance(questions, dict) else questions
    by_id = {str(q["question_id"]): q for q in questions}

    answerable = sorted(
        (q for q in questions if q.get("answerable") and q["language"] == "en"),
        key=lambda q: str(q["question_id"]),
    )[0]

    # The same question in the same language the caption claims. An earlier version captured the
    # English text with `lang=tr` and captioned it "the same question in Turkish", which showed a
    # language filter rather than a translation and would have been read as the second.
    parallel = {
        str(q["language"]): str(q["text"])
        for q in questions
        if q["parallel_group"] == answerable["parallel_group"]
    }

    documents = json.loads((CORPUS / "documents.json").read_text(encoding="utf-8"))
    documents = documents["documents"] if isinstance(documents, dict) else documents
    corrected = sorted(
        (d for d in documents if d.get("corrected_by") and d["language"] == "en"),
        key=lambda d: str(d["document_id"]),
    )
    correction = corrected[0] if corrected else None
    bitemporal = _bitemporal_question(questions, documents, correction)

    # Which question demonstrates which state is decided by **asking the running system**, in
    # `_probe` below. gate.json's `unsupported_examples` is capped at ten of the forty-seven, so
    # picking "an unanswerable question not in that list" selected one of the other thirty-seven
    # answered ones and captioned it a refusal. Reading a truncated list and calling it the result
    # is the same class of error as reading a metric and not its denominator.
    refused: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    _ = by_id

    def described(record: dict[str, object] | None, prefix: str) -> dict[str, str]:
        if record is None:
            return {}
        return {
            f"{prefix}": str(record["text"]),
            f"{prefix}_variant": str(record.get("variant_id") or ""),
            f"{prefix}_as_of": str(record["as_of"]),
            f"{prefix}_lang": str(record["language"]),
            f"{prefix}_kind": str(record.get("unanswerable_kind") or ""),
        }

    return {
        "answerable": str(answerable["text"]),
        "answerable_tr": parallel.get("tr", str(answerable["text"])),
        "answerable_variant": str(answerable.get("variant_id") or ""),
        "answerable_as_of": str(answerable["as_of"]),
        **bitemporal,
        **described(refused, "refused"),
        **described(failure, "failure"),
        "corrected_valid_from": str(correction["valid_from"]) if correction else "",
        "corrected_valid_to": str(correction["valid_to"]) if correction else "",
        "corrected_known_to": str(correction["known_to"]) if correction else "",
        "corrected_family": str(correction["family_id"]) if correction else "",
    }


def _outcome(base_url: str, record: dict[str, Any]) -> str:
    """What the system actually did with this question, asked through the shipped endpoint."""
    params = {
        "q": str(record["text"]),
        "variant": str(record.get("variant_id") or ""),
        "as_of": str(record["as_of"]),
        "lang": str(record["language"]),
    }
    try:
        with urllib.request.urlopen(  # noqa: S310
            f"{base_url}/api/ask?{urlencode(params)}", timeout=60
        ) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return "unknown"
    return str(payload.get("decision", {}).get("outcome", "unknown"))


def _probe(base_url: str, example: dict[str, str]) -> dict[str, str]:
    """Fill in the refusal and the failure by observing the system, in corpus order.

    Bounded at forty questions per role so the capture does not turn into a second evaluation run.
    If neither is found the corresponding screen is simply not captured, rather than a screen being
    captured under a caption that does not describe it.
    """
    questions = json.loads((CORPUS / "questions.json").read_text(encoding="utf-8"))
    questions = questions["questions"] if isinstance(questions, dict) else questions
    unanswerable = [q for q in questions if not q.get("answerable")]
    unanswerable.sort(key=lambda q: str(q["question_id"]))

    found: dict[str, str] = {}

    # A refusal, in English, that the system genuinely withheld.
    for record in [q for q in unanswerable if q["language"] == "en"][:40]:
        if _outcome(base_url, record) in {"abstain", "review"}:
            found.update(
                {
                    "refused": str(record["text"]),
                    "refused_variant": str(record.get("variant_id") or ""),
                    "refused_as_of": str(record["as_of"]),
                    "refused_lang": str(record["language"]),
                    "refused_kind": str(record.get("unanswerable_kind") or ""),
                }
            )
            break

    # A failure the gate actually made, in Turkish or Russian, which is where they concentrate.
    for record in [q for q in unanswerable if q["language"] in {"tr", "ru"}][:40]:
        if _outcome(base_url, record) == "answer":
            found.update(
                {
                    "failure": str(record["text"]),
                    "failure_variant": str(record.get("variant_id") or ""),
                    "failure_as_of": str(record["as_of"]),
                    "failure_lang": str(record["language"]),
                    "failure_kind": str(record.get("unanswerable_kind") or ""),
                }
            )
            break

    return {**example, **found}


def _shots(example: dict[str, str]) -> list[Shot]:
    def url(text: str, variant: str, as_of: str, lang: str, **extra: str) -> str:
        return "/?" + urlencode(
            {"q": text, "variant": variant, "as_of": as_of, "lang": lang, **extra}
        )

    shots = [
        Shot("01-ask", "/", "The console before a question is asked."),
        Shot(
            "02-answered",
            url(
                example["answerable"],
                example["answerable_variant"],
                example["answerable_as_of"],
                "en",
            ),
            "An answered question: the extracted span, the citation it came from with its "
            "character offset, and every gate signal that permitted it.",
        ),
    ]

    if example.get("refused"):
        shots.append(
            Shot(
                "03-refused",
                url(
                    example["refused"],
                    example.get("refused_variant", ""),
                    example["refused_as_of"],
                    example.get("refused_lang", "en"),
                ),
                "A question the corpus cannot support, and the system withholds. The kind is "
                f"{example.get('refused_kind') or 'unanswerable'}. This is the screen the project "
                "exists for.",
            )
        )

    shots += [
        Shot(
            "04-evidence",
            "/evidence",
            "Every measured number, the twelve kill conditions with their verdicts, and the "
            "release-gate banner that does not go away.",
        ),
        Shot("05-failures", "/failures", "The failure cases, published rather than hidden."),
        Shot(
            "06-provenance",
            "/provenance",
            "Corpus provenance, the hold-out freeze digest, and the synthetic notice.",
        ),
        Shot(
            "07-turkish",
            url(
                example["answerable_tr"],
                example["answerable_variant"],
                example["answerable_as_of"],
                "tr",
            ),
            "The same question in Turkish -- the corpus's own parallel rendering, not the English "
            "text with a language filter -- against the same index.",
        ),
    ]

    if example.get("failure"):
        shots.append(
            Shot(
                "10-unsupported-answer-failure",
                url(
                    example["failure"],
                    example.get("failure_variant", ""),
                    example["failure_as_of"],
                    example.get("failure_lang", "en"),
                ),
                "**Kill condition E failing, live.** The corpus contains no passage that answers "
                f"this ({example.get('failure_kind') or 'unanswerable'}), and the gate answers it "
                "anyway. Every one of these failures is Turkish or Russian: term coverage is "
                "computed with a bidirectional prefix match that errs towards covering.",
            )
        )

    if example.get("bitemporal"):
        shots += [
            Shot(
                "08-knowledge-now",
                url(
                    example["bitemporal"],
                    example["bitemporal_variant"],
                    example["bitemporal_as_of"],
                    "en",
                ),
                "A past date asked with today's knowledge. The revision in force on that date was "
                "later corrected, and it is the correction that answers.",
            ),
            Shot(
                "09-knowledge-then",
                url(
                    example["bitemporal"],
                    example["bitemporal_variant"],
                    example["bitemporal_as_of"],
                    "en",
                    known_as_of=example["bitemporal_known_as_of"],
                ),
                "The same question, the same date, pinned to what was known at the time: the "
                f"answer is {example['bitemporal_then_value']} again, because a later correction "
                "does not rewrite what the technician had in front of them. Two axes, one query.",
            ),
        ]
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

        example = _probe(base_url, example)
        print(
            f"[screenshots] refusal: {'found' if example.get('refused') else 'none found'} · "
            f"unsupported answer: {'found' if example.get('failure') else 'none found'}"
        )

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

        # Where a screenshot was taken is part of what it evidences. A local capture and one
        # from the public deployment are different claims and a reader cannot tell them apart
        # by looking, so the file says which.
        (args.output / "captions.json").write_text(
            json.dumps(
                {
                    "captured_from": (
                        base_url
                        if args.base_url
                        else "a server this script started against the local database"
                    ),
                    "captions": captions,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
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
