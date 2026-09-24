"""The hold-out rule, and the freeze that turns a rule into evidence.

The tests that matter are the drift ones. A rule plus a corpus implies a membership only while both
are unchanged, and a hold-out that silently follows the corpus is not a hold-out — the number
measured over it means whatever the last edit decided. `freeze` refuses to overwrite a different
membership, and these assert that it really refuses rather than logging and continuing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from parts_answer_gate.holdout import (
    HOLDOUT_RULE,
    HOLDOUT_SHARE,
    HoldoutDriftError,
    compute,
    freeze,
    is_held_out,
    load_frozen,
    split_of,
    verify_partition,
)


def _documents(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"document_id": d, "family_id": f} for d, f in pairs]


def _questions(*triples: tuple[str, str, list[str]]) -> list[dict[str, Any]]:
    return [
        {"question_id": q, "family_id": f, "supporting_document_ids": docs}
        for q, f, docs in triples
    ]


class TestTheRule:
    def test_it_is_a_pure_function_of_the_family_id(self) -> None:
        """No seed, no score, nothing to re-draw. Called twice, it must agree with itself."""
        for family in ("FAM-01", "FAM-07", "PUMP-XP", "GEARBOX-9"):
            assert is_held_out(family) == is_held_out(family)

    def test_it_is_stable_across_processes(self) -> None:
        """blake2b, not `hash()`.

        Python's built-in hash is salted per process, so a rule built on it would put a family in
        the hold-out on Monday and in development on Tuesday — and every score either side of that
        would be measured over a different set without anybody noticing.
        """
        assert is_held_out("FAM-02") is True
        assert is_held_out("FAM-01") is False

    def test_the_share_is_roughly_what_was_declared(self) -> None:
        """Over enough families. A rule that held out everything or nothing would satisfy every
        other test in this file."""
        families = [f"FAM-{i:04d}" for i in range(2000)]
        held = sum(is_held_out(f) for f in families)
        share = held / len(families)
        assert 0.28 < share < 0.40, f"{held}/{len(families)} is not near {HOLDOUT_SHARE}%"

    def test_split_of_names_the_two_sides(self) -> None:
        assert split_of("FAM-02") == "holdout"
        assert split_of("FAM-01") == "development"

    def test_the_rule_string_describes_what_the_code_does(self) -> None:
        """The README quotes this. A description that drifted from the implementation is a lie with
        a citation."""
        assert "blake2b" in HOLDOUT_RULE
        assert str(HOLDOUT_SHARE) in HOLDOUT_RULE
        assert "no seed and no score" in HOLDOUT_RULE


class TestPartitioning:
    def test_a_family_takes_its_documents_and_questions_with_it(self) -> None:
        """The whole reason for splitting by family rather than by question."""
        documents = _documents(("D1", "FAM-02"), ("D2", "FAM-02"), ("D3", "FAM-01"))
        questions = _questions(
            ("Q1", "FAM-02", ["D1"]), ("Q2", "FAM-02", ["D2"]), ("Q3", "FAM-01", ["D3"])
        )
        frozen = compute(documents, questions)

        assert frozen.families == ("FAM-02",)
        assert set(frozen.documents) == {"D1", "D2"}
        assert set(frozen.questions) == {"Q1", "Q2"}

    def test_a_document_in_two_families_is_reported_as_a_leak(self) -> None:
        """Kill condition L is a zero, and the only way to earn it is to look."""
        documents = _documents(("D1", "FAM-02"), ("D1", "FAM-01"))
        problems = verify_partition(documents, [])
        assert any("appears in both splits" in p for p in problems)

    def test_a_question_supported_by_the_other_sides_document_is_reported(self) -> None:
        """The subtle leak. The question is in development, its evidence is in the hold-out, and
        every aggregate number hides it."""
        documents = _documents(("D1", "FAM-02"), ("D2", "FAM-01"))
        questions = _questions(("Q1", "FAM-01", ["D1"]))
        problems = verify_partition(documents, questions)
        assert any("but its supporting document" in p for p in problems)

    def test_a_clean_corpus_reports_nothing(self) -> None:
        documents = _documents(("D1", "FAM-02"), ("D2", "FAM-01"))
        questions = _questions(("Q1", "FAM-02", ["D1"]), ("Q2", "FAM-01", ["D2"]))
        assert verify_partition(documents, questions) == []


class TestTheFreeze:
    @pytest.fixture
    def corpus(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        documents = _documents(("D1", "FAM-02"), ("D2", "FAM-02"), ("D3", "FAM-01"))
        questions = _questions(
            ("Q1", "FAM-02", ["D1"]), ("Q2", "FAM-02", ["D2"]), ("Q3", "FAM-01", ["D3"])
        )
        return documents, questions

    def test_it_writes_the_membership_enumerated(
        self, tmp_path: Path, corpus: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        """A rule implies a membership; an enumerated membership can be checked without rerunning
        anything, by a reader who does not trust the author."""
        frozen = freeze(tmp_path, *corpus)
        written = json.loads((tmp_path / "holdout.json").read_text(encoding="utf-8"))

        assert written["families"] == list(frozen.families)
        assert written["documents"] == list(frozen.documents)
        assert written["questions"] == list(frozen.questions)
        assert written["split_by"] == "product_family"
        assert written["digest"] == frozen.digest

    def test_freezing_twice_over_the_same_corpus_is_a_no_op(
        self, tmp_path: Path, corpus: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        first = freeze(tmp_path, *corpus)
        second = freeze(tmp_path, *corpus)
        assert first.digest == second.digest

    def test_a_corpus_that_moves_a_family_is_refused(
        self, tmp_path: Path, corpus: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        """The test this module exists for.

        A hold-out that silently follows the corpus is not frozen, and every score taken against it
        means whatever the last edit decided.
        """
        documents, questions = corpus
        freeze(tmp_path, documents, questions)

        grown = [*documents, *_documents(("D4", "FAM-03"))]
        with pytest.raises(HoldoutDriftError, match="already holds a different membership"):
            freeze(tmp_path, grown, questions)

    def test_a_refreeze_must_be_asked_for_explicitly(
        self, tmp_path: Path, corpus: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        documents, questions = corpus
        freeze(tmp_path, documents, questions)
        grown = [*documents, *_documents(("D4", "FAM-03"))]

        refrozen = freeze(tmp_path, grown, questions, allow_refreeze=True)
        assert "FAM-03" in refrozen.families

    def test_a_leaking_corpus_cannot_be_frozen_at_all(self, tmp_path: Path) -> None:
        """Freezing a leak would make the leak permanent and give it a timestamp."""
        documents = _documents(("D1", "FAM-02"), ("D1", "FAM-01"))
        with pytest.raises(HoldoutDriftError, match="does not partition cleanly"):
            freeze(tmp_path, documents, [])

    def test_load_frozen_returns_none_when_nothing_is_frozen(self, tmp_path: Path) -> None:
        assert load_frozen(tmp_path) is None

    def test_a_frozen_file_round_trips(
        self, tmp_path: Path, corpus: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        written = freeze(tmp_path, *corpus)
        loaded = load_frozen(tmp_path)
        assert loaded is not None
        assert loaded.digest == written.digest
        assert loaded.families == written.families
        assert loaded.rule == HOLDOUT_RULE
