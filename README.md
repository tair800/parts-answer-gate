# parts-answer-gate

**A pre-registered RAG safety experiment that refused to rewrite its release criteria when the
system failed them.**

> **ORIGINAL RELEASE GATE: FAILED.** Twelve kill conditions and their thresholds were committed to
> `DECISIONS.md` **before any source file existed**. Four do not hold. A fifth passes for a reason
> that makes the pass nearly worthless. None was lowered, none was deleted, nothing was tuned
> against the hold-out after it was scored, and the project is closed here rather than adjusted
> until it cleared its own bar.

---

## Why an ordinary RAG answer can be wrong while looking right

A technician asks *what is the case drain pressure limit for this machine?*

The expensive failure is not a vague answer or a hallucination. It is a **correct quotation from the
wrong document**: a bulletin withdrawn two years ago, or the manual for the variant standing next to
theirs on the shop floor. The citation is real. The span is verbatim. The system can show you
exactly where it came from. And the answer is still a wrong part shipped, a voided warranty, or a
machine returned to service outside its limits.

Citation faithfulness does not catch this, because the citation is faithful. Groundedness does not
catch it, because the answer is grounded. What is wrong is **which documents were eligible to be
retrieved at all**, and that is a question about time and about machine identity, not about text.

### What effectivity and bitemporal filtering solve

- **Effectivity** — a passage applies to a variant and a serial range. The right revision of the
  wrong machine is still wrong. This is decided in SQL, **before anything is ranked**, so an
  ineligible passage is never scored rather than scored and discarded.
- **Valid time** — a revision is withdrawn rather than deleted, so *what was the approved procedure
  on the date of the incident* stays answerable.
- **Knowledge time** — *what was true in March 2021* and *what did we know in March 2021* are
  different questions. A correction issued in 2025 about a 2021 procedure is valid in 2021 and known
  from 2025, and an `UPDATE` that recorded it would have destroyed the ability to answer the second.

### What the answer gate solves

A deterministic decision, computed from seven signals **none of which is a model output**, taken
**before** generation. The rule loop can only *withhold*; the answering outcome is reachable from
exactly one place in the code, by falling through every rule. `REVIEW` is not a hedge — it is two
in-force sources disagreeing, and it withholds exactly as abstention does.

The shipped answerer is extractive: the answer is literally the quoted spans, joined. An ungrounded
part number is therefore impossible rather than improbable.

### What it did not solve

- **The gate over-covers in Turkish and Russian.** It answers a small number of questions the corpus
  cannot support, and **every one of those failures is Turkish or Russian; none is English.**
- **Four of the twelve predeclared criteria do not hold, and a fifth is near-vacuous** — and for
  most of them the reason is the same one, which turned out to be the most interesting result in
  the project. See [Where it falls short](#where-it-falls-short).

### Why nothing was tuned afterwards

The hold-out was frozen by a rule that takes no seed and no score, committed before any number
existed over it, and scored once. After that: no retrieval weight, effectivity rule, bitemporal
rule, embedding, chunking, gate threshold, citation rule or corpus membership was changed. A
threshold moved to clear a line it did not clear is not a measurement, and a portfolio project that
reports 12 of 12 passing has usually chosen its thresholds afterwards.

Everything below is emitted from `artifacts/*.json` by a committed code path. The verdict table is
produced by **running the graded test**, not typed.

---

## Filtering before ranking, which is the whole design

The predicate goes into the `WHERE` clause of **both** ranking queries — the lexical candidate
fetch and the pgvector search — so a passage outside the asked-for variant, serial range or in-force
window is never scored at all:

```sql
chunk.valid_from <= :as_of  AND (chunk.valid_to  IS NULL OR :as_of       < chunk.valid_to)
AND chunk.known_to IS NULL                      -- or the knowledge interval, when one is asked for
AND chunk.language = ANY(:languages)
AND chunk.variant_id = :variant_id
AND (chunk.serial_first IS NULL OR :serial >= chunk.serial_first)
AND (chunk.serial_last  IS NULL OR :serial <= chunk.serial_last)
```

The adversarial test for this takes a passage that **is** superseded and uses that passage's own
text as the query. Against an unfiltered retriever that is the best possible match — the query is
the document, so BM25 scores it top because every term matches and the dense signal scores it top
because the query vector is the chunk vector. The test establishes that premise first, then asserts
the passage is absent from the candidate set: not demoted, not dropped after ranking, never scored.
It checks the count of admitted rows rather than the output, because a passage that was ranked and
then discarded is indistinguishable from one that was never ranked if you only look at the answer.

## Bitemporal, and why one date is not enough

A revision is withdrawn rather than deleted, so *what was the approved procedure on the date of the
incident* stays answerable. That is **valid time**, and it is the axis most systems have.

**Knowledge time** is the one they do not. A correction issued in 2025 about a 2021 procedure is
valid in 2021 and known from 2025. A store with one date can express *what was true then* or *what
we believe now*, and silently answers the other; an audit asking *what did the technician have in
front of them* cannot be answered at all, because the `UPDATE` that recorded the correction
destroyed the belief it replaced.

Here a correction is a re-issue of one revision carrying the **same validity window** and a later
`known_from`. The original gains a `known_to` and a `corrected_by`, and keeps its text. Asked about
a date inside that window:

| | `as_of` | `known_as_of` | returns |
|---|---|---|---|
| a technician | 2021-06-26 | *(blank — current knowledge)* | the **correction** |
| an auditor | 2021-06-26 | 2022-03-12 | the **original**, because that is what was believed |
| after the fix | 2021-06-26 | 2022-03-14 | the **correction** |

Both dates are exposed in the console and on `/api/ask`. A second axis the interface cannot reach is
a second axis nobody can check.

## What the gate actually does

```
retrieve (already filtered)  ->  seven signals  ->  rules, in order, first match wins
                                                     |
                                    ABSTAIN / REVIEW  |  fall through -> ANSWER
```

`REVIEW` is not a hedge. It is the system saying two in-force sources disagree and a person must
decide, and it withholds exactly as `ABSTAIN` does. An abstaining decision carries **no approved
chunks**, so an answerer cannot be handed evidence for a question the gate refused.

None of the seven signals is a model output, and no model can move a question from `ABSTAIN` to
`ANSWER`. Adding a reason to answer would mean deleting a rule, which is visible in a diff.

---

## Measured

Every figure below is emitted from `artifacts/*.json` by `scripts/readme_numbers.py`, and
`--check` fails the build if this block and the artifacts disagree. Nothing here is typed by hand.

<!-- BEGIN MEASURED -->
<!-- END MEASURED -->

---

## Where it falls short

### Four fail, and a fifth passes for the wrong reason

`DECISIONS.md` fixed twelve kill conditions and their thresholds **before any source file existed**.
None has been lowered, none deleted, none marked `xfail`. Four do not hold.

**E — the gate answers questions the corpus cannot support.** The weakness is not spread evenly, and
where it concentrates is the finding:

| unanswerable kind | refused |
|---|---|
| superseded, replacement asked for | 45/45 |
| a product family that does not exist | 44/45 |
| an identifier that nearly matches a real one | 44/45 |
| two in-force sources contradict each other | 35/36 |
| a specification absent from the manual | 37/45 |
| a malformed part number | 37/45 |
| **an attribute absent for a product that exists** | **17/45** |

When the machine is real and the attribute simply is not documented, the gate answers 62% of the
time. `gate.term_is_covered` stands in for a stemmer with a bidirectional prefix match — its own
docstring says it errs towards covering — and a question about a real machine shares enough terms
with that machine's *other* specifications to clear the coverage floor. **The failures are
concentrated in Turkish and Russian**, which is what the per-language table above shows.

**I — abstention on the unanswerable set falls below the 0.90 floor.** Same mechanism as E.

**F — the system does not beat every baseline.** `dense_only` wins by **0.0093**; `bm25_only` ties
exactly; `ungated_rag` ties by construction, because it removes only the gate and therefore runs the
identical retriever, which makes that comparison impossible rather than hard.

And recall@10 is the wrong question. Removing the effectivity predicate leaves recall roughly where
it was — on development it goes *up* — while **MRR collapses from 0.9097 to 0.3695**. What the
filter buys is that the right passage is at rank 1 instead of rank 8. F does not ask about MRR.

**K — the query plan does not use a vector index.** pgvector 0.8.6 is installed, the column is a
real `vector`, and the executed statement uses `<=>`. But the effectivity filter has already reduced
the candidate set to **21 rows**, for which PostgreSQL correctly prefers a sequential scan — forcing
the ANN plan measures **29.2 ms against the planner's 3.1 ms**. The criterion demanded an index scan
on a query that should not have one, and the planner is right.

**G passes, and the pass is near-vacuous.** For the 297 hold-out questions that name a variant — 297
of 312 — the numerator is empty by construction: the effectivity predicate excludes revision- and
variant-incorrect passages *upstream of the gate*, so no answer the system can give for those is
*capable* of being wrong in either sense. The measured rate comes entirely from the 15 questions
about a family the corpus does not contain, where the filter has nothing to constrain.

### One root cause under most of it

E and I are the gate's Turkish and Russian over-covering. F, G, H and K are all the same thing:
**the criteria were written as if the effectivity filter sat beside the thing being measured, and it
sits upstream of everything.** It makes F's strongest baseline unbeatable, it empties G's numerator,
it emptied H's until a corpus correction restored the one class of question the filter cannot reach,
and it shrinks the candidate set until an index scan is the wrong plan and K cannot hold.

That is a design error in criteria written before the implementation existed. Pre-registration is
what made it visible rather than convenient, and it is a more interesting result than a system that
cleared its own bar.

### What was not done about it

No third benchmark. No tuning against this hold-out — not one retrieval weight, effectivity rule,
bitemporal rule, embedding, chunking parameter, gate threshold, citation rule or corpus membership
was changed after it was scored. The failing set is **pinned** in `scripts/release_gate.py`, which
fails the build if reality diverges from it in either direction, so a regression cannot hide behind
a disclosed failure and a disclosed failure cannot quietly become a pass.

### Honestly not built

| | |
|---|---|
| Managed vector store comparison | a **local Qdrant container**, not Qdrant Cloud. The artifact says so in its own body and `qdrant_cloud_tested` is `false`. The cost block publishes measured byte counts and formulas with `null` where a price would go, because no vendor price list was read. |
| Live LLM inference | never called. No key exists, the abstractive arm **raises** rather than returning a stub, and no cost, latency or quality figure is published for any live arm. |
| Native-speaker evaluation | not performed. The corpus is **synthetic parallel text**, no native speaker reviewed it, and no LLM judge was used — so there is no judge to calibrate. The per-language scores measure retrieval and gating, not whether the Turkish reads like Turkish. |
| A typed front end | server-rendered HTML. The abstention and citation states are built; a bundler and a second language were not worth the day. |

### The first benchmark was invalid, and its history is intact

The first iteration was fully scored and then found not to be a valid retrieval test at all: after
filtering, most hold-out questions left fewer candidates than `top_k`, so ranking could not change
recall@10 and four arms scored exactly 1.0000. The `hybrid_without_effectivity` baseline did not
remove the filter. The split-leak check reduced to `split_of(F) != split_of(F)` and could not report
a leak. Three evidence artifacts had no producer. The corpus cleared its size floors only by
counting the English, Turkish and Russian renderings of one fact as three.

All of that is **ADR-002** in `DECISIONS.md`, and none of it was squashed out of git history. The
corpus was regenerated, a fresh hold-out was drawn, and the whole ordering is checkable — including
the part that is least flattering:

```
61b3d3c, eab2955   corpus generated
3ae15bf            hold-out materialised and FROZEN — this commit contains zero scores
eecdbf5            the first score over it
44589bc            RE-FROZEN, after that score existed — this commit also contains zero scores
d7cb4d4            the confirmed findings of three independent reviews, fixed
```

**Read `44589bc` sceptically, because it is a re-freeze after a score.** The rule did not change and
the draw was not re-taken: `holdout.compute` enumerated the held families from documents alone, so
questions about a product family the corpus deliberately does not contain — the whole
`different_product_family` failure mode — could never enter the frozen membership. Fifteen
questions, every one of them unanswerable and none answerable, which means the correction can only
make the hold-out **harder**: more chances for the gate to answer something unsupported, a larger
abstention denominator, and no effect at all on recall, which skips questions with no supporting
passage. A correction that cannot flatter a result is safe to apply to a scored hold-out. The
reverse would not be, and rule 2 of `CLAUDE.md` forbids it.

---

## Falsifiability

A passing suite is evidence that the tests pass, not that they would fail. So every guarantee is
broken on purpose, one at a time, by **replacing a function the running system calls** — the
candidate predicate, the gate's decision, the citation builder, the content hash — and a detector
has to observe the behaviour change. A guard that greps source text does not count.

`scripts/plant_breaches.py` plants ten: effectivity bypass, wrong variant, an off-by-one that makes
the validity interval inclusive, a knowledge-time bypass, a pgvector substitution, an unsupported
answer, a tampered citation, a missing disclosure, a frozen content hash, and a split leak.
`tests/test_falsifiability.py` requires each to be caught **and** requires the detector to have been
quiet before it was planted — a breach whose baseline was already dirty proves nothing.

---

## Running it

```sh
make setup        # uv sync
make db           # PostgreSQL with pgvector, and the local Qdrant the comparison measures against
make migrate
make corpus       # regenerated from a committed seed; the corpus is not in git
make index        # embed and load
make candidates   # how many passages the filter leaves, against top_k — run this before any score
make artifacts    # the evidence
make bitemporal   # both time axes, against the live database
make breaches     # break every guarantee and check each is caught
make test         # the engineering suite
make release-gate # the predeclared kill test, reported
make console      # http://127.0.0.1:8071
```

**`make test` and `make release-gate` are different questions and the distinction is deliberate.**

`make test` is the engineering suite — does the software work. It passes, and CI's engineering lanes
are green: lint, types, unit tests, a real pgvector integration, the bitemporal behaviour, citation
validation, artifact reproduction, the planted breaches and the image.

`make release-gate` runs the twelve predeclared kill conditions and prints
`ORIGINAL RELEASE GATE: FAILED`. It is **not** marked `xfail` and nothing is skipped — that would
convert a failed criterion into an accepted pass, which is the one thing `DECISIONS.md` forbids.
Instead the failing set is **pinned**, and the command exits non-zero if reality diverges from it in
either direction: a condition that starts failing is a regression, and a disclosed failure that
starts passing needs a human to decide whether the fix was legitimate or the hold-out was tuned
against.

`make evidence` runs the whole chain and ends with that report.

## Architecture

```
                      question + as_of + known_as_of + variant + serial
                                          |
                    exact part-number lookup (no model involved)
                                          |
          effectivity filter, in SQL  ---- both axes, variant, serial, language
                                          |
              +---------------------------+---------------------------+
              |                                                       |
         BM25 over the survivors                      pgvector over the same survivors
              |                                                       |
              +------------------ reciprocal rank fusion -------------+
                                          |
                                 deterministic rerank
                                          |
                          the gate: seven non-model signals
                                          |
                        ABSTAIN / REVIEW  |  ANSWER
                                          |
                        extractive answerer + verbatim citations
                                          |
                            Article 50 disclosure on the response
```

- **PostgreSQL 16 + pgvector**, HNSW, `vector_cosine_ops`. The vectors are not unit-norm — the
  quantised ONNX build does not normalise, and `retrieval_config.json` publishes the **measured**
  probe norm rather than an assumption.
- **fastembed**, `paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions, local, no network call at
  query time. Chosen because the corpus is parallel EN/TR/RU: a stronger English-only encoder would
  win the English split and make the one cross-lingual result meaningless.
- **No model provider.** Nothing in this build has ever called a model API.

`CLAUDE.md` is the operating contract, `DECISIONS.md` the record of what was fixed in advance and
what was got wrong.

## Licence

MIT. The corpus is synthetic, generated from a committed seed, and is not derived from any
manufacturer's documentation.
