# parts-answer-gate

**A parts and service-procedure answering system that refuses more often than it answers, and can
prove why.**

A technician asks *what is the case drain pressure limit for this machine?* The expensive failure is
not a vague answer. It is a confident one taken from a bulletin that was withdrawn two years ago, or
from the manual for the variant next to theirs on the shop floor. That answer is a wrong part
shipped, a voided warranty, or a machine returned to service outside its limits — and it looks
exactly like a right answer.

So this system is built around the refusal rather than around the answer:

- **Eligibility is decided in SQL, before anything is ranked.** A passage that was not in force on
  the date asked about, or that belongs to another variant or serial range, is never scored at all.
  Not retrieved and discarded — never a candidate.
- **The decision to answer is deterministic and precedes generation.** Seven signals, none of them a
  model output. The rule loop can only *withhold*; the answering outcome is reachable from exactly
  one place in the code, by falling through every rule.
- **The answer is its citations, character for character.** The shipped answerer is extractive, so
  the set of part numbers in an answer is a subset of the set in its quotes as string arithmetic
  rather than as a measured rate.
- **Two time axes, not one.** *What was true of the machine in March 2021* and *what did we know in
  March 2021* are different questions with different right answers, and a later correction does not
  rewrite the historical one.

Everything below is measured by a committed code path against a hold-out that was frozen before any
score over it existed. **Two of the twelve predeclared kill conditions fail, and they are reported
as failures.** See [Where it falls short](#where-it-falls-short).

---

## The sole-home skill: filtering before ranking

Most retrieval systems rank first and filter afterwards, because that is what a vector database
makes easy. Post-filtering is wrong here in a way that does not show up in the output: ask an index
for ten, discard six that belong to another variant or to a withdrawn revision, answer from four —
and the evaluation still records that ten were retrieved. The recall you lost is invisible, and the
four survivors look like a confident result.

The predicate therefore goes into the `WHERE` clause of **both** ranking queries, the lexical
candidate fetch and the pgvector search:

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

This section is the point of the project, not an appendix to it.

### Two predeclared kill conditions fail

`DECISIONS.md` fixed twelve kill conditions and their thresholds **before any source file existed**.
Thresholds may be raised and may never be lowered, and none has been touched. Two fail, and both
fail because the criterion was wrong rather than because the system is:

- **F — "recall@10 above every predeclared baseline."** One of the four baselines, `ungated_rag`,
  removes only the gate and therefore runs the identical retriever; beating it on a *retrieval*
  metric is not a hard target but an impossible one. More substantively, recall@10 turns out to be
  the wrong measure of what effectivity filtering buys: removing the filter enlarges the candidate
  pool, so recall can go *up* while the right passage falls from rank 1 to rank 8. The MRR column in
  the table above is where the difference is, and F does not ask about MRR.
- **E — "the gate never answers a question the corpus cannot support."** It does, ten times out of
  306, and **every one of them is Turkish or Russian**. `gate.term_is_covered` approximates stemming
  with a bidirectional prefix match for terms of four characters or more, because an exact-match
  rule would score coverage near zero on two of the three languages and call that caution. Its own
  docstring says it errs towards covering. The first benchmark's corpus was too easy to show what
  that costs; this one prices it.

The gate was **not** adjusted after the hold-out was scored. That is the whole discipline: a
threshold moved to clear a line it did not clear is not a measurement.

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
corpus was regenerated, a fresh hold-out was drawn, and the ordering is checkable:

```
61b3d3c, eab2955   corpus generated
3ae15bf            hold-out MATERIALISED AND FROZEN — this commit contains zero scores
   ...             first score, in a later commit
```

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
make setup       # uv sync
make db          # PostgreSQL with pgvector, and the local Qdrant the comparison measures against
make migrate
make corpus      # regenerated from a committed seed; the corpus is not in git
make index       # embed and load
make artifacts   # the evidence
make breaches    # break every guarantee and check each is caught
make test
make console     # http://127.0.0.1:8071
```

`make evidence` runs the whole chain, and is what CI runs. It is the only command whose success
means anything about the claims above.

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
