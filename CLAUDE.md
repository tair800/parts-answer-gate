# CLAUDE.md — parts-answer-gate

The operating contract for this repository. Read it before touching anything here.

`DECISIONS.md` is authoritative and this file is subordinate to it. ADR-001 is the predeclared
contract and its twelve kill conditions; ADR-002 records what the first benchmark iteration got
wrong. Read both before changing retrieval, the store, the gate, the corpus or the evaluation.
Where this file and `DECISIONS.md` disagree, `DECISIONS.md` wins and this file is the thing to fix.

---

## 1. Purpose

Answer a technician's parts or procedure question with a **variant-correct, revision-correct**
citation, or abstain. The expensive failure is not a vague answer but a confident one taken from a
withdrawn bulletin or the wrong machine variant, so revisions are withdrawn rather than deleted and
*"what was the approved procedure on the date of the incident"* stays answerable. The system
publishes its own coverage-versus-wrong-answer curve in English, Turkish and Russian rather than a
single flattering point.

---

## 2. Architecture

### 2.1 The retrieval path

The order is the argument, and it is fixed:

```
deterministic identifier lookup        retrieval/pipeline.py :: Retriever._exact_stage
  -> effectivity filter, in SQL        store/effectivity.py  :: candidate_filter
  -> BM25 over the survivors           retrieval/bm25.py
  -> pgvector over the same survivors  retrieval/dense.py
  -> reciprocal rank fusion            retrieval/fusion.py
  -> deterministic rerank              retrieval/rerank.py
```

Stages 3 and 4 never see a row stage 2 rejected. That is why `RetrievalTrace.filter_applied` can
say `before_ranking` truthfully: there is no code path in which a score is computed for a row the
predicate refused. `pipeline.retrieve` raises if a fused chunk id falls outside the filtered
candidate set, so "both stages read the same `WHERE` clause" is asserted rather than assumed.

Exact part-number lookup runs **first** and finishes before the encoder is touched. Sending a
catalogue lookup through an embedding model wastes money and accuracy at once.

Lexical and dense scores are carried separately to the end. A single fused number cannot tell a
question that failed with a strong lexical signal from one where both signals were weak, and the
evaluation's job is to attribute error to a stage.

Fusion is RRF (`RRF_K = 60`, unchanged from Cormack et al.) with stage weights
`exact_identifier: 2.0`, `lexical: 1.0`, `dense: 1.0`. Each stage returns
`max(top_k * 5, 50)` candidates, because a stage truncated to `top_k` can only confirm what the
other stage already found. Ties break on `(-score, chunk_id)` everywhere; kill condition J does not
survive a tie resolved by whatever the database returned first.

The reranker is four features and four constants. No cross-encoder: a neural model deciding the
final ordering would put a model output upstream of the gate's inputs.

### 2.2 The bitemporal store

Two independent time axes, both denormalised onto `chunk` so the candidate predicate stays a
single-table `WHERE`:

- **Valid time** (`valid_from`, `valid_to`, `superseded_by`) — when a revision was in force for the
  machine.
- **Knowledge time** (`known_from`, `known_to`, `corrected_by`) — when the organisation believed it.

A `Query` carries `as_of` (validity) and `known_as_of` (knowledge; `None` means *current
knowledge*, which is `known_to IS NULL`, not today's date). The two questions are genuinely
different: *what was the correct torque in March 2021* is `as_of=2021-03` with current knowledge;
*what did the technician have in front of them in March 2021* pins both.

Both predicates are half-open — `valid_from <= as_of < valid_to` — so the day a revision is
withdrawn belongs to its successor and not to both.

Effectivity is `variant_id` plus an open-ended `SerialRange`, flattened into `variant_id`,
`serial_first`, `serial_last` so a btree can read it. The serial rule is deliberately asymmetric: a
query naming a variant but no serial matches only passages that apply to every serial.
`store/effectivity.py :: applies_in_python` mirrors the SQL so a test can assert the two are one
rule, and it is never used by the retrieval path.

`chunk.embedding` is a real `vector(384)` with an HNSW index declared in `store/schema.py`
(`m=16`, `ef_construction=200`, `vector_cosine_ops`). `store/engine.py` sets `hnsw.iterative_scan`
per connection, which is what makes "filter in SQL, then an indexed vector search over what
survives" a real plan rather than a description.

Alembic owns the shipped schema (`0001_initial_schema`, `0002_knowledge_time`).
`schema.create_schema` exists only for throwaway test databases and builds from the same metadata.

### 2.3 The gate

`gate.py`. Deterministic, computed **before** any generation, from seven signals **none of which is
a model output**: top fused score, supporting-chunk count, term coverage, variant agreement,
superseded-passage presence, conflicting measured values, exact-identifier hit.

Outcomes are `ANSWER`, `ABSTAIN`, `REVIEW`. `REVIEW` is not a hedge — it is the system saying the
evidence conflicts and a person must decide — and both `ABSTAIN` and `REVIEW` withhold
(`gate.withholds_answer`). An `ABSTAIN` decision carries no approved chunks; `domain.GateDecision`
validates that, so an answerer cannot be handed evidence for a question the gate refused.

Rules are evaluated in order, first match wins, and the loop only ever **withholds**. `ANSWER` is
constructed in exactly one place, the fall-through, and `test_gate.py` asserts that over the AST.
Adding a reason to answer would mean deleting a rule, which is visible in a diff.

`superseded_present` is checked ahead of the strength rules: if a passage outside the as-of window
reached the gate at all, the SQL filter failed, and abstaining would bury that under a message
about weak evidence. Every other conflict is checked after the strength rules, so a contradiction
between two passages too thin to support anything does not fill a review queue.

Thresholds are module constants carried into a call through `GateThresholds` (frozen,
`extra="forbid"`). The primary, swept threshold is `min_term_coverage`, shipped at 0.60, swept over
ten points from 0.0 to 0.9 to draw the published curve. It is primary because coverage is a share in
[0, 1] that this module owns; sweeping the fused score would sweep another module's units.
`min_top_fused_score` is deliberately 0.0 and deliberately untuned.

### 2.4 The extractive answerer and citations

`answerer.py`. The shipped arm is `ExtractiveAnswerer`: the answer **is** its citations, character
for character, joined by a blank-line separator that no part number can contain. The set of part
numbers in the text is therefore a subset of the set in the quotes as string arithmetic — kill
condition C made impossible rather than measured. The cost is fluency, and it was chosen
deliberately over a phrasing model that could not be verified.

`AbstractiveAnswerer` is a port with nothing behind it. Calling `answer` **raises**. It exposes
`prompt_payload` so the exact request is inspectable, and a test asserts the payload contains the
gate's approved chunks and nothing else. No stub, no canned string: either would flow into the
evaluation as a number attributed to a model that was never called.

`PostValidatedAnswerer` wraps any arm and enforces the grounding rule and the Article 50 disclosure
on the way out. It is a wrapper, not a base class, so it applies to an arm this repository does not
own.

`citations.py`. A citation is a **verbatim span plus its offsets into the document's own text**.
`verify_citation` slices at `start_offset` and compares; it does not test membership, because a
quote that appears elsewhere in the same manual is a coordinate that has drifted. Quote selection
uses the gate's own tokeniser and coverage rule, so the passage displayed is chosen by the measure
the gate scored with. There is no placeholder revision label: `MissingRevisionError` rather than
`unknown`.

Every `Answer` carries `ai_disclosure` with `min_length=20` on a frozen model — EU AI Act Article 50
transparency, a property of the response rather than of the UI. It is a transparency obligation and
nothing else is asserted.

### 2.5 The corpus and the hold-out

Synthetic, openly so, generated from a committed seed (`corpus/`), and never described as
real-world data. Ground truth is **construction metadata**: the generator knows which revision
superseded which and which questions have no supporting passage because it built them that way.
Nothing is labelled by a model.

`corpus/rng.py` seeds every draw from a hash of the thing being decided, never from a counter, so
inserting one variant does not shift every subsequent draw. `generate.py` checks offsets, the split
and the ADR-001 contract minimums and refuses to write a corpus that fails. No timestamps,
hostnames or paths in the output: two runs must be byte-identical.

The unanswerable set is seven distinct failures (`corpus/questions.py`), only one of which is the
easy case where nothing scores well.

The hold-out rule lives in exactly one place and consults no seed and no score:

> a product family is held out iff `blake2b(family_id, digest_size=8) % 100 < 34` (big-endian)

Split by **family**, not by question row: two questions about the same pump share passages, so a
random row split measures memorisation of a document the system was tuned on.
`holdout.freeze` materialises the membership into `artifacts/holdout.json` and raises
`HoldoutDriftError` if the corpus would now produce a different one.

### 2.6 The evaluation harness

`evaluation/runner.py` runs five arms that differ in **exactly one thing each**, through one
retriever and one corpus:

| arm | what is removed |
|---|---|
| `system` | nothing |
| `bm25_only` | the dense signal (`weights={"dense": 0.0}`) |
| `dense_only` | the lexical signal |
| `hybrid_without_effectivity` | the effectivity predicate itself (`apply_effectivity=False`) |
| `ungated_rag` | the gate, by replacing the decision — never by a flag inside `gate.decide` |

`evaluation/metrics.py` imports `domain` and the standard library and nothing else. It reads no
file and touches no database, so every number that decides whether this project ships can be
unit-tested against hand-written inputs.

`evaluation/artifacts.py :: build_all` writes the evidence. `tests/test_kill_criteria.py` grades
it and imports none of this package. The absolute-zero guarantees (C, D, E, I, Article 50) are
graded over the whole corpus; F, G and H stay hold-out-only, because those are comparative scores.

Artifacts under `artifacts/`:

| file | written by | carries |
|---|---|---|
| `corpus.json` | `scripts/generate_corpus.py` | counts, split rule, leak check, `is_synthetic` |
| `holdout.json` | `scripts/freeze_holdout.py` | the frozen membership and its digest |
| `effectivity.json` | `build_all` | A, B, the replay, `filter_applied` |
| `groundedness.json` | `build_all` | C, D |
| `gate.json` | `build_all` | E, I, the disclosure check |
| `evaluation.json` | `build_all` | F, G, H, the curve, error attribution, both splits |
| `multilingual.json` | `build_all` | per-language scores, the ablated fix, the judge caveat |
| `determinism.json` | `build_all` | J |
| `pgvector.json` | `build_all` | K, from the server's own plan |
| `index_lifecycle.json` | `build_all` | measured incremental re-embedding |
| `retrieval_config.json` | `build_all` | the configuration the numbers were taken under |
| `storage_comparison.json` | `store/comparison.py` | pgvector against a local Qdrant |
| `breaches.json` | `scripts/plant_breaches.py` | every planted breach and whether it was caught |
| `candidate_profile.json` | `scripts/candidate_profile.py` | candidates left after filtering, against `top_k` |

`candidate_profile.json` is not graded by the kill test. It exists because ADR-002 found the
retrieval task saturated by construction, and it measures a property of the **corpus and the
filter**, never of performance. Run it before reading any recall figure.

### 2.7 The service

`api/app.py`. Four server-rendered screens (`/`, `/evidence`, `/failures`, `/provenance`), a JSON
endpoint (`/api/ask`) and `/healthz`. Read-only **by construction**: there is no route that ingests,
re-indexes or edits. `PAG_READ_ONLY` is a label on the header, not a gate in front of a write, and
the code says so rather than implying a permission system with nothing to protect.

`/evidence` reads `artifacts/*.json` and nothing else, so it works with no database at all.
`_kill_rows` renders a missing artifact as a row saying so, never as a blank cell a reader would
take for a pass.

---

## 3. Key commands

`make help` lists them. The chain, in the order a reviewer would run it:

```
make setup       uv sync
make db          PostgreSQL with pgvector on 127.0.0.1:15440, Qdrant on 16333/16334
make migrate     alembic upgrade head
make corpus      generate the synthetic trilingual corpus from its committed seed
make determinism build it twice and diff every byte
make index       embed and load into pgvector
make artifacts   run the evaluation and write the evidence
make breaches    plant a defect into every guarantee and check each is caught
make test        the whole suite, kill criteria included
make fast        lint, types, and everything that needs no infrastructure
make lint        ruff check + ruff format --check
make types       mypy --strict
make console     the Answer Gate Lab on http://127.0.0.1:8071
make evidence    the full chain, and what CI runs
```

`make evidence` is the only command whose success means anything about the project's claims.

---

## 4. Conventions

- **Typed Python 3.12**, `from __future__ import annotations` everywhere, `src/` layout, `uv` with a
  committed `uv.lock`, `hatchling` build backend.
- **Pydantic v2 models are frozen with `extra="forbid"`** (`domain._Frozen`, `GateThresholds`,
  `PromptPayload`). `extra="forbid"` does real work here: a mistyped `varient` silently becoming an
  ignored attribute is how an effectivity filter comes to filter nothing.
- Dataclasses are `frozen=True` where they carry results (`CandidateFilter`, `RetrievalTrace`,
  `RunSet`, `FrozenHoldout`).
- **ruff at 100 columns**, `target-version = "py312"`, the rule set in `pyproject.toml`.
  Per-file ignores are argued in place; do not widen them silently. `ruff format --check` is part of
  lint, so formatting is not optional.
- **mypy `--strict`** over `src`. Not advisory.
- **Docstrings argue WHY, at length, and name the failure they prevent.** This is the house style
  and it is load-bearing: the reasoning for a threshold, a byte order, a half-open interval or a
  rejected alternative lives beside the code, not in a commit message nobody reads. A rejected
  option is recorded as rejected, with its reason. Match this voice — British English, plain,
  no exclamation marks, no marketing adjectives — in any file you add.
- Constants that appear in an artifact are **imported from the code that implements them**
  (`FILTER_STAGE`, `HOLDOUT_RULE`, `PRIMARY_THRESHOLD_NAME`, `HNSW_BUILD_PARAMETERS`), so the claim
  and the implementation are the same string.
- One rule, one place. Where a rule is unavoidably expressed twice (the SQL predicate and its Python
  mirror; the SQL predicate and the Qdrant payload filter), a test compares the two rather than a
  comment asserting they agree.
- Structured, deterministic output: no wall clock in a benchmark, no `date.today()` default buried
  in a model, no set iteration deciding an order.

---

## 5. Testing requirements

- `tests/test_kill_criteria.py` grades `artifacts/*.json` and **imports nothing from
  `parts_answer_gate`**. `test_predeclaration.py` asserts that over the AST.
- `tests/test_predeclaration.py` parses the kill test: thresholds must remain module-level
  constants, the zeros must be zero, the floors must not have been lowered, and once the package
  imports, nothing in the kill test may skip, xfail or importorskip.
- `tests/conftest.py` inspects collected items with `tryfirst=True` and raises `pytest.UsageError`
  if a mark disabled the kill test — before `-m` deselection can hide it. Deselection is not
  disablement, and a failing test can be deselected too.
- `scripts/plant_breaches.py` breaks each guarantee in turn by **replacing a function the running
  system calls**, and `tests/test_falsifiability.py` requires every breach in `REQUIRED_BREACHES` to
  be planted, to be caught, and to have had a clean baseline before it was planted. A breach whose
  baseline was already dirty proves nothing.
- Store and retrieval tests skip without a database; the gate, citations, metrics, domain and
  Article 50 tests are real coverage and run with no infrastructure.
- Before any commit: `make lint`, `make types`, `make fast`. Before any claim about a measured
  result: `make evidence`.

---

## 6. Deployment shape

- **Docker**, two stages. The build stage has `uv` and a compiler; the runtime stage has neither.
  The embedding model (~220MB of ONNX) is baked in at build time so a cold start does not depend on
  a model host. Runs as a non-root system user with no home and no shell.
- `docker-compose.yml` for local infrastructure: `pgvector/pgvector:pg16` on
  `127.0.0.1:15440` and `qdrant/qdrant:v1.19.0` on `127.0.0.1:16333`/`16334`. Both bound to
  loopback. Postgres is tuned (`shared_buffers`, `maintenance_work_mem`, `work_mem`) so an HNSW
  build does not spill and kill condition K does not become a question about container defaults.
  Qdrant deliberately mounts **no volume**: the comparison rebuilds its collection from the vectors
  already in PostgreSQL every run, and an index that outlives a corpus change is how a flattering
  recall figure survives the data it was measured on.
- `docker-entrypoint.sh` indexes, then `exec`s uvicorn on `$PORT` (default 8000). `set -e` is
  deliberately absent: a seeding failure must not crash-loop a console whose evidence screen works
  without a database, and `/healthz` reports the real state (200 with a database, 503 without).
- Target deployment is **Render free tier**, not the Cloudflare Tunnel the blueprint scoped.
  Indexing at container start is an exemption earned by a single-instance plan, not a design: more
  than one replica must move migration and loading back out of the serving process, or two replicas
  race through the same embedding work.
- CI (`.github/workflows/ci.yml`) has four lanes, none `continue-on-error`: `fast` (lint, types,
  infrastructure-free tests), `evidence` (real pgvector + Qdrant services, corpus, determinism,
  index, artifacts, kill criteria, full suite), `falsifiability` (planted breaches, then
  `git diff --exit-code` scoped to `src tests scripts alembic`), and `docker` (the image builds,
  serves degraded with no database, then migrates and serves against a real one).
  The `evidence` lane must use `pgvector/pgvector`, never plain `postgres`: kill condition K is the
  claim that the extension participates, and a plain image would make the lane green and the claim
  unrunnable.

---

## 7. Current integrations

- **PostgreSQL 16 + pgvector** — the primary store and the vector index. Required.
- **Qdrant, local container** — the second backend behind the same storage port, for
  `storage_comparison.json` only. Nothing that serves a request reads it. **Qdrant Cloud has never
  been reached**, and the artifact says so in its own body; the cost column is published-list-price
  arithmetic, clearly labelled, not a measured bill.
- **fastembed / `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**, 384 dimensions,
  quantised ONNX, local. No PyTorch, no CUDA, no network call at query time. Chosen because the
  corpus is parallel EN/TR/RU and a stronger English-only encoder would win the English split and
  make the one cross-lingual result meaningless.
- **rank-bm25** — lexical scoring, index built per query over the *filtered* candidate set. A
  persistent global index would compute IDF over rows that are not candidates and would score rows
  the effectivity filter rejected.
- **No model provider.** `PAG_LLM_API_KEY` has no default and nothing in this build has ever called
  a model API.
- Configuration is `PAG_*` environment variables through `config.py`; see `.env.example`.
  `read_only` defaults to true and the API key has no default at all — both fail closed.

---

## 8. NON-NEGOTIABLE RULES

1. **`tests/test_kill_criteria.py` and `tests/test_predeclaration.py` are PREDECLARED.** Thresholds
   may be raised. They may never be lowered. Neither file may be edited to make a build pass. If a
   measurement misses, the honest outcomes are to fix the system or to record the failure in
   `DECISIONS.md` — not to move the line, not to add a skip, not to relax an assertion, not to
   rename the field the assertion reads.

2. **The hold-out must never be re-drawn to improve a score.** `artifacts/holdout.json` is frozen
   and its position in git history is the evidence. Re-freezing invalidates every score taken
   against the old membership, and those scores must be re-run rather than carried over.
   `allow_refreeze` exists for a corpus that legitimately grew; it is not a way to make an
   inconvenient `HoldoutDriftError` go away.

3. **Nothing may be tuned against the hold-out after it has been scored.** Not retrieval weights,
   not the effectivity logic, not the bitemporal logic, not the embedding model, not chunking, not
   the gate thresholds, not the citation rules, not ranking, not corpus membership. If a change is
   needed after scoring, it is a new experiment and the hold-out is re-scored under a recorded
   decision, not quietly reused.

4. **Never claim a number that was not measured, and never describe an unbuilt thing as built.**
   Every figure in a README, an artifact, the console or a commit message must come from a
   committed code path that reproduces it. Evidence only a person can regenerate by hand is not
   evidence.

5. **The effectivity predicate lives in exactly one place — `store/effectivity.py` — and must run in
   SQL before ranking, never as a post-filter over a ranked list.** Post-filtering a top-k list
   silently shrinks it: ask for ten, discard six, answer from four, and the recall you lost is
   invisible. Any new retrieval path takes a `CandidateFilter`; none re-spells the predicate.

6. **Guards must fail from BEHAVIOUR, not from source-text inspection.** A breach is planted by
   replacing a function the running system calls, and the detector observes rows that should not
   have been returned, an answer that should not have been given, a re-embed that should have
   happened. A grep over source text does not count as a guard.

7. **No model API key, and the shipped answerer is extractive.** The abstractive arm raises. It must
   never be made to return a stub, an echo or a canned string, because that number would enter the
   evaluation attributed to a model that was never called. No cost, latency or quality figure may be
   published for any live arm.

8. **No model may move a question from `ABSTAIN` to `ANSWER`.** The gate decides before the answerer
   is constructed, the answerer sees only `decision.approved_chunks`, and an abstention approves
   nothing. `ANSWER` stays reachable from exactly one place in `gate.py`.

9. **The corpus is synthetic and is described as synthetic everywhere**, including in the body of
   every generated file. It is never presented as real-world data or as a manufacturer publication,
   and the multilingual result is never implied to be a human-validated benchmark: no native speaker
   reviewed this corpus.

10. **Counts are counts of distinct content.** A translation of a passage is not a second passage.
    `corpus.json` reports distinct content under the plain keys and per-language rows under the
    `*_rows` keys, and `counted_by` says which is which. Clearing a contract floor by multiplying
    by three languages is how the first iteration failed its own corpus contract.

11. **No Redis, no queue, no background worker, no scheduler.** The skill matrix does not assign
    them to this project, and a dependency nothing needs is still a dependency somebody has to
    patch.

12. **No secrets in the repository.** `.env` is gitignored; `.env.example` carries placeholders. The
    compose and CI credentials are local-only values with no reach outside their own network
    namespace, and they stay that way.

---

## 9. The two benchmark iterations

Ordering, and it matters: **corpus generated → hold-out frozen at commit `3ae15bf` → scored
afterwards.** The freeze commit precedes the first commit carrying a score, which is what makes the
hold-out result mean anything and what a reader who does not trust the author can check.

**Iteration one was scored, and then found invalid.** ADR-002 has it in full; briefly:

- Kill conditions **F** and **H** are reported **FAIL** and stay FAIL. Both failures are in the
  criteria, not in the split. F requires the system to beat `ungated_rag` on a *retrieval* metric,
  but that arm removes only the gate and therefore runs the identical retriever — an impossible
  target, and a defect in a criterion written before the arms existed. H requires an ungated
  wrong-answer rate above zero, but the effectivity filter runs before the gate and already
  saturates the quantity H measures, so the ungated rate is zero by construction rather than by
  merit.
- The `hybrid_without_effectivity` baseline did not remove the filter. It moved `as_of` to
  2099-12-31 and left the half-open predicate in place, which is strictly *more* filtered: it
  deleted the gold passage for every question whose answer had since been superseded. Its recall
  measured a query issued at the wrong date, and no claim may be built on it. The arm now removes
  the predicate (`apply_effectivity=False`).
- The retrieval task was saturated by construction: after filtering, most hold-out questions left
  fewer candidates than `top_k`, so ranking could not change recall@10 at all. Not fixable by
  tuning. `scripts/candidate_profile.py` exists to make that visible before any score is read.
- The corpus cleared ADR-001's size floors only by counting EN, TR and RU renderings of the same
  content as three items. On distinct content it missed several floors, and the supersession floor
  was missed on any reading. The corpus was regenerated, which invalidated the first freeze.
- `pgvector.json`, `index_lifecycle.json` and `storage_comparison.json` existed only as helper
  functions with no caller; the numbers quoted for them came from scripts run by hand.
- One harness defect was corrected *after* scoring: wrong-answer scoring compared bare revision
  labels, which are not unique across families, so citing another family's `C` scored as correct.
  The comparison is now keyed `family/revision` (`runner.revision_key`). This is a correction to the
  measuring instrument, not to the system under test, and it can only move a measured rate upward —
  which is the property that made it safe to apply to an already-scored hold-out.
- **No re-draw, and the system was not touched.** A fresh hold-out would have changed neither
  result, so re-drawing one could only have looked like shopping for a friendlier number. No
  retrieval threshold, rule, effectivity condition, gate condition, chunking parameter, embedding or
  citation rule was changed after the hold-out was scored.

**Iteration two is the current one.** The corpus is regenerated and counted by distinct content; the
hold-out is frozen again at `3ae15bf` and scored after that; the baseline removes the predicate
rather than simulating its removal; the three orphaned artifacts are produced by the ordinary build;
`bitemporal_knowledge_time` and `split_leak` were added to the required breaches, the latter because
its guard had been vacuous — it read a key the corpus does not use and so inspected nothing.

When quoting a result, quote the artifact by filename. Do not copy a number into prose: every
current figure is being replaced.

---

## 10. Recorded gaps

From ADR-001's own table, recorded so they cannot look like omissions discovered later. Nothing in
this list may be described as built.

| blueprint item | status |
|---|---|
| Managed vector store comparison | local Qdrant container, **not** Qdrant Cloud; cost column is list-price arithmetic |
| Live LLM inference | not called; no key; extractive arm only; no cost, latency or quality figure published |
| Judge calibration against native-speaker checks | not performed; no native speaker reviewed this corpus; no LLM judge was used, so there is no judge to calibrate |
| Next.js / TypeScript technician UI | server-rendered HTML; the abstention and citation states are built |
| Cloudflare Tunnel self-hosted deployment | Render free tier |
| HNSW tuning sweep | one committed configuration with its build parameters recorded |

Open items in the repository itself, to be closed rather than papered over:

- `make artifacts-check` and `make screenshots` call `scripts/check_artifacts_current.py` and
  `scripts/screenshots.py`, neither of which exists.
- `store/comparison.py :: build_storage_comparison_artifact` has no caller, so
  `storage_comparison.json` is not produced by `make evidence` although the kill test grades it.
- `docker-entrypoint.sh` does not run `alembic upgrade head`, although its header says it migrates
  and the Dockerfile copies the migrations in for that purpose.
- CI passes `PAG_SKIP_SEED` to the bare-container step; nothing in the codebase reads it.
- There is no `README.md` and no `PROJECT_STATUS.md`, both of which the workspace contract requires
  and which ADR-001 repeatedly says will publish specific figures. `docs/` is empty.
