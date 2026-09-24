# DECISIONS

## ADR-001 — The contract, recovered from the portfolio sources, and the gates that can kill it

**Status: accepted. Written and committed before a single source file exists.** Nothing in this
document may be revised downward once an implementation has been scored against it.

---

### Where this came from

The brief that started this build described a generic "answer gate" and guessed at the domain. The
authoritative sources say something considerably more specific, and they win. Recovered from
`portfolio-control/PORTFOLIO_BLUEPRINT.md` §7, `portfolio-control/SKILL_MATRIX.md` and
`PORTFOLIO_MASTER_SPEC.md`:

**Single purpose.** Answer a technician's parts or procedure question with a **variant-correct,
revision-correct** citation — or abstain — and publish the coverage-versus-wrong-answer curve in
English, Turkish and Russian.

**The expensive failure is not a vague answer.** It is a confident one taken from a **withdrawn
bulletin or the wrong variant**: wrong part shipped, component scrapped, warranty voided. Older
revisions are withdrawn rather than deleted, so *"what was the approved procedure on the date of the
incident"* must still be answerable.

**Headline claim, verbatim from the blueprint.** *An as-of query provably cannot return a superseded
revision or a wrong-variant procedure — replayed at multiple query dates in CI against a committed
golden supersession set — and the system abstains rather than guessing, with the
coverage-versus-wrong-answer curve published for the same corpus and question set in English, Turkish
and Russian, the degradation split into retrieval and generation, one ablated cross-lingual fix with
its measured effect, and the caveat that the LLM judge itself degrades off English, backed by a
human-scored calibration sample.*

---

### What the skill matrix assigns to this project

Project 7 is the **sole home** of five skills. If this build does not deliver them, the portfolio has
no evidence of them anywhere:

| sole-home skill | what that obliges |
|---|---|
| Metadata / effectivity filtering **before** ranking | model, serial range, revision and in-force window must constrain the candidate set before a score is computed, not filter a ranked list afterwards |
| Bitemporal / as-of retrieval | every chunk carries a validity interval **and** a knowledge interval; a query carries an as-of date; the answer is what was in force then |
| Managed vector store comparison | one alternative backend behind the same storage port, with a published recall@k, p95 and cost comparison against pgvector |
| Index lifecycle and incremental re-embedding | re-embedding only what changed, with cost accounting and a staleness measure |
| Multilingual evaluation + judge calibration | the same corpus and question set in EN, TR and RU, with the degradation split into retrieval and generation |

Shared but required (`●`): pgvector · RAG end-to-end through citation · embeddings · hybrid BM25 +
vector with RRF · reranking · groundedness/faithfulness scoring · citation-span faithfulness ·
retrieval-vs-generation error attribution · recall@k / nDCG / MRR · committed golden dataset · eval
as a build-failing CI gate · recorded-cassette eval with no live key in CI · cost per operation ·
p50/p95 latency · structured outputs · **guardrails / abstention** · type-level containment of model
output · regulatory reasoning on in-force obligations only · FastAPI · PostgreSQL · Docker · CI.

**Redis is NOT assigned to project 7.** The matrix gives Redis to 1, 4, 8 and 9, and queues to 1, 8
and 9. None is added here. A dependency nothing needs is still a dependency somebody has to patch.

**Background workers are NOT assigned to project 7.** No queue, no worker, no scheduler.

---

### The three claims

**Claim 1 — an as-of query cannot return a superseded revision or a wrong-variant procedure.**
Not "is unlikely to": the effectivity filter runs in SQL, before ranking, and a chunk outside the
asked-for variant, serial range or in-force window is not in the candidate set at all. Replayed at
several query dates against a committed golden supersession set.

**Claim 2 — the system abstains rather than guessing, and the curve is published.**
Coverage and wrong-answer rate are reported together, as a curve over the gate's threshold, not as a
single flattering point. A wrong answer is defined as **revision-incorrect or variant-incorrect**,
which is the definition the domain cares about and the one commercial vendors avoid publishing.

**Claim 3 — a part number the retrieved evidence does not contain cannot appear in an answer.**
Structural, not statistical. The default answerer is extractive: it returns spans from the cited
chunk. An abstractive arm sits behind the same port and is post-validated against the same rule.

---

### The answer gate

Deterministic. Computed **before** any generation, from signals none of which is a model output:

| signal | why it is in the gate |
|---|---|
| top fused retrieval score | the cheapest evidence that nothing relevant was found |
| number of chunks above the support floor | one lucky chunk is not corroboration |
| citation coverage of the question's key terms | a chunk about the right product but the wrong attribute scores well and supports nothing |
| variant and effectivity agreement across the supporting chunks | evidence drawn from two different variants is a conflict, not a consensus |
| presence of a superseding edge on any supporting chunk | the single failure this project exists to prevent |
| deterministic exact-identifier hit | an exact part-number match is a different kind of evidence from a semantic one |

Outcomes are `ANSWER`, `ABSTAIN` and `REVIEW`. `REVIEW` is not a hedge: it is the system saying the
evidence conflicts and a person must decide, and nothing is presented as an answer while a question
sits in it.

**No model may move a question from `ABSTAIN` to `ANSWER`.** The gate's decision is computed and
recorded before the answerer is called, and the answerer receives only the chunks the gate approved.

---

### The AI boundary

The model **may** phrase an answer from chunks the gate approved, and summarise a conflict.

The model **may not** invent a specification, choose evidence that was not retrieved, override the
gate, fabricate a citation, turn an abstention into an answer, or mutate source truth.

`ANSWER` is reachable from exactly one place in the gate module, and a test asserts it over the AST.

---

### Deterministic-first

Exact part-number and serial-range lookup is deterministic and runs **first**. The README publishes
what share of the question set never needs a model at all. A retrieval system that sends an exact
catalogue lookup through an embedding model is wasting money and accuracy at once.

---

### The corpus

**Synthetic, and openly so.** No redistributable corpus of multi-revision service manuals with
variant effectivity, serial ranges, supersession edges and parallel EN/TR/RU text exists — real ones
are a manufacturer's commercial property. This project generates its own from a committed seed and
**never describes the result as real-world data**. Every generated file says `is_synthetic` in its
own body.

Ground truth is **construction metadata**, not an opinion: the generator knows which revision
superseded which, which variant a procedure applies to, and which questions have no supporting
passage, because it built them that way.

| the corpus must contain | minimum |
|---|---|
| product variants | 6 |
| documents | 40 |
| revisions, with supersession edges | 25 superseding edges |
| chunks | 1,500 |
| languages | 3 — EN, TR, RU, as parallel text |
| benchmark questions | 300 |
| of which genuinely unanswerable | 90 |

**The unanswerable set is not padding.** It contains: a requested specification absent from the
corpus; a product that exists with the requested attribute absent; a near-match part with a
different identifier; a superseded part whose replacement is not asked about; contradictory
passages; a question about another product family; and a malformed part number.

---

### The hold-out

**Frozen by product family, not by random question rows.** Random splitting leaks: two questions
about the same pump share the same passages, so a random split tests memorisation of a document the
system has already been tuned on.

The rule is committed here, before anything is scored, and it consults no score:

> A product family is held out if `blake2b(family_id, digest_size=8) % 100 < 34`.

Everything about a held-out family — its documents, its chunks, its questions — is in the hold-out.
**Once scored, nothing in retrieval, the gate or the corpus is tuned against it.**

---

### Predeclared baselines

Named now so the strongest cannot quietly be dropped later:

1. **BM25 only** — lexical retrieval, no embeddings.
2. **Dense only** — pgvector similarity, no lexical signal.
3. **Hybrid without effectivity filtering** — the same fusion this system uses, with the temporal and
   variant constraints removed. This is the baseline that matters: it isolates what the sole-home
   skill actually buys.
4. **Ungated RAG** — hybrid retrieval that always answers, never abstains. This is what an ordinary
   RAG demo is, and the wrong-answer rate it produces is the number this project exists to beat.

---

### Kill conditions

Each is a specific number or a zero, fixed now. The build fails if any is false.

| | condition | threshold |
|---|---|---|
| **A** | an as-of query returns a chunk that was superseded at that date | **0** |
| **B** | an as-of query returns a chunk belonging to a variant the question did not ask for | **0** |
| **C** | a part number appears in an answer that is absent from every cited chunk | **0** |
| **D** | a cited span does not appear in the document it is attributed to | **0** |
| **E** | the gate answers a question whose supporting evidence the corpus does not contain | **0** |
| **F** | hybrid retrieval recall@10 on the hold-out | **≥ 0.85**, and above every predeclared baseline |
| **G** | wrong-answer rate on the hold-out, at the shipped gate threshold | **≤ 0.02** |
| **H** | wrong-answer rate of the ungated baseline, to prove the gate is not decorative | **≥ 5× the gated rate** |
| **I** | abstention on the unanswerable set | **≥ 0.90** |
| **J** | two runs of the whole pipeline produce identical retrieval and identical gate decisions | byte-identical |
| **K** | vector retrieval executes through pgvector, not through an in-process fallback | proven by a planted breach |
| **L** | a document appears in both the development and hold-out splits | **0** |

**Thresholds may be raised. They may never be lowered.** If a measurement misses, the honest
outcomes are to fix the system or to record the failure — not to move the line.

---

### Falsifiability

A passing suite is evidence the tests pass, not that they would fail. Breaches are planted into every
guarantee above and the suite must catch each one, including: bypassing the gate; swapping a citation
for a non-supporting chunk; returning another variant's evidence; answering on empty retrieval;
disabling the effectivity filter; replacing the pgvector query with an in-memory sort; leaking a
hold-out family into development.

**Guards that only read source text do not count.** A breach must change behaviour and a test must
observe the behaviour change.

---

### What this build does not include, against the blueprint

Recorded now so it cannot look like an omission discovered later. The blueprint scopes ~11 sessions;
the instruction for this increment is a 1–2 day fast-track.

| blueprint item | status |
|---|---|
| Managed vector store comparison | **Delivered against a local Qdrant container, not Qdrant Cloud.** The blueprint says the comparison is deleted rather than trimmed if it consumes days; a cloud account, its credentials and its network variance are exactly that risk. Two real backends behind one port give a genuine recall@k and p95 comparison. **The cost column is published-list-price arithmetic, clearly labelled, not a measured bill.** |
| Live LLM inference | **Not called.** No key exists. The default answerer is **extractive** — it returns spans from the cited chunk and therefore *structurally* cannot emit an absent part number — and the abstractive arm sits behind the same port, unexercised. **No cost, latency or quality figure is published for any live arm.** |
| Judge calibration against native-speaker spot checks | **Not performed.** No native speaker reviewed this corpus. The multilingual result is therefore a measurement over *synthetic parallel text* whose ground truth is construction metadata, and the README says exactly that rather than implying a human-validated multilingual benchmark. |
| Next.js / TypeScript technician UI | **Server-rendered HTML.** A bundler and a second language are a day this increment does not have. The abstention and citation states the blueprint asks for are built. |
| Cloudflare Tunnel self-hosted deployment | **Render free tier instead.** A tunnel needs an always-on machine belonging to the owner. |
| HNSW index tuning sweep | **One committed configuration with its build parameters recorded**, not a swept comparison. |

**Nothing in the README, the UI or this file may describe an unbuilt item as built, or publish a
number that was not measured.**

---

### Article 50 transparency

The EU AI Act's Article 50 transparency obligation is in force (applicable 2 August 2026, grace to
2 December 2026) and this surface talks to end users, so it is implemented: every answer the system
returns is labelled as machine-generated, the disclosure is present on the response object rather
than only in the user interface, and a test asserts it cannot be removed from a response.

This is a **transparency** obligation and nothing more. Implementing it asserts no conformity with
any other instrument, and this project claims none.

---

## ADR-002 — Two predeclared kill conditions are unsatisfiable as written

**Status:** accepted · **Date:** 2026-09-24 · **Written after the hold-out had been scored.**

ADR-001 is above, unedited. This is an addendum, and everything in it was learned *after* the first
hold-out score existed. Nothing here changes a threshold, because a threshold changed after seeing a
number is not a threshold. Conditions **F** and **H** are reported **FAIL**.

### F — "recall@10 ≥ 0.85, and above every predeclared baseline"

Measured on the hold-out: system **1.0000**, `bm25_only` **1.0000**, `dense_only` **1.0000**,
`ungated_rag` **1.0000**, `hybrid_without_effectivity` **0.1714**.

The floor is cleared. The comparison is not, for two separate reasons:

1. **`ungated_rag` cannot be beaten on retrieval, by construction.** ADR-001 defines that arm as the
   one that removes *the gate*. It removes no retrieval component, so it runs the identical
   retriever and returns the identical ranking. `system > ungated_rag` on a retrieval metric is not
   a hard target, it is an impossible one. That is a defect in the criterion, written before the
   arms existed, and it was mine.
2. **The retrieval task saturates at k=10.** Once effectivity filtering has cut the candidate set to
   one variant of one family at one date, ten slots are more than the task needs, and a lexical-only
   retriever finds the supporting chunk too. On the hold-out the system is in fact *below*
   `bm25_only` on MRR — 0.9952 against 1.0000 — because fusion demotes a top-1 lexical hit in 1 of
   105 scored questions.

**The one comparison that appeared to carry the project's argument does not survive inspection
either.** `hybrid_without_effectivity` scores 0.1714, and that number is not "what the effectivity
filter buys". The arm does not remove the filter: it moves `as_of` to 2099-12-31 and leaves the
half-open predicate in place, so `valid_from <= as_of < valid_to` still excludes every superseded
chunk — and now also excludes the *gold* chunk for every question whose answer has since been
superseded. 0.1714 is the recall of a query issued at the wrong date. It measures a broken baseline,
not a filter's contribution, and no claim may be built on it.

### H — "ungated wrong-answer rate ≥ 5× the gated rate"

Measured on the hold-out: gated **0.0000**, ungated **0.0000**. The test also requires
`ungated > 0`, on the reasoning that an ungated baseline with no wrong answers means the corpus is
too easy. The corpus is not too easy. The metric is blind.

ADR-001 defines a wrong answer as **revision-incorrect or variant-incorrect**. The effectivity
filter runs *before* the gate, in SQL, and constrains family, variant and as-of date. Every hold-out
question names its variant (153 of 153), and each variant belongs to exactly one family. So every
candidate that survives the filter is right-family, right-variant and in force — and an arm that
removes only the gate is therefore *incapable* of producing a revision- or variant-incorrect answer.
Its rate is zero by construction, not by merit.

H asks the gate to improve a quantity the filter has already saturated. It conflates two components
the same ADR is careful to separate everywhere else.

What the gate does prevent is answering with no supporting passage at all. The ungated arm answered
all 48 unanswerable hold-out questions; the shipped system answered none of them. That is measured
and published as `answering.supplementary` in `artifacts/evaluation.json`, and it is **disclosed,
not substituted**: it was declared after scoring, it carries none of the pre-registration weight of
the twelve conditions, and H stays FAIL.

### The retrieval task is saturated by construction

Measured directly against the database, over all 153 hold-out questions: after the effectivity
filter the candidate set has a **median of 9 chunks and a maximum of 10**, and **84.3% of questions
leave fewer candidates than `top_k` = 10**.

When the candidate set is smaller than k, every arm that shares the filter returns the same set and
ranking cannot change recall@10 at all. The 1.0000 is a property of the corpus geometry, not
evidence that retrieval works. BM25, the dense signal, fusion and the reranker are all measured on a
task where none of them can be wrong. This is the root cause of F, and it is not fixable by tuning —
it needs a corpus with enough in-force material per variant and date for ranking to matter.

### The corpus does not meet its own predeclared contract

ADR-001's size floors are cleared only by counting the EN, TR and RU renderings of the same content
as three separate items. Measured on distinct content:

| | published | distinct | ADR-001 floor | |
|---|---|---|---|---|
| documents | 108 | **36** | 40 | miss |
| chunks | 1941 | **647** | 1500 | miss |
| questions | 432 | **144** | 300 | miss |
| unanswerable | 132 | **44** | 90 | miss |
| supersession edges | 69 | **23** | 25 | miss |

A translation of a passage is not a new passage. `corpus.json` discloses the ×3 for questions only,
and not for documents, chunks or supersession edges. The supersession floor is missed on any
reading. **The corpus must be regenerated,** which invalidates the committed freeze and every number
measured against it.

### Why no re-draw, and why the system was not touched

A fresh hold-out would change neither result — both failures are in the criteria, not in the split —
so re-drawing one could only look like shopping for a friendlier number. No retrieval threshold,
rule, effectivity condition, gate condition, chunking parameter, embedding or citation rule was
changed after the hold-out was scored.

### One harness defect was corrected after scoring

Wrong-answer scoring compared **bare revision labels**. There are five of them — `A`, `B`, `C`, `D`,
`FB1` — across 108 documents in 9 families, because a revision letter identifies a document's place
in its own family's history and nothing more. Citing another family's `C` compared equal to this
family's in-force `C` and scored as correct. The comparison is now keyed `family/revision`.

This is a correction to the *measuring instrument*, not to the system under test, and it can only
move a measured rate upward. A change that cannot flatter a result is safe to apply to a hold-out
that has already been scored; the reverse would not be. It is recorded here rather than folded in
silently.

---

## ADR-003 — The second benchmark iteration, and what the fresh hold-out exposed

**Status:** accepted · **Date:** 2026-09-24 · **Supersedes nothing. ADR-001 and ADR-002 stand
unedited above.**

> **Written between the first score of the second hold-out and the final one.** It names E, F and H
> as the failures; the final scoring found **E, F, I and K**, and H passes. ADR-004 has the measured
> result and explains what moved. This ADR is left exactly as it was written rather than edited to
> agree with what came later — the reasoning in it is what produced the corrections, and a decision
> record rewritten after the fact is not a record.

ADR-002 ended by recording that the first benchmark was not a valid retrieval test. This is the
record of the second one: what was rebuilt, what the fresh hold-out found, and which of it was
fixed against which of it was published.

### What was rebuilt

| | first iteration | second |
|---|---|---|
| eligible candidates after filtering (hold-out median) | **9**, with 84.3% at or below `top_k` | **21**, with 0.0% at or below `top_k` |
| corpus floors | cleared only by counting EN/TR/RU as three | cleared on **distinct content** |
| supersession edges (distinct) | 23, against a floor of 25 | 56 |
| `hybrid_without_effectivity` | `as_of=2099-12-31`, which is *more* filtered | the predicate removed |
| knowledge time | did not exist | a second axis, with corrections |
| split-leak check | `split_of(F) != split_of(F)` | resolves question → chunk → document → family |
| `pgvector.json`, `index_lifecycle.json`, `storage_comparison.json` | no producer | written by the ordinary build |

The retrieval task is no longer saturated, and the arms separate. That was the point.

### The ordering, and it is checkable

```
61b3d3c, eab2955   corpus generated
3ae15bf            hold-out materialised and frozen — zero scores in that commit
eecdbf5            the first score over it
44589bc            re-freeze: the enumeration corrected (below), zero scores in that commit
<this build>       the score this repository publishes
```

### Corrections made after a score existed, and why each was safe

Three. Each was found by reading code against its own stated contract, none by looking at a number
and searching for a way to move it, and each can only move a measured rate **adversely or not at
all** — which is the property that makes it safe to apply to a hold-out that has already been
scored. The reverse would not be.

1. **The conflict signal compared units, not quantities.** `_conflicting_evidence` grouped
   measurements by unit while its docstring said "different values for the same quantity", and the
   two coincided only while the corpus had exactly one topic per unit. The second corpus has three
   torque specifications, so one manual page states three different torques for three different
   things; the gate called that a contradiction and sent **59.93%** of the hold-out to review.
   Conflicts are now grouped by the passage's own heading as well as its unit.
2. **The hold-out enumeration dropped a whole failure mode.** `compute` derived the held families
   from documents alone, so the `different_product_family` negatives — whose families own no
   documents by definition — were absent from the frozen membership. Fifteen questions, all
   unanswerable. Re-frozen at `44589bc`.
3. **`freeze` ran its leak check without the chunk map**, so the reference check resolved nothing
   and reported clean because it had looked at nothing. The same shape as the defect ADR-002
   recorded on the corpus side.

### What was *not* fixed, and is published instead

Nothing below was touched after the hold-out was scored. ADR-001's thresholds are unchanged.

**E fails: the gate answered ten questions the corpus cannot support.** Every one of the ten is
Turkish or Russian; none is English. `gate.term_is_covered` approximates stemming with a
bidirectional prefix match for terms of four characters or more, because an exact-match rule would
score coverage near zero on two of the three languages and call that caution. Its own docstring says
it errs towards covering. The first corpus was too easy to price that; this one prices it. Adjusting
the gate now would be tuning against a scored hold-out, so the number is published and the
mechanism named.

**F fails, for two reasons and neither is the retriever.** `ungated_rag` removes only the gate, so
it runs the identical retriever and cannot be beaten on a retrieval metric — an impossible target
rather than a hard one, and a defect in a criterion written before the arms existed. More
substantively, **recall@10 is the wrong measure of what effectivity filtering buys**: removing the
predicate enlarges the candidate pool, so recall barely moves while MRR collapses. F does not ask
about MRR.

**H fails on its `ungated > 0` clause**, for the structural reason ADR-002 already set out.

**G passes, and passes vacuously.** This is the finding an independent review surfaced and it
deserves to be stated plainly rather than banked. A wrong answer is defined as revision-incorrect
or variant-incorrect. The effectivity predicate runs *before* the gate and constrains family,
variant and validity in SQL; every hold-out question names a variant; every variant belongs to
exactly one family. So no answer the gated system can give is capable of being wrong in either
sense, and G's numerator is empty by construction. **G cannot distinguish this system from a broken
one**, and a criterion that cannot fail is not evidence. It is reported as PASS because that is what
it measured, with this paragraph attached.

F, G and H share one root cause: all three were written as if the effectivity filter were beside the
thing being measured, and it is upstream of everything.

**Condition I counts `REVIEW` as withholding.** That is the reading `gate.withholds_answer` states —
nothing reaches the technician while a question is in review — but it is the lenient reading of the
word "abstention", and the floor of 0.90 leaves only ten points of headroom. `gate.json` now
publishes the strict split beside it so a reader can see how much of the margin is refusal and how
much is a review queue.

### The coverage curve was measuring the wrong thing

The published curve re-decides every question at every threshold. A refused question carried no
citations, `metrics.revision_incorrect` returns `True` on an empty cited set by design, and the
curve's wrong-answer axis therefore rose as the threshold fell because *more refusals entered the
numerator* — not because any lower-threshold answer would have been incorrect. `AnswerOutcome` had
documented from the start that these fields must be filled for every question including the refused
ones. The runner did not fill them; it does now, from the evidence the gate looked at and declined.

---

## ADR-004 — The final result, and the close

**Status:** accepted · **Date:** 2026-09-24 · **ADR-001, ADR-002 and ADR-003 stand unedited above.**

ADR-003 was written between the first score of the second hold-out and the final one, and named E, F
and H as the failures. The final scoring changed that, and this ADR records what was actually
measured rather than editing ADR-003 to look prescient.

**ORIGINAL RELEASE GATE: FAILED — 4 of 12.**

| | condition | verdict |
|---|---|---|
| A | an as-of query returns a superseded chunk | PASS |
| B | an as-of query returns another variant's chunk | PASS |
| C | a part number in an answer that its evidence lacks | PASS |
| D | a cited span absent from the document it names | PASS |
| **E** | the gate answers an unsupportable question | **FAIL** |
| **F** | recall@10 ≥ 0.85 and above every baseline | **FAIL** |
| G | hold-out wrong-answer rate ≤ 0.02 | PASS, near-vacuous |
| H | ungated wrong-answer rate ≥ 5× gated | PASS |
| **I** | abstention on the unanswerable set ≥ 0.90 | **FAIL** |
| J | two runs agree byte for byte | PASS |
| **K** | vector retrieval executes through pgvector | **FAIL** |
| L | a document appears in both splits | PASS |

### What changed between the first score and the final one, and why

Three corrections, all argued in ADR-003, none of them a response to a number:

1. **The conflict signal compared units rather than quantities.** Fixing it moved coverage from
   0.1481 to **0.6859** and the review rate from 0.5993 to **0.0288**. The first figure was a gate
   drowning in contradictions it had invented between three different torque specifications on one
   page.
2. **The hold-out enumeration dropped every question about a family the corpus does not contain.**
   Restoring them added 15 questions, all unanswerable. This is what turned G and H from vacuous
   into measurable — and it is what made I fail.
3. **The coverage curve scored refusals as wrong answers**, because an abstention cites nothing and
   an empty cited set reads as revision-incorrect.

**H now passes at 10.3×** (ungated 0.0481 against gated 0.0047) and the pass is real: it is measured
on the one class of question where the effectivity filter cannot help, which is exactly the class
correction 2 restored.

### The four failures, exactly

**E — 47 of 306 unanswerable questions were answered.** The weakness is not spread evenly:

| unanswerable kind | refused |
|---|---|
| superseded, replacement asked for | 45/45 |
| a product family that does not exist | 44/45 |
| an identifier that nearly matches a real one | 44/45 |
| two in-force sources contradict each other | 35/36 |
| a specification absent from the manual | 37/45 |
| a malformed part number | 37/45 |
| **an attribute absent for a product that exists** | **17/45** |

The last row is the finding. When the machine is real and the attribute simply is not documented,
the gate answers 62% of the time. `gate.term_is_covered` approximates stemming with a bidirectional
prefix match — its own docstring says it errs towards covering — and a question about a real machine
shares enough terms with that machine's other specifications to clear the coverage floor.

**F — the system does not beat every baseline.** `dense_only` reaches **0.9352** against the
system's **0.9259**: it wins by **0.0093**. `bm25_only` ties exactly at 0.9259. `ungated_rag` ties by
construction, because it removes only the gate and runs the identical retriever, which makes that
comparison impossible rather than hard.

And recall@10 is the wrong question. Removing the effectivity predicate leaves recall almost
unchanged — `hybrid_without_effectivity` scores 0.9167 on the hold-out and 0.9432 on development,
*above* the system — while **MRR collapses from 0.9097 to 0.3695**. What the filter buys is that the
right passage is at rank 1 rather than rank 8. F does not ask about MRR.

**I — abstention on the unanswerable set is 0.8464 against a floor of 0.90.** Same mechanism as E.
Counting only `ABSTAIN` and not `REVIEW` it is 0.7255; both figures are published, because `REVIEW`
withholds but "abstention" is the lenient reading of it.

**K — the query plan does not mention a vector index.** pgvector 0.8.6 is installed, the column is a
real `vector`, and the executed statement uses `<=>` — but the effectivity filter has already cut
the candidate set to **21 rows**, for which PostgreSQL correctly prefers a sequential scan. Forcing
the ANN plan measures **5.6 ms against the planner's 3.1 ms**. The criterion demanded an index scan
on a query that should not have one, and the planner is right.

### G passes, and the pass is near-vacuous

For the 297 hold-out questions that name a variant — 297 of 312 — G's numerator is empty by
construction: the effectivity predicate excludes revision- and variant-incorrect passages upstream
of the gate, so no answer the system can give for those is *capable* of being wrong in either sense.
The measured 0.0047 is a single wrong answer, and it comes from the 15 questions about a family the
corpus does not contain, where the filter has nothing to constrain. **G measures the gate only on
the class of question the filter cannot help with.**

### One root cause under five of the six

E and I are the gate's Turkish and Russian over-covering. F, G, H and K are all the same thing:
**the criteria were written as if the effectivity filter sat beside the thing being measured, and it
sits upstream of everything.** It makes F's strongest baseline unbeatable, it empties G's numerator,
it emptied H's until a corpus correction restored the one class it cannot reach, and it shrinks the
candidate set until an index scan is the wrong plan and K's assertion cannot hold.

That is the result this project has to report, and it is more interesting than a system that cleared
its own bar. Pre-registration is what made it visible instead of convenient.

### Closed

No third benchmark. No tuning against this hold-out. The twelve conditions are unchanged, none was
lowered, none was deleted, and none was marked `xfail`. The failing set is pinned in
`scripts/release_gate.py`, which fails the build if reality diverges from it in either direction —
so a regression cannot hide behind a disclosed failure, and a disclosed failure cannot quietly
become a pass.

**PROJECT 7 — CLOSED AS A PRE-REGISTERED NEGATIVE RESULT.**
