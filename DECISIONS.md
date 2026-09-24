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
