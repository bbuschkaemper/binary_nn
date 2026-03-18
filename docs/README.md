# Docs Index

Last updated: 2026-03-18

This folder is the repository memory for one main research objective:

- find a new low-bit training idea that improves the GPU quality/speed frontier
- make that idea useful for both training and inference, not just as a CPU-only trick

## Document Role

Use this file as the entry point to the docs set.

## 1. New North Star

The repo is no longer mainly asking:

- can low-bit CPU inference be faster than dense BF16?

The stronger question now is:

- can we discover a binary or ternary training method that matches or beats tuned
  dense BF16 quality
- lowers GPU training cost in a way that survives repeated measurement
- and produces a representation that can later support faster GPU inference too

That means the docs should now bias toward:

- training-rule ideas
- GPU measurement quality
- representation and kernel co-design
- benchmark honesty about both quality and runtime

## 2. Recommended Reading Order

If the goal is to resume active research work, read in this order:

1. `CURRENT_STATUS.md`
2. `ROADMAP.md`
3. `IDEA.md`
4. `ARCHITECTURE.md`
5. `TERNARY_RESEARCH_EXPERIMENT_LOG.md`
6. `BINARY_REGRESSION_EXPERIMENT_LOG.md`
7. `1BIT_NN_STATE_OF_THE_ART.md`
8. `1BIT_NN_PAPER_INVENTORY.md`

## 3. Document Roles

### 3.1 Active working memory

- `CURRENT_STATUS.md`
  - shortest operational handoff
  - what is implemented now
  - what is currently believed
  - what the next session should assume

- `ROADMAP.md`
  - near-term plan
  - open design decisions
  - recommended next experiments under the new GPU-first goal

- `IDEA.md`
  - current working research hypothesis
  - the main new low-bit training idea worth testing next

- `ARCHITECTURE.md`
  - code structure map for `src/`
  - main execution flows
  - where to change what when implementing the next idea

- `TERNARY_RESEARCH_EXPERIMENT_LOG.md`
  - detailed chronology of the ternary workstream
  - the projected, STE, shadow-free, and hybrid findings
  - which artifacts remain decision-grade

- `BINARY_REGRESSION_EXPERIMENT_LOG.md`
  - detailed chronology of the binary baseline work
  - what the binary branch still contributes to the new GPU-first direction

### 3.2 Research background memory

- `1BIT_NN_STATE_OF_THE_ART.md`
  - outside-literature framing
  - what the field currently says about native low-bit training and kernels
  - what is still missing for GPU training wins

- `1BIT_NN_PAPER_INVENTORY.md`
  - compact reading list
  - fastest way to recover which papers matter for the new goal

## 4. Practical Update Rules

When updating docs in future sessions:

- put current implementation state and best current numbers in `CURRENT_STATUS.md`
- put next-step priorities and unresolved decisions in `ROADMAP.md`
- put the current best working research hypothesis in `IDEA.md`
- put code-structure and flow knowledge in `ARCHITECTURE.md`
- keep the experiment logs chronological, but always interpret them through the
  current GPU-first goal rather than the old CPU-first framing
- update the longer research notes only when the external framing or the
  repository's top-level objective changes

## 5. What This Folder Is Optimized For

This docs folder is optimized for:

- fast session-to-session continuity
- preserving measured evidence that would otherwise be trapped in terminal output
- keeping the new GPU-first objective visible even when local results are noisy
- making future design decisions easier to justify from both local evidence and
  external literature

It is still internal working memory, not polished public documentation.
