# Docs Index

Last updated: 2026-03-18

This directory is the repository memory for binary and ternary neural network
research. The goal is to make future sessions fast to resume:

- know what the repo currently does
- know what has already been tried and measured
- know which design decisions are still open
- know where to look for binary-baseline results versus ternary follow-up work

## 1. Recommended Reading Order

If the goal is to resume active engineering work, read in this order:

1. `CURRENT_STATUS.md`
2. `TERNARY_RESEARCH_EXPERIMENT_LOG.md`
3. `ARCHITECTURE.md`
4. `ROADMAP.md`
5. `BINARY_REGRESSION_EXPERIMENT_LOG.md`
6. `1BIT_NN_STATE_OF_THE_ART.md`
7. `1BIT_NN_PAPER_INVENTORY.md`
8. `IDEA.md`

## 2. Document Roles

### 2.1 Active working memory

- `CURRENT_STATUS.md`
  - short operational handoff
  - what is implemented now
  - what has already been validated
  - what the next session should assume

- `ROADMAP.md`
  - near-term plan
  - open design decisions
  - recommended next experiments

- `ARCHITECTURE.md`
  - code structure map for `src/`
  - main execution flows
  - where to change what in future sessions

- `BINARY_REGRESSION_EXPERIMENT_LOG.md`
  - detailed chronology of the binary baseline work
  - measurements, ablations, and conclusions from the binary regression branch

- `TERNARY_RESEARCH_EXPERIMENT_LOG.md`
  - detailed chronology of the new ternary branch
  - shadow-free versus STE findings
  - concrete artifact paths and conclusions from the 2026-03-18 ternary work

### 2.2 Research background memory

- `1BIT_NN_STATE_OF_THE_ART.md`
  - longer-form research summary
  - where the public frontier currently is
  - why BitNet-style ternary systems matter

- `1BIT_NN_PAPER_INVENTORY.md`
  - compact reading list
  - quickest way to recover the relevant papers and their purpose

- `IDEA.md`
  - original repository concept note
  - long-horizon research direction around discrete FFN training and packing

## 3. Practical Use Rules

When updating docs in future sessions:

- put current implementation state in `CURRENT_STATUS.md`
- put next-step priorities and unresolved decisions in `ROADMAP.md`
- put code-structure and workflow knowledge in `ARCHITECTURE.md`
- put binary-specific experiment history in `BINARY_REGRESSION_EXPERIMENT_LOG.md`
- put ternary-specific experiment history in `TERNARY_RESEARCH_EXPERIMENT_LOG.md`
- update the longer research notes only when the external literature or the
  high-level research framing changes

## 4. What This Docs Folder Is Optimized For

This docs folder is optimized for:

- fast session-to-session continuity
- preserving experimental findings that would otherwise be trapped in terminal
  output
- keeping research framing separate from implementation status
- making future design decisions easier to justify

It is not optimized for polished public documentation. The emphasis is internal
memory and research continuity.
