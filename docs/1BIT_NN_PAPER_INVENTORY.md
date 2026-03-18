# 1-Bit Neural Networks Paper Inventory

Last updated: 2026-03-18

This is a compact reading list for the current 1-bit and ternary landscape, organized
around the repository's new goal: finding a low-bit training idea that improves GPU
training and eventually GPU inference.

## Document Role

Use this file as the fastest way to recover which papers matter and why.

## 1. GPU-First Anchor Papers

| Paper | Link | Why it matters for the current goal |
| --- | --- | --- |
| BitNet: Scaling 1-bit Transformers for Large Language Models | <https://arxiv.org/abs/2310.11453> | Foundational `BitLinear` paper. Best starting point for the quality-anchor side of the repo. |
| The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits | <https://arxiv.org/abs/2402.17764> | Establishes the practical ternary `1.58-bit` regime. Most directly aligned with the repo's projected ternary path. |
| BitNet b1.58 2B4T Technical Report | <https://arxiv.org/abs/2504.12285> | Strongest public open flagship result for native low-bit quality at scale. |
| BitNet a4.8: 4-bit Activations for 1-bit LLMs | <https://arxiv.org/abs/2411.04965> | Important because it makes activation precision part of the speed story, not just weight precision. |

## 2. Inference And Systems Co-Design Papers

| Paper | Link | Why it matters for the current goal |
| --- | --- | --- |
| 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs | <https://arxiv.org/abs/2410.16144> | Strong reminder that bits alone do not create speed; kernels and representation co-design do. |
| Bitnet.cpp: Efficient Edge Inference for Ternary LLMs | <https://arxiv.org/abs/2502.11880> | Useful as a concrete example of how a deployment stack is built around a stable low-bit representation. |

## 3. Training-Idea Papers

| Paper | Link | Why it matters for the current goal |
| --- | --- | --- |
| BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training | <https://arxiv.org/abs/2512.04189> | Useful because it revisits the backward path directly instead of treating training as standard float optimization plus quantization. |
| Layerwise Progressive Freezing Enables STE-Free Training of Deep Binary Neural Networks | <https://arxiv.org/abs/2601.22660> | Relevant because staged or scheduled discrete updates are a plausible direction for a GPU-first training idea. |
| Quadratic Unconstrained Binary Optimisation for Training and Regularisation of Binary Neural Networks | <https://arxiv.org/abs/2601.00449> | Helps frame low-bit training as explicit discrete optimization rather than only as surrogate-gradient engineering. |

## 4. Comparator And Caution Papers

| Paper | Link | Why it matters for the current goal |
| --- | --- | --- |
| BiLLM: Pushing the Limit of Post-Training Quantization for LLMs | <https://arxiv.org/abs/2402.04291> | Strong comparator showing how far a non-native low-bit path can go. Good for avoiding overclaiming. |
| Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens | <https://arxiv.org/abs/2411.17691> | Important caution that low-bit stories can look worse as the dense baseline gets more fully trained. |

## 5. Recommended Reading Order For This Repository

If the goal is to pick the next research move, read in this order:

1. `The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits`
2. `BitNet: Scaling 1-bit Transformers for Large Language Models`
3. `BitNet b1.58 2B4T Technical Report`
4. `BitNet a4.8`
5. `1-bit AI Infra`
6. `Bitnet.cpp`
7. `Layerwise Progressive Freezing...`
8. `BEP`
9. `QUBO for BNN training`
10. `BiLLM` and `Low-Bit Quantization Favors Undertrained LLMs`

## 6. What These Papers Suggest For The Repo

The paper set points to four practical questions.

- How can we keep BitNet-like ternary quality while reducing GPU training cost?
- Can a staged or refresh-scheduled discrete update rule work better than rebuilding
  the low-bit state every step?
- When do low-bit activations become necessary for a real systems win?
- What measurements would actually count as a believable GPU speed claim?

## 7. Bottom Line

The literature suggests that the next useful local contribution is not “another low-bit
baseline.” It is a training procedure that connects low-bit quality, training cost, and
kernel-friendly inference representation in one coherent story.
