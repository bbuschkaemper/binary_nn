# 1-Bit Neural Networks Paper Inventory

Last updated: 2026-03-16

This is a compact reading list for the current 1-bit neural network landscape.
It separates papers into the ones that look foundational for this repository and
the ones that broaden the picture around training, quantization, inference, and
hardware.

## Document Role

Use this file as the fastest way to recover the relevant papers and why they
matter. It is the quickest literature-entry document in the folder.

## 1. Core 1-Bit LLM Papers

| Year | Paper | Link | Why it matters |
| --- | --- | --- | --- |
| 2023 | BitNet: Scaling 1-bit Transformers for Large Language Models | <https://arxiv.org/abs/2310.11453> | Original BitNet paper. Introduces `BitLinear` and the core scaling claim. |
| 2024 | The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits | <https://arxiv.org/abs/2402.17764> | Establishes the practical ternary `1.58-bit` regime as the main public frontier. |
| 2024 | BitNet a4.8: 4-bit Activations for 1-bit LLMs | <https://arxiv.org/abs/2411.04965> | Extends the BitNet line into lower-bit activations and sparse activation handling. |
| 2024 | 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs | <https://arxiv.org/abs/2410.16144> | Shows that BitNet-style models can get real CPU speedups with specialized kernels. |
| 2025 | Bitnet.cpp: Efficient Edge Inference for Ternary LLMs | <https://arxiv.org/abs/2502.11880> | Formalizes the `bitnet.cpp` inference stack for edge deployment. |
| 2025 | BitNet b1.58 2B4T Technical Report | <https://arxiv.org/abs/2504.12285> | Current flagship open native 1-bit LLM result. |

## 2. Strong Comparator Papers

These are not BitNet papers, but they are important because they either compete
with BitNet's claims or reveal failure modes in low-bit assumptions.

| Year | Paper | Link | Why it matters |
| --- | --- | --- | --- |
| 2024 | BiLLM: Pushing the Limit of Post-Training Quantization for LLMs | <https://arxiv.org/abs/2402.04291> | Strong post-training 1-bit-style quantization baseline for existing LLMs. |
| 2024 | Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens | <https://arxiv.org/abs/2411.17691> | Warns that quantization can degrade more as models become more fully trained. |

## 3. Training-Centric BNN Papers

These are especially relevant for this repo because they attack the optimization
problem directly instead of treating binary models as a pure inference format.

| Year | Paper | Link | Why it matters |
| --- | --- | --- | --- |
| 2025/2026 | BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training | <https://arxiv.org/abs/2512.04189> | Binary-valued backward propagation with bitwise forward and backward passes. |
| 2026 | Layerwise Progressive Freezing Enables STE-Free Training of Deep Binary Neural Networks | <https://arxiv.org/abs/2601.22660> | Replaces STE with progressive stochastic binarization. |
| 2026 | Quadratic Unconstrained Binary Optimisation for Training and Regularisation of Binary Neural Networks | <https://arxiv.org/abs/2601.00449> | Treats BNN training explicitly as discrete optimization via QUBO. |

## 4. Hardware and Verification Papers

These broaden the landscape beyond training quality. They matter because the
real value of binary representations depends on the full stack.

| Year | Paper | Link | Why it matters |
| --- | --- | --- | --- |
| 2026 | Scalable Digital Compute-in-Memory Ising Machines for Robustness Verification of Binary Neural Networks | <https://arxiv.org/abs/2603.05677> | Hardware-accelerated robustness verification for BNNs. |
| 2026 | Robustness Verification of Binary Neural Networks: An Ising and Quantum-Inspired Framework | <https://arxiv.org/abs/2602.13536> | Another sign that trustworthy BNN behavior is becoming an active topic. |
| 2026 | PiC-BNN: A 128-kbit 65 nm Processing-in-CAM-Based End-to-End Binary Neural Network Accelerator | <https://arxiv.org/abs/2601.19920> | End-to-end accelerator for fully binary inference, relevant to hardware co-design. |

## 5. Practical Reading Order

If the goal is to decide where this repository should invest effort, the most
useful reading order is:

1. `BitNet: Scaling 1-bit Transformers for Large Language Models`
2. `The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits`
3. `BitNet b1.58 2B4T Technical Report`
4. `1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs`
5. `Bitnet.cpp: Efficient Edge Inference for Ternary LLMs`
6. `BiLLM: Pushing the Limit of Post-Training Quantization for LLMs`
7. `Low-Bit Quantization Favors Undertrained LLMs`
8. `BEP`
9. `Layerwise Progressive Freezing Enables STE-Free Training of Deep Binary Neural Networks`
10. `Quadratic Unconstrained Binary Optimisation for Training and Regularisation of Binary Neural Networks`

## 6. Key Takeaways From the Inventory

- The public frontier for 1-bit LLMs is currently the BitNet family, not a wide
  field of equally strong alternatives.
- The best practical results are ternary, not strictly binary.
- The most open research gap is training, especially training without heavy
  reliance on floating-point latent states and STE-style approximations.
- Hardware-aware representation design is now central, not optional.
- Broader BNN research is useful here because it is beginning to revisit the
  core optimization problem directly.

## 7. Repository-Relevant Questions Raised by the Literature

- Can a discrete FFN method outperform shadow-weight quantization-aware training
  on stability or sample efficiency?
- Can the representation be packed in a way that looks attractive to custom GPU
  or CPU kernels from the beginning?
- Can the update rule operate on the discrete state itself rather than on a
  hidden continuous copy?
- Is ternary the right first target for this repo, even if the longer-term goal
  is stricter binary training?

The current literature suggests the answer to the last question is probably yes.
