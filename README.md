# Manifold-Parameterized Weight Compression for Vision-Language Models

This repository explores a post-hoc compression framework for large Vision-Language Models (VLMs), focusing on models such as **LLaVA-1.5**. The goal is to significantly reduce parameter count while preserving downstream task performance, without training models from scratch.

The core idea is to represent pretrained model weights using compact latent codes decoded by a shared lightweight network, effectively constraining weights to lie on a learned low-dimensional manifold.

---

## Motivation

Modern VLMs combine large language models (LLMs) with vision encoders (e.g., ViTs), resulting in hundreds of millions to billions of parameters. Retraining such models from scratch is often infeasible due to data and compute constraints.  

This work targets **post-hoc compression**: starting from a pretrained VLM, we replace its original weight tensors with a compact parameterization that approximates their function while enabling high compression ratios.

---

## Notation and Setup

Let the pretrained model consist of $N$ parameterized layers:

$$
\{ W^{(1)}, W^{(2)}, \dots, W^{(N)} \}.
$$

Each layer weight $W^{(l)} \in \mathbb{R}^{d_l^{\text{out}} \times d_l^{\text{in}}}$ is assumed to be fixed and pretrained.

### Weight Chunking

To enable practical compression, each weight tensor is partitioned into smaller chunks:

$$
W^{(l)} = \{ W^{(l)}_1, W^{(l)}_2, \dots, W^{(l)}_{K_l} \}.
$$

Each chunk $W^{(l)}_k$ is a small block (e.g., block-wise or row-wise). Chunking avoids the need to generate large tensors directly and allows fine-grained allocation of model capacity.

---

## Core Parameterization

We introduce a shared **weight generator** (decoder) network:

$$
\psi(\alpha; s): \mathbb{R}^{d_\alpha} \rightarrow \mathbb{R}^{d_{\text{chunk}}},
$$

where:
- $\alpha \in \mathbb{R}^{d_\alpha}$ is a low-dimensional latent code,
- $s$ are the learnable parameters of the generator,
- the output dimension matches the flattened chunk size.

Each weight chunk is approximated as:

$$
\hat{W}^{(l)}_k = \psi(\alpha^{(l)}_k; s),
$$

and the reconstructed layer $\hat{W}^{(l)}$ is formed by assembling its reconstructed chunks.

A single generator $\psi$ is typically shared across many layers and chunks, ensuring that compression gains come primarily from the low-dimensional latent codes rather than from duplicating decoder parameters.

---

## Weight-Space Reconstruction

A basic reconstruction objective directly matches pretrained weights:

$$
\mathcal{L}_{\text{weight}} =
\sum_{l=1}^{N} \sum_{k=1}^{K_l}
\left\|
W^{(l)}_k - \psi(\alpha^{(l)}_k; s)
\right\|_2^2.
$$

This data-free objective provides a simple baseline for evaluating the expressiveness of the generator and the effectiveness of the manifold parameterization.

---

## Activation-Aware Reconstruction

Direct weight matching may not preserve model behavior. Instead, we can match the *functional outputs* of layers using a small calibration dataset.

Let $X^{(l)}$ denote the inputs to layer $l$, collected by running the original pretrained model on calibration data. The reconstruction objective becomes:

$$
\mathcal{L}_{\text{act}} =
\sum_{l=1}^{N}
\mathbb{E}_{X^{(l)}} \left[
\left\|
W^{(l)} X^{(l)} -
\hat{W}^{(l)} X^{(l)}
\right\|_2^2
\right].
$$

This objective aligns reconstructed weights with the original model’s behavior and is often more robust to small parameter perturbations.

---

## Light Post-Training

After reconstruction, a small amount of task-specific fine-tuning can further recover performance. In this stage:
- latent codes $\alpha^{(l)}_k$ are optimized,
- generator parameters $s$ may be partially or fully updated,
- training is performed on a limited downstream dataset.

The objective is the standard VLM task loss:

$$
\mathcal{L}_{\text{task}} =
\mathcal{L}_{\text{VLM}}(f_{\hat{W}}(x), y).
$$

This stage preserves the post-hoc nature of the method while improving end-task accuracy.

---

## Generator Architectures

### MLP Generator

A standard choice is a multi-layer perceptron:

$$
\psi(\alpha) = W_L \sigma(\dots \sigma(W_1 \alpha)).
$$

The depth, width, and activation function can be varied to study the expressiveness–compression trade-off.

### Random Fourier Feature Generator

A structured alternative uses random Fourier features:

$$
\phi(\alpha) = [\sin(B\alpha), \cos(B\alpha)],
$$

$$
\psi(\alpha) = W \phi(\alpha),
$$

where $B$ is a fixed random Gaussian matrix. This design reduces learnable parameters while providing a strong inductive bias.

---

## Modality-Specific Compression

Vision and language components in VLMs have different structural properties. We therefore allow separate generators:

$$
\psi_{\text{LLM}}(\cdot; s_{\text{LLM}}), \quad
\psi_{\text{ViT}}(\cdot; s_{\text{ViT}}),
$$

with potentially different chunking schemes, latent dimensions, architectures, and reconstruction objectives.

---

## Importance-Aware Allocation

Different layers and chunks contribute unequally to model performance. Let $\omega^{(l)}_k$ denote an importance weight for chunk $k$ in layer $l$, estimated using sensitivity or gradient-based measures.

A weighted reconstruction objective is then used:

$$
\mathcal{L} =
\sum_{l,k}
\omega^{(l)}_k
\left\|
W^{(l)}_k - \psi(\alpha^{(l)}_k; s)
\right\|_2^2.
$$

More important chunks can also be assigned higher-dimensional latent codes.

---

## Compressing the Generator

At high compression rates, the generator network itself may become a significant parameter contributor. In such cases, a second-stage compression can be applied to the generator parameters using techniques such as low-rank factorization or parameter sharing, resulting in a hierarchical compression pipeline.

---

## Scope

This repository focuses on:
- post-hoc compression of pretrained VLMs,
- preserving functional behavior rather than retraining from scratch,
- exploring trade-offs between compression rate, reconstruction fidelity, and downstream accuracy.

The primary target model is **LLaVA-1.5**, with extensions to other VLM architectures.

---
