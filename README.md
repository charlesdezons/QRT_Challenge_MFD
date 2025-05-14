# Challenge Context

A classic prediction problem in finance is to predict future **returns** (i.e. relative price variations) of a **stock market**. Given a market of *N* stocks with returns **Rₜ ∈ ℝᴺ** at time *t*, the goal is to design at each time *t* a vector **Sₜ₊₁ ∈ ℝᴺ**, using information available up to time *t*, such that the **prediction overlap** ⟨Sₜ₊₁, Rₜ₊₁⟩ is often positive.

This is not an easy task. In this challenge, we tackle it using a **linear factor model**, where the **factors are learned over an exotic non-linear parameter space**.

> There is a **dedicated forum** for this challenge.

## Linear Factor Model

A common linear prediction model takes the form:

**Sₜ₊₁ = Σ (βₗ · Fₜ,ₗ), for ℓ = 1 to F**

- Here, **Fₜ,ₗ ∈ ℝᴺ** are *explicative factors* (a.k.a. features)
- **β₁, ..., β_F ∈ ℝ** are parameters learned from training data

### Designing the Factors Fₜ,ₗ

Typical financial features include:

- **5-day normalized mean returns:** Rₜ⁽⁵⁾
- **Momentum:** Mₜ = Rₜ⁽²³⁰⁾ − 20

Where the m-day average return is defined as:

**Rₜ⁽ᵐ⁾ = (1 / √m) · Σ Rₜ₊₁₋ₖ, for k = 1 to m**

Instead of using predefined features, you can **learn the factors** as linear combinations of past returns:

**Fₜ,ₗ = Σ (Aₖₗ · Rₜ₊₁₋ₖ), for k = 1 to D**

- Aₗ = (A₁ₗ, ..., A_Dₗ) ∈ ℝᴰ
- D is the *time depth* (e.g. 250 days)

To ensure factor independence, the vectors Aₗ should be **orthonormal**:

**⟨A_k, A_ℓ⟩ = δₖₗ** (i.e. dot product is 1 if k = ℓ, 0 otherwise)

---

## Model Parameters

The model is defined by:

- A matrix **A ∈ ℝᴰˣᶠ** with orthonormal columns: A = [A₁, ..., A_F]
- A vector **β ∈ ℝᶠ**: β = (β₁, ..., β_F)

This structure includes:
- The two-factor model with Rₜ⁽⁵⁾ and Mₜ
- The autoregressive (AR) model from time series

---

## Challenge Goals

The goal is to learn model parameters (A, β) from the **training dataset** (3 years of daily returns for 50 stocks), and test them on **a different set of 50 stocks** over the same period.

- Time depth: **D = 250**
- Number of factors: **F = 10**

### Evaluation Metric

The metric to maximize is:

**Metric(A, β) = (1 / 504) · Σ [⟨Sₜ, Rₜ⟩ / (‖Sₜ‖ · ‖Rₜ‖)], for t = 250 to 753**

Where:
- Rₜ ∈ ℝ⁵⁰ is the return vector of test stocks
- Sₜ is your prediction

If A does **not satisfy orthonormality** to within tolerance (|⟨Aᵢ, Aⱼ⟩ − δᵢⱼ| ≤ 1e-6), then:

**Metric(A, β) = −1**

---

## Output Format

Expected output is a single vector:

**Output ∈ ℝ²⁵¹⁰ = [A₁; ...; A₁₀; β]**

That is, 10 vectors Aₗ ∈ ℝ²⁵⁰ followed by β ∈ ℝ¹⁰.

---

## Data Description

- **X_train:** DataFrame with daily returns for 50 stocks over 754 days  
  (rows = stocks, columns = days)

- **Y_train:** Target return vectors (redundant, also contained in X_train)

Use this data to learn the model parameters A and β.

---

## Benchmark Strategy

A brute-force baseline works as follows:

1. Generate random A₁, ..., A₁₀ ∈ ℝ²⁵⁰ with orthonormal columns
2. Fit β using linear regression
3. Repeat many times
4. Keep the (A, β) with best metric

The **QRT benchmark** does:

- Repeat N_iter = 1000:
  - Sample matrix M ∈ ℝ²⁵⁰ˣ¹⁰ with N(0,1) entries
  - Apply Gram-Schmidt to get orthonormal matrix A
  - Fit β to minimize mean square error on training set
  - Evaluate metric on training set
- Return A and β that maximize this metric

> **Note:** The orthonormality condition Aᵀ A = I defines the **Stiefel manifold** — a generalization of the orthogonal group. The benchmark samples uniformly from this space.

---
