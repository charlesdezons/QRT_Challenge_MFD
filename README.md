# 📊 Challenge Context

A classic prediction problem from finance is to predict the next *returns* (i.e. relative price variations) from a *stock market*. That is, given a stock market of \(N\) stocks having returns \(R_t \in \mathbb{R}^N\) at time \(t\), the goal is to design at each time \(t\) a vector \(S_{t+1} \in \mathbb{R}^N\) from the information available up to time \(t\) such that the prediction overlap \(\langle S_{t+1}, R_{t+1} \rangle\) is quite often positive. To be fair, this is not an easy task. In this challenge, we attack this problem armed with a **linear factor model where one learns the factors over an exotic non-linear parameter space**.

> NB: There is a [dedicated forum](#) for this challenge.

More precisely, the simplest estimators being the linear ones, a typical move is to consider a parametric model of the form

$
S_{t+1} := \sum_{\ell=1}^{F} \beta_\ell F_{t,\ell}
$

where the vectors \(F_{t,\ell} \in \mathbb{R}^N\) are **explicative factors** (a.k.a. features), usually designed from financial expertise, and \(\beta_1, \ldots, \beta_F \in \mathbb{R}\) are model parameters that can be fitted on a training data set.

---

## ❓ But how to design the factors \(F_{t,\ell}\)?

Factors that are “well known” in the trading world include the 5-day (normalized) mean returns \(R_t^{(5)}\) or the **Momentum** \(M_t := R_{t-20}^{(230)}\), where \(R_t^{(m)} := \frac{1}{\sqrt{m}} \sum_{k=1}^m R_{t+1-k}\). But if you know no finance and have developed enough taste for mathematical elegance, you may aim at learning the factors themselves within the simplest class of factors, namely linear functions of the past returns:

\[
F_{t,\ell} := \sum_{k=1}^{D} A_{k\ell} R_{t+1-k}
\]
where the vectors \(F_{t,\ell} \in \mathbb{R}^N\) are *explicative factors* (a.k.a. *features*), usually designed from financial expertise, and \(\beta_1, \ldots, \beta_F \in \mathbb{R}\) are model parameters that can be fitted on a training data set.

---

## **But how to design the factors \(F_{t,\ell}\)?**

Factors that are “well known” in the trading world include the 5-day (normalized) mean returns \(R_t^{(5)}\) or the **Momentum** \(M_t := R_{t-20}^{(230)}\), where 

\[
R_t^{(m)} := \frac{1}{\sqrt{m}} \sum_{k=1}^m R_{t+1-k}.
\]

But if you know no finance and have developed enough taste for mathematical elegance, you may aim at learning the factors themselves within the simplest class of factors, namely linear functions of the past returns:

\[
F_{t,\ell} := \sum_{k=1}^D A_{k\ell} R_{t+1-k}
\]

for some vectors \(A_\ell := (A_{k\ell}) \in \mathbb{R}^D\) and a fixed *time depth* parameter \(D\).

Well, we need to add a condition to create enough independence between the factors, since otherwise they may be redundant. One way to do this is to **assume the vectors \(A_\ell\)'s are orthonormal**, 

\[
\langle A_k, A_\ell \rangle = \delta_{k\ell} \text{ for all } k, \ell,
\]

which adds a non-linear constraint to the parameter space of our predictive model.

---

All in all, we thus have at hand a predictive parametric model with parameters:

- a \(D \times F\) matrix \(A := [A_1, \ldots, A_F]\) with orthonormal columns,  
- a vector \(\beta := (\beta_1, \ldots, \beta_F) \in \mathbb{R}^F\).

Note that it contains the two-factor model using \(R_t^{(5)}\) and \(M_t\) defined above, or the **autoregressive model AR** from time series analysis, as submodels.

# Challenge Goals

The goal of this challenge is to design/learn factors for stock return prediction using the exotic parameter space introduced in the context section.

Participants will be able to use three-year data history of 50 stock from the same stock market (**training data set**) to provide the model parameters \((A, \beta)\) as outputs. Then the predictive model associated with these parameters will be tested to predict the returns of 50 **other** stocks over the **same** three-year time period (**testing data set**).

> **We allow \(D = 250\)** days for the time depth and  
> **\(F = 10\)** for the number of factors.

---

## Metric

More precisely, we assess the quality of the predictive model with parameters \((A, \beta)\) as follows. Let \(\widetilde{R}_t \in \mathbb{R}^{50}\) be the returns of the 50 stocks of the testing data set over the three-year period (\(t = 0 \ldots 753\)) and let \(\widetilde{S}_t := \widetilde{S}_t(A, \beta)\) be the participants’ predictor for \(\widetilde{R}_t\). The metric to maximize is defined by

\[
\text{Metric}(A, \beta) := \frac{1}{504} \sum_{t=250}^{753} \frac{\langle \widetilde{S}_t, \widetilde{R}_t \rangle}{\| \widetilde{S}_t \| \| \widetilde{R}_t \|}
\]

if \(| \langle A_i, A_j \rangle - \delta_{ij} | \leq 10^{-6}\) for all \(i, j\) and  
\[
\text{Metric}(A, \beta) := -1
\]
otherwise.

By construction the metric takes its values in \([-1, 1]\) and equals to \(-1\) as soon as there exists a couple \((i, j)\) breaking too much the orthonormality condition.

---

## Output Structure

The output expected from the participants is a vector where the model parameters  
\(A = [A_1, \ldots, A_{10}] \in \mathbb{R}^{250 \times 10}\) and  
\(\beta \in \mathbb{R}^{10}\) are stacked as follows:

