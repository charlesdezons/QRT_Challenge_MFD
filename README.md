# Challenge context

(In VS Code, type (Ctrl+Shift+V or Cmd+Shift+V on Mac) to preview the Readme in Latex format)

A classic prediction problem from finance is to predict the next *returns* (i.e. relative price variations) from a *stock market*. That is, given a stock market of *N* stocks having returns $ R_t \in \mathbb{R}^N $ at time $ t $, the goal is to design at each time $ t $ a vector $ S_{t+1} \in \mathbb{R}^N $ from the information available up to time $ t $ such that the prediction overlap $ \langle S_{t+1}, R_{t+1} \rangle $ is quite often positive. To be fair, this is not an easy task. In this challenge, we attack this problem armed with a **linear factor model where one learns the factors over an exotic non-linear parameter space**.

NB: There is a **dedicated forum** for this challenge.

More precisely, the simplest estimators being the linear ones, a typical move is to consider a parametric model of the form:

$ S_{t+1} := \sum_{\ell=1}^{F} \beta_\ell F_{t,\ell} $

where the vectors $ F_{t,\ell} \in \mathbb{R}^N $ are *explicative factors* (a.k.a. *features*), usually designed from financial expertise, and $ \beta_1, \dots, \beta_F \in \mathbb{R} $ are model parameters that can be fitted on a training data set.

### But how to design the factors $ F_{t,\ell} $?

Factors that are "well known" in the trading world include the 5-day (normalized) mean returns $ R_t^{(5)} $ or the **Momentum** $ M_t := R_t^{(230)} - 20 $, where $ R_t^{(m)} := \frac{1}{\sqrt{m}} \sum_{k=1}^{m} R_{t+1-k} $. But if you know no finance and have developed enough taste for mathematical elegance, you may aim at learning the factors themselves within the simplest class of factors, namely linear functions of the past returns:

$ F_{t,\ell} := \sum_{k=1}^{D} A_{k\ell} R_{t+1-k} $

for some vectors $ A_\ell := (A_{k\ell}) \in \mathbb{R}^D $ and a fixed **time depth** parameter $ D $.

One way to ensure independence between the factors is to assume the vectors $ A_\ell $ are **orthonormal**, i.e., $ \langle A_k, A_\ell \rangle = \delta_{k\ell} $ for all $ k, \ell $, which adds a non-linear constraint to the parameter space of our predictive model.

### Model Parameters

Thus, we define our predictive parametric model with:

- A $ D \times F $ matrix $ A := [A_1, \dots, A_F] $ with orthonormal columns.
- A vector $ \beta := (\beta_1, \dots, \beta_F) \in \mathbb{R}^F $.

This setup includes the two-factor model using $ R_t^{(5)} $ and $ M_t $ defined above, or the **autoregressive model (AR)** from time series analysis, as submodels.

## Challenge goals

The goal of this challenge is to design/learn factors for stock return prediction using the exotic parameter space introduced above.

Participants will use three-year data history of 50 stocks from the same stock market (**training data set**) to provide the model parameters $ (A, \beta) $ as outputs. Then, the predictive model associated with these parameters will be tested to predict the returns of **50 other stocks** over the **same** three-year time period (**testing data set**).

We allow $ D = 250 $ days for the time depth and $ F = 10 $ for the number of factors.

### Metric

The quality of the predictive model with parameters $ (A, \beta) $ is assessed using:

$ \text{Metric}(A, \beta) := \frac{1}{504} \sum_{t=250}^{753} \frac{\langle \tilde{S}_t, \tilde{R}_t \rangle}{\| \tilde{S}_t \| \| \tilde{R}_t \|} $

where $ \tilde{R}_t \in \mathbb{R}^{50} $ represents the returns of 50 testing stocks and $ \tilde{S}_t $ is the participants' predictor for $ \tilde{R}_t $.

If $ |\langle A_i, A_j \rangle - \delta_{ij}| \leq 10^{-6} $ for all $ i, j $, the metric takes values in $[-1,1]$; otherwise, $ \text{Metric}(A, \beta) := -1 $.

### Output Structure

The expected output is a vector containing the model parameters:

$ \text{Output} = \begin{bmatrix} A_1 \\ \vdots \\ A_{10} \\ \beta \end{bmatrix} \in \mathbb{R}^{2510} $

## Data description

The training input $ X_{train} $ is a dataframe containing the (cleaned) daily returns of 50 stocks over a period of 754 days (three years). Each row represents a stock, and each column refers to a day. $ X_{train} $ should be used to find the predictive model parameters $ A, \beta $.

The returns to be predicted in the training data set are provided in $ Y_{train} $ for convenience, but they are also contained in $ X_{train} $.

## Benchmark description

A possible "brute force" approach to tackle this problem is:

1. Generate orthonormal vectors $ A_1, \dots, A_{10} \in \mathbb{R}^{250} $ at random.
2. Fit $ \beta $ on the training data set using linear regression.
3. Repeat this operation many times.
4. Select the best result from these attempts.

More precisely, the QRT benchmark strategy to beat is (see the notebook in the supplementary material):

Repeat $ N_{iter} = 1000 $ times the following.

- Sample a $ 250 \times 10 $ matrix $ M $ with iid Gaussian $ N(0,1) $ entries.
- Apply the Gram-Schmidt algorithm to the columns of $ M $ to obtain a matrix  
  $ A = [A_1, \ldots, A_{10}] $ with orthonormal columns (see the randomA function).
- Use the columns of $ A $ to build the factors and then take $ \beta $ with minimal mean square error on the training data set (with fitBeta).
- Compute the metric on the training data (metricTrain).

Return the model parameters $ (A, \beta) $ that maximize this metric.

**Remark:** The orthonormality condition for the vectors $ A_1, \ldots, A_F $ reads  
$ A^T A = I_F $  
for the matrix $ A := [A_1, \ldots, A_F] $. The space of matrices satisfying this condition is known as the *Stiefel manifold*, a generalization of the orthogonal group, and one can show that the previous procedure generates a sample from the uniform distribution on this (compact symmetric) space.
