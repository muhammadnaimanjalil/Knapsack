# Zalando Gift Portfolio Optimizer

A Gurobi-free Streamlit application for solving Zalando's Gift Problem with a
Gaussian chance constraint and the open-source HiGHS mixed-integer optimizer.

## 1. Problem description

Sergey's birthday is approaching, and Ahmad wants to fill his backpack with the
most valuable collection of gifts from the Zalando website. Each available item
has a known price, but the backpack can carry at most **40 liters**.

This would normally be a binary knapsack problem: select the combination of
items with maximum total price while keeping total volume within the backpack's
capacity. The difficulty is that the database containing the individual item
volumes was accidentally erased.

The remaining data consists of historical customer packages. For every package,
we know:

- which items were included; and
- the package's measured total volume.

The volume-measuring machine is imperfect. Its measurement error follows a
normal distribution with mean 0 and variance 2. Consequently, individual item
volumes must first be inferred from noisy aggregate measurements, and the gift
selection should allow for the resulting statistical uncertainty. Merely using
point estimates could produce a selection whose estimated volume is below 40
liters but whose true volume has an unacceptably high probability of exceeding
the capacity.

The application uses the same data to study three decision settings. Case 1
selects a portfolio before exact volumes are observed. Case 2 makes an
irreversible accept/reject decision after each arriving item's volume is
revealed. Case 3 repeatedly exposes both policies to matched volume
realizations so that their long-run value, capacity use, and overflow risk can
be compared fairly.

### Input data

`items.json` contains the item identifiers and prices:

```json
[
  {"name": "A1", "price": 98},
  {"name": "A2", "price": 108}
]
```

`packages.json` contains historical package compositions and measured volumes:

```json
[
  {"total_volume": 36.04, "items": ["A1", "A2"]},
  {"total_volume": 22.15, "items": ["A1"]}
]
```

For the individual volumes to be identifiable, the package-item incidence
matrix must have full column rank. The implementation also requires more
historical packages than items so that the regression has positive residual
degrees of freedom.

## 2. Case 1 — Baseline Offline Setting

In the baseline setting, Ahmad must choose the complete gift portfolio before
the true item volumes are known. Historical package measurements are used to
estimate those volumes and their joint uncertainty. A chance-constrained
stochastic binary knapsack model then maximizes total price while requiring the
selected portfolio to fit within the user-specified capacity with at least the
chosen confidence level.

The decision is *offline*: once the portfolio has been selected, it remains
fixed and cannot react to later volume realizations. This makes Case 1 a useful
benchmark for evaluating the value of sequential information in Cases 2 and 3.

### 2.1 Statistical model for the missing volumes

Suppose there are $n$ items and $m$ historical packages. Define:

- $p_i$: known price of item $i$;
- $v_i$: unknown true volume of item $i$;
- $y_j$: measured total volume of historical package $j$;
- $A_{ji}=1$ if package $j$ contains item $i$, and $0$ otherwise;
- $\varepsilon_j$: measurement error for package $j$; and
- $\sigma^2=2$: the measurement-error variance stated in the problem.

In vector form, the historical measurements follow the linear model

$$
y = Av + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,\sigma^2 I_m).
$$

When $A$ has full column rank, ordinary least squares gives

$$
\widehat v=(A^\top A)^{-1}A^\top y.
$$

Under the Gaussian measurement model, the estimator has covariance

$$
\Sigma
=\operatorname{Cov}(\widehat v)
=\sigma^2(A^\top A)^{-1}.
$$

The off-diagonal elements of $\Sigma$ are important. Estimated item volumes are
generally correlated because the same historical packages provide information
about several items. The uncertainty of a selected collection must therefore
use the full covariance matrix, not a sum of independent item variances.

The application also calculates the residual variance, regression RMSE, matrix
rank, and condition number as diagnostics. It rejects rank-deficient data and
non-positive fitted physical volumes rather than silently optimizing an
unreliable model.

### 2.2 Chance-constrained stochastic programming formulation

Let

$$
x_i=
\begin{cases}
1, & \text{if item }i\text{ is selected},\\
0, & \text{otherwise}.
\end{cases}
$$

Let $C$ denote the user-specified backpack capacity. Let $1-\alpha$ be the
required probability of fitting; the application default is
$1-\alpha=0.95$.

The stochastic binary knapsack model is

$$
\begin{aligned}
\max_{x}\quad & \sum_{i=1}^{n}p_i x_i \\
\text{subject to}\quad
& \mathbb{P}\left(v^\top x\le C\right)\ge 1-\alpha,\\
& x_i\in\{0,1\},\qquad i=1,\ldots,n.
\end{aligned}
$$

For a fixed selection $x$, the estimated total volume has mean
$\widehat v^\top x$ and variance $x^\top\Sigma x$. If $\Phi$ is the standard
normal cumulative distribution function and
$z_{1-\alpha}=\Phi^{-1}(1-\alpha)$, the probability constraint has the
deterministic equivalent

$$
\widehat v^\top x
+z_{1-\alpha}\sqrt{x^\top\Sigma x}
\le C.
$$

The complete deterministic optimization problem solved by the application is

$$
\begin{aligned}
\max_{x}\quad & p^\top x \\
\text{subject to}\quad
& \widehat v^\top x
+z_{1-\alpha}\sqrt{x^\top\Sigma x}\le C,\\
& x\in\{0,1\}^n.
\end{aligned}
$$

The first term in the capacity constraint is the nominal estimated volume. The
second term is an uncertainty buffer. It becomes larger when the requested
confidence increases or when the historical packages provide weak information
about the selected combination.

For a nonzero standard error, the model-implied fit probability reported by the
application is

$$
\Phi\left(
\frac{C-\widehat v^\top x}{\sqrt{x^\top\Sigma x}}
\right).
$$

This formulation treats the given Gaussian measurement variance as known and
uses uncertainty in the estimated item volumes. It does not model unrelated
real-world effects such as compressible goods, alternative packing layouts,
damaged items, or future measurement noise unless those effects are represented
in the uploaded data and variance parameter.

### 2.3 Stochastic programming solution approach

#### 2.3.1 Why a special algorithm is required

The square-root term is a covariance norm. The resulting constraint is convex
in the continuous relaxation but is not linear, so the model is a
mixed-integer second-order-cone problem rather than an ordinary MILP.

HiGHS is an open-source LP/MIP solver and does not directly accept this conic
constraint. Replacing it with only the nominal constraint would ignore risk,
while using a loose fixed safety factor for every item would be unnecessarily
conservative and would lose the covariance structure.

The application preserves the original chance constraint through an iterative
outer-approximation algorithm. Each subproblem is a standard MILP solved by
HiGHS through `scipy.optimize.milp`.

#### 2.3.2 Convex constraint function

Define

$$
f(x)=\widehat v^\top x+z_{1-\alpha}\sqrt{x^\top\Sigma x}.
$$

The chance constraint is $f(x)\le C$. Because $\Sigma$ is positive
semidefinite, $\sqrt{x^\top\Sigma x}$ is a convex norm and $f$ is convex.

At a candidate $\bar x$ with
$s=\sqrt{\bar x^\top\Sigma\bar x}>0$, its gradient is

$$
\nabla f(\bar x)
=\widehat v
+z_{1-\alpha}\frac{\Sigma\bar x}{s}.
$$

Convexity gives the supporting-hyperplane inequality

$$
f(x)\ge
f(\bar x)+\nabla f(\bar x)^\top(x-\bar x).
$$

The covariance norm is positively homogeneous, so the tangent expression
simplifies to a linear inequality:

$$
\left(
\widehat v
+z_{1-\alpha}\frac{\Sigma\bar x}{
\sqrt{\bar x^\top\Sigma\bar x}}
\right)^\top x
\le C.
$$

Every point satisfying the original chance constraint must satisfy this cut.
The cut is therefore a valid relaxation: it removes the violated candidate
$\bar x$ without removing any chance-feasible binary selection.

#### 2.3.3 Outer-approximation algorithm

The implementation performs the following steps:

1. **Build an initial relaxation.** Start with the nominal capacity constraint
   $\widehat v^\top x\le C$. Since the uncertainty buffer is nonnegative, every
   chance-feasible selection satisfies this constraint.
2. **Solve the MILP globally.** Maximize $p^\top x$ over the current linear
   relaxation with binary variables using HiGHS. The requested MIP relative gap
   is set to zero.
3. **Check the true chance constraint.** For the returned binary candidate
   $\bar x$, evaluate
   $\widehat v^\top\bar x+z_{1-\alpha}
   \sqrt{\bar x^\top\Sigma\bar x}$ directly.
4. **Stop if feasible.** If the value is at most $C$ within numerical tolerance,
   return the candidate.
5. **Cut off a violation.** If the candidate is not chance-feasible, add its
   supporting-hyperplane cut to the MILP and solve again.
6. **Repeat** until a globally optimal chance-feasible selection is found or a
   configured time or cut limit is reached.

#### 2.3.4 Why the returned solution is globally optimal

At every iteration, the MILP feasible region contains every feasible solution
of the original chance-constrained problem. Its optimal objective value is
therefore an upper bound on the best achievable gift value.

When the MILP optimizer returns a candidate that also satisfies the original
chance constraint, that candidate is feasible for the original problem and
already attains the relaxation's upper bound. No other chance-feasible binary
selection can have a greater price. The candidate is consequently globally
optimal, not merely locally optimal or the output of a heuristic.

Because the decision set is binary and finite, each violated binary candidate
can be excluded by a valid cut. In exact arithmetic the process therefore
terminates after finitely many distinct candidates. The implementation also
detects repeated candidates and enforces time and cut limits to handle numerical
or pathological cases safely.

#### 2.3.5 Solver diagnostics

The downloaded result records:

- total price and selected items;
- estimated total volume;
- standard error and uncertainty buffer;
- complete chance-constraint left-hand side and capacity slack;
- model-implied fit probability;
- OLS residual diagnostics and design-matrix condition number;
- number of HiGHS solves and uncertainty cuts;
- explored MIP nodes, final MIP gap, and runtime; and
- all model parameters needed to interpret the result.

## 3. Case 2 — Online Setting

Case 2 changes the timing of both information and decisions. Every item arrives
once in a random order. Before an item arrives, its volume is known only through
the probability model estimated from package history. At arrival, its exact
volume is revealed and Ahmad must immediately accept or reject it. An accepted
item consumes its realized volume, while a rejected item cannot be reconsidered.

The online policy balances the value of the current item against the opportunity
cost of using capacity that may be more valuable for future arrivals. It does so
with an approximate dynamic programming value function derived from a
deterministic linear-programming relaxation.

### 3.1 Approximate dynamic programming approach

Let \(b_t\) be the exactly known remaining capacity immediately before the
decision at epoch \(t\), and let \(\mathcal R_t\) contain the items that have not
yet arrived. The true dynamic-programming value function is difficult to
compute because its state must account for remaining capacity, unobserved
volumes, and the set of future items. The application approximates the future
value of capacity by solving the following deterministic fractional-knapsack
LP for the items that have not yet arrived:

$$
\begin{aligned}
\max_y\quad & \sum_{j\in\mathcal R_t}p_jy_j\\
\text{subject to}\quad
& \sum_{j\in\mathcal R_t}\widehat v_jy_j\le b_t,\\
&0\le y_j\le1.
\end{aligned}
$$

Because this LP has one capacity resource, sorting the future items by
\(p_j/\widehat v_j\) solves it exactly. The density of the marginal fractional
item is the capacity bid price \(\lambda_t\), an approximation of the marginal
future value of one liter.

### 3.2 Bid-price acceptance rule

For an arriving item \(i\) with realized volume \(V_i\), the approximate
marginal value is

$$
\Delta_{it}=p_i-\lambda_tV_i.
$$

The item is accepted exactly when it fits and has nonnegative marginal value:

$$
V_i\le b_t
\quad\text{and}\quad
\Delta_{it}\ge0.
$$

After an acceptance, remaining capacity is updated using the exact realized
volume. The DLP approximation is then re-solved for the next arrival, producing
a state-dependent bid price. Each scenario records
the arrival order, expected and realized volumes, bid prices, opportunity
costs, marginal values, decisions, remaining capacity, and cumulative value.

### 3.3 Correlated volume realization

Volume vectors are drawn jointly from

$$
V\sim\mathcal N\left(\widehat v,\sigma^2(A^\top A)^{-1}\right).
$$

Draws containing a non-positive physical volume are rejected and resampled.
This retains the correlation implied by the package regression while enforcing
positive item volumes.

## 4. Case 3 — Simulation Comparison

Case 3 estimates how the fixed offline portfolio and the adaptive online policy
perform over many possible futures. It first solves Case 1 once. Each Monte
Carlo run then generates a correlated vector of exact item volumes and a random
arrival order for Case 2. Increasing the run count reduces Monte Carlo noise at
the cost of additional computation.

### 4.1 Paired simulation approach

The comparison uses common random numbers: on every run, both policies receive
the same joint volume realization. Pairing the scenarios removes avoidable
sampling variation and makes differences between the policies easier to
attribute to their decision rules:

- Case 1 evaluates its fixed chance-constrained portfolio. Its value is retained
  even if realized volume exceeds capacity; infeasibility is measured separately.
- Case 2 receives a random arrival order and applies the sequential bid-price
  policy. It rejects any item that cannot fit, so it never intentionally
  overflows.

Case 1's portfolio is not repaired after a realization. If its realized volume
exceeds capacity, the run retains the portfolio's value and separately records
its overflow. Case 2 updates capacity after every acceptance and rejects any
item that does not fit, so it does not intentionally exceed capacity.

### 4.2 Comparative statistics

The simulation reports the following statistics side by side:

- long-run average total value (and total item cost);
- value standard deviation, percentiles, and a 95% confidence interval;
- average used volume and capacity utilization;
- average overflow volume;
- probability of fitting and probability of overflow; and
- average number of selected or accepted items.

Run-level paired results and the complete comparison can be downloaded as JSON.

## 5. Application features

- Upload `items.json` and `packages.json` directly in the browser.
- Validate schemas, duplicate names, package references, regression rank, and
  fitted physical volumes.
- Navigate between baseline, sequential-scenario, and simulation tabs.
- Configure capacity, confidence, variance, random seeds, and simulation runs.
- Solve without Gurobi or a commercial solver license.
- Display selected gifts, online decisions, risk measures, and paired statistics.
- Download each case's complete result as a JSON file.

## 6. Run locally

Use Python 3.12 or newer:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## 7. Repository structure

```text
streamlit_app.py          Streamlit user interface
optimizer.py              validation, OLS estimation, and optimization logic
online_policy.py          online bid-price policy and paired simulation
requirements.txt          runtime dependencies
.streamlit/config.toml    Streamlit server and theme configuration
tests/test_optimizer.py   regression and small-instance optimizer tests
tests/test_online_policy.py online-policy and simulation tests
```
