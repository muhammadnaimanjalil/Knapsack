# Gift Portfolio Optimizer

A Streamlit application for Zalando's gift problem. It estimates missing item
volumes from noisy historical package measurements and solves a 0-1 knapsack
with a configurable Gaussian chance constraint.

## Features

- Upload `items.json` and `packages.json` in the browser.
- Validate schemas, item references, regression rank, and physical volumes.
- Configure backpack capacity, fit confidence, noise variance, and time limit.
- Solve with open-source HiGHS through `scipy.optimize.milp`.
- Preserve the exact binary chance constraint through iterative outer approximation.
- Review selected items, risk metrics, and diagnostics.
- Download the complete result as JSON.

## Expected inputs

`items.json`:

```json
[
  {"name": "A1", "price": 98},
  {"name": "A2", "price": 108}
]
```

`packages.json`:

```json
[
  {"total_volume": 36.04, "items": ["A1", "A2"]},
  {"total_volume": 22.15, "items": ["A1"]}
]
```

For identifiable volume estimates, the package incidence matrix must have full
column rank and there must be more package observations than items.

## Run locally

Use Python 3.12 or newer:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the entrypoint to `streamlit_app.py`.
4. Choose Python 3.12 or newer and deploy.

No solver license, API key, database, or external executable is required.

## Optimization method

Let `x` be the binary item-selection vector, `v_hat` the estimated item volumes,
and `Sigma` their estimation covariance. The app solves

```text
maximize     price' x
subject to   v_hat' x + z(confidence) * sqrt(x' Sigma x) <= capacity
             x is binary
```

HiGHS does not directly solve this mixed-integer conic constraint. The app
therefore solves a sequence of MILP relaxations. Each violated binary candidate
adds a supporting-hyperplane cut for the covariance norm. Once the MILP optimum
satisfies the original chance constraint, global optimality follows because the
MILP was a relaxation containing every chance-feasible selection.
