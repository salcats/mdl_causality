# Causal discovery by minimum description length

Bivariate causal discovery using the minimum description length principle: given
two variables, the causal direction is taken to be the one under which the data
compresses better.

Research carried out at the Image Processing Laboratory, Universitat de
València, under Prof. Gustau Camps-Valls.

## What's here

**Three MDL-based scoring variants** — `ciscm.py`, `ciscn.py` and
`cisc_combined.py` — with the supporting machinery for negative log-likelihood,
parametric complexity and multinomial recurrence relations in `utils.py`.

**Benchmarking against five established methods** — RECI, IGCI, CDS, ANM and
RCC — on two sets of problems:

- `benchmark_cep.py` — the Tübingen cause-effect pairs database, a standard
  benchmark of real-world variable pairs with expert-assigned causal direction.
- `benchmark_simulated.py` — simulated data with known ground truth, generated
  across several noise and function classes.


## Running it

```bash
pip install -r requirements.txt
python MDL_bivariate/benchmark_cep.py
python MDL_bivariate/benchmark_simulated.py
```

## Data

The benchmark datasets aren't included in this repository. The Tübingen CEP
pairs are downloaded automatically by the Causal Discovery Toolbox via
`cdt.data.load_dataset`. The simulated benchmark sets (SIM, AN, LS, MN-U and
their variants) come from the causal discovery literature and should be obtained
from their original sources; place them under `MDL_bivariate/data/`.

## Notes

Discretisation is handled with scikit-learn's `KBinsDiscretizer` and
equal-frequency binning from feature-engine. Benchmarks are parallelised with
joblib. Built with NumPy, SciPy, pandas, scikit-learn and the Causal Discovery
Toolbox.

## Related work

A separate kernel-based approach to the same problem, developed during the same
period, was published as:

> Varando, G., Catsis, S., Diaz, E., Camps-Valls, G. *Pairwise causal discovery
> with support measure machines.* Applied Soft Computing 150: 111030, 2024.
