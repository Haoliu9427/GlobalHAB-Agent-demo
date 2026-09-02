# GlobalHAB-Agent

GlobalHAB-Agent is a research prototype for cross-region harmful algal bloom (HAB) analysis. It combines environmental conditions, transport information, biological observations and aquaculture response variables in a common workflow for scenario analysis, lag testing, model comparison and forward validation.

## Main functions

The Streamlit interface contains six workspaces:

1. **Risk assessment** — 7/14/30-day scenario maps and relative risk ranking across representative marine regions.
2. **Observed-event analysis** — South Australia qPCR replay, Norway long-term monitoring, Florida/Gulf flow-constrained retrospective analysis, and field-data forward validation.
3. **Biological-response sandbox** — cage-fish pressure trajectories under HAB, temperature, dissolved oxygen, density and feeding scenarios.
4. **Scientific analysis** — multiscale anomalies, routing diagnostics, TE/CTE lag analysis and spatial Durbin decomposition.
5. **Exploration and validation** — constrained experiment selection, Bayesian policy comparison, negative controls, model benchmark and blocked holdout evaluation.
6. **Data and provenance** — data-quality metrics, sources, result boundaries and downloadable evidence files.

## Data structure

The synthetic benchmark contains four anonymous regions and daily records of:

- sea-surface temperature and seasonal climatology;
- marine heatwave status and intensity;
- nitrate, phosphate and silicate;
- a bounded transport/residence/convergence proxy;
- HAB event labels generated from an upstream lagged signal, nutrient background and stochastic noise.

The default generator contains a 14-day upstream-to-downstream lag as a known synthetic reference. That value is not available to the experiment-selection policy and is used only after a trajectory is complete to evaluate recovery.

Observed-data modules are kept separate from the synthetic benchmark:

- South Australia: 115 qPCR observations from the 2025 Karenia event;
- Norway: 5,919 observations from 2006–2019 toxic-algae and environmental monitoring;
- Florida/Gulf: NOAA HABSOS `Karenia brevis` observations combined with surface-current data from NOAA CoastWatch or uploaded HYCOM/Copernicus/HF-radar products;
- field forward validation: user-supplied station observations and current fields.

Third-party data attribution is documented in `THIRD_PARTY_DATA.md`.

## Experiment space

The core experiment space is:

```text
route ∈ {local, downstream}
lag   ∈ {3, 7, 14, 21, 30, 45} days
model ∈ {logistic, random_forest}
```

This gives 24 candidate experiments. The default budget is 8 experiments.

The experiment-selection page can compare:

- constrained heuristic policy;
- Bayesian Expected Improvement;
- Bayesian Information Gain proxy;
- Thompson Sampling;
- Random selection.

All policies use the same candidate table and budget. The synthetic 14-day reference is excluded from acquisition functions.

## Model benchmark

The model benchmark uses the same blocked holdout rows for all methods and includes statistical, machine-learning and compact neural models, including:

- Seasonal / persistence baselines;
- Logistic Regression;
- GAM;
- Gaussian Naive Bayes;
- kNN;
- RBF-SVM;
- Decision Tree;
- Random Forest;
- Extra Trees;
- AdaBoost;
- Gradient Boosting;
- HistGradientBoosting;
- XGBoost;
- LightGBM;
- MLP;
- STS-Interaction GLM;
- Lightweight TCN;
- STS-Gated TCN.

Hyperparameter selection is restricted to the training period. The final holdout is not used for tuning.

## Validation

Synthetic benchmark validation uses:

- a completely held-out region;
- a forward temporal test block;
- seasonal and persistence baselines;
- equal-budget random-search comparison;
- reverse-path and time-permutation controls;
- Average Precision, Brier Skill, ECE and fixed-capacity Top-k metrics.

The Norway module uses expanding forward windows. The Florida/Gulf module compares flow-constrained, no-flow and reverse-flow matching across candidate lags.

## Field forward validation

Templates are provided in `data/field_validation/`.

Minimum observation fields:

```text
date, station_id, latitude, longitude, cell_count
```

Minimum current fields:

```text
date, latitude, longitude, u_ms, v_ms
```

Optional fields include toxin concentration, SST, salinity, dissolved oxygen, nitrate, phosphate, silicate, chlorophyll and biological-response variables.

The field workflow checks sample count, temporal coverage, spatial support, event count and current-field overlap. When the quality criteria are met, an earlier time block is used for lag selection and a later block is used once for forward evaluation. Insufficient data return `DEFER`.

## Biological-response model

The cage-fish sandbox updates a relative pressure state using a bounded hourly process:

```text
P(t+1) = clip[P(t) + 1.45*C(t)*(1-P(t)/100)
              - 0.55*(1-C(t))*P(t)/100, 0, 100]
```

`C(t)` is a 0–1 composite challenge term and `P(t)` is a 0–100 relative physiological-pressure state. Parameters are transparent and listed in `outputs/cage_fish_sandbox_parameters.csv`. They are not species- or farm-calibrated.

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the web interface:

```bash
streamlit run app.py
```

Run the default command-line workflow:

```bash
python run_demo.py --config config/demo.json
```

Minimal synthetic reproduction:

```bash
python scripts/run_minimal_reproduction.py
```

Experiment-selection policy comparison:

```bash
python scripts/run_agent_policy_benchmark.py
```

Full model benchmark:

```bash
python scripts/run_broad_benchmark_audit.py
```

Florida/Gulf retrospective analysis:

```bash
python scripts/run_florida_sts_validation.py --online
```

Field forward validation:

```bash
python scripts/run_field_forward_validation.py \
  --observations <field_observations.csv> \
  --currents <field_currents.csv>
```

## Release checks

```bash
python scripts/verify_release.py
python -m pytest -q tests/test_release_smoke_fast.py tests/test_bayesian_design.py tests/test_florida_sts.py tests/test_broad_benchmark.py
```

## Repository layout

```text
app.py
run_demo.py
config/
data/
docs/
outputs/
prompts/
scripts/
src/globalhab_demo/
tests/
```

See `docs/MINIMAL_REPRODUCTION.md` for the shortest reproducible workflows and `docs/TECHNICAL_NOTE.md` for method details.

## Result boundaries

- Synthetic results are method-validation results, not real-ocean forecast performance.
- South Australia is an event replay and is not used to train the synthetic models.
- The Florida/Gulf flow matching uses a first-order surface-current displacement and is not a full 3-D particle-tracking model.
- The cage-fish response parameters are not calibrated to a specific species, life stage or farm.
- Outputs do not constitute mortality estimates, toxin thresholds, regulatory alerts or automatic operational commands.

## License

Code is released under the MIT License. Third-party datasets retain their original licenses; see `THIRD_PARTY_DATA.md`.
