# GE-STDGN
 Spatio-temporal weather prediction model based on graph evolution

## BJ history 1 hour to future 3 hours

`split_history1predict3.py` splits the original `data/BJ` events before making
six-minute windows. Events sharing raw calendar dates remain in one split.
Defaults are seed 20 and an 80/10/10 target ratio by raw event count.

```powershell
python -B split_history1predict3.py
python -B split_history1predict3.py --verify-only
```

The processed dataset is saved to `data/history1predict3/{train,valid,test}`.
Each NPZ contains 11 radar frames (60 minutes between first and last) and
30 strictly future RA frames (180 minutes), cropped to 128 by 128 pixels.
The current export has 1443 training, 81 validation and 221 test windows.
The event counts are 50/6/6; window ratios differ because event lengths vary.

Radar uses physical channel units and labels use millimeters. Missing pixels
have explicit validity masks. No training statistics are fitted during export.
Use the dedicated lazy loader/training entry (the old hourly entry is separate):

```powershell
python -B main_history1predict3.py --preflight --run_dir runs/history1predict3_preflight
python -B main_history1predict3.py --normalization runs/history1predict3_preflight/input_normalization.json --run_dir runs/history1predict3_pm1
```

`history1predict3_dataset.py` loads the fixed splits without cutting windows or
resplitting. Original RA uses 0.1 mm; exported `label` already uses mm and is
never multiplied by 0.1 again. Physical radar values are not clipped to [0,1].
The reader additionally masks reflectivity outside [-100,100] dBZ (the source
contains malformed values near -2866.5 dBZ), negative ET/VIL, and nonfinite data.
Normalization uses only valid pixels in unique training-history frames. Invalid
radar pixels and nonfinite/sentinel NWP inputs are filled with physical zero
before normalization (`physical-zero-v3`); their standardized values may be
nonzero. Radar validity masks remain available for audits/statistics. New runs
automatically use a separate cache for this cleaning version; old normalization
and checkpoints must not be mixed with the new reader. Leave `normalization=None`
to fit current statistics and retrain. Positive VIL values retain their existing
validity rules; no unverified upper cutoff is applied. Target 4x4 pooling requires all
pixels to be observed, and missing targets are excluded from both MSE and PM.
Hourly metrics sum ten six-minute predictions and require all ten masks.

For PyCharm, open `main_history1predict3.py`, leave Script parameters empty,
and click Run. The `PYCHARM_CONFIG` block near the top contains all settings:
the default is the 64-channel radar backbone plus a zero-start gated NWP residual.
It reuses the verified fixed KNN9 graph from
`runs/history1predict3_ge_20261022_133324_587584/graph_evolution/selected_adjacency.npy`;
`evolve_graph=False` prevents separate graph selection for each NWP subset.
It trains 100 epochs and evaluates `best.pt`. Set `evolve_graph=True` and
`adj_Set=None` explicitly to start another GE search.
Each run creates a timestamped output directory automatically. Paths
are relative to the script directory, independent of the working directory.
The cleaned training statistics are reused when present; otherwise they are
fitted from the training split. Explicit command-line invocations retain their
existing defaults and do not apply `PYCHARM_CONFIG`.

### NWP inputs and feature selection for 11 to 30

PyCharm enables `input_sources="radar_aux"` and the initial moisture subset
`PWAT, Q850, RH700` (not yet established as optimal for 11-to-30 prediction).
Edit the same `PYCHARM_CONFIG` block to choose an ordered subset:

```python
"input_sources": "radar_aux",
"nwp_features": ["PWAT", "Q850", "RH700"],
# Or NWP_FEATURE_GROUPS["moisture"], ["instability"], ["dynamics"], ["all"].
# Set input_sources="radar" for a radar-only experiment.
# Use list(NWP_FEATURES) for all 27 variables.
```

Supported names are `CAPE DVG200 DVG850 DVG925 HT0 HT10 HTw0 KI LCL LFC LI300
LI500 muLCL PE PWAT Q1000 Q700 Q850 RH1000 RH500 RH700 TdSfc850 TTdMax74
TTdMean74 WS500 WS700 WS925`. Unknown, duplicate or empty selections fail early.
`radar_aux` preserves all 64 radar channels and adds a small 16-channel NWP
encoder, a 16-to-64 projection and a pixelwise sigmoid gate. The projection
starts at zero, so its initial predictions and shared weights exactly match
the same-seed radar-only model. The auxiliary encoder has no dropout and is
initialized without advancing the backbone RNG stream. NWP gradients reach
the projection first, then its encoder/gate as the projection learns.
`radar_nwp` retains the earlier 32+32 concatenation mode for historical
checkpoint evaluation and explicit experiments. Both modes support the named
feature selection; data loading, training and evaluation use the same order.

The user confirmed that NWP time H describes the preceding hour `(H-1h,H]`.
For each six-minute historical radar time, use the latest NWP timestamp not
later than that radar time, after `nwp_lag_minutes` (default zero). Radar 16:18
uses NWP 16:00, covering 15:00--16:00; NWP 17:00 cannot be used before 17:00.
There is no forward interpolation. `nwp_max_age_minutes=60` rejects missing or
stale hourly coverage instead of silently dropping samples or using future data.
NWP is historical context, rather than future forecast covariates in this entry.

`history1predict3_nwp.py` builds independent sidecars under
`data/history1predict3_nwp/`; the existing event split, radar exports and radar
cache are reused. Spatial alignment uses metadata start coordinates/resolutions
and bilinear sampling at the exact centers of the 32x32 target pooling cells.
The center is used, without surrounding shifted windows. NWP values retain
their physical units; nonfinite and fill values <=-9999 are masked. A target
grid cell is masked if any nonzero bilinear neighbor is invalid. Mean filling
and normalization use unique hourly NWP frames from the training split only.
Feature names/order, alignment, source hashes and cache hashes are recorded;
changing any of these creates another sidecar. Checkpoints retain the NWP
configuration/statistics, so evaluation reconstructs the same input selection.
Each run records `nwp_alignment.json` and prints `[INPUT] sources=...`.

```powershell
# Optional cache preparation only; normal training also prepares automatically.
python -B history1predict3_nwp.py --nwp_features all
# Explicit CLI uses radar-only by default; enable NWP as follows:
python -B main_history1predict3.py --input_sources radar_aux --nwp_features PWAT Q850 RH700 --adj_Set runs/history1predict3_ge_20261022_133324_587584/graph_evolution/selected_adjacency.npy --cache_dir data/history1predict3_cache --num_workers 2 --run_dir runs/history1predict3_gated_moisture
```

This adds NWP preprocessing and model wiring, not a completed 100-epoch NWP
experiment. Forecast quality must be evaluated after training the new model.

### Three-case fixed-graph gated comparison

Open `main_history1predict3_compare.py` in PyCharm, leave Script parameters
empty, and click Run. `COMPARISON_CONFIG` reuses the main settings and runs
three cases sequentially, each for 100 epochs (300 total):

- `radar`: 64-channel radar backbone, no NWP.
- `q850`: the same backbone plus the gated Q850 residual.
- `moisture`: the same backbone plus gated PWAT/Q850/RH700 residual.

The runner verifies and snapshots one graph into `fixed_adjacency.npy`. Before
training it checks every shared initial tensor via a fingerprint and verifies
bit-identical initial predictions on the same training sample. Each training
entry checks the expected graph and shared initialization before the first
optimizer step. Epoch sample-order and RNG fingerprints must also match across
cases. Trainable adaptive-graph parameters retain the same initialization and
are learned within each model; only the base graph is fixed across cases.
Auxiliary cases add a small number of parameters, recorded in the audit.

The runner first trains/validates every case using `--train_only`, without
reading test labels or NWP test fields. It records `validation_selection.json`
before evaluating the three selected checkpoints. Test scores never determine
the selected case. Results are saved under
`runs/history1predict3_fixed_graph_gated_<timestamp>/`, including:
`initialization_audit.json`, per-case checkpoints/history,
`<case>/evaluation_best/fig_test_error_maps/`, and `comparison.json`/`comparison.md`.

```powershell
# Verify all three initializations without training or accessing test data.
python -B main_history1predict3_compare.py --prepare_only
# Wiring checks: one training batch per case, followed by evaluation.
python -B main_history1predict3_compare.py --smoke
```

Smoke results are labeled `smoke` and do not establish forecast quality.

PyCharm defaults to `num_workers=2` and `prefetch_factor=2`: two CPU worker
processes decode/clean upcoming batches while the GPU trains. Workers persist
across epochs and GE candidates, and close before the final test stage or on
failure. Candidate sample orders are reset independently of worker RNGs.
CUDA loaders use pinned CPU memory and nonblocking input transfers. Set
`num_workers=0` to fall back to synchronous loading; command-line mode also
supports `--num_workers 2 --prefetch_factor 2`. Training logs include epoch
`seconds` and time spent waiting for batches (`data_wait`), including initial
worker startup. Prefetching does not change data cleaning, splits, or losses.

PyCharm also enables `cache_dir=data/history1predict3_cache`. The first use of
each split builds a lossless float32 mmap cache of cleaned radar frames and
pooled targets. Overlapping input windows share identical cached radar frames,
instead of storing/decoding 11 copies per window. No normalization is fitted
by caching; validation/test remain separate and test caching starts only during
final evaluation. Original-source and cache hashes are checked before reuse.
The cache is keyed by dataset fingerprint and cleaning version, and interrupted
builds are ignored. Files are memory-mapped, rather than loading the whole cache
into RAM. Later candidates, training epochs and runs reuse these files. Disable
with `"cache_dir": None`; CLI mode supports `--cache_dir data/history1predict3_cache`.
PyTorch 1.13 checkpoint loading uses a restricted parser supporting float
metadata, which its experimental weights-only parser does not support.
PyTorch 2.x retains its standard weights-only loader.

After test evaluation the entry also writes `fig_test_error_maps/` with
`HOUR1_3_GT_PRED_ERROR_MAPS.png` (three hourly sums),
`SIX_MINUTE_GT_PRED_ERROR_MAPS.png` (native predictions at +6/+30/+60/+90/+120/+180
minutes), and `sample_metadata.json`. Rows show observations, predictions, and
signed errors. Gray cells are unobserved, not zero rain. The displayed test
case is selected by maximum observed mean hourly rainfall, independently of
prediction accuracy. To draw existing saved results without retraining, run
`python -B plot_history1predict3.py`; it selects the latest completed full run.
Use `--run_dir` and `--sample_index` to select another run or test case.

This entry reuses STDGN with 11 inputs/30 outputs. In command-line mode without
`--evolve_graph`, its default fixed graph is 8-neighbor KNN from row/column
coordinates; `--adj_Set` selects another graph
whose training provenance must be checked against the new split. The existing
sigmoid/log1p rainfall transform is retained, with a ceiling set by training
labels. `--smoke` checks one batch per split; it does not measure trained skill.
Normalization files carry a cleaning-rule version to reject outdated statistics.

Details and exact event/sample assignments are in the output README,
`split_manifest.json`, and `samples.jsonl`.
The script refuses to overwrite a nonempty output directory.

## Graph evolution for history1predict3

Search only (training and validation data, no test samples):

```powershell
python -B evolve_history1predict3.py --normalization runs/history1predict3_preflight_cleaned_20261021/input_normalization.json --run_dir runs/history1predict3_ge_search
```

Search then reinitialize and train the final model:

```powershell
python -B main_history1predict3.py --evolve_graph --normalization runs/history1predict3_preflight_cleaned_20261021/input_normalization.json --run_dir runs/history1predict3_ge
```

Or load a graph from a previous search:

```powershell
python -B main_history1predict3.py --adj_Set runs/history1predict3_ge_search/selected_adjacency.npy --normalization runs/history1predict3_preflight_cleaned_20261021/input_normalization.json --run_dir runs/history1predict3_ge_train
```

GE defaults are population 6, generations 3, and 3 training epochs per candidate.
Set `--ge_population`, `--ge_generations`, and `--ge_fitness_epochs` to change
these proxy budgets. At most 24 distinct candidates are evaluated with the
defaults; duplicate genotypes reuse cached fitness. Final training still uses
`--epochs` (default 100). Candidate ranking after short training is a proxy,
so a small search does not establish a forecasting improvement.

The history1predict3 search enables unique populations: initial seeds and
survivors are deduplicated, and repeated offspring are rejected before training.
Bounded retries and new legal graph immigrants fill vacant slots. Candidates
already evaluated in this search are excluded from new offspring; an exhausted
finite space stops early. Generation logs record unique graph counts, rejected
duplicates, immigrants and graph hashes. The PyCharm budget is at most 12 unique
graphs, each trained for 3 epochs (up to 36 proxy epochs), then 100 final epochs.
This costs more than the previous one-epoch screening setting.

New runs use deterministic algorithms, a fixed cuBLAS workspace, stable PM
sorting, and explicit per-epoch Python/NumPy/CPU/CUDA seeds reset after iterator
startup. The multiscale decoder uses equivalent separable matrix interpolation
instead of nondeterministic CUDA bilinear-interpolation backward. The selected
graph's proxy model/RNG/sample-order fingerprints must match the first proxy
budget epochs of fresh final training, or training stops with a
`proxy_reproduction_epoch*.json` diagnostic. The final matching proxy epoch
also checks its validation score. Proxy weights are never reused.

New training configs use `self_loop_policy=single`: the model strips the base
graph diagonal before adding one structural identity matrix. Historical configs
without this field retain their original double-self-loop convention and
interpolation implementation when evaluated. Shared legacy model defaults and
checkpoint parameter keys are preserved. These fixes require a new experiment
to assess forecasting improvement; they do not change existing checkpoints.

Genes encode undirected, non-self edges within the geometric KNN-15 graph.
KNN-6 edges form a mandatory backbone; `--ge_k_min` and `--ge_k_max` change
these limits. Initial graphs cover this K range. Objectives minimize validation
hourly RMSE plus 0.25 times hourly true>=7 mm RMSE, and base edge count excluding
self loops. Each candidate uses identical initialization, sample order,
normalization, masked MSE+PM, and epoch budget. Graph search uses its own RNG.
GE evolves the base graph; STDGN's dense adaptive graph remains trainable, so
base-edge sparsity does not imply a sparse total model or lower runtime.

Search exports `selected_adjacency.npy` (lowest prediction objective),
`sparsest_adjacency.npy`, `AdjSet(GE).npy` (prediction-best then sparsest),
`pareto_front.npz`, candidate/generation histories, and `ge_summary.json`.
Graph JSON sidecars record dataset/cleaning fingerprints and hashes, checked
when the graph is reused. Final training creates a new model and optimizer;
proxy weights are not reused. Only final evaluation reads the test split.
Evaluation currently runs serially on the requested device.

Append `--smoke` to either entry for a two-graph, one-generation wiring check
using one batch per proxy split. Integrated smoke also trains a fresh final
model for one batch and checks a test batch. Smoke graphs/results are marked
as such and should not be presented as a completed GE experiment.

Verification:

```powershell
python -B analysis/test_history1predict3_ge.py
```

## Radar-only precipitation baseline

Run the radar-only mode with:

```powershell
python -B main_cnn.py --input_sources radar
```

This mode reads the existing sample windows but loads only radar, labels and
time/event metadata. A radar CNN produces 64 features for the existing STDGN
and multiscale decoder, with all four forecast leads predicted together.
There is no NWP encoder or NWP normalization in this mode. The tiny `x` array
passed through the existing two-input trainer interface is only a placeholder.

Defaults are PM=1.0, equal lead MSE weights, seed 20, grouped event split,
100 epochs, no early stop, fixed adjacency, and final-epoch evaluation.
Radar-only outputs default to `runs/radar_only_pm1`; use a fresh `--run_dir`
for another experiment. This is a new architecture and requires fresh training;
Radar+NWP checkpoints cannot be used as radar-only weights. The radar encoder
is wider than in the original fusion model, so parameter counts and initial
weights are not identical despite using the same seed and backbone width.
The fixed graph is inherited from the original experiment: this comparison
ablates input sources without relearning the graph.
After training and evaluation, the generated map also overwrites the project's
`fig_step_error_maps/STEP0_3_GT_PRED_ERROR_MAPS.png`. The run retains its own copy.
Override the display directory with `--error_map_save_dir` if needed.

Radar-only is the CLI default. The original mode remains available with
`--input_sources radar_nwp`. Historical configs/checkpoints without
an `input_sources` field are interpreted as Radar+NWP by the evaluation helper.

Verification (synthetic data, without a full training run):

```powershell
python -B analysis/test_radar_only.py
```

## NWP moisture auxiliary experiment

```powershell
python -B main_cnn.py --input_sources radar_aux --nwp_features PWAT Q850 RH700 --run_dir runs/new_moisture_experiment
```

Variable names are checked against each file's `nwp_vars`, and only selected
channels are normalized and encoded. The 64-channel Radar path is retained.
A small NWP encoder, a sigmoid gate and a zero-initialized projection add an
auxiliary residual. Auxiliary initialization preserves the shared parameter
RNG stream, so same-seed Radar and auxiliary models have identical shared
initial weights and initial predictions. The added encoder has no dropout.
NWP remains an experimental option; the CLI default stays Radar-only.

The completed first-group run is `runs/radar_moisture_pm1_20261020`; its
comparison report uses the fixed epoch-100 Radar-only baseline. It is a
single-seed, fixed-split experiment, not completed cross-validated feature
selection. Adding an auxiliary branch also changes model capacity.

The second group uses the identical auxiliary structure with `--nwp_features CAPE KI`.
Its completed run and three-way comparison are in `runs/radar_instability_pm1_20261020`.
`analysis/run_instability_comparison.py` refuses to overwrite that run and verifies
the shared initialization and training-source parity with the moisture experiment.
`analysis/evaluate_nwp_group_comparisons.py` compares all three fixed epoch-100
models, with validation-selected best checkpoints reported separately. Forecast
comparison images contain only precipitation maps, without bottom RMSE bars.

## Requirement
* pytorch (see https://pytorch.org/get-started/locally/ for how to install it)
* GPU
* Numpy (see http://www.numpy.org/)

## Dataset

* The source dataset is in [Google Drive](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view) and [Baiduyun](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view) with code `ni44`.

* The source dataset and processed dataset are in [Google Drive](https://drive.google.com/drive/folders/1dWsPYqnkNcZi4s4WDTDAnOI359Lot2YE?usp=sharing)


## Directory description

    ├── Readme.md                   // help
    ├── requirements.txt            // 
    ├── GE                          // graph evolution
    │   ├── Geo_threshold.py        // the adjacency matrix generation method based on geographical threshold
    │   ├── KNN.py                  // the adjacency matrix generation method based on k nearest-neighbor 
    │   ├── TIN.py                  // the adjacency matrix generation method based on triangulated irregular network
    │   ├── parallel.py             // multi-GPUs parapall control
    │   ├── population.py           // define about population
    │   ├── utils.py                // define about genetic operations
    │   ├── individual.py           // define about individuals
    ├── baseline.py                 // baselines based on traditional machine learning methods
    ├── feature_selection.py        // feature selection
    ├── graph_evolution.py          // setup of graph evolution
    ├── main.py
    ├── measure.py                  // evaluation standard
    ├── model.py                    // model of STDGN
    ├── model_fs.py                 // model used in feature selection
    ├── tester.py
    └── trainer.py

## Expiremental setup

### graph generation
* k->[6,23], β is adopted to weigh distance and altitude is 0.8 in KNN
* threshold in distance is 300km, and altitude is 1200m

### graph evolution 
* mutation 0.1
* crossing 0.8


 
|
