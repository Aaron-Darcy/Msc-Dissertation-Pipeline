# Synthetic Patient Data: Generation and Evaluation Pipeline

A reproducible pipeline that generates synthetic hospital admission data with four different
methods and evaluates each one on three dimensions: how useful the data is for machine
learning, how closely it resembles the real data, and how much it reveals about the real
patients it was learned from.

Built for the MSc dissertation *Generating and Evaluating Synthetic Patient Data for
Privacy-Preserving Machine Learning*, ATU Donegal, 2026.

## What it found

The prediction task is unplanned readmission within 30 days, using a cohort of 534,182
admissions from MIMIC-IV. A model trained on the real data reaches a ROC AUC of 0.6154, and
that is the ceiling every synthetic dataset is measured against.

- The best synthetic dataset reached **98.0% of the real-data baseline**.
- The two leading configurations, CTGAN and TabDDPM at searched settings, were
  **statistically indistinguishable** from each other.
- **Tuning changed the answer.** TabDDPM went from the only dataset to fail the success
  criterion to joint best. Comparing methods at their library defaults alone would have
  produced a different and misleading ranking.
- **No method led on fidelity as a whole.** The methods that reproduced numeric columns best
  were the worst at categorical ones, and the reverse.
- **Privacy did not separate the methods.** All eight datasets passed all three privacy tests.
- **Cost varied by a factor of 107**, from 1.5 minutes to 161 minutes, and the cheapest method
  still retained more than 90% of the baseline.

Full results are in `Results/`.

## The pipeline

Seven notebooks, run in order. Each writes files the next one reads, so a failure in one does
not cost the others.

| Notebook | What it does |
|---|---|
| `01_EDA` | Extracts the cohort from MIMIC-IV on BigQuery and explores it |
| `02_data_preparation` | Cleans the data and splits it by patient into four partitions |
| `03_gaussian_copula` | Searches for settings, then generates two datasets |
| `04_ctgan` | The same, for CTGAN |
| `05_tvae` | The same, for TVAE |
| `06_tabddpm` | The same, for TabDDPM |
| `07_evaluation` | Scores all eight synthetic datasets on utility, fidelity and privacy |

Every method is run twice, once at its library defaults and once at settings chosen by a
bounded hyperparameter search. That gives eight synthetic datasets, and it is what allows the
question of whether the ranking survives tuning to be answered rather than assumed.

The data is split by patient rather than by admission, because 45.1% of patients have more
than one admission. Splitting by admission would put the same person on both sides of the
split, which inflates every utility figure.

| Split | Admissions | Used for |
|---|---|---|
| Training | 427,354 | Fitting the generators, and the real-data baseline |
| Test | 106,828 | Reported results only |
| Fitting | 342,004 | Fitting during the hyperparameter search |
| Validation | 85,350 | Selecting hyperparameters |

Fitting and Validation are a further patient-level division of Training. The test set takes no
part in selecting anything.

## Running it

The notebooks run on **Google Colab** with Google Drive mounted. Open them in order and run
each from top to bottom.

Before starting you need:

1. A **PhysioNet account approved for MIMIC-IV**. Approval requires completing the required
   training and signing the data use agreement.
2. A **Google Cloud project** to bill BigQuery queries to. Set `GCP_PROJECT` in notebook 01.
3. A **GPU runtime** for notebooks 04 to 06. All results here were produced on an A100.

Expect notebooks 03 to 06 to take a long time. The hyperparameter search and the two final
generation runs together took roughly 14 hours across the four methods, most of it TabDDPM.
Screening results and completed runs are written to disk as they finish, so a dropped session
costs only the run in progress rather than the whole search.

## Data access and what is not in this repository

MIMIC-IV is credentialed data, governed by a PhysioNet Data Use Agreement.

**No patient data is in this repository, and none may be added.** `.gitignore` excludes
`data/`, `*.parquet` and `*.csv`, with a single exception for `Results/tables/*.csv`, which
contain aggregate statistics only and no individual records. The cohort and the synthetic
datasets live in a private Drive folder and are not published.

## What is in `Results/`

- `Results/tables/` holds the result tables: utility, fidelity, privacy, the trade-off summary,
  the effect of tuning, training convergence and the timing log.
- `Results/figures/` holds the charts, including the distribution overlays, the correlation
  differences, the privacy distance histograms and the bootstrap confidence intervals.

These are the evidence base for the results chapter of the dissertation.

## Environment

Python 3.12 on Colab. The versions that produced these results:

```
sdv==1.17.0
synthcity==0.2.12
torch==2.2.2+cu121
torchvision==0.17.2
numpy==1.26.4
pandas==2.1.4
pyarrow==15.0.2
```

The pins are not arbitrary. `synthcity` requires numpy below 2 and torch below 2.3, and
installing torch without a matching torchvision produces an error about `torchvision::nms`
that looks unrelated to either. Notebook 06 installs the matched set in one resolution pass
and restarts the session, which is necessary because a compiled extension cannot bind to an
already imported library of the wrong version.

## Known limitations

- **TabDDPM had not converged** in either configuration when the training budget ran out. Both
  runs were still improving, so its results are a lower bound rather than its best.
- **The hyperparameter search was bounded, not exhaustive.** Twelve configurations per method.
- **The patient split depends on row order.** It shuffles the unique patient identifiers as
  returned by BigQuery. Sorting them before shuffling would make the partition reproducible
  regardless of that order, and is the correction a future version should make.
- **The privacy tests are indicators, not proof.** They rule out the clearest failure mode, a
  generator copying its training records, but they do not certify the data against a stronger
  attacker with auxiliary information.
- **The real-data baseline is modest**, at 0.6154. Ten coarse admission-level columns carry
  limited signal, and a weaker signal is easier for a generator to reproduce. These conclusions
  may not transfer to a richer clinical feature set.
