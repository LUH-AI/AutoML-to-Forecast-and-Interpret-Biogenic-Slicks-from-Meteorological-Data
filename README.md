# Marine Sea Slick Forecasting

Code for the paper *Automated Machine Learning to Forecast and Interpret Biogenic Slicks from Meteorological Data*: automated detection and forecasting of marine sea slicks from meteorological data.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Overview](#overview)
3. [Getting started](#getting-started)
4. [Data](#data)
5. [Forecasting](#forecasting)
6. [Evaluation](#evaluation)
7. [Shapley analysis](#shapley-analysis)
8. [Labelling your own images](#labelling-your-own-images)

---

## Abstract

Biogenic slicks, thin films of organic matter on the ocean surface, remain underexplored, despite their relevance to marine ecosystems. Forecasting slick occurrence could enable large-scale studies and help marine scientists reach slicks when they occur to analyse them more effectively. Prior work on biogenic slick detection with machine learning relied on images, which are subject to optical limitations and are resource-intensive to acquire. Here, we train a model that exclusively relies on meteorological data during inference, which is widely available and free of optical constraints. We use images only once to generate binary labels, which we pair with concurrent measurements of wind speed, wind direction, air temperature, air pressure, and humidity. We train tabular time-series models on data from two North Sea sites (Spiekeroog and Fino1). The dataset is temporally correlated and imbalanced because slicks are rare, making it unclear which models will perform well. We therefore use Automated Machine Learning (AutoML) rather than manually tuning a model. Shapley value analysis revealed which meteorological features are most predictive of slick occurrence, with recent wind speed as the most influential. The model recovered a documented wind-speed peak favouring slick formation, despite this relationship not being encoded in the labels. Additionally, Shapley value analysis revealed unreported associations with air temperature, humidity, and wind direction. This shows that sensor-based slick forecasting is feasible, as they outperform a wind-speed threshold baseline reported in literature. Since meteorological data is widely available, this work provides a foundation for automated, large-scale monitoring of marine surface dynamics.

---

## Overview

The `secX_Y` folders are named after the paper section they implement.

| Paper section | Path | Contents |
|---|---|---|
| 3 Datasets | [labels/](labels/) | Hourly slick labels ([hourly_labels.csv](labels/hourly_labels.csv)). |
| 3 Datasets | [sec3_sensor_datasets/](sec3_sensor_datasets/) | Weather preprocessing and label merging ([preprocess.py](sec3_sensor_datasets/preprocess.py)), dataset figures. |
| 4.1 Label creation | [sec4_1_label_creation/](sec4_1_label_creation/) | Image classifier ([train_image_model.py](sec4_1_label_creation/train_image_model.py), [classify_images.py](sec4_1_label_creation/classify_images.py)), hand-label check set and its scoring. |
| 5.1 Experimental setup | [sec5_1_experimental_setup/](sec5_1_experimental_setup/) | Tabular forecaster ([train_tabular_predictor.py](sec5_1_experimental_setup/train_tabular_predictor.py), [forecast_slick.py](sec5_1_experimental_setup/forecast_slick.py)). |
| 5.2 Slick prediction | [sec5_2_slick_prediction/](sec5_2_slick_prediction/) | Metrics and figures except the Shapley plots. |
| 5.3 Meteorological predictors | [sec5_3_meteorological_predictors/](sec5_3_meteorological_predictors/) | Shapley value computation and plots. |

---

## Getting started

Install the environment:

```bash
python -m venv .venv && source .venv/bin/activate     # or: conda create -n slick python=3.10
pip install -r requirements.txt
```
The paper used AutoGluon 1.3.1 for the image classifier and 1.4.0 for the tabular models and Shapley runs. `requirements.txt` pins 1.4.0 for both.


---

## Data

### Meteorological files (paper Section 3)

| Site | Source | Period used |
| --- | --- | --- |
| Spiekeroog 2021 | [PANGAEA 10.1594/PANGAEA.975977](https://doi.org/10.1594/PANGAEA.975977), *Meteorological data of Time Series Station Spiekeroog for 2021* | 2021-01-01 to 2021-12-31 |
| Spiekeroog 2024 | [PANGAEA 10.1594/PANGAEA.983002](https://doi.org/10.1594/PANGAEA.983002), *Meteorological data of Time Series Station Spiekeroog for 2024* | 2024-01-30 to 2024-06-10 |
| Fino1 2022 | BSH FINO database, [fino1.de](https://www.fino1.de) | 2022-03-07 to 2022-10-29 |

**Spiekeroog.** Download both PANGAEA datasets as tab-delimited text (they are under a moratorium until 2027-04-29, so a PANGAEA login is needed until then) and save them as `data/spiekeroog_2021_meteo.csv` and `data/spiekeroog_2024_meteo.csv`.

**Fino1.** Export 2022 from the BSH FINO database as semicolon separated text and save it as `data/fino1_2022_meteo.csv`. The script reads these columns as named in the export: ` WSPD_CUP_140_MC.Cup Anemometer.51.0 [m/s]`, ` WDIR_VANE_315deg.Wind Vane.91.0 [deg]`, ` DRYT.Thermometer.34.0 [degC]`, ` ATMP.Barometer.21.0 [hPa]` and ` RELH.Hygrometer.34.0 [%]` (leading space included).

As in the paper (Section 3.2), wind speeds are adjusted to 0.1 m with a sensor height of 41 m for Fino1 and 12 m for Spiekeroog (`-w` overrides it).

### Labels (paper Sections 3 and 4.1)

The images are not part of this repository. The resulting hourly labels are, in `labels/hourly_labels.csv`: 8,310 labelled hours (`Time`, `site`, `slick_tag`) across `fino1_2022` (2,383), `spiekeroog_2021` (4,442) and `spiekeroog_2024` (1,485). `Time` is the full hour that ends the labelled hour.

### Building the dataset (paper Section 3)

Merge the hourly labels with the weather data using [preprocess.py](sec3_sensor_datasets/preprocess.py):

```bash
for site in fino1_2022 spiekeroog_2021 spiekeroog_2024; do
    python sec3_sensor_datasets/preprocess.py weather -s $site \
        -f data/${site}_meteo.csv -o build/$site/meteo.csv
done

python sec3_sensor_datasets/preprocess.py hourly-labels -l labels/hourly_labels.csv -b build

python sec3_sensor_datasets/preprocess.py combine \
    build/1hour_fino1_2022.csv build/1hour_spiekeroog_2021.csv build/1hour_spiekeroog_2024.csv \
    -o build/1hour_combined_dataset.csv

python sec3_sensor_datasets/plot_dataset_overview.py -d build -o results/figures   # Figures 3 and 4
```

The `weather` step puts each site on its image grid (10 min at Fino1, 1 h at Spiekeroog 2021, 5 min at Spiekeroog 2024), adjusts the wind speed, encodes the wind direction as sine and cosine, and fills gaps of up to six hours (three forward, three backward). The combined file has nine columns: `Time`, `windspeed`, `airtemperature`, `airpressure`, `humidity`, `winddir_sin`, `winddir_cos`, `slick_tag` and `site`.

---


## Forecasting

Paper Section 5.1, code in [train_tabular_predictor.py](sec5_1_experimental_setup/train_tabular_predictor.py) and [sec5_1_experimental_setup/](sec5_1_experimental_setup/). Trains AutoGluon-Tabular models that predict whether a slick occurs at the next hour from a sliding window of past readings, flattened into one row per prediction.

```bash
python sec5_1_experimental_setup/train_tabular_predictor.py -d build/1hour_combined_dataset.csv -w 12 -p best \
    -o results/no_static/12h_combined_1hour_best/12h_combined_1hour.csv --seed 0 --cleanup
```

`-w` is the window length in hours, `--seed` varies from 0 to 4, `--cleanup` deletes the model after the predictions are written, and `--model_dir` keeps it under a name of your choice. The train/test split is 70/30 and independent of the seed.

The reported results are **4 datasets × 10 window lengths × 5 seeds = 200 runs**:

```bash
python sec5_1_experimental_setup/generate_jobs.py     # writes jobs.txt, one command per run
sbatch sec5_1_experimental_setup/run_array.sh         # SLURM array, one task per line
```

`--array` in `run_array.sh` must match the job count printed by `generate_jobs.py`. Each job asks for two nodes with 8 cores and 32 GB each (16 cores and 64 GB in total, 6 hours) and starts its own Ray cluster.

Each run writes to `results/no_static/<window>_<site>_1hour_best/`: the windowed dataset, the shared train and test split, and one `..._seed<N>_predictions.csv` per seed (`Time`, `windspeed_<w>`, `y_true`, `y_pred`, `proba_0`, `proba_1`). `eval_common.py` parses window length and site from it produces the figures.

A saved model is applied with `python sec5_1_experimental_setup/forecast_slick.py -m AutogluonModels/<model> -t <test.csv> -p <predictions.csv>`.

---

## Evaluation

Paper Section 5.2, code in [sec5_2_slick_prediction/](sec5_2_slick_prediction/).

```bash
python sec5_2_slick_prediction/plot_f1.py                       # Figures 5 and 6, plus metrics_f1_adjf1.csv
python sec5_2_slick_prediction/plot_pr_curves.py -d results/no_static -o results/figures
python sec5_2_slick_prediction/plot_per_month_f1_heatmap.py \
    results/no_static/12h_combined_1hour_best -o results/figures/per_month_f1.png
python sec5_2_slick_prediction/print_adj_slick_metric.py results/no_static/*_combined_1hour_best
```

The adjusted F1 counts a prediction as correct if a positive label exists within one hour (`--tolerance`), checked against the full, unwindowed ground truth.

---

## Shapley analysis

Paper Section 5.3, code in [sec5_3_meteorological_predictors/](sec5_3_meteorological_predictors/). Attributes each prediction of the 12 h combined model to the input features. The job scripts read the model from `AutogluonModels/<NAME>`, with `<NAME>` such as `12h_combined_1hour_best`, and the train and test split from `results/no_static/<NAME>/`. The grid deletes its models, so first run that configuration of the grid and then train and keep the model with `sbatch sec5_1_experimental_setup/train_keeper.sh` (edit `NAME`, `DATA` and `WINDOW` at its top).

**First-order values** (summary and dependence plots, Figures 9 to 11), computed on the cluster as 100 sample chunks:

```bash
sbatch sec5_3_meteorological_predictors/calculate_12h_shap.sh
ls results/shapley_values/12h_combined_1hour_best/shap_values_job_*.npy | wc -l   # must be 100
python sec5_3_meteorological_predictors/plot_shap.py -o results/shapley_values/12h_combined_1hour_best
```

`plot_shap.sh` runs the plotting as a SLURM job (use `--dependency=afterok`).

**Wind direction** (Figures 12 and 15). The sine and cosine components are recombined into one directional contribution through their pairwise interaction (50 chunks):

```bash
sbatch sec5_3_meteorological_predictors/calculate_12h_shapiq.sh
cd results/shapley_interactions/12h_combined_1hour_best
awk 'NR==1 || FNR>1' explained_samples_job_*.csv > explained_samples.csv && cd -

python -m sec5_3_meteorological_predictors.plot_wind_direction shap-binned \
    -d results/shapley_interactions/12h_combined_1hour_best -o results/figures
python -m sec5_3_meteorological_predictors.plot_wind_direction data-vs-shap \
    -d results/no_static/12h_combined_1hour_best/12h_combined_1hour_test.csv \
    -s results/shapley_interactions/12h_combined_1hour_best \
    -o results/figures/wdir_data_vs_shap.png --timestep 12
```

`plot_wind_direction.py` imports from `sec3_sensor_datasets/`, so run it as a module from the project root. Its `slick-rate` command draws the empirical rate alone, without Shapley output. The interaction analysis is hardcoded to the 12 h model.

---

## Labelling your own images

Paper Section 4.1, code in [sec4_1_label_creation/](sec4_1_label_creation/) and [sec3_sensor_datasets/preprocess.py](sec3_sensor_datasets/preprocess.py). For building the labels from your own camera images and classifier. It needs a per-site inventory `data/<site>_image_labels.csv` with a `file_name` column (image name without extension, `YYYYMMDD_HHMMSS`) and an empty `slick_tag` column.

```bash
# 1. Train the classifier. Each site folder under the dataset root holds train.csv and test.csv
#    (one row of image paths), list_of_labels.csv and the images. The held-out evaluation site
#    (AlphaVentus) must not be under the root.
python sec4_1_label_creation/train_image_model.py -d /path/to/images_dataset

# 2. Per site: weather table (see Data), then tag windy readings (> 5 m/s, adjusted wind speed) as
#    no-slick, attach each image to its nearest reading, clear the night (before 06:00 and from 20:00)
#    and list the images that still need a label in to_classify.csv
python sec3_sensor_datasets/preprocess.py align \
    -m build/$site/meteo.csv -i data/${site}_image_labels.csv -o build/$site

# 3. Classify the images in to_classify.csv (predictions.csv is written to the current directory)
mkdir -p to_classify
cut -d, -f1 build/$site/to_classify.csv | tail -n +2 | xargs -I{} cp /path/to/images/{}.jpg to_classify/
python sec4_1_label_creation/classify_images.py -m <model_dir> -d to_classify && mv predictions.csv build/$site/

# 4. Merge the labels back, aggregate to hours, and continue with `combine` as above
python sec3_sensor_datasets/preprocess.py merge -p build/$site/predictions.csv -o build/$site
python sec3_sensor_datasets/preprocess.py hourly \
    -d build/$site/sensors_with_labels.csv -o build/1hour_$site.csv

# Evaluate the classifier on a held-out site (Table 2)
python -m sec5_2_slick_prediction.evaluate_image_predictions -m <model_dir> -d <images_dataset> -p predictions.csv
```

[classify_images.sh](sec4_1_label_creation/classify_images.sh) applies the classifier on the cluster.

