# Multi-Level Anomaly Detection for Physiological Sound Analysis Using Autoencoders and Transformer Predictors

This is the code repository associated with the paper:

**Multi-Level Anomaly Detection for Physiological Sound Analysis Using Autoencoders and Transformer Predictors**

This repository provides a multi-level anomaly detection pipeline for bowel, lung, and heart sounds. The models are trained using healthy recordings and detect abnormalities at two levels:

- **Sequence level:** deviations in the order and timing of physiological sound events.
- **Pattern level:** deviations in the acoustic characteristics of individual sound patterns.

The pipeline is implemented using **PyTorch, Hugging Face Transformers, and Scikit-learn**.

## Features

**Sequence-level anomaly detection**

- Reconstruction using RNN and Transformer autoencoders.
- Next-event prediction using TCN and Transformer predictors.
- Probabilistic scoring of event labels and duration bins.
- Additional gap-duration modeling for bowel sounds.

**Pattern-level anomaly detection**

- LSTM, GRU, convolutional, and Transformer autoencoders.
- Raw waveform, MFCC, Wav2Vec2, and Audio Spectrogram Transformer representations.
- Gaussian negative log-likelihood for probabilistic reconstruction scoring.
- Independent analysis of each physiological sound pattern.

**Evaluation**

- Subject-level separation of training, validation, and test data.
- Training and threshold estimation using healthy data only.
- Recording- and subject-level evaluation.
- Disease-group analysis using linear classifiers.

## Installation

Install the required packages with:

```sh
pip install -r requirements.txt
```

Python 3.9 or later is recommended. A CUDA-enabled GPU can be used for faster training, but CPU execution is also supported.

## Usage

Run all analyses for all applications:

```sh
python main.py --mode all --application all
```

Run sequence reconstruction for bowel sounds:

```sh
python main.py --mode label --application bowel \
    --label-task reconstruction --label-model transformer
```

Run sequence prediction for lung sounds:

```sh
python main.py --mode label --application lung \
    --label-task prediction --label-model tcn
```

Run pattern-level analysis using AST features and a convolutional autoencoder:

```sh
python main.py --mode pattern --application bowel \
    --pattern-feature-backend ast --pattern-model conv
```

Run selected sound patterns only:

```sh
python main.py --mode pattern --application bowel --pattern-label SB MB
```

Use a custom annotation file:

```sh
python main.py --mode all --application bowel --csv /path/to/annotations.csv
```

The available arguments can be displayed using:

```sh
python main.py --help
```

Dataset paths, event labels, disease groups, and application-specific settings can be changed in `application_settings.py`. General model and evaluation settings are defined in `config.py`.

## Data Preparation

The annotation data should be provided in CSV format with the following columns:

| Column | Description |
|---|---|
| `Patient` | Subject identifier |
| `Wav_path` | Path to the audio recording |
| `Start` | Event start time in seconds |
| `End` | Event end time in seconds |
| `Label` | Physiological sound event or pattern label |

Example:

| Patient | Wav_path | Start | End | Label |
|---|---|---:|---:|---|
| 101 | `/path/recording_101.wav` | 1.20 | 1.42 | SB |
| 101 | `/path/recording_101.wav` | 1.42 | 2.10 | NONE |
| 102 | `/path/recording_102.wav` | 0.80 | 1.16 | MB |

Healthy and patient subject identifiers must be defined in `application_settings.py`.

## Evaluation Metrics

The pipeline reports:

- **AUROC:** overall discrimination between normal and anomalous samples across decision thresholds.
- **AUPRC:** precision--recall performance, particularly useful for imbalanced groups.
- **Accuracy:** proportion of correctly classified samples.
- **Balanced accuracy:** average of sensitivity and specificity.
- **Macro-F1:** average F1-score across classes with equal weight for each class.
- **MCC:** correlation between the predicted and reference labels.
- **Sensitivity:** proportion of patient samples detected as anomalous.
- **Specificity:** proportion of healthy samples identified as normal.

Confidence intervals are estimated using bootstrap resampling. The anomaly threshold is determined from the healthy validation scores according to the selected target false-positive rate.

## Outputs

Results are stored in the `outputs` directory and organized by application, analysis level, and model. The saved files include model checkpoints, anomaly scores, evaluation metrics, thresholds, disease-group results, and publication-ready plots.

## Citation

If you use this repository, please cite the corresponding paper:

**Multi-Level Anomaly Detection for Physiological Sound Analysis Using Autoencoders and Transformer Predictors**
