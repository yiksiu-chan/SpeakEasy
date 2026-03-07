# Speak Easy

**Eliciting Harmful Jailbreaks from LLMs with Simple Interactions**

[![arXiv](https://img.shields.io/badge/arXiv-2502.04322-b31b1b.svg)](https://arxiv.org/abs/2502.04322)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ICML 2025](https://img.shields.io/badge/ICML-2025-purple.svg)](https://icml.cc/virtual/2025)
[![CI](https://github.com/yiksiu-chan/speakeasy/actions/workflows/ci.yml/badge.svg)](https://github.com/yiksiu-chan/speakeasy/actions/workflows/ci.yml)

<p align="center">
  <img src="./Speak_Easy.png" width="800px" alt="Speak Easy overview"/>
</p>

This repository contains the code for our paper [Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions](https://arxiv.org/abs/2502.04322). 
We show that simple interactions such as multi-step, multilingual querying can elicit sufficiently harmful jailbreaks from LLMs. We design a metric (HarmScore) to measure the actionability and informativeness of jailbreak responses, and a straightforward attack method (Speak Easy) that significantly increases the success of these exploits across multiple benchmarks.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Running Attacks](#running-attacks)
  - [Scoring Responses](#scoring-responses)
- [Results](#results)
- [Data](#data)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [Contact](#contact)

## Installation

```bash
# Basic install
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development install
pip install -e ".[dev]"
```

Optional extras can be installed individually as needed:

| Extra | Description |
|-------|-------------|
| `vllm` | vLLM + Ray for high-throughput local inference |
| `azure` | Azure Translator for multilingual translation |
| `google` | Google Cloud Translation API |
| `deep-translator` | Free Google Translate wrapper |
| `tap` | Tree of Attacks with Pruning (TAP) dependencies |
| `all` | All of the above |
| `dev` | Testing and linting tools |

## Quick Start

All data files must be JSON arrays of objects with `query` and `response` keys:

```json
[
  {"query": "example query", "response": "example response"},
  ...
]
```

A sample file is provided at [`data/sample_data.json`](data/sample_data.json).

### Score existing responses

```bash
speakeasy score \
    --data-dir data/sample_data.json \
    --save-dir results/scored.json \
    --scorer harmscore
```

### Run an attack framework

```bash
speakeasy run \
    --data-dir data/sample_data.json \
    --save-dir results/ \
    --model vllm:meta-llama/Llama-3.3-70B-Instruct \
    --frameworks baseline_dr speakeasy_dr
```

## Usage

### Running Attacks

```bash
speakeasy run \
    --data-dir <path-to-data.json> \
    --save-dir <output-directory> \
    --model <source:model_name> \
    --frameworks <framework_1> [<framework_2> ...]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--data-dir` | Path to input JSON file |
| `--save-dir` | Directory for output results (default: `results/`) |
| `--model` | Model in `source:name` format (e.g., `openai:gpt-4o`, `vllm:meta-llama/Llama-3.3-70B-Instruct`) |
| `--frameworks` | One or more frameworks to run |
| `--device` | GPU device index (default: `0`) |
| `-v` | Enable verbose logging |

**Available frameworks:**

| Framework | Description |
|-----------|-------------|
| `baseline_dr` | Direct Request baseline |
| `baseline_gcg` | GCG suffix attack baseline |
| `baseline_tap` | Tree of Attacks with Pruning baseline |
| `speakeasy_dr` | Speak Easy with Direct Request |
| `speakeasy_gcg` | Speak Easy with GCG suffixes |
| `speakeasy_tap` | Speak Easy with TAP |

### Scoring Responses

```bash
speakeasy score \
    --data-dir <path-to-data.json> \
    --save-dir <output-path.json> \
    --scorer <scorer_name>
```

**Available scorers:**

| Scorer | Description |
|--------|-------------|
| `gpt4judge` | GPT-4-based judge for binary jailbreak detection |
| `harmscore` | HarmScore metric combining actionability and informativeness |

## Results

ASR and HarmScore across four benchmarks on GPT-4o. Speak Easy consistently and significantly increases ASR over all baselines:

| Method | HarmBench | | AdvBench | | Sorry-Bench | | MedHarm | |
|--------|:---------:|:---------:|:--------:|:---------:|:-----------:|:---------:|:-------:|:---------:|
| | ASR | HS | ASR | HS | ASR | HS | ASR | HS |
| Direct Request | 0.125 | 0.099 | 0.010 | 0.010 | 0.158 | 0.236 | 0.073 | 0.376 |
| + Speak Easy | **0.560** | **0.779** | **0.682** | **0.724** | **0.604** | **0.793** | **0.373** | **0.740** |
| GCG | 0.095 | 0.105 | 0.010 | 0.017 | 0.178 | 0.198 | 0.058 | 0.301 |
| + Speak Easy | **0.586** | **0.816** | **0.694** | **0.660** | **0.587** | **0.807** | **0.393** | **0.882** |
| TAP | 0.575 | 0.402 | 0.946 | 0.558 | 0.678 | 0.509 | 0.529 | 0.608 |
| + Speak Easy | **0.985** | **0.912** | **0.994** | **0.930** | **0.933** | **0.919** | **0.950** | **0.887** |

Results on Qwen2-72B and Llama-3.1-70B show similar trends. For full results and analysis, see [the paper](https://arxiv.org/abs/2502.04322).

## Data

### Sample Data

A sample input file is provided at [`data/sample_data.json`](data/sample_data.json).

### HarmScore Training Data

The `data/` directory includes the DPO preference datasets used to train the HarmScore reward models. Each dataset contains chosen/rejected conversation pairs in Parquet format.

| Dataset | Examples | Description |
|---------|----------|-------------|
| `actionable_dataset/` | 55,475 | Full training set for the actionable evaluator model |
| `informative_dataset/` | 55,475 | Full training set for the informative evaluator model |
| `actionable_dataset_select/` | 27,737 | Subset for training the actionable response selection model |
| `informative_dataset_select/` | 27,737 | Subset for training the informative response selection model |
| `informative_dataset_eval/` | 27,738 | Held-out subset for evaluator validation |

The **eval** models (`narutatsuri/evaluation-actionable`, `narutatsuri/evaluation-informative`) are used post-hoc to score final outputs. The **select** models are used mid-pipeline during the Speak Easy attack to choose the most harmful multilingual response. The full and subset splits are separated to avoid data leakage between selection and evaluation.

## Project Structure

```
speakeasy/
├── src/speakeasy/
│   ├── backends/          # Model inference backends (OpenAI, vLLM)
│   ├── config/            # Pydantic configuration classes and defaults
│   ├── evaluators/        # Response evaluation (GPT-4 Judge, HarmScore)
│   ├── frameworks/        # Attack frameworks (DR, GCG, TAP, Speak Easy)
│   │   └── tap/           # TAP sub-library (conversers, judges, prompts)
│   ├── scoring/           # Response selection models
│   ├── translation/       # Multilingual translation (Azure, Google, Deep)
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── constants.py       # Shared constants
│   └── utils.py           # Utility functions
├── tests/                 # Unit tests
├── data/                  # Sample data and HarmScore training datasets
├── pyproject.toml         # Package metadata and dependencies
├── Makefile               # Development shortcuts
└── README.md
```

## Configuration

Speak Easy uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration. All settings can be overridden via environment variables with the `SPEAKEASY_` prefix:

```bash
# API keys
export SPEAKEASY_OPENAI_API_KEY="sk-..."
export SPEAKEASY_HF_TOKEN="hf_..."

# Azure Translation
export SPEAKEASY_AZURE_TRANSLATOR_KEY="..."
export SPEAKEASY_AZURE_TRANSLATOR_REGION="eastus"

# Google Cloud Translation
export SPEAKEASY_GOOGLE_PROJECT_ID="my-project"
export SPEAKEASY_GOOGLE_CREDENTIALS_PATH="/path/to/creds.json"
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{chan2025speakeasy,
  title={Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions},
  author={Yik Siu Chan and Narutatsu Ri and Yuxin Xiao and Marzyeh Ghassemi},
  year={2025},
  url={https://arxiv.org/abs/2502.04322},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Contact

Questions about the code or the paper can be directed to:

- **Narutatsu Ri** — [nr3764@princeton.edu](mailto:nr3764@princeton.edu)
- **Yik Siu Chan** — [yik_siu_chan@brown.edu](mailto:yik_siu_chan@brown.edu)

If you encounter any issues, please [open an issue](https://github.com/yiksiu-chan/speakeasy/issues).
