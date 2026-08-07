# RS-Agent

RS-Agent is a research-oriented pipeline for multi-agent remote-sensing image change understanding. Its first objective is to reproduce the workflow described in our ICASSP paper: multi-model caption enrichment, candidate selection, a caption knowledge bridge, remote-sensing VQA, optional change-mask evidence, and model evaluation.

This project is independently maintained and is based in part on [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Curated legacy source is kept under `legacy/` as a migration reference; all new implementation belongs under `src/rs_agent/`.

## Implemented

- Strict schemas for image-pair requests, caption candidates, model responses, VQA records, selections, and pipeline results.
- YAML model/provider configuration with API keys resolved from environment variables.
- An asynchronous OpenAI-compatible provider for OpenRouter, SiliconFlow, and similar APIs.
- Concurrency limits, timeout handling, bounded retries, non-retryable 4xx handling, usage metadata, and raw-response provenance.
- A write-once JSON artifact store with SHA-256 integrity verification.
- The paper evaluation semantics migrated from `TextAgent-LLMasJudge.py`.
- Separate selector and evaluator model configuration.
- Deterministic `C*` selection: highest score, judge choice as the tie-breaker, then stable label order.
- Compatibility export for the original full-result, best-caption, and model-mapping files.

## Not Yet Implemented

- Five-model RS-CC candidate generation from the original bi-temporal images and pair caption.
- RS-VQA question templates and user-question normalization.
- The Knowledge Bridge and complete Main-Agent workflow.
- Optional MCI mask integration.
- Streamlit/Lagent interactive demo integration.

## Environment

The core environment follows the Change-Agent Python 3.9 baseline.

```bash
conda env create -f environment.yml
conda activate rs-agent
pip install -e ".[dev,evaluation]"
pytest -q
```

The MCI model has a separate legacy CUDA/OpenMMLab dependency set in `requirements/mci-legacy.txt`. It is intentionally not installed as part of the lightweight API and evaluation environment.

## Configuration

Copy `.env.example` to `.env` and provide only the API keys that are needed. Never put credentials in YAML or Python source.

```bash
cp .env.example .env
```

Model IDs, provider endpoints, selector settings, and evaluator settings are defined in `configs/models.example.yaml`. The paper profile is documented in `configs/paper_reproduction.example.yaml`.

## Repository Layout

```text
src/rs_agent/core/          Typed contracts, configuration, and artifacts
src/rs_agent/providers/     External model API adapters
src/rs_agent/evaluation/    LLM-as-Judge parsing, selection, and exports
src/rs_agent/domains/       Domain-specific prompts, templates, and validation
legacy/                     Curated migration reference from the original project
configs/                    Reproducible experiment configuration examples
docs/progress/              Stage-by-stage implementation records
```

## Reproducibility Rules

- Paper reproduction runs do not use cross-sample memory.
- Every candidate and raw model response is retained.
- Runtime candidate selection and offline evaluation remain separately configured.
- Artifacts are write-once and integrity checked.
- API credentials are never stored in experiment artifacts or Git history.

## Acknowledgement

RS-Agent builds on ideas and selected code from [Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Please cite the original Change-Agent work when using the migrated MCI components.