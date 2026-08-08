# RS-Agent

RS-Agent is a research-oriented pipeline for multi-agent remote-sensing image change understanding. Its first objective is to reproduce and extend the workflow described in our ICASSP paper: multi-model caption enrichment, LLM-as-Judge evaluation, a selected-caption knowledge bridge, remote-sensing VQA, optional change-mask evidence, and model evaluation.

This project is independently maintained and is based in part on [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Curated legacy source is kept under `legacy/` as a migration reference; new implementation belongs under `src/rs_agent/` and `scripts/`.

## Implemented

- A text-only RS-CC stage that accepts one Change-Agent caption per image pair.
- Five concurrent RS-CC candidates, independent selector and evaluator roles, and deterministic `C*` selection.
- A separate five-model RS-VQA stage that reads the original before/after images.
- Preset, user, and hybrid question resolution for change summary, presence, structure, buildings, roads, vegetation, water, and location.
- Image-grounded VQA selection and 1-10 scoring that preserve the paper's LLM-as-Judge logic.
- A minimal Knowledge Bridge that passes only selected `C*` and its provenance.
- An end-to-end `RS-CC -> C* -> RS-VQA` application entry point.
- OpenRouter and SiliconFlow through one OpenAI-compatible provider interface.
- Per-model request options, including OpenRouter reasoning parameters and preserved `reasoning_details` for follow-up turns.
- Bounded retries and detection of errors nested inside HTTP 200 completion choices.
- Write-once JSON artifacts with SHA-256 integrity verification.
- Reproducible Change-Agent inference for 100 LEVIR-MCI test pairs.

## Pipeline Contract

The base workflow is:

```text
original Change-Agent caption
  -> five text-only RS-CC candidates
  -> selector + evaluator
  -> selected C*
  -> Knowledge Bridge
  -> original before/after images + question + auxiliary C*
  -> five RS-VQA candidates
  -> image-grounded selector + evaluator
  -> selected answers and immutable evidence artifacts
```

RS-CC never receives images. RS-VQA visual inputs contain only the original bi-temporal images; masks are reserved for a later structured-evidence stage. `C*` is auxiliary text and may be corrected when it conflicts with visible evidence.

## Environments

```bash
conda env create -f environment.yml
conda activate rs-agent
pip install -e ".[dev,evaluation]"
pytest -q
```

Change-Agent MCI inference remains isolated in `rs-agent-mci`; it does not add legacy Torch dependencies to the API environment.

## Provider Profiles

Paper profiles preserve the experimental model roles:

- `configs/rs_cc.paper.yaml`
- `configs/rs_vqa.paper.yaml`

Operational profiles use models reachable from the current server and exist only to validate orchestration:

- `configs/rs_cc.smoke.yaml`
- `configs/rs_vqa.smoke.yaml`

Operational outputs must not be reported as paper reproduction results. Paper profiles never silently substitute a blocked or retired model.

The current AutoDL egress receives region-related HTTP 403 responses for OpenRouter-hosted OpenAI, Google, and Anthropic models, while the same key works locally. This does not block operational testing. A compliant proxy can be supplied through `HTTPS_PROXY`, or provider `base_url` can point to an OpenAI-compatible gateway. Credentials remain in `.env` and never belong in YAML, source, artifacts, or Git history.

## End-to-End Usage

Validate the complete operational workflow without API calls:

```bash
rs-agent-run \
  --cc-config configs/rs_cc.smoke.yaml \
  --vqa-config configs/rs_vqa.smoke.yaml \
  --image-a /path/to/before.png \
  --image-b /path/to/after.png \
  --caption "the scene is the same as before" \
  --dry-run
```

Run one complete item:

```bash
rs-agent-run \
  --cc-config configs/rs_cc.smoke.yaml \
  --vqa-config configs/rs_vqa.smoke.yaml \
  --env-file .env \
  --image-a /path/to/before.png \
  --image-b /path/to/after.png \
  --caption "the scene is the same as before" \
  --question "Did any meaningful structural change occur?" \
  --item-id test_000001 \
  --artifact-dir artifacts
```

The individual stages remain available as `rs-agent-cc` and `rs-agent-vqa`.

## Change-Agent Batch Inference

```bash
conda activate rs-agent-mci
python scripts/generate_change_agent_captions.py \
  --source-root /root/autodl-tmp/Change-Agent-upstream/Multi_change \
  --dataset-root /root/autodl-tmp/datasets/LEVIR-MCI/LEVIR-MCI-dataset \
  --checkpoint /root/autodl-tmp/models/change-agent/MCI_model.pth \
  --output /root/autodl-tmp/rs-agent-data/change-agent/levir_mci_test_100.jsonl \
  --split test \
  --limit 100
```

## Repository Layout

```text
src/rs_agent/applications/    CLI and later web entry points
src/rs_agent/core/            Typed contracts, configuration, and artifacts
src/rs_agent/providers/       External model API adapters
src/rs_agent/evaluation/      LLM-as-Judge evaluation and parsing
src/rs_agent/domains/         Domain-specific agents, prompts, and adapters
src/rs_agent/orchestration/   Stage and end-to-end pipeline composition
scripts/                      Reproducible data-generation utilities
legacy/                       Curated migration reference
configs/                      Paper and operational profiles
docs/progress/                Chinese implementation records
```

## Reproducibility Rules

- Paper runs do not use cross-sample memory.
- Every candidate, raw response, score, selection, error, and model ID is retained.
- Paper and operational profiles are never mixed in one result identity.
- VQA image payloads are not persisted; image hashes are recorded instead.
- Artifacts are write-once and integrity checked.
- API credentials never enter Git history or experiment artifacts.

## Next Milestones

- Add Main-Agent conflict arbitration across `C*`, VQA, and optional mask evidence.
- Add optional mask statistics without adding masks to VLM image inputs.
- Add batch resume, cache, provider preflight, and evaluation exports.
- Integrate the Streamlit demo after the experimental evidence pipeline is stable.

## Acknowledgement

RS-Agent builds on ideas and selected code from [Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Please cite the original Change-Agent work when using the migrated MCI components.
