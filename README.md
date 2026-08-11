# RS-Agent

RS-Agent is a research-oriented pipeline for multi-agent remote-sensing image change understanding. Its first objective is to reproduce and extend the workflow described in our ICASSP paper: multi-model caption enrichment, LLM-as-Judge evaluation, a selected-caption knowledge bridge, remote-sensing VQA, optional change-mask evidence, and model evaluation.

This project is independently maintained and is based in part on [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Curated legacy source is kept under `legacy/` as a migration reference; new implementation belongs under `src/rs_agent/` and `scripts/`.

## Implemented

- A text-only RS-CC stage that accepts one Change-Agent caption per image pair.
- Five concurrent RS-CC candidates, independent selector and evaluator roles, and deterministic `C*` selection.
- A separate five-model RS-VQA stage that reads only the original before/after images.
- Preset, user, and hybrid question resolution for change summary, presence, structure, buildings, roads, vegetation, water, and location.
- Image-grounded VQA selection and 1-10 scoring that preserve the paper's LLM-as-Judge logic.
- A minimal Knowledge Bridge that passes only selected `C*` and its provenance.
- An auditable Main Agent planner for caption, VQA, combined, and no-Knowledge-Bridge ablation paths.
- Optional single-channel mask evidence with class statistics, connected components, and bounding boxes.
- A domain-neutral Evidence Bundle that records conservative cross-source conflicts without overriding model outputs.
- OpenRouter and SiliconFlow through one OpenAI-compatible provider interface.
- Per-model request options, bounded retries, preserved OpenRouter reasoning details, and nested error detection.
- Write-once JSON artifacts with SHA-256 integrity verification.
- Reproducible Change-Agent caption and three-class mask inference for 100 LEVIR-MCI test pairs.

## Pipeline Contract

The Main Agent maps each explicit task to the smallest required workflow:

| Task | Stages |
| --- | --- |
| `caption` | `RS-CC` |
| `vqa` with Knowledge Bridge | `RS-CC -> C* -> RS-VQA` |
| `vqa` without Knowledge Bridge | `RS-VQA` |
| `combined` | `RS-CC -> C* -> RS-VQA` |
| any task with a mask | requested stages plus structured mask evidence |

The complete path is:

```text
Main Agent plan
  -> original Change-Agent caption
  -> five text-only RS-CC candidates
  -> selector + evaluator -> selected C*
  -> optional Knowledge Bridge
  -> original before/after images + question + optional C*
  -> five RS-VQA candidates
  -> image-grounded selector + evaluator
  -> optional external/Change-Agent mask statistics
  -> Evidence Bundle + immutable result artifacts
```

RS-CC never receives images. RS-VQA visual messages contain only the original bi-temporal images. Masks are parsed separately and never enter VLM messages. `C*` is auxiliary text and may be corrected when it conflicts with visible evidence.

The paper describes the Main Agent as a task coordinator, not as an additional LLM answer-fusion model. Task routing and `C*` transfer are therefore part of the paper-aligned baseline. The deterministic Evidence Bundle is explicitly marked as an engineering enhancement; it records conflicts but does not silently create a new experimental verdict.

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

Operational outputs must not be reported as paper reproduction results. Paper profiles never silently substitute a blocked or retired model. Credentials remain in `.env` and never belong in YAML, source, artifacts, or Git history.

## End-to-End Usage

Validate a VQA run without the Knowledge Bridge and without API calls:

```bash
rs-agent-run \
  --cc-config configs/rs_cc.smoke.yaml \
  --vqa-config configs/rs_vqa.smoke.yaml \
  --image-a /path/to/before.png \
  --image-b /path/to/after.png \
  --caption "the scene is the same as before" \
  --task-type vqa \
  --without-knowledge-bridge \
  --dry-run
```

Run caption enrichment with an optional predicted mask:

```bash
rs-agent-run \
  --cc-config configs/rs_cc.smoke.yaml \
  --vqa-config configs/rs_vqa.smoke.yaml \
  --env-file .env \
  --image-a /path/to/before.png \
  --image-b /path/to/after.png \
  --caption "a new road appears in the middle of the scene" \
  --task-type caption \
  --mask /path/to/predicted-mask.png \
  --mask-source predicted \
  --item-id test_000068 \
  --artifact-dir artifacts
```

Use `--question` one or more times for user questions. If no question is supplied to a VQA task, the configured preset question set is used. The individual stages remain available as `rs-agent-cc` and `rs-agent-vqa`.

## Change-Agent Batch Inference

```bash
conda activate rs-agent-mci
python scripts/generate_change_agent_captions.py \
  --source-root /root/autodl-tmp/Change-Agent-upstream/Multi_change \
  --dataset-root /root/autodl-tmp/datasets/LEVIR-MCI/LEVIR-MCI-dataset \
  --checkpoint /root/autodl-tmp/models/change-agent/MCI_model.pth \
  --output /root/autodl-tmp/rs-agent-data/change-agent/levir_mci_test_100_with_masks.jsonl \
  --mask-output-dir /root/autodl-tmp/rs-agent-data/change-agent/masks/test-100 \
  --split test \
  --limit 100
```

The mask output is the `argmax` of the MCI model's three-class logits: background, road change, and building change.

## Repository Layout

```text
src/rs_agent/applications/    CLI and later web entry points
src/rs_agent/core/            Typed contracts, configuration, and artifacts
src/rs_agent/providers/       External model API adapters
src/rs_agent/evaluation/      LLM-as-Judge evaluation and parsing
src/rs_agent/domains/         Domain-specific agents, prompts, and adapters
src/rs_agent/orchestration/   Main Agent planning, evidence, and pipeline composition
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
- Ground-truth masks must be labeled as `ground_truth` and are not inference evidence.
- Artifacts are write-once and integrity checked.
- API credentials never enter Git history or experiment artifacts.

## Next Milestones

- Add batch resume, cache, provider preflight, and evaluation exports.
- Add quantitative caption/VQA/mask consistency reports while retaining the existing Judge logic.
- Design an optional, separately reported synthesis/arbitration Agent after the paper baseline is frozen.
- Integrate the Streamlit demo after the experimental evidence pipeline is stable.

## Acknowledgement

RS-Agent builds on ideas and selected code from [Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Please cite the original Change-Agent work when using the migrated MCI components.
