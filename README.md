# RS-Agent

RS-Agent is a research-oriented pipeline for multi-agent remote-sensing image change understanding. Its first objective is to reproduce and extend the workflow described in our ICASSP paper: multi-model caption enrichment, LLM-as-Judge evaluation, a selected-caption knowledge bridge, remote-sensing VQA, optional change-mask evidence, and model evaluation.

This project is independently maintained and is based in part on [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Curated legacy source is kept under `legacy/` as a migration reference; new implementation belongs under `src/rs_agent/` and `scripts/`.

## Implemented

- A paper-aligned text-only RS-CC baseline plus a capability-aware enhancement where selected VLMs can read the original image pair and reference caption.
- Five concurrent RS-CC candidates, independent selector and evaluator roles, and deterministic `C*` selection.
- A separate five-model RS-VQA stage that reads only the original before/after images.
- Preset, user, and hybrid question resolution for change summary, presence, structure, buildings, roads, vegetation, water, and location.
- Image-grounded VQA selection and 1-10 scoring that preserve the paper's LLM-as-Judge logic.
- A minimal Knowledge Bridge that passes only selected `C*` and its provenance.
- An auditable Main Agent planner for caption, VQA, combined, and no-Knowledge-Bridge ablation paths.
- Optional single-channel mask evidence with class statistics, connected components, and bounding boxes.
- A domain-neutral Evidence Bundle that records conservative cross-source conflicts without overriding model outputs.
- Resumable JSONL batch experiments with bounded concurrency, atomic state, retry controls, and opt-in verified result caching.
- Content-derived experiment identities covering inputs, image/mask content, configs, source code, runtime versions, and run options.
- Credential-safe provider preflight with optional `/models` endpoint and model-visibility checks.
- Checksum-verified exports for per-item results, complete candidate ledgers, Judge scores, LEVIR-MCI reference metrics, request telemetry, failures, and model summaries.
- Explicit `paper`, `operational`, and `enhancement` experiment protocols embedded in configs and artifacts.
- OpenRouter and SiliconFlow through one OpenAI-compatible provider interface.
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
  -> five RS-CC candidates (text-only in the paper baseline)
  -> selector + evaluator -> selected C*
  -> optional Knowledge Bridge
  -> original before/after images + question + optional C*
  -> five RS-VQA candidates
  -> image-grounded selector + evaluator
  -> optional external/Change-Agent mask statistics
  -> Evidence Bundle + immutable result artifacts
```

Paper-track RS-CC never receives images. The separately labeled enhancement profile may mix text-only candidates with image-text candidates; only models configured with `input_mode: image_text` receive the original image pair. RS-VQA visual messages contain only the original bi-temporal images. Masks are parsed separately and never enter VLM messages. `C*` is auxiliary text and may be corrected when it conflicts with visible evidence. Image payloads are never persisted; artifacts retain only image hashes and actual input modes.

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

Adapted profiles use currently available models and keep substitutions explicit:

- `configs/rs_cc.adapted.yaml`: enhancement profile with two text-only and three image-text candidates.
- `configs/rs_vqa.adapted.yaml`: operational profile with five currently available multimodal models.

Operational outputs must not be reported as paper reproduction results. Paper profiles never silently substitute a blocked or retired model. Credentials remain in `.env` and never belong in YAML, source, artifacts, state, cache indexes, or Git history.

Check credentials and configuration without network requests:

```bash
rs-agent-preflight \
  --cc-config configs/rs_cc.smoke.yaml \
  --vqa-config configs/rs_vqa.smoke.yaml \
  --env-file .env
```

Add `--network` to query each provider's `/models` endpoint and report configured model visibility. This check does not print API keys and does not send inference prompts.

## Single-Item Usage

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

Use `--question` one or more times for user questions. If no question is supplied to a VQA task, the configured preset question set is used. The individual stages remain available as `rs-agent-cc` and `rs-agent-vqa`.

## Batch Experiments

The batch input is JSONL with one item per line:

```json
{"item_id":"test_000001","original_caption":"the scene is the same as before","image_a":"/path/A.png","image_b":"/path/B.png","predicted_mask":"/path/mask.png"}
```

Validate all input files, both configs, and the experiment identity without API calls:

```bash
rs-agent-batch \
  --input /path/to/input.jsonl \
  --batch-id levir-mci-smoke \
  --cc-config configs/rs_cc.smoke.yaml \
  --vqa-config configs/rs_vqa.smoke.yaml \
  --task-type combined \
  --dry-run
```

Run or resume a batch:

```bash
rs-agent-batch \
  --input /path/to/input.jsonl \
  --batch-id levir-mci-smoke \
  --cc-config configs/rs_cc.smoke.yaml \
  --vqa-config configs/rs_vqa.smoke.yaml \
  --env-file .env \
  --task-type combined \
  --concurrency 2 \
  --artifact-dir /path/to/artifacts \
  --state-dir /path/to/batch-state \
  --cache-dir /path/to/cache
```

Reuse the exact command and `batch-id` to resume pending items. Use `--retry-failed` to retry failed items. `--max-items N` intentionally stops after at most `N` eligible items and leaves the batch in `partial` state.

Cross-batch caching is disabled unless `--cache-dir` is supplied. A cache hit requires the same sample content, captions/questions, configs, source tree, runtime versions, and inference options, plus a checksum-valid result artifact. Paper experiments should preserve the generated state and identity alongside their exports.

## Evaluation Exports

```bash
rs-agent-export \
  --state /path/to/batch-state/levir-mci-smoke/state.json \
  --output-dir /path/to/exports/levir-mci-smoke \
  --references /path/to/levir_mci_test_references.jsonl
```

Exports are created only in an empty directory and include:

- `summary.json`: batch status, conflicts, caption-mask consistency, generation failures, supplementary caption metrics, metric version, and export-source hash.
- `experiment_identity.json`: exact input/config/source/runtime identity.
- `items.jsonl`: selected captions, answers, masks, evidence consensus, and result provenance.
- `failures.jsonl`: pending or failed batch items and errors.
- `caption_scores.csv` and `vqa_scores.csv`: every configured candidate, including failed generations with empty scores and preserved errors.
- `model_summary.csv`: Judge statistics, selection counts, success rate, retries, latency, and token use by stage, role, model, and input mode.
- `request_telemetry.csv`: every generator and Judge request with outcomes, HTTP statuses, retries, latency, and available token usage.
- `caption_reference_metrics.csv`: selected-caption BLEU-1, unsmoothed BLEU-4, ROUGE-L, and change-flag agreement against normalized LEVIR-MCI references.

Every referenced artifact is checksum-verified before export. A model that failed to generate is kept in the ledger and is never treated as a score of zero. LLM-as-Judge remains the primary method; reference metrics are supplementary and their definitions are versioned.

## Phase 8 Pilot

A balanced 10-item LEVIR-MCI combined pilot (five change and five no-change samples) completed with 10/10 successful items, 50/50 RS-CC generations, and 50/50 RS-VQA generations. Across RS-CC candidates, text-only inputs averaged 8.45 Judge points and were selected 8/10 times; image-text inputs averaged 4.50 and were selected 2/10 times. The image-text models were especially vulnerable to treating seasonal appearance as change, so this capability remains an enhancement and ablation rather than a baseline replacement.

The selected captions achieved BLEU-1 0.3897, unsmoothed sentence BLEU-4 0.0141, ROUGE-L 0.3190, and change-flag accuracy 1.0 on this small pilot. These numbers validate the evaluation path; they are not paper-scale results.

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
src/rs_agent/applications/    Single-item, batch, preflight, and export CLIs
src/rs_agent/core/            Typed contracts, configuration, and artifacts
src/rs_agent/experiments/     Batch identity, state, cache, and orchestration
src/rs_agent/providers/       External model adapters and provider preflight
src/rs_agent/evaluation/      LLM-as-Judge logic and verified result exports
src/rs_agent/domains/         Domain-specific agents, prompts, and adapters
src/rs_agent/orchestration/   Main Agent planning, evidence, and pipeline composition
scripts/                      Reproducible data-generation utilities
legacy/                       Curated migration reference
configs/                      Paper and operational profiles
docs/progress/                Chinese implementation records
```

## Reproducibility Rules

- Paper runs do not use cross-sample semantic memory.
- Every candidate, raw response, score, selection, error, and model ID is retained.
- Paper, operational, and enhancement profiles are explicitly labeled and never silently mixed in one result identity.
- VQA image payloads are not persisted; image hashes are recorded instead.
- Ground-truth masks must be labeled as `ground_truth` and are not inference evidence.
- Scientific artifacts are write-once and integrity checked; mutable batch state is stored separately and updated atomically.
- API credentials never enter Git history, experiment artifacts, state, exports, or cache indexes.

## Next Milestones

- Add larger paper-profile pilot runs and dataset-level metrics against available references.
- Freeze paper baseline schemas and add explicit baseline/operational/enhancement run labels.
- Design an optional, separately reported synthesis/arbitration Agent after the paper baseline is frozen.
- Integrate the Streamlit demo after the experimental evidence pipeline is stable.

## Acknowledgement

RS-Agent builds on ideas and selected code from [Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent). Please cite the original Change-Agent work when using the migrated MCI components.
