# Architecture baseline

## Paper protocols

The paper contains two RS-CC descriptions that must remain distinct in results:

1. The formal experiment implemented by `legacy/elvaluation/TextAgent-LLMasJudge.py` sends only the original pair caption to five candidate LLMs. `configs/rs_cc.paper.yaml` preserves this exact text-only protocol and the reported model identities.
2. The method prose and formula also describe image embeddings together with the original caption. `configs/rs_cc.method_image_text.yaml` implements this capability-aware interpretation with explicitly labeled substitute VLMs. It is an enhancement protocol, not an exact reproduction result.

Both protocols retain all five candidates. An independent Selector and Evaluator produce 1-10 scores, and deterministic resolution selects `C*`. The profile's protocol metadata is copied into experiment identity and result artifacts.

## Pipeline contract

1. The Main Agent validates the original image pair, original pair caption, task, and optional user question.
2. RS-CC runs only when the plan requires a new caption. A VQA-only run may instead load `C*` from a checksum-valid prior state.
3. Candidate outputs are preserved unchanged. A separate Selector and Evaluator score them, and deterministic resolution selects `C*`.
4. The Knowledge Bridge passes only `C*` and its provenance to RS-VQA when enabled.
5. RS-VQA receives only the original two images as visual input. Its text input contains a user question or a system template and may contain `C*`.
6. The MCI mask is optional evidence and an optional final artifact. It is never injected into VLM image inputs.
7. The result composer returns captions, answers, mask references, scores, conflicts, request telemetry, and provenance.
8. Offline evaluation reads checksum-verified artifacts without rerunning candidate inference.

## Memory policy

Paper profiles use run memory and session provenance only. Run memory consists of write-once artifacts, atomic batch state, and input/config/source/runtime fingerprints. Streamlit session memory retains the current provenance-linked `C*` so a follow-up executes RS-VQA without rerunning RS-CC. Cross-sample semantic retrieval is disabled because it would make results depend on sample order and history; a future domain may introduce it only behind a separate protocol and ablation.

## Extension boundary

Framework-neutral contracts, providers, artifact storage, experiment state, and evaluation belong to `core/`, `providers/`, `experiments/`, and `evaluation/`. Remote-sensing prompts, question templates, validation rules, and MCI adapters belong to `domains/remote_sensing/`. A future medical domain should add `domains/medical/` and domain-specific configs instead of modifying the orchestration engine.

## Entry points

The application service is the canonical entry point. Single-item, batch, preflight, export, and Streamlit CLIs all call the same typed pipeline. Legacy Lagent internals are retained only as migration reference and are not part of the new core. The Streamlit adapter lives under `src/rs_agent/web/` and contains no provider or orchestration logic.