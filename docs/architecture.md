# Architecture baseline

## Paper reproduction path

1. The Main Agent validates the image pair, original pair caption, task, and
   optional user question.
2. In the paper baseline, RS-CC sends only the original pair caption to five
   configured candidate models. Explicit enhancement profiles may allow selected
   multimodal models to read the original image pair as well.
3. Candidate outputs are preserved unchanged. A separate selector scores them
   and selects the highest-scoring caption `C*`.
4. The Knowledge Bridge passes only `C*` to RS-VQA when enabled.
5. RS-VQA receives only the original two images as visual input. Its text input
   contains a user question or a system question template and may contain `C*`.
6. The MCI mask is optional evidence and an optional final artifact. It is not
   silently injected into VLM image inputs.
7. The result composer returns captions, answers, mask references, scores, and
   provenance.
8. Offline evaluation reads saved artifacts and applies the migrated evaluation
   logic without rerunning inference.

## Extension boundary

Framework-neutral state, providers, artifact storage, and evaluation execution
belong to `core/`, `providers/`, and `evaluation/`. Remote-sensing prompts,
question templates, validation rules, and MCI adapters belong to
`domains/remote_sensing/`. A future medical domain should add another domain
package instead of modifying the orchestration engine.

## Entry points

The application service is the canonical entry point. Single-item, batch,
preflight, export, and Streamlit CLIs all call the same typed pipeline. Legacy
Lagent internals are retained only as migration reference and are not part of
the new core. The Streamlit adapter lives under `src/rs_agent/web/` and does not
contain provider or orchestration logic.

