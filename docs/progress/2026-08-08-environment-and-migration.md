# Environment and migration baseline

## Completed

- Connected to the SeetaCloud server over SSH and verified the RTX 4090 GPU.
- Created the `rs-agent` Conda environment under `/root/miniconda3/envs/`.
- Matched the official Change-Agent baseline by selecting Python 3.9.
- Confirmed the official MCI setup uses the Multi_change dependency list and
  installs Lagent separately.
- Designed a new independent `RS-Agent` repository layout.
- Prepared a curated legacy migration that excludes datasets, checkpoints,
  pretrained weights, caches, IDE files, and embedded API credentials.
- Added configuration examples for five caption generators and one judge.
- Recorded the paper-reproduction workflow and future domain extension boundary.

## Current problems

- The server's default Conda configuration contains an obsolete `pkgs/free`
  mirror; project commands must use `--override-channels` or a corrected channel
  configuration.
- Full cloning from GitHub timed out or returned 503 even with AutoDL network
  acceleration. The official README was retrieved successfully.
- The copied Change-Agent tree contains large model and metric assets that are
  intentionally not part of the Git repository.
- The legacy code contains expired credentials and hard-coded API endpoints;
  these must not be used as new configuration.
- The official MCI dependency stack is old and has not yet been installed or
  validated against the server driver.

## Next work after confirmation

1. Finalize and install the core API/evaluation dependencies.
2. Validate the MCI CUDA, PyTorch, MMCV, and MMSEG dependency combination.
3. Define typed pipeline state and immutable artifact schemas.
4. Implement the provider registry and OpenRouter-compatible async client.
5. Reimplement RS-CC with five parallel candidate models and separated judging.
6. Migrate the existing evaluation logic into tested modules.
7. Add RS-VQA templates, Knowledge Bridge ablations, and optional mask routing.
8. Integrate the canonical pipeline into the Streamlit/Lagent demo.

## Decisions still required

- Whether the five caption candidates must always come from five distinct models
  or whether controlled same-model sampling is allowed as an experiment profile.
- Whether candidate selection uses the same judge model as offline evaluation or
  a separately configured selector model.
- Which exact model IDs will form the first paper-reproduction profile after API
  keys are renewed.


## Baseline verification results

- Installed the editable package with core, evaluation, and development dependencies through the Tsinghua PyPI mirror.
- Verified `rs_agent==0.1.0` imports successfully.
- `pytest -q`: 1 passed.
- `pip check`: no broken requirements found.
- Python compilation completed for `src/` and `legacy/`.
- Ruff checks passed for `src/` and `tests/`.
- Server-side credential scan found no `sk-...` tokens.
- LangGraph, Streamlit, Lagent, and the MCI GPU dependency group remain intentionally uninstalled pending confirmation.