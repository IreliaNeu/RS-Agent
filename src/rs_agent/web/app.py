"""Streamlit application for interactive RS-Agent analysis."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from rs_agent.core.schemas import TaskType
from rs_agent.web.io import DemoSample, load_demo_manifest, save_uploaded_image
from rs_agent.web.service import DemoPipelineRequest, DemoRunView, run_demo_pipeline

TASK_MODES = {
    "Full analysis": TaskType.COMBINED,
    "Caption": TaskType.CAPTION_ENRICHMENT,
    "VQA": TaskType.VQA,
}
PROFILE_PATHS = {
    "Adapted mixed": ("configs/rs_cc.adapted.yaml", "configs/rs_vqa.adapted.yaml"),
    "Operational": ("configs/rs_cc.smoke.yaml", "configs/rs_vqa.smoke.yaml"),
    "Paper baseline": ("configs/rs_cc.paper.yaml", "configs/rs_vqa.paper.yaml"),
}


def _session_directory() -> Path:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
    root = Path(os.getenv("RS_AGENT_WEB_WORKDIR", ".rs-agent-web"))
    return root.resolve() / st.session_state.session_id


@st.cache_data(show_spinner=False)
def _manifest(path: str) -> list[DemoSample]:
    return load_demo_manifest(Path(path))


def _run_id(item_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_item = "".join(character if character.isalnum() else "_" for character in item_id)
    return "web-{}-{}-{}".format(safe_item[:36], timestamp, uuid4().hex[:8])


def _candidate_rows(view: DemoRunView) -> list[dict]:
    return [
        {
            "Selected": candidate.selected,
            "Label": candidate.label,
            "Score": candidate.score,
            "Model": candidate.model_name,
            "Input": candidate.input_mode,
            "Mode": candidate.generation_mode,
            "Caption": candidate.text,
            "Error": candidate.error,
        }
        for candidate in view.caption_candidates
    ]


def _vqa_rows(view: DemoRunView) -> list[dict]:
    return [
        {
            "Selected": candidate.selected,
            "Question": candidate.question,
            "Label": candidate.label,
            "Score": candidate.score,
            "Model": candidate.model_name,
            "Answer": candidate.text,
            "Error": candidate.error,
        }
        for candidate in view.vqa_candidates
    ]


def _render_view(view: DemoRunView, mask_path: Optional[Path]) -> None:
    status_columns = st.columns(4)
    status_columns[0].metric("Stages", len(view.stages))
    status_columns[1].metric("Caption candidates", len(view.caption_candidates))
    status_columns[2].metric("VQA candidates", len(view.vqa_candidates))
    status_columns[3].metric(
        "Consensus", str(view.evidence.get("consensus", "unknown")).replace("_", " ")
    )

    summary_tab, caption_tab, vqa_tab, evidence_tab, artifact_tab = st.tabs(
        ["Summary", "RS-CC", "RS-VQA", "Evidence", "Artifacts"]
    )
    with summary_tab:
        st.subheader("Selected change description")
        st.write(view.selected_caption or "No RS-CC stage in this run.")
        if view.knowledge_caption:
            st.caption("Knowledge Bridge C*")
            st.write(view.knowledge_caption)
        if view.selected_answers:
            st.subheader("Selected answers")
            for answer in view.selected_answers:
                with st.chat_message("assistant"):
                    st.markdown("**{}**".format(answer.question))
                    st.write(answer.answer)
                    st.caption(answer.model_name)
        if mask_path is not None and view.mask_summary is not None:
            left, right = st.columns([1, 2])
            with left:
                st.image(str(mask_path), caption="Change mask", width="stretch")
            with right:
                st.metric(
                    "Changed area",
                    "{:.2%}".format(view.mask_summary["changed_ratio"]),
                )
                st.metric(
                    "Components", view.mask_summary["significant_component_count"]
                )
    with caption_tab:
        if view.caption_candidates:
            st.dataframe(
                _candidate_rows(view),
                hide_index=True,
                width="stretch",
                column_config={
                    "Selected": st.column_config.CheckboxColumn(width="small"),
                    "Label": st.column_config.TextColumn(width="small"),
                    "Score": st.column_config.NumberColumn(width="small"),
                    "Caption": st.column_config.TextColumn(width="large"),
                },
            )
        else:
            st.info("This run did not execute RS-CC.")
    with vqa_tab:
        if view.vqa_candidates:
            st.dataframe(
                _vqa_rows(view),
                hide_index=True,
                width="stretch",
                column_config={
                    "Selected": st.column_config.CheckboxColumn(width="small"),
                    "Label": st.column_config.TextColumn(width="small"),
                    "Score": st.column_config.NumberColumn(width="small"),
                    "Answer": st.column_config.TextColumn(width="large"),
                },
            )
        else:
            st.info("This run did not execute RS-VQA.")
    with evidence_tab:
        st.dataframe(
            view.evidence.get("claims", []),
            hide_index=True,
            width="stretch",
        )
        conflicts = view.evidence.get("conflicts", [])
        if conflicts:
            st.warning("{} evidence conflict(s) detected.".format(len(conflicts)))
            st.dataframe(conflicts, hide_index=True, width="stretch")
        else:
            st.success("No cross-source conflict detected.")
    with artifact_tab:
        st.json(view.artifacts, expanded=False)
        st.caption("Run ID: {}".format(view.run_id))


def _resolve_inputs(source_mode: str, manifest_path: str):
    if source_mode == "Dataset sample":
        samples = _manifest(manifest_path)
        selected_id = st.selectbox(
            "Sample",
            [sample.item_id for sample in samples],
            key="sample_id",
        )
        sample = next(item for item in samples if item.item_id == selected_id)
        image_columns = st.columns(2)
        image_columns[0].image(str(sample.image_a), caption="Before", width="stretch")
        image_columns[1].image(str(sample.image_b), caption="After", width="stretch")
        return (
            sample.item_id,
            sample.image_a,
            sample.image_b,
            sample.predicted_mask,
            sample.original_caption,
        )

    upload_columns = st.columns(2)
    before_upload = upload_columns[0].file_uploader(
        "Before image", type=["png", "jpg", "jpeg"], key="before_upload"
    )
    after_upload = upload_columns[1].file_uploader(
        "After image", type=["png", "jpg", "jpeg"], key="after_upload"
    )
    mask_upload = st.file_uploader(
        "Change mask", type=["png"], key="mask_upload"
    )
    session_dir = _session_directory() / "uploads"
    before_path = (
        save_uploaded_image(
            before_upload.getvalue(), before_upload.name, session_dir, kind="rgb"
        )
        if before_upload
        else None
    )
    after_path = (
        save_uploaded_image(
            after_upload.getvalue(), after_upload.name, session_dir, kind="rgb"
        )
        if after_upload
        else None
    )
    mask_path = (
        save_uploaded_image(mask_upload.getvalue(), mask_upload.name, session_dir, kind="mask")
        if mask_upload
        else None
    )
    if before_path and after_path:
        preview_columns = st.columns(2)
        preview_columns[0].image(str(before_path), caption="Before", width="stretch")
        preview_columns[1].image(str(after_path), caption="After", width="stretch")
    return "uploaded_pair", before_path, after_path, mask_path, ""


def render_app() -> None:
    st.set_page_config(page_title="RS-Agent", layout="wide")
    st.markdown(
        """
        <style>
        :root { --rs-ink: #172026; --rs-teal: #087f8c; --rs-coral: #d95d39; }
        .stApp { color: var(--rs-ink); }
        h1, h2, h3 { letter-spacing: 0 !important; }
        [data-testid="stMetric"] { border: 1px solid #d7dee2; border-radius: 6px; padding: 10px; }
        [data-testid="stSidebar"] { border-right: 1px solid #d7dee2; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("RS-Agent")
    st.caption("Remote-sensing change understanding")

    repo_root = Path(os.getenv("RS_AGENT_REPO_ROOT", Path.cwd())).resolve()
    default_manifest = os.getenv(
        "RS_AGENT_DEMO_MANIFEST",
        "/root/autodl-tmp/rs-agent-data/change-agent/phase10-20260820/"
        "levir_mci_test_100_with_masks.jsonl",
    )
    with st.sidebar:
        st.header("Run")
        profile = st.selectbox("Profile", list(PROFILE_PATHS), index=0)
        task_label = st.radio("Task", list(TASK_MODES), horizontal=True)
        use_bridge = st.toggle("Knowledge Bridge", value=True)
        source_options = ["Upload pair"]
        if Path(default_manifest).is_file():
            source_options.insert(0, "Dataset sample")
        source_mode = st.radio("Input", source_options)
        with st.expander("Paths"):
            manifest_path = st.text_input("Manifest", value=default_manifest)
            env_file = st.text_input("Environment", value=str(repo_root / ".env"))
            artifact_dir = st.text_input(
                "Artifacts", value=str(repo_root / "artifacts" / "web")
            )

    try:
        item_id, image_a, image_b, mask_path, default_caption = _resolve_inputs(
            source_mode, manifest_path
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    original_caption = st.text_area(
        "Reference caption",
        value=default_caption,
        height=80,
    )
    question = st.text_input("Question", placeholder="Leave empty to use preset questions")
    mask_enabled = st.toggle("Use mask evidence", value=mask_path is not None)
    run_clicked = st.button("Run analysis", type="primary")

    if run_clicked:
        if image_a is None or image_b is None:
            st.error("Both before and after images are required.")
        elif not original_caption.strip():
            st.error("A reference caption is required.")
        else:
            cc_relative, vqa_relative = PROFILE_PATHS[profile]
            load_dotenv(Path(env_file), override=False)
            request = DemoPipelineRequest(
                run_id=_run_id(item_id),
                item_id=item_id,
                image_a=image_a,
                image_b=image_b,
                original_caption=original_caption,
                task_type=TASK_MODES[task_label],
                questions=[question] if question.strip() else [],
                use_knowledge_bridge=use_bridge,
                mask=mask_path if mask_enabled else None,
            )
            try:
                with st.status("Running RS-Agent", expanded=True) as status:
                    status.write("Executing {}".format(task_label))
                    view = asyncio.run(
                        run_demo_pipeline(
                            request,
                            cc_config_path=repo_root / cc_relative,
                            vqa_config_path=repo_root / vqa_relative,
                            artifact_dir=Path(artifact_dir),
                        )
                    )
                    status.update(label="Analysis complete", state="complete")
                st.session_state.latest_view = view
                st.session_state.latest_mask = str(mask_path) if mask_path else None
                st.session_state.current_inputs = request.model_dump(mode="json")
            except Exception as exc:
                st.exception(exc)

    if "latest_view" in st.session_state:
        _render_view(
            st.session_state.latest_view,
            Path(st.session_state.latest_mask) if st.session_state.latest_mask else None,
        )

    follow_up = st.chat_input("Ask a follow-up question")
    if follow_up and "current_inputs" in st.session_state:
        previous = dict(st.session_state.current_inputs)
        latest_view = st.session_state.latest_view
        previous.update(
            {
                "run_id": _run_id(previous["item_id"]),
                "task_type": TaskType.VQA,
                "questions": [follow_up],
                "provided_knowledge": (
                    {
                        "run_id": latest_view.run_id,
                        "item_id": latest_view.item_id,
                        "c_star": latest_view.knowledge_caption,
                        "source_result_artifact": (
                            latest_view.knowledge_source_artifact
                        ),
                    }
                    if latest_view.knowledge_caption
                    and latest_view.knowledge_source_artifact
                    else None
                ),
            }
        )
        cc_relative, vqa_relative = PROFILE_PATHS[profile]
        load_dotenv(Path(env_file), override=False)
        with st.chat_message("user"):
            st.write(follow_up)
        try:
            with st.spinner("Running RS-VQA"):
                follow_up_view = asyncio.run(
                    run_demo_pipeline(
                        DemoPipelineRequest.model_validate(previous),
                        cc_config_path=repo_root / cc_relative,
                        vqa_config_path=repo_root / vqa_relative,
                        artifact_dir=Path(artifact_dir),
                    )
                )
            with st.chat_message("assistant"):
                for answer in follow_up_view.selected_answers:
                    st.write(answer.answer)
                    st.caption(answer.model_name)
            st.session_state.latest_view = follow_up_view
        except Exception as exc:
            st.exception(exc)


render_app()
