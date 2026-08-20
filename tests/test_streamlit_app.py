import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_streamlit_app_renders_dataset_sample_without_exceptions(
    tmp_path: Path, monkeypatch
) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(before)
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(after)
    Image.new("L", (8, 8), color=0).save(mask)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "item_id": "sample-1",
                "original_caption": "The scene is unchanged.",
                "image_a": str(before),
                "image_b": str(after),
                "predicted_mask": str(mask),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RS_AGENT_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("RS_AGENT_DEMO_MANIFEST", str(manifest))
    monkeypatch.setenv("RS_AGENT_WEB_WORKDIR", str(tmp_path / "web"))

    app = AppTest.from_file(str(repo_root / "src" / "rs_agent" / "web" / "app.py"))
    app.run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "RS-Agent"
    assert "sample-1" in [widget.value for widget in app.selectbox]
