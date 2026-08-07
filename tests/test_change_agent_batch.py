import importlib.util
from pathlib import Path


def load_script_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "generate_change_agent_captions.py"
    spec = importlib.util.spec_from_file_location("change_agent_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_decode_caption_uses_vocab_ids_and_removes_special_tokens() -> None:
    module = load_script_module()
    vocab = {"<NULL>": 0, "<START>": 2, "<END>": 3, "two": 7, "buildings": 10}
    assert module.decode_caption([2, 7, 10, 3, 0], vocab) == "two buildings"


def test_read_names_applies_limit(tmp_path: Path) -> None:
    module = load_script_module()
    path = tmp_path / "test.txt"
    path.write_text("a.png\n\nb.png\nc.png\n", encoding="utf-8")
    assert module.read_names(path, 2) == ["a.png", "b.png"]
