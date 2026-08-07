from pathlib import Path

from rs_agent.applications.rs_cc import main


def test_cli_dry_run_needs_no_api_key(capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    exit_code = main(
        [
            "--config",
            str(root / "configs" / "rs_cc.paper.yaml"),
            "--input",
            str(root / "examples" / "rs_cc_input.jsonl"),
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"record_count": 1' in output
    assert '"input_mode": "text_only_original_caption"' in output
