import csv
import json
from pathlib import Path

from rs_agent.evaluation.human_eval import (
    prepare_blind_trials,
    score_blind_responses,
)


def test_blind_trials_keep_key_separate_and_score_preferences(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(
        json.dumps(
            {
                "item_id": "pair-1",
                "original_caption": "Original caption.",
                "selected_caption": "Agent caption.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "blind"
    protocol = prepare_blind_trials(items, output, seed=7)
    key = json.loads((output / "answer_key.jsonl").read_text(encoding="utf-8"))
    with (output / "blind_trials.csv").open("r", encoding="utf-8") as handle:
        trial = next(csv.DictReader(handle))

    assert protocol["trial_count"] == 1
    assert "agent_side" not in trial
    responses = tmp_path / "responses.csv"
    responses.write_text(
        "trial_id,rater_id,preferred_side\n{},{},{}\n".format(
            trial["trial_id"], "rater-1", key["agent_side"]
        ),
        encoding="utf-8",
    )
    summary = score_blind_responses(output / "answer_key.jsonl", responses)
    assert summary["agent_wins"] == 1
    assert summary["agent_preference_rate_excluding_ties"] == 1.0
