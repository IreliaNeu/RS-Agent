from rs_agent.evaluation.prompts import score_messages


def test_score_prompt_contains_literal_json_example() -> None:
    messages = score_messages("one building was added", {"A": "first", "B": "second"})
    assert '{"A": 8, "B": 7}' in messages[1]["content"]

