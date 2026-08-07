import asyncio
from typing import List

from rs_agent.core.config import ModelConfig
from rs_agent.core.schemas import CaptionCandidate, ModelRef, ModelResponse
from rs_agent.evaluation.caption_judge import CaptionJudge
from rs_agent.evaluation.parsing import parse_choice, parse_scores, select_caption


def candidate(label: str, text: str) -> CaptionCandidate:
    return CaptionCandidate(
        candidate_id="candidate-{}".format(label.lower()),
        label=label,
        model=ModelRef(name="generator-{}".format(label), provider="test", model="m"),
        text=text,
    )


class FakeProvider:
    def __init__(self, outputs: List[str]):
        self.outputs = iter(outputs)

    async def complete(self, messages, model, temperature, max_tokens, metadata=None):
        return ModelResponse(
            provider="test",
            model=model,
            content=next(self.outputs),
        )


def test_score_parser_handles_fenced_json() -> None:
    scores = parse_scores('```json\n{"A": 7, "B": 10, "C": 20}\n```', ["A", "B", "C"])
    assert scores == {"A": 7, "B": 10, "C": None}
    assert parse_choice("The answer is B.", ["A", "B"]) == "B"


def test_highest_score_wins_even_when_judge_choice_differs() -> None:
    candidates = [candidate("A", "a"), candidate("B", "b"), candidate("C", "c")]
    selection = select_caption(candidates, {"A": 7, "B": 8, "C": 10}, "B")
    assert selection.selected_label == "C"


def test_judge_choice_breaks_score_tie() -> None:
    candidates = [candidate("A", "a"), candidate("B", "b")]
    selection = select_caption(candidates, {"A": 9, "B": 9}, "B")
    assert selection.selected_label == "B"


def test_caption_judge_preserves_choice_and_selects_highest_score() -> None:
    candidates = [candidate("A", "first"), candidate("B", "second")]
    selector = FakeProvider(["A"])
    evaluator = FakeProvider(['{"A": 8, "B": 10}'])
    model = ModelConfig(name="judge", provider="test", model="judge-model", max_tokens=50)
    judge = CaptionJudge(selector, model, evaluator, model)

    record = asyncio.run(judge.evaluate("one building was added", candidates))

    assert record.judge_choice == "A"
    assert record.selection.selected_label == "B"
    assert record.legacy_best_record()["Best Caption"] == "second"

