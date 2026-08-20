import pytest

from rs_agent.evaluation.corpus_metrics import corpus_caption_metrics
from rs_agent.evaluation.reference_metrics import CaptionReference


def references() -> dict:
    return {
        "one": CaptionReference(
            item_id="one",
            split="test",
            references=["a new building appears", "one building was constructed"],
            change_flag=1,
        ),
        "two": CaptionReference(
            item_id="two",
            split="test",
            references=["a road was removed", "the old road disappeared"],
            change_flag=1,
        ),
    }


def test_corpus_metrics_reward_exact_references() -> None:
    exact = corpus_caption_metrics(
        {"one": "a new building appears", "two": "a road was removed"},
        references(),
    )
    unrelated = corpus_caption_metrics(
        {"one": "water covers fields", "two": "trees remain green"},
        references(),
    )

    assert exact["corpus_bleu_4"] == pytest.approx(1.0)
    assert exact["cider"] > unrelated["cider"]
    assert unrelated["corpus_bleu_1"] == 0.0
