import pytest

from rs_agent.evaluation.embedding_similarity import (
    caption_embedding_rows,
    cosine_similarity,
    summarize_embedding_rows,
)


def test_embedding_similarity_reports_human_and_agent_baselines() -> None:
    embeddings = {
        "agent": [1.0, 0.0],
        "human one": [1.0, 0.0],
        "human two": [0.0, 1.0],
    }
    rows = caption_embedding_rows(
        {"item": "agent"},
        {"item": ["human one", "human two"]},
        embeddings,
    )
    summary = summarize_embedding_rows(rows)

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert summary["mean_human_agent_similarity"] == pytest.approx(0.5)
    assert summary["mean_human_human_similarity"] == pytest.approx(0.0)
