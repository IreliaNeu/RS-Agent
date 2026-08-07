from pathlib import Path

import pytest
from pydantic import ValidationError

from rs_agent.core.schemas import ImagePair, TaskRequest, TaskType


def test_caption_enrichment_requires_original_pair_caption() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(
            task_type=TaskType.CAPTION_ENRICHMENT,
            images=ImagePair(before=Path("before.png"), after=Path("after.png")),
        )


def test_vqa_can_defer_question_to_system_templates() -> None:
    request = TaskRequest(
        task_type=TaskType.VQA,
        images=ImagePair(before=Path("before.png"), after=Path("after.png")),
    )
    assert request.user_questions == []

