from rs_agent.core.schemas import QuestionSource
from rs_agent.domains.remote_sensing.questions import resolve_questions


def test_default_templates_are_used_when_user_question_is_absent() -> None:
    questions = resolve_questions([], ["change_summary", "spatial_location"])
    assert [question.template_id for question in questions] == [
        "change_summary",
        "spatial_location",
    ]
    assert all(question.source == QuestionSource.TEMPLATE for question in questions)


def test_matching_user_question_keeps_exact_wording_and_marks_hybrid() -> None:
    text = "How many buildings were newly constructed?"
    question = resolve_questions([text], ["change_summary"])[0]
    assert question.text == text
    assert question.source == QuestionSource.HYBRID
    assert question.template_id == "building_change"


def test_unmatched_user_question_is_not_rewritten() -> None:
    question = resolve_questions(
        ["Is the acquisition season visibly different?"], ["change_summary"]
    )[0]
    assert question.source == QuestionSource.USER
    assert question.template_id is None
