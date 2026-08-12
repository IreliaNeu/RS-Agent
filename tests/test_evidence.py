from rs_agent.domains.remote_sensing.mask_evidence import MaskEvidenceSummary, MaskSource
from rs_agent.orchestration.evidence import (
    EvidenceStance,
    build_evidence_bundle,
    classify_statement,
)


def no_change_mask() -> MaskEvidenceSummary:
    return MaskEvidenceSummary(
        path="/tmp/mask.png",
        source=MaskSource.PREDICTED,
        sha256="0" * 64,
        width=256,
        height=256,
        class_stats=[],
        changed_pixel_count=0,
        changed_ratio=0.0,
        component_count=0,
        significant_component_count=0,
        has_change=False,
        interpretation=(
            "The optional mask contains no significant non-background change region."
        ),
    )


def test_no_change_phrase_takes_priority_over_change_token() -> None:
    assert classify_statement("There is no significant change.") == EvidenceStance.NO_CHANGE
    assert (
        classify_statement("The scene exhibits no discernible changes.")
        == EvidenceStance.NO_CHANGE
    )
    assert classify_statement("A new road appeared.") == EvidenceStance.CHANGE
    assert classify_statement("Several buildings are visible.") == EvidenceStance.UNCERTAIN


def test_caption_mask_disagreement_is_recorded_without_forced_verdict() -> None:
    bundle = build_evidence_bundle(
        original_caption="A new road appears in the middle of the scene.",
        caption=None,
        vqa=None,
        mask=no_change_mask(),
    )

    assert bundle.has_conflict is True
    assert bundle.consensus == EvidenceStance.UNCERTAIN
    assert len(bundle.conflicts) == 1
    assert bundle.conflicts[0].left_source_id == "caption:original"
    assert bundle.conflicts[0].right_source_id.startswith("mask:")
