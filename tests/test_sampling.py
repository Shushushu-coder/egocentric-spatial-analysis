import pytest

from egocentric_spatial_analysis.preprocessing.sampling import FrameCandidate, select_frames


def _candidates(n: int = 10) -> list[FrameCandidate]:
    return [
        FrameCandidate(frame_index=i, timestamp_sec=float(i), quality_score=float(i * 10))
        for i in range(n)
    ]


def test_uniform_returns_requested_count() -> None:
    selected = select_frames(
        _candidates(), num_frames=4, strategy="uniform", start_time=0, end_time=10
    )
    assert len(selected) == 4
    assert selected[0].timestamp_sec <= selected[-1].timestamp_sec


def test_quality_weighted_picks_high_quality_in_segments() -> None:
    selected = select_frames(
        _candidates(), num_frames=2, strategy="quality_weighted", start_time=0, end_time=10
    )
    assert len(selected) == 2
    assert selected[1].quality_score >= selected[0].quality_score


def test_bad_window_raises() -> None:
    with pytest.raises(ValueError):
        select_frames(_candidates(), num_frames=2, strategy="uniform", start_time=5, end_time=1)


def test_empty_candidates() -> None:
    assert select_frames([], num_frames=3, strategy="uniform", start_time=0, end_time=1) == []
