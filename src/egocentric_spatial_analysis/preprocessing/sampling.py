"""Frame sampling strategies independent of observer identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SamplingStrategy(str, Enum):
    UNIFORM = "uniform"
    QUALITY_WEIGHTED = "quality_weighted"
    COMPLEMENTARY = "complementary"


@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    timestamp_sec: float
    quality_score: float
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    feature_count: float = 0.0
    similarity: float = 0.0
    feature_matches: int = 0


def select_frames(
    candidates: list[FrameCandidate],
    *,
    num_frames: int,
    strategy: SamplingStrategy | str,
    start_time: float,
    end_time: float,
) -> list[FrameCandidate]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if not candidates:
        return []

    strategy = SamplingStrategy(strategy)
    if len(candidates) <= num_frames:
        return list(candidates)

    if strategy is SamplingStrategy.UNIFORM:
        return _select_uniform(candidates, num_frames)
    if strategy is SamplingStrategy.QUALITY_WEIGHTED:
        return _select_quality_weighted(candidates, num_frames, start_time, end_time)
    if strategy is SamplingStrategy.COMPLEMENTARY:
        return _select_complementary(candidates, num_frames)
    raise ValueError(f"unsupported sampling strategy: {strategy}")


def _select_uniform(candidates: list[FrameCandidate], num_frames: int) -> list[FrameCandidate]:
    if num_frames == 1:
        return [candidates[len(candidates) // 2]]
    step = (len(candidates) - 1) / (num_frames - 1)
    indices = [int(round(i * step)) for i in range(num_frames)]
    # Keep order and drop accidental duplicates from rounding.
    unique: list[int] = []
    for idx in indices:
        if not unique or idx != unique[-1]:
            unique.append(idx)
    while len(unique) < num_frames:
        for idx in range(len(candidates)):
            if idx not in unique:
                unique.append(idx)
            if len(unique) >= num_frames:
                break
    unique.sort()
    return [candidates[i] for i in unique[:num_frames]]


def _select_quality_weighted(
    candidates: list[FrameCandidate],
    num_frames: int,
    start_time: float,
    end_time: float,
) -> list[FrameCandidate]:
    duration = end_time - start_time
    segment = duration / num_frames
    selected: list[FrameCandidate] = []
    used: set[int] = set()
    for i in range(num_frames):
        seg_start = start_time + i * segment
        seg_end = seg_start + segment
        in_segment = [c for c in candidates if seg_start <= c.timestamp_sec < seg_end]
        if in_segment:
            best = max(in_segment, key=lambda c: c.quality_score)
        else:
            midpoint = seg_start + segment / 2
            best = min(candidates, key=lambda c: abs(c.timestamp_sec - midpoint))
        if best.frame_index in used:
            unused = [c for c in candidates if c.frame_index not in used]
            if unused:
                best = max(unused, key=lambda c: c.quality_score)
        used.add(best.frame_index)
        selected.append(best)
    selected.sort(key=lambda c: c.timestamp_sec)
    return selected


def _select_complementary(candidates: list[FrameCandidate], num_frames: int) -> list[FrameCandidate]:
    remaining = list(candidates)
    selected: list[FrameCandidate] = []
    first = max(remaining, key=lambda c: c.quality_score)
    selected.append(first)
    remaining = [c for c in remaining if c.frame_index != first.frame_index]
    while len(selected) < num_frames and remaining:
        scores = []
        for candidate in remaining:
            time_span = sum(abs(candidate.timestamp_sec - s.timestamp_sec) for s in selected) / len(
                selected
            )
            scores.append(time_span * 0.6 + candidate.quality_score * 0.4)
        pick = remaining[int(max(range(len(remaining)), key=lambda i: scores[i]))]
        selected.append(pick)
        remaining = [c for c in remaining if c.frame_index != pick.frame_index]
    selected.sort(key=lambda c: c.timestamp_sec)
    return selected
