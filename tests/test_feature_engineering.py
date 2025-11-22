import pytest

from xgoal_tutor.modeling.feature_engineering import (
    _compute_cutback_for_shot,
    _dist_point_to_segment,
)


def test_dist_point_to_segment_perpendicular():
    distance = _dist_point_to_segment(5.0, 5.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(distance - 5.0) < 0.01


def test_dist_point_to_segment_on_segment():
    distance = _dist_point_to_segment(5.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(distance - 0.0) < 0.01


def test_dist_point_to_segment_endpoint():
    distance = _dist_point_to_segment(0.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(distance - 0.0) < 0.01


def test_dist_point_to_segment_beyond_endpoint():
    distance = _dist_point_to_segment(15.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(distance - 5.0) < 0.01


def test_dist_point_to_segment_zero_length():
    distance = _dist_point_to_segment(5.0, 5.0, 3.0, 3.0, 3.0, 3.0)
    expected = ((5.0 - 3.0)**2 + (5.0 - 3.0)**2)**0.5
    assert abs(distance - expected) < 0.01


def test_compute_cutback_no_freeze_frame():
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=[]
    )
    assert has_cutback == 0
    assert target_x == 0.0
    assert target_y == 0.0


def test_compute_cutback_no_teammates_in_zone():
    freeze_frame_rows = [
        (110.0, 30.0, 1, 0),
        (90.0, 40.0, 1, 0),
        (115.0, 20.0, 0, 0),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 0


def test_compute_cutback_clear_lane():
    freeze_frame_rows = [
        (110.0, 40.0, 1, 0),
        (115.0, 20.0, 0, 0),
        (115.0, 60.0, 0, 0),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 1
    assert target_x == 110.0
    assert target_y == 40.0


def test_compute_cutback_blocked_lane():
    freeze_frame_rows = [
        (110.0, 40.0, 1, 0),
        (105.0, 40.0, 0, 0),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 0


def test_compute_cutback_multiple_candidates_picks_closest_to_goal():
    freeze_frame_rows = [
        (110.0, 40.0, 1, 0),
        (115.0, 38.0, 1, 0),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 1
    assert target_x == 115.0
    assert target_y == 38.0


def test_compute_cutback_no_opponents_all_lanes_clear():
    freeze_frame_rows = [
        (110.0, 40.0, 1, 0),
        (105.0, 38.0, 1, 0),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 1
    assert target_x == 110.0
    assert target_y == 40.0


def test_compute_cutback_ignores_goalkeeper():
    freeze_frame_rows = [
        (110.0, 40.0, 1, 0),
        (120.0, 40.0, 0, 1),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 1


def test_compute_cutback_ignores_teammates_behind_shooter():
    freeze_frame_rows = [
        (90.0, 40.0, 1, 0),
        (110.0, 40.0, 1, 0),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 1
    assert target_x == 110.0


def test_compute_cutback_gold_zone_boundaries():
    freeze_frame_rows = [
        (110.0, 32.0, 1, 0),
        (110.0, 48.0, 1, 0),
        (110.0, 31.9, 1, 0),
        (110.0, 48.1, 1, 0),
    ]
    has_cutback, target_x, target_y = _compute_cutback_for_shot(
        shot_x=100.0,
        shot_y=40.0,
        freeze_frame_rows=freeze_frame_rows
    )
    assert has_cutback == 1
    assert target_y in [32.0, 48.0]
