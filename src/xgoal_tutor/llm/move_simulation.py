from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from xgoal_tutor.modeling.constants import GOAL_HALF_WIDTH_SB, GOAL_Y_CENTER_SB, PITCH_LENGTH_SB, PITCH_WIDTH_SB

if TYPE_CHECKING:
    from xgoal_tutor.api.models import LogisticRegressionModel
else:
    LogisticRegressionModel = object  # type: ignore[assignment]


PITCH_BOUNDS = np.array([[0.0, 0.0], [PITCH_LENGTH_SB, PITCH_WIDTH_SB]])
GOAL_CENTER = np.array([PITCH_LENGTH_SB, GOAL_Y_CENTER_SB])
LEFT_POST = np.array([PITCH_LENGTH_SB, GOAL_Y_CENTER_SB - GOAL_HALF_WIDTH_SB])
RIGHT_POST = np.array([PITCH_LENGTH_SB, GOAL_Y_CENTER_SB + GOAL_HALF_WIDTH_SB])
GK_MIN_X = 120.0 - (5.5 * (PITCH_LENGTH_SB / 105.0))
GK_MIN_Y = 33.5
GK_MAX_Y = 46.5


def _default_logistic_model():
    from xgoal_tutor.api.models import DEFAULT_LOGISTIC_REGRESSION_MODEL

    return DEFAULT_LOGISTIC_REGRESSION_MODEL


def goal_biased_headings(start: Sequence[float], n_dir: int = 8) -> List[np.ndarray]:
    """Generate unit vectors prioritising directions toward goal posts and nearby angles."""
    direction = _unit_vector(GOAL_CENTER - np.asarray(start, dtype=float))
    post_left = _unit_vector(LEFT_POST - np.asarray(start, dtype=float))
    post_right = _unit_vector(RIGHT_POST - np.asarray(start, dtype=float))
    base_angle = math.atan2(direction[1], direction[0])
    offsets = [0.0, -0.218, 0.218, math.radians(30.0), -math.radians(30.0), math.radians(60.0), -math.radians(60.0)]
    headings = [direction, post_left, post_right]
    for off in offsets[3:]:
        headings.append(_unit_vector(_rotate_from_angle(base_angle + off)))
    lateral_sign = 1.0 if float(start[1]) <= GOAL_Y_CENTER_SB else -1.0
    headings.append(np.array([0.0, lateral_sign]))
    return headings[:n_dir]


def assign_marking(
    defenders: Sequence[Tuple[float, float]],
    teammates: Sequence[Tuple[float, float]],
    pass_threats: Sequence[float],
    *,
    r_mark: float = 1.8,
    thr: float = 0.10,
) -> Dict[int, int]:
    """Allocate defenders to dangerous teammates within a marking radius."""
    busy: Dict[int, int] = {}
    dangerous = [idx for idx, threat in enumerate(pass_threats) if threat >= thr]
    taken: set[int] = set()
    for t_idx in dangerous:
        nearest_idx = _nearest_index(defenders, teammates[t_idx], taken)
        if nearest_idx is None:
            continue
        if _distance(defenders[nearest_idx], teammates[t_idx]) <= r_mark:
            busy[nearest_idx] = t_idx
            taken.add(nearest_idx)
    return busy


def choose_defenders_for_shooter(
    shooter: Sequence[float],
    defenders: Sequence[Tuple[float, float]],
    busy: Dict[int, int],
    shooter_xg: float,
    pass_threats: Sequence[float],
    *,
    k: int = 2,
    delta_realloc: float = 0.05,
) -> List[int]:
    """Select up to ``k`` defenders to engage the shooter, reallocating marks if justified."""
    ordered = sorted(range(len(defenders)), key=lambda idx: _distance(defenders[idx], shooter))
    free = [idx for idx in ordered if idx not in busy][:k]
    if len(free) >= k:
        return free[:k]
    chosen = list(free)
    for idx in ordered:
        if idx in busy and idx not in chosen:
            threat_idx = busy[idx]
            threat_val = pass_threats[threat_idx] if threat_idx < len(pass_threats) else 0.0
            if shooter_xg >= threat_val + delta_realloc:
                chosen.append(idx)
                if len(chosen) == k:
                    break
    return chosen


def defender_response_options(defender: Sequence[float], shooter: Sequence[float], delta_def: float) -> List[Tuple[float, float]]:
    """Enumerate discrete best-response moves for an individual defender."""
    base = np.asarray(defender, dtype=float)
    shooter_point = np.asarray(shooter, dtype=float)
    bisector = _unit_vector(GOAL_CENTER - shooter_point)
    drop_target = np.array(
        [
            PITCH_LENGTH_SB,
            _clamp(base[1], GOAL_Y_CENTER_SB - GOAL_HALF_WIDTH_SB, GOAL_Y_CENTER_SB + GOAL_HALF_WIDTH_SB),
        ]
    )
    left_target = _nearest_on_ray(shooter_point, LEFT_POST, base)
    if np.allclose(left_target, shooter_point):
        left_target = shooter_point + _unit_vector(LEFT_POST - shooter_point) * delta_def
    right_target = _nearest_on_ray(shooter_point, RIGHT_POST, base)
    if np.allclose(right_target, shooter_point):
        right_target = shooter_point + _unit_vector(RIGHT_POST - shooter_point) * delta_def
    options = [
        _move_towards(base, shooter_point, delta_def),
        _move_towards(base, shooter_point + bisector, delta_def),
        _move_towards(base, drop_target, delta_def),
        _move_towards(base, left_target, delta_def),
        _move_towards(base, right_target, delta_def),
    ]
    unique: List[Tuple[float, float]] = []
    for opt in options:
        clamped = tuple(_clamp_pitch(opt))
        if clamped not in unique:
            unique.append(clamped)
    if len(unique) < 5:
        perp = _unit_vector(np.array([-bisector[1], bisector[0]]))
        for scale in (0.5, -0.5, 1.0, -1.0):
            if len(unique) >= 5:
                break
            candidate = tuple(_clamp_pitch(base + perp * delta_def * scale))
            if candidate not in unique:
                unique.append(candidate)
    return unique


def gk_response_options(gk: Optional[Sequence[float]], shooter: Sequence[float], delta_gk: float) -> List[Optional[Tuple[float, float]]]:
    """Enumerate goalkeeper adjustments within the goal-area band."""
    if gk is None:
        return [None]
    base = _clamp_goalkeeper(np.asarray(gk, dtype=float))
    shooter_point = np.asarray(shooter, dtype=float)
    bisector = _unit_vector(GOAL_CENTER - shooter_point)
    lateral = np.array([0.0, math.copysign(delta_gk, bisector[1] if bisector[1] != 0 else 1.0)])
    options = [
        base,
        base + lateral,
        _move_towards(base, shooter_point, delta_gk),
        _move_towards(base + lateral, shooter_point, 0.5 * delta_gk),
    ]
    unique: List[Tuple[float, float]] = []
    for option in options:
        clamped = _clamp_goalkeeper(option)
        if tuple(clamped) not in unique:
            unique.append(tuple(clamped))
    return [(opt if opt is None else (float(opt[0]), float(opt[1]))) for opt in unique]


def predict_xg_from_point(
    shooter: Sequence[float],
    defenders: Sequence[Tuple[float, float]],
    gk: Optional[Sequence[float]],
    features_fn,
    model,
) -> float:
    """Estimate xG from arbitrary positions using either a logistic model or ``predict_proba`` API."""
    features = features_fn(shooter, defenders, gk)
    if isinstance(features, pd.DataFrame):
        frame = features
    elif isinstance(features, dict):
        frame = pd.DataFrame([features])
    else:
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        frame = arr
    if hasattr(model, "coefficients"):
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        coeffs = getattr(model, "coefficients", {})
        intercept = float(getattr(model, "intercept", 0.0))
        ordered = frame.reindex(columns=list(coeffs.keys()), fill_value=0.0)
        coef_series = pd.Series(coeffs)
        linear = ordered.mul(coef_series, axis=1).sum(axis=1) + intercept
        probability = 1.0 / (1.0 + np.exp(-linear.iloc[0]))
        return float(probability)
    proba = model.predict_proba(frame)
    value = proba[0, 1] if hasattr(proba, "shape") and proba.shape[1] > 1 else proba[0, 0]
    return float(value)


def simulate_one_direction(
    start: Sequence[float],
    heading: np.ndarray,
    defenders: List[Tuple[float, float]],
    teammates: List[Tuple[float, float]],
    pass_threats: Sequence[float],
    gk: Optional[Sequence[float]],
    features_fn,
    model,
    *,
    delta_att: float = 1.0,
    delta_def: float = 1.2,
    delta_gk: float = 0.72,
    k_max: int = 6,
    eps_gain: float = 0.01,
):
    """Simulate iterative attacker steps along one heading with defender and GK best responses."""
    current_point = np.asarray(start, dtype=float)
    defender_positions = list(defenders)
    best_xg = predict_xg_from_point(current_point, defender_positions, gk, features_fn, model)
    trace = [best_xg]
    best_point = tuple(current_point)
    best_gk = gk
    for _ in range(k_max):
        candidate_point = _clamp_pitch(current_point + heading * delta_att)
        busy = assign_marking(defender_positions, teammates, pass_threats)
        selected = choose_defenders_for_shooter(candidate_point, defender_positions, busy, best_xg, pass_threats)
        xg_min = math.inf
        defender_best = defender_positions
        gk_best = best_gk
        for gk_opt in gk_response_options(best_gk, candidate_point, delta_gk):
            if not selected:
                xg_try = predict_xg_from_point(candidate_point, defender_positions, gk_opt, features_fn, model)
                if xg_try < xg_min:
                    xg_min = xg_try
                    defender_best = defender_positions
                    gk_best = gk_opt
            elif len(selected) == 1:
                i = selected[0]
                for opt1 in defender_response_options(defender_positions[i], candidate_point, delta_def):
                    defenders_try = list(defender_positions)
                    defenders_try[i] = opt1
                    xg_try = predict_xg_from_point(candidate_point, defenders_try, gk_opt, features_fn, model)
                    if xg_try < xg_min:
                        xg_min = xg_try
                        defender_best = defenders_try
                        gk_best = gk_opt
            else:
                i, j = selected[:2]
                for opt1 in defender_response_options(defender_positions[i], candidate_point, delta_def):
                    for opt2 in defender_response_options(defender_positions[j], candidate_point, delta_def):
                        defenders_try = list(defender_positions)
                        defenders_try[i] = opt1
                        defenders_try[j] = opt2
                        xg_try = predict_xg_from_point(candidate_point, defenders_try, gk_opt, features_fn, model)
                        if xg_try < xg_min:
                            xg_min = xg_try
                            defender_best = defenders_try
                            gk_best = gk_opt
        if xg_min > best_xg + eps_gain:
            current_point = candidate_point
            defender_positions = defender_best
            best_xg = xg_min
            best_point = tuple(current_point)
            best_gk = gk_best
            trace.append(best_xg)
        else:
            break
    return best_xg, trace, best_point


def hill_climb_best_move(
    start: Sequence[float],
    defenders: List[Tuple[float, float]],
    teammates: List[Tuple[float, float]],
    pass_threats: Sequence[float],
    gk: Optional[Sequence[float]],
    features_fn,
    model,
    *,
    delta_att: float = 1.0,
    delta_def: float = 1.2,
    delta_gk: float = 0.72,
    k_max: int = 6,
    eps_gain: float = 0.01,
) -> Dict[str, object]:
    """Search goal-biased headings to find the max–min xG-improving short move."""
    base_xg = predict_xg_from_point(start, defenders, gk, features_fn, model)
    best: Dict[str, object] = {
        "xg": base_xg,
        "dist_m": 0.0,
        "heading": None,
        "trace": [base_xg],
        "S_best": tuple(start),
    }
    for heading in goal_biased_headings(start):
        xg_dir, trace, end_point = simulate_one_direction(
            start,
            heading,
            list(defenders),
            list(teammates),
            pass_threats,
            gk,
            features_fn,
            model,
            delta_att=delta_att,
            delta_def=delta_def,
            delta_gk=delta_gk,
            k_max=k_max,
            eps_gain=eps_gain,
        )
        if xg_dir > best["xg"]:
            best["xg"] = xg_dir
            best["dist_m"] = (len(trace) - 1) * delta_att
            best["heading"] = heading
            best["trace"] = trace
            best["S_best"] = end_point
    return {
        "xg_current": base_xg,
        "xg_best": float(best["xg"]),
        "xg_gain": float(best["xg"] - base_xg),
        "best_distance_m": float(best["dist_m"]),
        "best_heading_vec": None if best["heading"] is None else tuple(float(v) for v in best["heading"]),
        "xg_trace": [float(v) for v in best["trace"]],
        "S_best": tuple(float(v) for v in best["S_best"]),
    }


def build_point_feature_row(
    shooter: Sequence[float], defenders: Sequence[Tuple[float, float]], gk: Optional[Sequence[float]], model: LogisticRegressionModel
) -> pd.DataFrame:
    """Construct a single-row feature frame from positional inputs for the logistic model."""
    shooter_point = np.asarray(shooter, dtype=float)
    row: Dict[str, float] = {}
    row["dist_sb"] = _distance_to_goal(shooter_point)
    row["angle_deg_sb"] = _opening_angle_deg(shooter_point)
    row["ff_opponents"] = float(len(defenders))
    row["blockers_in_cone"] = float(_blockers_in_cone(shooter_point, defenders))
    if gk is not None:
        row["gk_depth_sb"] = max(0.0, PITCH_LENGTH_SB - float(gk[0]))
        row["gk_offset_sb"] = float(gk[1]) - GOAL_Y_CENTER_SB
    else:
        row["gk_depth_sb"] = 0.0
        row["gk_offset_sb"] = 0.0
    for feature in model.coefficients:
        row.setdefault(feature, 0.0)
    return pd.DataFrame([row], columns=list(model.coefficients.keys()))


def simulate_best_move_with_defaults(
    shooter: Sequence[float],
    defenders: List[Tuple[float, float]],
    teammates: List[Tuple[float, float]],
    gk: Optional[Sequence[float]],
    pass_threats: Optional[Sequence[float]] = None,
    *,
    model: Optional[LogisticRegressionModel] = None,
    delta_att: float = 1.0,
    delta_def: float = 1.2,
    delta_gk: float = 0.72,
    k_max: int = 6,
    eps_gain: float = 0.01,
) -> Dict[str, object]:
    """Convenience wrapper using default model and step sizes for hill-climb move search."""
    logistic_model = model or _default_logistic_model()
    threats = list(pass_threats) if pass_threats is not None else _estimate_pass_threats(teammates, defenders, gk, logistic_model)

    def features_fn(point, local_defenders, local_gk):
        return build_point_feature_row(point, local_defenders, local_gk, logistic_model)

    return hill_climb_best_move(
        shooter,
        defenders,
        teammates,
        threats,
        gk,
        features_fn,
        logistic_model,
        delta_att=delta_att,
        delta_def=delta_def,
        delta_gk=delta_gk,
        k_max=k_max,
        eps_gain=eps_gain,
    )


def _estimate_pass_threats(
    teammates: Sequence[Tuple[float, float]],
    defenders: Sequence[Tuple[float, float]],
    gk: Optional[Sequence[float]],
    model: LogisticRegressionModel,
) -> List[float]:
    threats: List[float] = []
    if not teammates:
        return threats

    def features_fn(point, local_defenders, local_gk):
        return build_point_feature_row(point, local_defenders, local_gk, model)

    for teammate in teammates:
        threats.append(predict_xg_from_point(teammate, defenders, gk, features_fn, model))
    return threats


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return np.array([1.0, 0.0])
    return vector / norm


def _rotate_from_angle(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


def _nearest_index(points: Sequence[Tuple[float, float]], target: Sequence[float], excluded: Iterable[int]) -> Optional[int]:
    best_idx: Optional[int] = None
    best_dist = math.inf
    for idx, point in enumerate(points):
        if idx in excluded:
            continue
        dist = _distance(point, target)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _move_towards(source: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
    direction = _unit_vector(target - source)
    return source + direction * step


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return float(max(min_value, min(max_value, value)))


def _clamp_pitch(point: np.ndarray) -> np.ndarray:
    lower, upper = PITCH_BOUNDS
    return np.array([
        _clamp(point[0], lower[0], upper[0]),
        _clamp(point[1], lower[1], upper[1]),
    ])


def _clamp_goalkeeper(point: np.ndarray) -> np.ndarray:
    return np.array([
        _clamp(point[0], GK_MIN_X, PITCH_LENGTH_SB),
        _clamp(point[1], GK_MIN_Y, GK_MAX_Y),
    ])


def _nearest_on_ray(origin: np.ndarray, target: np.ndarray, point: np.ndarray) -> np.ndarray:
    direction = target - origin
    denom = float(np.dot(direction, direction))
    if denom == 0:
        return origin
    t = float(np.dot(point - origin, direction) / denom)
    t = max(0.0, t)
    return origin + direction * t


def _distance_to_goal(point: np.ndarray) -> float:
    return float(np.hypot(PITCH_LENGTH_SB - point[0], GOAL_Y_CENTER_SB - point[1]))


def _opening_angle_deg(point: np.ndarray) -> float:
    v_left = LEFT_POST - point
    v_right = RIGHT_POST - point
    dot = float(np.dot(v_left, v_right))
    norm_left = float(np.linalg.norm(v_left))
    norm_right = float(np.linalg.norm(v_right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (norm_left * norm_right)))
    return math.degrees(math.acos(cosang))


def _blockers_in_cone(shooter: np.ndarray, defenders: Sequence[Tuple[float, float]]) -> int:
    if not defenders:
        return 0
    edge_left = LEFT_POST - shooter
    edge_right = RIGHT_POST - shooter
    cross_edges = float(np.cross(edge_left, edge_right))
    blockers = 0
    for defender in defenders:
        vec = np.asarray(defender, dtype=float) - shooter
        cross_left = float(np.cross(edge_left, vec))
        cross_right = float(np.cross(vec, edge_right))
        if cross_edges >= 0:
            inside = cross_left >= 0 and cross_right >= 0
        else:
            inside = cross_left <= 0 and cross_right <= 0
        if inside and np.dot(vec, edge_left) >= 0 and np.dot(vec, edge_right) >= 0:
            blockers += 1
    return blockers
