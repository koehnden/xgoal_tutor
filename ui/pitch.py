"""Utilities for plotting the football pitch and player positions."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Circle, Rectangle

from ui.models import Pitch, Player

PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
CENTER_CIRCLE_RADIUS = 9.15
GRID_SPACING_METERS = 5.0


def plot_pitch_with_players(pitch: Pitch, players: List[Player]) -> plt.Figure:
    """Render a football pitch with player markers."""

    fig, ax = plt.subplots(figsize=(8, 5))

    # Pitch background
    pitch_rect = Rectangle((0, 0), pitch.length_m, pitch.width_m, linewidth=1, edgecolor="black", facecolor="#f0f8ff")
    ax.add_patch(pitch_rect)

    # Grid lines every GRID_SPACING_METERS meters
    x_grid = [x for x in frange(0.0, pitch.length_m, GRID_SPACING_METERS)]
    y_grid = [y for y in frange(0.0, pitch.width_m, GRID_SPACING_METERS)]
    for x in x_grid:
        ax.axvline(x, color="#d3d3d3", linewidth=0.4, zorder=0)
    for y in y_grid:
        ax.axhline(y, color="#d3d3d3", linewidth=0.4, zorder=0)

    # Penalty areas and goal line (assuming origin bottom-left)
    ax.add_patch(
        Rectangle(
            (pitch.length_m - PENALTY_AREA_DEPTH, (pitch.width_m - PENALTY_AREA_WIDTH) / 2),
            PENALTY_AREA_DEPTH,
            PENALTY_AREA_WIDTH,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
    )
    ax.add_patch(
        Rectangle(
            (0, (pitch.width_m - PENALTY_AREA_WIDTH) / 2),
            PENALTY_AREA_DEPTH,
            PENALTY_AREA_WIDTH,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
    )
    ax.axvline(0, color="black", linewidth=1.5)
    ax.axvline(pitch.length_m, color="black", linewidth=1.5)

    # Center line and circle
    ax.axvline(pitch.length_m / 2, color="black", linewidth=1)
    center_circle = Circle((pitch.length_m / 2, pitch.width_m / 2), radius=CENTER_CIRCLE_RADIUS, edgecolor="black", facecolor="none", linewidth=1)
    ax.add_patch(center_circle)

    teams = {player.team for player in players}
    team_colors = _team_colors(sorted(teams))
    role_markers = {"attacker": "o", "defender": "^", "gk": "s"}

    handles: List[Artist] = []
    labels: List[str] = []
    for player in players:
        marker = role_markers.get(player.role.lower(), "o")
        color = team_colors.get(player.team, "#1f77b4")
        scatter = ax.scatter(
            player.x,
            player.y,
            marker=marker,
            color=color,
            edgecolor="black",
            s=120,
            label=f"{player.name} ({player.team})",
        )
        ax.annotate(
            player.name,
            (player.x, player.y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )
        handles.append(scatter)
        labels.append(f"{player.name} ({player.team})")

    if handles:
        unique = _unique_handles_labels(handles, labels)
        ax.legend(*unique, loc="upper right", fontsize=8, frameon=True)

    ax.set_xlim(0, pitch.length_m)
    ax.set_ylim(0, pitch.width_m)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    """Generate a range of floating point values inclusive of stop."""

    value = start
    while value <= stop + 1e-9:
        yield round(value, 10)
        value += step


def _team_colors(teams: Iterable[str]) -> Dict[str, str]:
    """Generate deterministic colors per team using matplotlib defaults."""

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colors: Dict[str, str] = {}
    for index, team in enumerate(teams):
        colors[team] = color_cycle[index % len(color_cycle)]
    return colors


def _unique_handles_labels(handles: Iterable[Artist], labels: Iterable[str]) -> Tuple[List[Artist], List[str]]:
    """Return unique handles and labels preserving order."""

    seen = set()
    unique_handles: List[Artist] = []
    unique_labels: List[str] = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    return unique_handles, unique_labels


__all__ = [
    "CENTER_CIRCLE_RADIUS",
    "GRID_SPACING_METERS",
    "PENALTY_AREA_DEPTH",
    "PENALTY_AREA_WIDTH",
    "frange",
    "plot_pitch_with_players",
]
