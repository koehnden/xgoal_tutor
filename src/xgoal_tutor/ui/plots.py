"""Plotting helpers for visualising shots on a football pitch."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import plotly.graph_objects as go

PITCH_LENGTH = 120
PITCH_WIDTH = 80


_DEF_COLOR = "#1f77b4"
_ATT_COLOR = "#ff7f0e"
_GK_COLOR = "#2ca02c"
_ACTOR_BORDER = "#000000"

_ROLE_STYLE = {
    "attacker": {"color": _ATT_COLOR, "symbol": "circle"},
    "defender": {"color": _DEF_COLOR, "symbol": "square"},
    "goalkeeper": {"color": _GK_COLOR, "symbol": "diamond"},
}


def _add_pitch_shapes(fig: go.Figure) -> None:
    """Add common football pitch markings to the figure."""
    # Outer boundaries
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH, line=dict(color="#222"))
    # Midline
    fig.add_shape(type="line", x0=PITCH_LENGTH / 2, y0=0, x1=PITCH_LENGTH / 2, y1=PITCH_WIDTH)
    # Penalty boxes
    penalty_box_length = 18
    penalty_box_width = 44
    six_yard_length = 6
    six_yard_width = 20
    fig.add_shape(
        type="rect",
        x0=0,
        y0=(PITCH_WIDTH - penalty_box_width) / 2,
        x1=penalty_box_length,
        y1=(PITCH_WIDTH + penalty_box_width) / 2,
        line=dict(color="#444"),
    )
    fig.add_shape(
        type="rect",
        x0=PITCH_LENGTH - penalty_box_length,
        y0=(PITCH_WIDTH - penalty_box_width) / 2,
        x1=PITCH_LENGTH,
        y1=(PITCH_WIDTH + penalty_box_width) / 2,
        line=dict(color="#444"),
    )
    # Six-yard boxes
    fig.add_shape(
        type="rect",
        x0=0,
        y0=(PITCH_WIDTH - six_yard_width) / 2,
        x1=six_yard_length,
        y1=(PITCH_WIDTH + six_yard_width) / 2,
        line=dict(color="#666"),
    )
    fig.add_shape(
        type="rect",
        x0=PITCH_LENGTH - six_yard_length,
        y0=(PITCH_WIDTH - six_yard_width) / 2,
        x1=PITCH_LENGTH,
        y1=(PITCH_WIDTH + six_yard_width) / 2,
        line=dict(color="#666"),
    )
    # Center circle
    fig.add_shape(type="circle", x0=60 - 9.15, y0=40 - 9.15, x1=60 + 9.15, y1=40 + 9.15)


def _player_hover_text(player: Mapping[str, Any], team_name: str) -> str:
    name = player.get("name", "Unknown")
    jersey = player.get("jersey_number")
    if jersey is not None:
        return f"{name} (#{jersey}, {team_name})"
    return f"{name} ({team_name})"


def create_shot_positions_figure(
    positions: Mapping[str, Any],
    *,
    actor_id: str | None,
) -> go.Figure:
    """Create a football pitch visualisation of shot freeze-frame positions."""
    fig = go.Figure()
    _add_pitch_shapes(fig)

    coordinate_system = positions.get("coordinate_system") if isinstance(positions, Mapping) else None
    if isinstance(coordinate_system, Mapping):
        length = float(coordinate_system.get("length") or coordinate_system.get("pitch_length") or PITCH_LENGTH)
        width = float(coordinate_system.get("width") or coordinate_system.get("pitch_width") or PITCH_WIDTH)
    else:
        length = PITCH_LENGTH
        width = PITCH_WIDTH

    items: Sequence[Mapping[str, Any]] = positions.get("items", []) if isinstance(positions, Mapping) else []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        player = item.get("player", {})
        team_name = player.get("team", {}).get("name") or item.get("team", {}).get("name") or "Team"
        role = item.get("role")
        style = _ROLE_STYLE.get(role, {"color": "#7f7f7f", "symbol": "circle"})
        x = item.get("x")
        y = item.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            location = item.get("location") or item.get("coordinates")
            if isinstance(location, Mapping):
                x = location.get("x") or location.get("lon") or location.get(0)
                y = location.get("y") or location.get("lat") or location.get(1)
            elif isinstance(location, Sequence) and len(location) >= 2:
                x, y = location[0], location[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if 0.0 <= x <= 1.0 and length > 1:
            x = x * length
        if 0.0 <= y <= 1.0 and width > 1:
            y = y * width
        marker = dict(
            size=12,
            color=style["color"],
            symbol=style["symbol"],
            line=dict(color=_ACTOR_BORDER, width=2) if actor_id and player.get("id") == actor_id else None,
        )
        hover_text = _player_hover_text(player, team_name)
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=marker,
                name=f"{team_name} ({role})" if role else team_name,
                hovertemplate=f"%{{x:.1f}}, %{{y:.1f}}<br>{hover_text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Freeze-frame positions",
        showlegend=True,
        xaxis=dict(range=[0, PITCH_LENGTH], visible=False),
        yaxis=dict(range=[0, PITCH_WIDTH], visible=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
    )
    return fig
