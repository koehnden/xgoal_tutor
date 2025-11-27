from xgoal_tutor.llm.models import (
    EventMetadata,
    FreezeFrameEntry,
    MatchMetadata,
    MoveSimulationContext,
    ShooterMetadata,
)


def test_llm_public_dataclasses_are_importable() -> None:
    frame_entry = FreezeFrameEntry(
        player_id=9,
        player_name="Jordan Smith",
        position_name="Striker",
        teammate=True,
        keeper=False,
        x=102.0,
        y=38.0,
    )
    move_context = MoveSimulationContext(
        block="- move_simulation_note: example",
        gain=0.25,
        heading_label="left channel",
    )
    match_metadata = MatchMetadata(
        home="Attacking FC",
        score_home="1",
        score_away="0",
        away="Defensive SC",
        competition="Champions League",
        season="2023/24",
        home_team_id=1,
        away_team_id=2,
    )
    event_metadata = EventMetadata(period="1", minute=23, second=12, play_pattern="open_play")
    shooter_metadata = ShooterMetadata(
        name="Jordan Smith",
        team_name="Attacking FC",
        position="Striker",
        body_part="right foot",
        technique="volley",
        start_x=102.0,
        start_y=38.0,
    )

    assert frame_entry.teammate is True
    assert move_context.gain == 0.25
    assert match_metadata.home == "Attacking FC"
    assert event_metadata.play_pattern == "open_play"
    assert shooter_metadata.position == "Striker"
