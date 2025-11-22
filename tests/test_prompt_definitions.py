from pathlib import Path


def test_prompts_include_cutback_definition():
    prompts_dir = Path("src/xgoal_tutor/prompts")
    definition = (
        "has_cutback: Flag if a central teammate in front of the ball (towards goal) "
        "has a reasonably clear passing lane for a cutback (high-value square/rule-back pass)."
    )

    defense_prompt = (prompts_dir / "xgoal_defense_prompt.md").read_text()
    offense_prompt = (prompts_dir / "xgoal_offense_prompt.md").read_text()

    assert definition in defense_prompt
    assert definition in offense_prompt


def test_offense_prompt_includes_move_simulation_section():
    prompts_dir = Path("src/xgoal_tutor/prompts")
    offense_prompt = (prompts_dir / "xgoal_offense_prompt.md").read_text()

    expected = "- move_simulation_note: Brief summary of whether a better short move exists."
    assert expected in offense_prompt
