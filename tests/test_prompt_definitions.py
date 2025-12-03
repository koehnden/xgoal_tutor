from pathlib import Path


def test_prompts_include_cutback_definition():
    prompts_dir = Path("src/xgoal_tutor/prompts")
    defense_prompt = (prompts_dir / "xgoal_defense_prompt.md").read_text()
    offense_prompt = (prompts_dir / "xgoal_offense_prompt.md").read_text()

    assert "has_cutback:" in defense_prompt
    assert "has_cutback:" in offense_prompt
    assert "reasonably clear passing lane for a cutback" in defense_prompt
    assert "High-value pass availability" in offense_prompt


def test_offense_prompt_includes_move_simulation_section():
    prompts_dir = Path("src/xgoal_tutor/prompts")
    offense_prompt = (prompts_dir / "xgoal_offense_prompt.md").read_text()

    gain_line = "- move_simulation_gain: The xG gain if the shooter had dribbled/moved instead of shooting."
    assert gain_line in offense_prompt

    label_line = "- move_simulation_heading_label: Plain-language label for the best heading direction."
    assert label_line in offense_prompt
