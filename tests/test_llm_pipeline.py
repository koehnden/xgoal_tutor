from typing import Dict, List

from xgoal_tutor.llm.pipeline import (
    EventExplanationInput,
    ExplanationPipeline,
    normalize_feature_contributions,
)


class FakeLLM:
    """Simple stand-in for :class:`OllamaLLM` used in integration-style tests."""

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def generate(self, prompt: str, *, model=None, options: Dict[str, object] | None = None):
        self.prompts.append(prompt)
        if "Feature contributions" in prompt:
            features = [
                line[2:].split(":", 1)[0].strip()
                for line in prompt.splitlines()
                if line.startswith("- ")
            ]
            text = f"Explained contributions from {', '.join(features)}"
            return text, "fake-primary"
        if "Important attacking moments" in prompt:
            assert "Explained contributions" in prompt, "event explanations should feed the summary"
            return "Match summary built from event narratives.", "fake-primary"
        if "player-facing feedback" in prompt:
            assert "Explained contributions" in prompt, "player notes should include event text"
            return "Player feedback built from event notes.", "fake-primary"
        if "Provide a tactical summary for each team" in prompt:
            assert "Explained contributions" in prompt, "team notes should include event text"
            return "Team feedback built from event notes.", "fake-primary"
        raise AssertionError(f"Unexpected prompt: {prompt}")


def test_pipeline_generates_explanations_and_summaries():
    llm = FakeLLM()
    pipeline = ExplanationPipeline(llm, top_features=2)

    match_metadata = {
        "competition": "Champions League",
        "season": "2023/24",
        "teams": {"home": "Home FC", "away": "Away FC"},
    }

    events = [
        EventExplanationInput(
            event_id="1",
            minute=12,
            second=30,
            team="Home FC",
            player="Alex Nine",
            xg=0.45,
            contributions={
                "shot_distance": -0.2,
                "pressure": 0.5,
                "keeper_position": 0.1,
            },
            context="quick transition from the left half-space",
        ),
        EventExplanationInput(
            event_id="2",
            minute=78,
            second=4,
            team="Away FC",
            player="Jamie Ten",
            xg=0.28,
            contributions={
                "shot_distance": -0.1,
                "angle": 0.2,
                "defenders_between_ball_and_goal": -0.3,
            },
        ),
    ]

    result = pipeline.run(match_metadata, events)

    explanations = [r.explanation for r in result.event_explanations]
    assert explanations == [
        "Explained contributions from pressure, shot_distance",
        "Explained contributions from defenders_between_ball_and_goal, angle",
    ]

    assert result.match_summary == "Match summary built from event narratives."
    assert result.player_summaries == "Player feedback built from event notes."
    assert result.team_summaries == "Team feedback built from event notes."
    assert len(result.models_used) == len(events) + 3

    first_event_prompt = llm.prompts[0]
    assert "Additional context: quick transition from the left half-space" in first_event_prompt
    assert "Feature contributions" in first_event_prompt

    match_summary_prompt = llm.prompts[2]
    assert "Explained contributions from pressure, shot_distance" in match_summary_prompt
    assert "Explained contributions from defenders_between_ball_and_goal, angle" in match_summary_prompt

    player_summary_prompt = llm.prompts[3]
    assert "Player: Alex Nine" in player_summary_prompt
    assert "Player: Jamie Ten" in player_summary_prompt

    team_summary_prompt = llm.prompts[4]
    assert "Team: Home FC" in team_summary_prompt
    assert "Team: Away FC" in team_summary_prompt


def test_pipeline_limits_feature_list():
    llm = FakeLLM()
    pipeline = ExplanationPipeline(llm, top_features=3)

    match_metadata = {"teams": {"home": "Home FC", "away": "Away FC"}}

    event = EventExplanationInput(
        event_id="42",
        minute=5,
        second=15,
        team="Home FC",
        player="Alex Nine",
        xg=0.62,
        contributions={
            "big_chance": 0.6,
            "keeper_position": -0.4,
            "defensive_pressure": 0.25,
            "shot_angle": 0.1,
            "distance": -0.05,
        },
    )

    pipeline.run(match_metadata, [event])

    event_prompt = llm.prompts[0]
    feature_lines = [line for line in event_prompt.splitlines() if line.startswith("- ")]
    assert feature_lines == [
        "- big_chance: +0.600 (higher xG)",
        "- keeper_position: -0.400 (lower xG)",
        "- defensive_pressure: +0.250 (higher xG)",
    ]


def test_pipeline_surfaces_cutback_feature():
    llm = FakeLLM()
    pipeline = ExplanationPipeline(llm, top_features=1)

    match_metadata = {"teams": {"home": "Home FC", "away": "Away FC"}}

    events = [
        EventExplanationInput(
            event_id="cutback-1",
            minute=15,
            second=45,
            team="Home FC",
            player="Alex Nine",
            xg=0.31,
            contributions={
                "has_cutback": 0.9,
                "angle": 0.2,
                "defenders_between_ball_and_goal": -0.4,
            },
        )
    ]

    result = pipeline.run(match_metadata, events)

    first_prompt = llm.prompts[0]
    assert "- has_cutback: +0.900 (higher xG)" in first_prompt
    assert result.event_explanations[0].explanation == "Explained contributions from has_cutback"


def test_normalize_feature_contributions_from_coefficients():
    raw = [
        {"feature": "dist_sb", "coefficient": -0.9},
        {"feature": "angle_deg_sb", "coefficient": 0.37},
        {"feature": "__intercept__", "coefficient": -2.68},
    ]

    contributions = normalize_feature_contributions(raw)

    assert contributions == {
        "dist_sb": -0.9,
        "angle_deg_sb": 0.37,
        "__intercept__": -2.68,
    }


def test_normalize_feature_contributions_from_shap_payload():
    shap_payload = {
        "feature_names": ["dist_sb", "angle_deg_sb", "is_set_piece"],
        "expected_value": -2.81,
        "shap_values": [[-1.56, -2.60, -0.05]],
    }

    contributions = normalize_feature_contributions(shap_payload)

    assert contributions == {
        "dist_sb": -1.56,
        "angle_deg_sb": -2.60,
        "is_set_piece": -0.05,
        "__expected_value__": -2.81,
    }
