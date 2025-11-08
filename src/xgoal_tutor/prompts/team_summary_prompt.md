---
id: team_summary_prompt
version: v1
description: Prompt for summarising team attacking tendencies
requires:
  - home
  - away
  - teams
---
Provide a tactical summary for each team covering chance creation patterns, build-up tendencies, and finishing quality.
Use the notes below and relate them to broader team-level trends. Aim for 2-3 bullet points per team.

Fixture: {{ home }} vs {{ away }}.

{% if teams %}
{% for entry in teams %}Team: {{ entry.team }}
{% for note in entry.notes %}    - {{ note }}
{% endfor %}

{% endfor %}
{% else %}No team summaries available.
{% endif %}
