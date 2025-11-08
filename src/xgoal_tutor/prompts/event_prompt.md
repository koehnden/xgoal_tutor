---
id: event_prompt
version: v1
description: Baseline event explanation prompt for pipeline usage
requires:
  - header_lines
  - home
  - away
  - competition
  - season
  - minute
  - second
  - player
  - team
  - xg
  - context_block
  - feature_block
word_limit: 120
---
{{ header_lines | join('\n') }}

Match: {{ home }} vs {{ away }} ({{ competition }} {{ season }}).
Event time: {{ "%02d"|format(minute) }}:{{ "%02d"|format(second) }}.
Player: {{ player }} ({{ team }}).
Model xG: {{ "%.3f"|format(xg) }}.
{% if context_block %}{{ context_block }}{% endif %}Feature contributions (positive increases xG, negative lowers it):
{{ feature_block }}
Provide a concise explanation referencing the most influential factors.
