---
id: player_summary_prompt
version: v1
description: Prompt for creating player-specific attacking feedback
requires:
  - players
---
You are writing player-facing feedback for an attacking coach. Summarise each player's involvement
using the provided notes. Be specific about movement, decision making, and shot quality. Use bullet lists
per player and keep the tone constructive.

{% if players %}
{% for entry in players %}Player: {{ entry.player }}
{% for note in entry.notes %}    - {{ note }}
{% endfor %}

{% endfor %}
{% else %}No player events captured.
{% endif %}
