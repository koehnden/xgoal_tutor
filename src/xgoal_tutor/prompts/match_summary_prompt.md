---
id: match_summary_prompt
version: v1
description: Prompt for summarising match attacking themes
requires:
  - home
  - away
  - competition
  - season
  - moments
---
You are preparing a post-match tactical summary for coaching staff.
Using the event explanations below, highlight the overall attacking patterns, key tactical themes,
and how chance quality evolved across the match. Provide 3-4 short paragraphs.

Match: {{ home }} vs {{ away }} ({{ competition }} {{ season }}).
Important attacking moments:
{% if moments %}
{% for moment in moments %}- {{ moment }}
{% endfor %}
{% else %}- None provided.
{% endif %}
