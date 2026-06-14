# Jaeger — Agent Instructions

## Project background

Jaeger (`jaeger-bio`) is a homology-free deep-learning command-line tool for identifying bacteriophage genome sequences. The repository lives at `/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger` and uses Python, TensorFlow, NumPy, and YAML training configs.

## Memory protocol — MemPalace

This project uses [MemPalace](https://mempalaceofficial.com/) for durable, searchable memory across Kimi Code sessions. You MUST follow this protocol on every turn:

1. **On wake-up / session start**: call `mcp__mempalace__mempalace_status` to load the palace overview and orient yourself.
2. **Before answering** any question about past plans, decisions, people, or project facts: call `mcp__mempalace__mempalace_search` or `mcp__mempalace__mempalace_kg_query`. Do not guess.
3. **After creating or updating a plan**: save it immediately with `mcp__mempalace__mempalace_add_drawer`:
   - `wing`: `jaeger`
   - `room`: `plans`
   - `topic`: `plan`
   - `content`: the full plan text, including goals, tasks, exact file paths, and decisions.
4. **After making a decision**: record it with `mcp__mempalace__mempalace_kg_add`:
   - Example: `subject="Jaeger project"`, `predicate="decided_to"`, `object="adopt AGENTS.md memory protocol"`, `valid_from="YYYY-MM-DD"`.
5. **After each session**: write a diary entry with `mcp__mempalace__mempalace_diary_write`:
   - `agent_name`: `kimi-code`
   - `wing`: `jaeger`
   - `topic`: `session-summary`
   - `entry`: a concise, factual summary of what was discussed, decided, implemented, and left unfinished.
6. **When facts change**: invalidate the old fact with `mcp__mempalace__mempalace_kg_invalidate` and add the corrected fact with `mcp__mempalace__mempalace_kg_add`.
7. **Do not bulk-delete drawers or knowledge-graph facts** without explicit user approval.

## Tool permission note

Read/search MemPalace tools (`mempalace_status`, `mempalace_search`, `mempalace_get_*`, `mempalace_kg_query`) are pre-approved in `~/.kimi-code/config.toml`. Write tools (`add_drawer`, `kg_add`, `diary_write`) will trigger approval prompts unless the user enables YOLO mode.
