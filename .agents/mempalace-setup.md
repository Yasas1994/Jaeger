# MemPalace Setup for Kimi Code

MemPalace is configured for this project. To recreate the setup on a new machine, follow the steps below.

## 1. Install MemPalace

```bash
uv tool install mempalace
```

`pipx install mempalace` works too. Plain `pip` is fine inside an activated virtualenv.

## 2. Register the MemPalace MCP server with Kimi Code

Add to `~/.kimi-code/mcp.json`:

```json
{
  "mcpServers": {
    "mempalace": {
      "command": "mempalace-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

Use a custom palace path if needed:

```json
{
  "mcpServers": {
    "mempalace": {
      "command": "mempalace-mcp",
      "args": ["--palace", "/path/to/palace"],
      "env": {}
    }
  }
}
```

## 3. Initialize and mine the project

From the repository root:

```bash
mempalace init . --yes
mempalace mine .
```

The project already includes a `mempalace.yaml` that maps the `jaeger` wing to the repository folders.

## 4. Verify the integration

Test the MCP server directly:

```bash
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | mempalace-mcp | python -m json.tool
```

In a fresh Kimi Code session:

```bash
kimi -p "List the tools from the mempalace MCP server."
```

You can then ask Kimi to search the palace, e.g.:

```bash
kimi -p "Search the palace for training speed comparisons."
```

## 5. Optional — pre-approve safe MemPalace tools

Add to `~/.kimi-code/config.toml`:

```toml
[[permission.rules]]
decision = "allow"
pattern = "mcp__mempalace__status"

[[permission.rules]]
decision = "allow"
pattern = "mcp__mempalace__list_*"

[[permission.rules]]
decision = "allow"
pattern = "mcp__mempalace__search"

[[permission.rules]]
decision = "allow"
pattern = "mcp__mempalace__get_*"

[[permission.rules]]
decision = "deny"
pattern = "mcp__mempalace__delete_*"

[[permission.rules]]
decision = "deny"
pattern = "mcp__mempalace__kg_invalidate"
```

This allows read-only/search operations and keeps destructive operations behind manual approval.

## 6. Always save plans and decisions

Create `AGENTS.md` in the project root with explicit MemPalace memory instructions. Kimi Code loads it as a project-level system prompt, so every session is reminded to:

- Search the palace before answering questions about past plans or decisions.
- Save every new or updated plan to the `jaeger/plans` room via `mcp__mempalace__add_drawer`.
- Record decisions as knowledge-graph facts via `mcp__mempalace__kg_add`.
- Write a session diary entry at the end of each session via `mcp__mempalace__diary_write`.

See the current `AGENTS.md` in this repo for the exact wording.

## 7. Autosave hooks

MemPalace's autosave hooks are adapted to Kimi Code via a small shim because Kimi Code's hook JSON and transcript layout differ from Claude Code/Codex.

### 7.1 Install the shim

Create `~/.kimi-code/hooks/mempalace-autosave.py` with the contents from the project plan or from the canonical version maintained in this repository (if committed). The shim:

- Receives Kimi Code `Stop` and `PreCompact` hook events.
- Looks up the session directory in `~/.kimi-code/session_index.jsonl`.
- Converts `agents/main/wire.jsonl` to a Claude-Code-style transcript.
- Writes the transcript under `~/.mempalace/kimi_transcripts/<session_id>/` so MemPalace only mines that directory.
- Invokes `mempalace hook run --hook stop|precompact --harness claude-code`.
- Does **not** re-mine the project automatically (to avoid lock contention); run `mempalace mine .` manually after significant code changes.

Make it executable:

```bash
chmod +x ~/.kimi-code/hooks/mempalace-autosave.py
```

### 7.2 Register the hooks

Append to `~/.kimi-code/config.toml`:

```toml
[[hooks]]
event = "Stop"
command = "/home/yasas-wijesekara/.kimi-code/hooks/mempalace-autosave.py"
timeout = 60

[[hooks]]
event = "PreCompact"
command = "/home/yasas-wijesekara/.kimi-code/hooks/mempalace-autosave.py"
timeout = 60
```

Adjust the absolute path to match your home directory.

### 7.3 Verify hooks

After a fresh Kimi Code session reaches ~15 user messages, check:

```bash
tail -20 ~/.mempalace/hook_state/hook.log
```

You should see `TRIGGERING SAVE at exchange 15` and a diary checkpoint entry. Trigger `/compact` and check the log again for a `PRE-COMPACT triggered` line.

## References

- MemPalace: <https://mempalaceofficial.com/>
- MemPalace hooks guide: <https://mempalaceofficial.com/guide/hooks.html>
- Kimi Code hooks docs: <https://www.kimi.com/code/docs/en/kimi-code-cli/customization/hooks.html>
