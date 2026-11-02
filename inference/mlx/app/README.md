# Zoro-1 Chat (macOS)

A native SwiftUI chat app for the on-device MLX server in `../trida_mlx`. It starts
`trida-mlx-server` in the background, streams replies over the local OpenAI-compatible API, and
shows the decode stats of every reply: self-spec vs AR, tok/s, tokens per forward, and how much
of the prompt came from the cache.

- **Get a Model.** Server → Get a Model… (or the button shown when no model is found)
  downloads Zoro-1 (or another Trida checkpoint) from Hugging Face, converts it to MLX q8, and
  loads it. For gated repos, paste an access token; it is used for that download only.
- **Models.** Lists every checkpoint in `~/models` (for example `Zoro-1-mlx-q8`,
  `Trida-2.0-4B-1006-mlx-q8`). The default is Zoro-1 q8. Switching models restarts the server.
- **Decoding.** Self-spec / AR toggle per message, with no restart: it sends `trida_mode`.
  Thinking is always on; the reasoning streams above each reply and collapses when it is done.
- **Images.** Drag an image onto the composer, choose one, or paste one. Only enabled when the
  served model has a vision encoder (Zoro-1, Trida-2.0-4B-1006).
- **Agent mode.** The model gets the same tools as `trida_mlx/agent.py`, confined to a
  workspace folder you choose:
  - always on: `list_dir`, `read_file`, `find_files` (by name), `search_files` (inside files),
    `calculator`, `now`. Paths may be relative or absolute, but must be inside the workspace;
  - `write_file` (opt-in in Settings → Agent; asks before every call) and `run_shell` (on by default), a terminal
    for the model: each chat has its own, whose working directory and exported variables carry over
    between commands (it starts in the workspace). Read-only commands (`ls`, `cat`, `grep`, `find`
    without `-exec`/`-delete`, `git status`/`log`/`diff`, …, also piped or chained, with no redirection to
    files) run without asking (Settings → Agent; on by default); everything else asks. Timeout 300 s by
    default, up to 1,800 s per call; long output keeps its first 4,000 and last 8,000 characters;
  - with a vision model (Zoro-1, Trida-2.0-4B-1006): `view_image` (an image file in the
    workspace) and `screenshot` (main display; on by default, asks every time, and macOS asks once for
    Screen Recording permission). The picture goes back to the model inside the tool result,
    and the tool card shows a thumbnail.
- **Slash commands.** Typing `/` in the composer opens a command menu, as in Claude Code: ↑↓ to
  choose, Return to run (`/de` + Return runs `/decide`), Tab to complete, Esc to close.
  `/decide [question]`, `/new`, `/agent` (agent mode on/off), `/mode`
  (self-spec ↔ AR). Unknown `/words` are sent as normal messages.
- **`/decide` in the chat.** Run `/decide` (or `/decide <question>`): the conversation so far, and
  its latest image, becomes the decision state, and the app asks step by step: the question, then
  its options (comma-separated, `key: description` allowed; or `yes/no`; or `score: low, medium,
  high`), then whether to add another question or **decide**. The answers appear as a card with a
  probability bar per option (below 0.7 the model reasons first; the reasoning is shown collapsed),
  and a text summary of them stays in the conversation for the model. The set-up messages are never
  sent to the model. `cancel` stops; an empty chat asks for the situation first.
- **Decide window** (Shift-⌘D, or **Decide** in the toolbar). Typed decisions by hand, as in the
  decision demos. It opens empty: write a situation (optionally with an image), add questions of type *choice*,
  *yes / no* or *score*, and get a probability bar per option. Reasoning: *when unsure* (below 0.7 by
  default, then decide again with the reasoning, shown live), *always* or *never*. The *Examples*
  menu loads one of four samples on request.
- **Tool routing** (Settings → Agent; on by default with a decision head). Before the model answers a
  message in agent mode, the decision model answers one choice question about the last six lines of the
  conversation: "What does replying to the user's last message need?" with the options `answer` (directly
  from the conversation) and the tool kinds that are enabled (`files`, `screen`, `time`, `math`, `shell`,
  `decide`, `app`). It decides directly (no reasoning pass, so routing stays fast). `answer` → the
  request carries `tool_choice: "none"` (the server keeps the schemas in the prompt, so the prompt cache still
  matches, and bans `<tool_call>`); a tool kind → only that kind (plus any other at ≥0.15) can be called.
  Below 0.4 → every tool stays available. The choice shows above the reply.
  Without `write_file`, `files` also allows `run_shell` (the terminal is then how files get written). When
  the router has picked a tool kind, and after a tool result, the request relaxes the server's tool-call
  gate to 0.2 (`tool_call_min_prob`), since a call is expected there; 0.7 guards everything else. The app starts
  the server with the loop guard and the 0.7 gate on (`--loop-guard --tool-call-gate 0.7`); both are off in a
  plain `trida-mlx-server`, including one you start yourself and the app connects to.
- **Reply review** (Settings → Agent → *Check replies*, on by default with a decision head). After each
  finished text reply, the decision model answers five yes/no checks about it: written in the language the
  user asked for (or wrote in); does what was asked; complete; claims match the tool results (no "saved
  the file" when the call failed or was refused); not a repeat of the previous reply. Below
  0.7 it reasons first. A check below 0.5 fails: the model is asked once to rewrite the reply with that
  feedback (the user's image goes with the checks; "claims match results" only runs when tools ran), the
  draft stays visible but collapsed and is no longer sent to the model, and the rewrite is
  checked again (no second rewrite). The result shows under each reply, e.g. `Check: ✓ language 0.97 · …`.
- **Feedback and training data.** Under each reply: 👍 / 👎, and ✏️ to edit the reply into what it should
  have been (the model sees your version from then on; the original is kept). Server → **Export Training
  Data…** asks first, then runs `trida-mlx-datamine` on the saved chats and request logs and writes JSONL to
  a folder you pick: preference pairs (drafts rewritten after a reply check, replies you edited), rated
  replies, tool-use turns with routing and call outcomes, the decision model's verdicts, failed generations,
  and every logged generation as a trajectory. Account names, emails and tokens are replaced and images
  left out; messages and tool output stay, so review before sharing. Labels are candidates, not gold.
- **Simulated users** (Settings → Agent → *Allow automation*, off by default). A local control API on
  127.0.0.1:8765 (token in `~/Library/Application Support/Zoro-1 Chat/automation-token`, readable by your user
  only) lets a program use the app like a person: new chat, send a message, wait for the whole turn, read what
  happened. `trida-mlx-simuser` uses it with an LLM playing the user (persona + goal per scenario, pushing
  back on wrong replies, `[DONE]` when finished) to collect more trajectories through the real harness.
  Approvals still show the normal sheet. Episode logs go to `~/Library/Logs/Zoro-1 Chat/simuser`.
- **Decision tools.** With a checkpoint that has the decision head, agent mode also offers tools
  answered by the decision model rather than by generated text, each reasoning first when its top
  probability is below 0.7: `decide_choice(question, options, context)`, `decide_yes_no(question,
  context)`, `pick_file(goal, path)` (over a folder's names) and, with the UI agent, `ui_check(goal)`
  (is it done on screen?). They change nothing, so they never ask.
- **UI agent (beta).** Settings → Agent → *UI agent*. The model can operate the app you used last
  before switching to Zoro-1 Chat (shown in the status bar), through the macOS Accessibility API:
  - `ui_elements` lists the focused window's and menu bar's clickable elements, numbered, as the
    screen-agent demo describes them (`button 'Send'`, `checkbox 'Wi-Fi' (on)`, `text field 'To' (empty)`);
  - `ui_click(goal)` does **not** let the chat model name the element. It sends a typed decision to
    the server's `/v1/decide`: the state is the app, the goal, the actions so far and the element list;
    the options are the elements; Zoro-1's pointer head picks one. When its top probability is below
    0.7 the server first reasons (thinking on, self-spec, at most 8,192 tokens), adds the reasoning to
    the state and decides again. The tool card shows the pick, its probability, the runner-up options
    and the reasoning;
  - `ui_type(text, field?, submit?)` (refuses password fields; with several fields and none focused,
    a decision picks the field), `ui_key("cmd+s")` and `open_app("Notes")`.

  Every action asks first; **Allow All for This Message** skips the questions for the rest of that
  reply, except picks with probability below 0.4. **⌘⎋** stops at any time, also from the other app.
  It needs the Accessibility permission (Settings → Agent → *Allow…*; macOS asks again after a
  rebuild of an ad-hoc-signed app) and a checkpoint with the decision head `jev_head.pt`.
  Folders converted before decision support lack it: Settings → Agent → *Decision head → Download*
  fetches it from `trillionlabs/Zoro-1` and restarts the server. Sending the window screenshot with
  each decision is optional and off: in the demo the element list alone completed more tasks.
- **Server.** If a server is already running on the port (for example one you started in a
  terminal), the app connects to it instead of launching its own. Server → Server Log shows the
  process output.

## Share it (self-contained build)

```bash
cd inference/mlx/app
zsh build_app.sh --bundle --zip
```

This puts a relocatable CPython 3.12, `trida_mlx`, MLX and the other locked dependencies from
`../uv.lock` inside the app (`Contents/Resources/python`, about 0.5 GB), smoke-tests the result,
and writes `dist/Zoro-1-Chat-<version>.zip`. Building it needs `uv` and network access. The
person you send it to needs only an Apple Silicon Mac on macOS 14 or later: no Python, no repo,
no `uv`. On first launch the app offers **Get a Model**, which uses the same built-in runtime.

**Opening it on another Mac.** The default build is signed ad-hoc, so macOS blocks it at first
("can't be opened" or "is damaged"). The recipient either:

- clicks **Open Anyway** in System Settings → Privacy & Security after the first attempt, or
- runs `xattr -dr com.apple.quarantine "/Applications/Zoro-1 Chat.app"` once.

To avoid the warning entirely, sign it with a Developer ID certificate and notarize it. Set it
up once with `xcrun notarytool store-credentials`, then:

```bash
SIGN_IDENTITY="Developer ID Application: <Team> (<TEAMID>)" NOTARY_PROFILE=<profile> \
  zsh build_app.sh --bundle --zip
```

This signs every binary in the runtime with the hardened runtime and `entitlements.plist`,
then notarizes and staples the app. This path has not been run yet.

A bundled build can still run from a checkout: Settings → Server → *Use a trida-stack checkout
instead*.

## Requirements (development build)

- An Apple Silicon Mac on macOS 14 or later.
- Xcode, or the Command Line Tools (`xcode-select --install`), for `swift`.
- The MLX project synced once: `cd inference/mlx && uv sync`.
- About 15 GB free for the first model: the ~10 GB bf16 download plus the ~5 GB q8 copy. The
  original download can be deleted once the conversion is done.

## Build and run

```bash
cd inference/mlx/app
zsh build_app.sh --install    # builds "Zoro-1 Chat.app" and copies it to ~/Applications (or /Applications)
open "build/Zoro-1 Chat.app"
```

For development, `swift run` starts the app directly, and `swift test` runs the core tests
(the streaming parser, tool confinement, the calculator and the download log). `swift test` needs Xcode.

On first launch, open **Settings (⌘,)** and check:

- **MLX project:** where `trida-stack/inference/mlx` lives. The default is
  `~/Desktop/trillionlabs/trida-stack/inference/mlx`.
- **Models folder:** the default is `~/models`.

The app runs `<MLX project>/.venv/bin/trida-mlx-server` when that exists, and otherwise falls
back to `uv run trida-mlx-server` through a login shell.

## Layout

| Path | Contents |
|---|---|
| `Sources/TridaChatCore` | Foundation-only code: the SSE streaming client, server process, model catalog and setup (download + convert), agent tools |
| `Sources/TridaChat` | SwiftUI: window, chat, composer, settings, tool approval |
| `Tests/TridaChatCoreTests` | Parser and tool tests (the parser is fed the server's exact SSE lines) |
| `build_app.sh` | Builds the `.app`; `--bundle` embeds the Python runtime; `--zip` packages it for sharing; optional Developer ID signing and notarization |
| `entitlements.plist` | Hardened-runtime exceptions for the embedded Python (Developer ID builds only) |

## Notes

- The app is not sandboxed: it launches a Python process and reads your workspace.
- **Context and compaction.** The context window is 128K tokens (Settings → Server → Context length;
  the reply budget, thinking included, is capped at whatever the prompt leaves). When a chat passes 80% of
  it, the app first asks the model to summarize the conversation, as Claude Code does, and from then on
  sends the summary (in the system message) instead of the messages before it. The earlier messages stay
  visible above a "conversation summarized" notice, whose summary can be expanded. It also compacts once
  when the server rejects a prompt as too long, and on `/compact`. At 128K the attention KV cache of one
  conversation can reach about 4 GB (8 attention layers × 4 KV heads × 256 × bf16, ~32 KB per token).
- Chats are saved in `~/Library/Application Support/Zoro-1 Chat/chats` (one JSON file per chat, with its images, tool results and decision cards) and come back in the sidebar after a restart; deleting a chat deletes its file. A reply that was still running when the app quit is kept up to where it stopped.
- Generation runs one request at a time, like the server. **Stop** (⌘.) closes the connection,
  which also stops decoding on the server.
- Sampling defaults follow the Zoro-1 card's tool-use setting: temperature 0.6, top-k 20,
  top-p 0.95. The same values are the started server's defaults, so Hermes or any other client
  pointed at it gets them too. Set temperature to 0 for greedy decoding, which reproduces AR
  output up to floating-point ties.
- **Request logs.** One JSON file per request in `~/Library/Logs/Zoro-1 Chat/requests`
  (Server → Open Request Logs; on by default, Settings → Server to turn off). Each holds the
  messages and tools sent, the model's raw output before tool-call parsing, the parsed result
  and the decode stats. Use these to diagnose tool-format failures.
- SwiftUI's `@State` is a compiler macro on recent SDKs, and its plugin ships with Xcode but not
  with the Command Line Tools. The app keeps its view state in `AppState` instead, so it builds
  with either.
