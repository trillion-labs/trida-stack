# Personal knowledge graph for Zoro-1 Chat

A standalone Swift package with the user's personal knowledge graph, learned from their chats. The design is
in the project doc *zoro1-knowledge-graph-design*. Nothing here depends on the app, so the package builds and
tests on its own, and wiring it into the app takes a handful of small hooks.

| Target | What it is |
|---|---|
| `TridaKnowledge` | The graph (`KnowledgeGraph`: nodes, aliases, dated facts, undo records) and the learning pipeline (`Learner`). Foundation only. |
| `TridaKnowledgeUI` | `KnowledgeModel` (file persistence, undo, background learning, window state), `KnowledgeView` (the Knowledge window) and `LearnedChip` (under a reply). |
| `kg-selftest` | The checks, runnable with only the Command Line Tools (no XCTest). |

```sh
cd inference/mlx/app/Knowledge
swift build
swift run kg-selftest      # "kg-selftest: all checks passed"
```

## How it works

- **Facts** are `subject —relation→ object` with a fixed vocabulary of 21 relations (`Relations.all`).
  - Each fact keeps when it became true, when it ended, where it came from (you said / from the web / you
    edited), the source message, and, for web facts, the page, the date and the supporting quote.
  - A *functional* relation (president, CEO, works at, lives in…) has one current object: a newer fact ends
    the older one, and the older one is kept with an end date instead of being deleted.
- **Learning**, after each finished reply, in the background:
  1. A strict yes/no gate (P ≥ 0.8): is there a lasting fact, preference, correction, or a fact looked up on
     the web?
  2. A short completion proposes up to 3 facts as JSON.
  3. Each fact is checked against its source:
     - From the user: it must be about the user or name something the user wrote.
     - From the web: only when the reply passed "backed by the pages", only with a quote that is really in
       the web results, and never about the user.
     - Secrets (keys, passwords, card or ID numbers) are dropped.
  4. Entity linking: an exact name or alias match, otherwise Zoro-1 chooses between the closest entries and
     "new" (P ≥ 0.6 to link).
- **Using it:** for each message, the facts about the user and the current facts about the entities it
  mentions are appended to it, with dates and sources. Hangul names match with particles attached, and Latin
  names match whole words.
  - **Priority:** the user's statements and edits outrank web facts. Web facts can't replace the user's.
  - **Possibly outdated:** time-sensitive facts older than 30 days are marked "may have changed since".
- **Storage:** `~/Library/Application Support/Zoro-1 Chat/knowledge.json`, written shortly after each change.
  Every change goes through `KnowledgeModel.mutate` and can be undone (⌘Z in the window). Each reply's
  learning can be undone from its chip.

## Wiring it into the app (feat/zoro-chat-app)

1. **Package.swift:** add the package and the two libraries to the app target.
   ```swift
   dependencies: [.package(path: "Knowledge")],
   // in the TridaChat target:
   dependencies: ["TridaChatCore", .product(name: "TridaKnowledge", package: "TridaKnowledge"),
                  .product(name: "TridaKnowledgeUI", package: "TridaKnowledge")],
   ```
2. **The glue file:** copy `Integration/KnowledgeGlue.swift.txt` to `Sources/TridaChat/KnowledgeGlue.swift`.
   Then in `AppState` (with `import TridaKnowledge` and `import TridaKnowledgeUI`):
   ```swift
   let knowledge = KnowledgeModel()
   lazy var knowledgeBackend = AppKnowledgeBackend(app: self)
   ```
   In `init()`, let the window's "Open the message it came from" select the chat:
   ```swift
   knowledge.openSource = { [weak self] chat, _ in self?.selectedChatID = UUID(uuidString: chat) }
   ```
3. **Learn after each reply:** where a reply is final (after `review(...)` returns, at the end of the turn), call
   `learnFromReply(chatID: chatID, replyID: reply.id)`.
4. **Use the facts:** at the end of `history(_:tools:)`, before returning the messages, call
   `addKnowledge(to: &messages)`.
5. **The window:** in `TridaChatApp`:
   ```swift
   Window("Knowledge", id: "knowledge") {
       KnowledgeView(model: delegate.state.knowledge)
   }
   // and in CommandGroup(after: .sidebar):
   OpenKnowledgeButton().keyboardShortcut("k", modifiers: [.command, .shift])
   ```
6. **The chip:** in `MessageView`, under a reply's footer (with `@Environment(\.openWindow) var openWindow`):
   ```swift
   LearnedChip(model: state.knowledge, messageID: message.id.uuidString) { edge in
       state.knowledge.show(edge: edge)
       openWindow(id: "knowledge")
   }
   ```
7. **Settings (optional):** `Toggle("Learn from chats automatically", isOn: $state.knowledge.enabled)`.

## Not in this version

- The "as of" date slider and two-hop retrieval.
- Per-workspace graphs, and chats excluded from learning.
- A `recall` tool. `KnowledgeGraph.search(_:)` is there for it.
