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
cd inference/zoro-1/mlx/app/Knowledge
swift build
swift run kg-selftest      # "kg-selftest: all checks passed"
```

## How it works

- **Facts** are `subject —relation→ object`. There are 21 built-in relations (`Relations.all`), and Zoro-1 adds new
  ones while learning when none fits:
  - It first checks whether an existing relation means the same thing (a choice, P ≥ 0.6), so synonyms such as
    "employed by" map onto `works_at`.
  - Otherwise it adds the relation as *provisional* and decides two flags with yes/no questions: one at a time,
    and changes over time.
  - A provisional relation becomes regular after 3 facts use it, or when you approve it in the Knowledge
    window's **Relations** tab. That tab can also rename a relation, change its flags, merge it into another, or
    delete it (its facts are kept as "related to", with the old name as their note).
  - At most 40 provisional relations wait at once; after that, new ones become "related to" with a note.
  - Each fact keeps when it became true, when it ended, where it came from (you said / from the web / you
    edited), the source message, and, for web facts, the page, the date and the supporting quote.
  - A *functional* relation (president, CEO, works at, lives in…) has one current object: a newer fact ends
    the older one, and the older one is kept with an end date instead of being deleted.
- **Learning**, after each finished reply, in the background:
  1. A yes/no gate (P ≥ 0.5): is there a lasting fact, preference, correction, or a fact looked up on the web?
     Documents the reply read get their own question first. A grounded web answer, `remember` and
     `/update-kg` skip the gate.
  2. A completion proposes facts as JSON: up to 3, 8 with pages or documents, 15 for `/update-kg`. Broken
     JSON (a `"` or a line break inside a string) is repaired.
  3. Each fact is checked against its source:
     - From the user: it must be about the user or name something the user wrote.
     - From the web or a document: only with a quote that is really in what was read (a quote labeled with
       the wrong one of the two is moved to the right one), and never about the user. Web facts also need
       the reply to have passed "backed by the pages".
     - Secrets (keys, passwords, card or ID numbers) are dropped.
  4. Entity linking: an exact name or alias match, otherwise Zoro-1 chooses between the closest entries and
     "new" (P ≥ 0.6 to link). A new entry's type is Zoro-1's choice (P ≥ 0.5), not the proposal's guess.
  5. Types and direction: each built-in relation has types for its two ends (`Relations.signatures`). A fact
     that fits only the other way round is turned around, one that fits neither way is dropped, and for one
     that fits both ways Zoro-1 says which way it goes.
- **Using it:** for each message, the facts about the user and the current facts about the entities it
  mentions are appended to it, with dates and sources. Hangul names match with particles attached, and Latin
  names match whole words.
  - **Priority:** the user's statements and edits outrank web facts. Web facts can't replace the user's.
  - **Possibly outdated:** time-sensitive facts older than 30 days are marked "may have changed since".
- **Storage:** `~/Library/Application Support/Zoro-1 Chat/knowledge.json`, written shortly after each change.
  Every change goes through `KnowledgeModel.mutate` and can be undone (⌘Z in the window). Each reply's
  learning can be undone from its chip.

## Wiring it into the app (feat/zoro-chat-app)

The app is wired up already: `Sources/TridaChat/KnowledgeGlue.swift` is the glue file, and the hooks below are in
place. The list stays here for reference.

1. **Package.swift:** add the package and the two libraries to the app target.
   ```swift
   dependencies: [.package(path: "Knowledge")],
   // in the TridaChat target:
   dependencies: ["TridaChatCore", .product(name: "TridaKnowledge", package: "Knowledge"),
                  .product(name: "TridaKnowledgeUI", package: "Knowledge")],
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
