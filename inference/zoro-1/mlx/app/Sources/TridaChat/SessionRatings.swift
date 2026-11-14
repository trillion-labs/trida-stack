import SwiftUI
import TridaChatCore

/// The user's answer to "How's the agent doing so far?", for online RL (phase 1: rewards only, no training).
/// `trida-mlx-rewards` (trida_mlx/rewards.py) spreads each rating over the turns since the previous one.
struct SessionRating: Codable, Equatable {
    var afterMessageID: UUID
    /// 1 bad, 2 good, 3 very good; 0 = the card was closed (no signal, but it waits another few messages).
    var value: Int
    var date = Date()
}

extension AppState {
    /// Messages since the last rating (or closed card) before the card shows again.
    static let ratingEvery = 4

    /// The reply the rating card sits under, when one is due: the latest reply is finished and real (not a notice,
    /// an error or a /decide step), and the user has sent at least `ratingEvery` messages since the last rating.
    func ratingDue(_ chat: Chat) -> UUID? {
        guard askRatings, !isGenerating, let last = chat.messages.last, last.role == .assistant,
              last.decide == nil, last.error == nil, !last.content.isEmpty else { return nil }
        let ratings = chat.ratings ?? []
        let since = ratings.last.flatMap { r in chat.messages.firstIndex { $0.id == r.afterMessageID } } ?? -1
        let sent = chat.messages.indices.filter { $0 > since && chat.messages[$0].role == .user
                                                  && chat.messages[$0].decide == nil }.count
        return sent >= Self.ratingEvery ? last.id : nil
    }

    func rateSession(_ chatID: Chat.ID, after: UUID, value: Int) {
        guard let ci = chats.firstIndex(where: { $0.id == chatID }) else { return }
        chats[ci].ratings = (chats[ci].ratings ?? []) + [SessionRating(afterMessageID: after, value: value)]
        if value > 0 { react(value == 1 ? .notes : .love) }
    }
}

/// "How's the agent doing so far?" under the latest reply, a minute after it ends (see `ratingDue`).
struct SessionRatingCard: View {
    @EnvironmentObject var state: AppState
    let chat: Chat
    @State private var shownFor: UUID?

    var body: some View {
        let due = state.ratingDue(chat)
        Group {
            if let due, shownFor == due {
                HStack(spacing: 10) {
                    Text("How's the agent doing so far?")
                        .font(.callout).foregroundStyle(.secondary)
                    Button("Bad") { state.rateSession(chat.id, after: due, value: 1) }
                    Button("Good") { state.rateSession(chat.id, after: due, value: 2) }
                    Button("Very good") { state.rateSession(chat.id, after: due, value: 3) }
                    Spacer(minLength: 0)
                    Button { state.rateSession(chat.id, after: due, value: 0) } label: {
                        Image(systemName: "xmark")
                    }
                    .buttonStyle(.borderless)
                    .help("Not now")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                .transition(.opacity)
            } else {
                Color.clear.frame(height: 0)
            }
        }
        .task(id: due) {
            shownFor = nil
            guard let due else { return }
            try? await Task.sleep(for: .seconds(60))
            if !Task.isCancelled { withAnimation { shownFor = due } }
        }
    }
}
