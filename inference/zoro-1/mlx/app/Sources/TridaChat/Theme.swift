import AppKit
import SwiftUI

/// Zoro-1 Chat's palette: forest green and sand. The model's side of the conversation is forest (moss accent,
/// sage paper in light mode, deep pine in dark), the person's side is sand: their messages are the only warm
/// surfaces. Every color has a light and a dark value and follows the system appearance (NSColor's dynamic
/// provider), so no view needs to know which mode is on.
enum Theme {
    /// A color with one value in light mode and one in dark mode (hex RGB).
    static func dynamic(light: UInt32, dark: UInt32) -> Color { Color(nsColor: ns(light: light, dark: dark)) }

    static func ns(light: UInt32, dark: UInt32) -> NSColor {
        NSColor(name: nil) { appearance in
            NSColor(hex: appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light)
        }
    }


    // surfaces, from the back to the front
    static let sidebar = dynamic(light: 0xE2E9DF, dark: 0x14271F)
    static let canvas = dynamic(light: 0xEEF2EA, dark: 0x0F1F19)
    static let surface = dynamic(light: 0xFAFBF7, dark: 0x17302A)      // composer, code, decision cards
    static let surfaceRaised = dynamic(light: 0xDDE6D8, dark: 0x1E3B32) // code / table headers
    static let border = dynamic(light: 0xC9D5C3, dark: 0x2B4A40)

    // text
    static let text = dynamic(light: 0x1C2B23, dark: 0xE6EFEA)
    static let muted = dynamic(light: 0x5F7168, dark: 0x93AEA2)

    // the user's messages
    static let userBubble = dynamic(light: 0xE9DCC2, dark: 0x3B3322)
    static let userText = dynamic(light: 0x3A2E16, dark: 0xF2E8D3)

    // accents and states
    static let accent = dynamic(light: 0x1F8A5B, dark: 0x5ED6A0)
    static let accentSoft = dynamic(light: 0xCFE3D3, dark: 0x1E4237)  // selected rows, the meter's track
    static let onAccent = dynamic(light: 0xFFFFFF, dark: 0x0F1F19)    // icons on an accent fill
    static let warning = dynamic(light: 0xB7791F, dark: 0xE7B45A)
    static let danger = dynamic(light: 0xC2410C, dark: 0xF08A5D)
    static let success = accent
}

extension NSColor {
    convenience init(hex: UInt32) {
        self.init(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }
}

/// A rounded card: surface fill and a hairline border.
struct CardBackground: ViewModifier {
    var radius: CGFloat = 10
    var fill: Color = Theme.surface

    func body(content: Content) -> some View {
        content
            .background(RoundedRectangle(cornerRadius: radius, style: .continuous).fill(fill))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).stroke(Theme.border, lineWidth: 1))
    }
}

extension View {
    func card(radius: CGFloat = 10, fill: Color = Theme.surface) -> some View {
        modifier(CardBackground(radius: radius, fill: fill))
    }
}
