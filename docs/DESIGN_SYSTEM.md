# SignSage — Design System

How the interface looks, behaves and speaks. The product is a **research-grade assistive instrument that looks serious without hiding experimental state.**
Tokens and component names below are the documented ones; the Tailwind 4 theme and `frontend/src/components` are the implementation — if they differ, update this file.

## 1. Principles

1. **Communication first, telemetry second.** The result of the conversation is large and readable; status and diagnostics are quiet.
2. **Honest states.** Verified, experimental, unsupported and unavailable each look different. Never a fake translation to make the screen look alive.
3. **Fail loudly.** Camera denied, poor tracking, hand out of frame, socket down, model unavailable, uncertain result — each gets a clear, specific message.
4. **Turn-taking is obvious.** Whose turn it is (signing or typing) is always visible.
5. **No decoration that implies capability.** The avatar is a research representation, never styled as a validated signer.

## 2. Foundations

| Token | Value |
|---|---|
| Headline font | Space Grotesk |
| Body font | Inter |
| Data / labels | JetBrains Mono |
| Spacing scale | 4 · 8 · 16 · 24 · 32 px |
| Desktop layout | 12-column "workbench" |
| Mobile layout | Single stream: turn → camera → output → status → controls → telemetry |
| Dark palette (documented) | `#090E13` · `#101720` · `#253342` · `#151E28` — role mapping to confirm in the theme |
| Surfaces | Flat, bordered cards; **no diffuse glows** |
| Theme | Dark by default with an Appearance toggle in the sidebar |

UI kit: React 19, Tailwind 4, shadcn/ui primitives, Three.js for the avatar.

## 3. Layout and navigation

Persistent left sidebar, collapsible:

```text
SignSage
Interpret   → Live Interpreter · Sign Generator
Library     → Translation Log
            Settings · Appearance (theme) · Sign in / account
```

Pages share one header pattern: a plain title (for example "Text → Sign"), a one-line statement of what the page does **and does not** do ("Dictionary lookup — not natural ISL grammar."), and a right-aligned status line ("14 signs · Human avatar ready").

## 4. Patterns

| Pattern | Rule |
|---|---|
| **Numbered section card** | Bordered card with a small uppercase mono label (`01 · YOUR MESSAGE`) and one muted helper line |
| **Segmented control** | Mutually exclusive modes (for example Dictionary / Assisted) |
| **Example chips** | Pill buttons that fill the input with a supported phrase |
| **Primary action** | One per card; disabled until valid input exists |
| **Caption on generated content** | Always visible, never in a tooltip (§6) |
| **Counters** | Show input limits (`17/280`) |
| **Help** | Short dialog linking to an external ISL reference; no embedded sign catalogue implying coverage |

## 5. Live Interpreter

Structure, top to bottom: **Recognition mode → Recognised words → Sentence construction → Interpretation → Audio output.** Camera and controls stack vertically below the `lg` breakpoint.

| State | Badge | Behaviour |
|---|---|---|
| Waiting | waiting | Nothing shown as a result |
| Recognising | recognising | Interim words visible, marked as interim |
| Composing | composing | Sentence being built from the recognised words |
| Done | composed | Final sentence; Speak available |
| Low confidence | not confidently recognised | Reason shown; offer fingerspelling when available; **no guessed text** |
| Model off | unavailable | Distinct panel explaining the gate; not an error toast |
| No camera / hand lost / socket down | specific message | Say what failed and what to do |

Auto-Speak plays completed interpretations only, once each. A manual Speak button replays the current interpretation. The static alphabet mode is a separate toggle in the page header and is labelled as image classification.

## 6. Text → Sign and the avatar

- Show the **gloss sequence** and the playing gloss ("Signing HELLO (1 of 3)") with Play / Stop / Replay / Clear.
- Unknown words are listed as dropped: **"Unknown words are dropped, never guessed."**
- A gloss with no motion shows a **"No clip yet"** alert and is skipped. Never substitute a generic animation.
- Persistent caption on the avatar: *"Human rig · motion from recorded/authored data — not a validated ISL sign."* Remove or soften it only after a fluent signer has reviewed that motion, per gloss.
- Authored approximations are labelled as such in data (`needsReview`) and in the UI.
- The avatar model is fixed (MakeHuman rig); only poses change. See [SIGN_ANIMATION](SIGN_ANIMATION.md).

## 7. Voice and copy

- Precise, scoped statements: "Interpreted from 3 recognised words", not "Perfect translation".
- Allowed vocabulary: *Implemented, Verified, Experimental, Unavailable, Not validated.* Never *accurate, reliable, fluent, production-ready* without evidence ([PRD §6](PRD.md#6-honesty-and-claims-policy)).
- Landing and About content must cover: what it is, how it works, what is implemented, what research showed, privacy, limitations, and the entry to the app — and advertise no accuracy that is not documented.
- Errors name the cause and the next step.

## 8. Credits and attribution (required)

A visible **Credits** surface (About page and footer link) must list: the V1 contributors, the FDMSE-ISL authors with the full citation, other datasets, and third-party components with their licenses — content from [CREDITS.md](../CREDITS.md), kept in sync. The Live Interpreter and Text → Sign pages carry a short "powered by research data from…" line linking to it once a model trained on a credited dataset is served.

## 9. Accessibility checklist

Targets to meet and verify (specific tools and results still to be recorded): sufficient colour contrast in both themes; full keyboard operation including camera controls; visible focus; live-region announcements for new interpretations; captions never conveyed by colour alone; motion that can be paused; readable text size without zoom on mobile.

## 10. Adding or changing UI

1. Reuse an existing shadcn primitive before writing a new one.
2. Every new state of an AI-generated result needs a row in §5 or §6.
3. Update this file and add a test for any state that decides whether a claim is shown.
