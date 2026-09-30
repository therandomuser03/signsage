# SignSage — Design Specification

**Status:** Current application design

## 1. Product character

SignSage is a research-grade assistive communication instrument.

It should look like a serious technical product without hiding the experimental state of the models.

## 2. Primary interaction

The shared-device workflow must make turn-taking obvious:

```text
ISL turn
→ camera capture
→ translation
→ English output
```

and:

```text
English turn
→ text input
→ gloss generation
→ pose/skeleton output
```

## 3. Communication hierarchy

The communication result is primary.

Telemetry is secondary.

Therefore:

- captions must be large and readable,
- output should remain visible long enough to understand,
- confidence/status information should not overwhelm the communication surface,
- errors should be explicit.

## 4. Fail loudly

Show meaningful states for:

- camera permission failure,
- poor landmark tracking,
- hand outside frame,
- weak tracking,
- WebSocket failure,
- unavailable model,
- uncertain/experimental result.

Never replace a model-unavailable state with a fake successful translation.

## 5. Visual identity

The documented visual system uses:

- **Space Grotesk** for headlines,
- **Inter** for body/UI text,
- **JetBrains Mono** for technical data.

Core spacing:

```text
4 / 8 / 16 / 24 / 32 px
```

Desktop layout uses a 12-column workbench with 24px gutters and 32px canvas margins.

The dark technical palette is structured around:

```text
Level 0: #090E13
Level 1: #101720
Border:  #253342
Level 2: #151E28
```

Avoid diffuse glows and oversized decorative shadows.

## 6. Responsive behavior

### Desktop

Prioritize:

- camera/input,
- translation output,
- diagnostics,
- turn state.

### Tablet

Reduce telemetry density while preserving the primary communication loop.

### Mobile

Use a single vertical stream:

1. active turn,
2. camera/input,
3. primary output,
4. status/error,
5. secondary controls,
6. telemetry.

## 7. Skeleton output

The Text→Pose visualization is a research representation.

It must not be styled as though it were a photorealistic or validated sign-language avatar.

## 8. Landing page

The public landing page should explain:

1. what SignSage is,
2. how the bidirectional workflow works,
3. what is technically implemented,
4. what the research has actually demonstrated,
5. privacy/data handling,
6. limitations,
7. how to enter the application.

The landing page should not advertise unsupported model accuracy.
