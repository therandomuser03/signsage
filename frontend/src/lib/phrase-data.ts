import type { ReactNode } from "react"

export type PhraseData = {
  id: string
  name: string
  instruction: string
  svg: ReactNode
  pending_verification: boolean
}

// Demo lexicon phrases from the backend — must match the prototype inventory
// Ponytail: NO real iSign pose data embedded (requires HF_TOKEN for gated HF dataset)
// Ponytail: NO hand-authored SVG wireframes included — marked pending_verification=true
export const PHRASE_INSTRUCTIONS: PhraseData[] = [
  {
    id: "hello",
    name: "Hello",
    instruction: "Touch your chest with the flat hand, then move it outward and down in a gentle arc.",
    svg: null,
    pending_verification: true,  // No verified iSign pose or hand-authored SVG available
  },
  {
    id: "thank_you",
    name: "Thank you",
    instruction: "Chin + flat hand: start at chin, move forward and slightly down as if pushing air away.",
    svg: null,
    pending_verification: true,
  },
  {
    id: "yes",
    name: "Yes",
    instruction: " nod head slightly while making a fist with thumb tucked in, moving forward slightly.",
    svg: null,
    pending_verification: true,
  },
  {
    id: "no",
    name: "No",
    instruction: "Shake head side-to-side while extending index finger from fist (N handshape).",
    svg: null,
    pending_verification: true,
  },
  {
    id: "help",
    name: "Help",
    instruction: "Thumb and pinky extended, other fingers folded, move hand up and down slightly (like waving for help).",
    svg: null,
    pending_verification: true,
  },
  {
    id: "please",
    name: "Please",
    instruction: "Chest + circular motion: flat hand rotates chest-high in a small vertical circle.",
    svg: null,
    pending_verification: true,
  },
  {
    id: "sorry",
    name: "Sorry",
    instruction: "Fist over chest, rotate clockwise a few times over the heart area.",
    svg: null,
    pending_verification: true,
  },
  {
    id: "good_morning",
    name: "Good morning",
    instruction: "Good + morning: flat hand touches chin, then moves forward as if pushing morning air.",
    svg: null,
    pending_verification: true,
  },
  {
    id: "how_are_you",
    name: "How are you",
    instruction: "Brow + hands: furrow brow, then open palms up at chest height, rotating slightly.",
    svg: null,
    pending_verification: true,
  },
  {
    id: "i_need_help",
    name: "I need help",
    instruction: "Point to self ( chest ), then use the help handshape (thumb+pinky) up and down.",
    svg: null,
    pending_verification: true,
  },
]

export const PHRASE_SVGS: Record<string, ReactNode> = {
  hello: null,
  thank_you: null,
  yes: null,
  no: null,
  help: null,
  please: null,
  sorry: null,
  good_morning: null,
  how_are_you: null,
  i_need_help: null,
}