import { useMemo, useState } from "react";
import { Search, Sparkles } from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../../components/ui/accordion";
import { Input } from "../../components/ui/input";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "../../components/ui/sheet";
import { PROCEDURAL_SIGNS } from "../text-to-sign/services/proceduralSigns";

export type GestureStatus = "browse-only" | "live-supported";

export interface GestureEntry {
  id: string;
  label: string;
  description: string;
  status: GestureStatus;
}

const DESCRIPTION_BY_ID: Record<string, string> = {
  HELLO: "Open-palm wave; greeting motion.",
  THANK_YOU: "Hand to chest then forward; appreciation gesture.",
  SORRY: "Head bow with palms together; apology gesture.",
  PLEASE: "Palm circles to the chest; polite request.",
  YES: "Head nod and small right-hand pump; affirmation.",
  NO: "Head shake with forearms crossed; negation.",
  HELP: "Both arms raised high; emergency or assist request.",
  WATER: "Hand to the mouth; drinking motion.",
  FOOD: "Hand to the mouth twice; eating motion.",
  NAME: "Index finger taps the chest; self-reference.",
  YOU: "Point forward toward the viewer; second-person reference.",
  ME: "Point to the chest; self-reference.",
  WHERE: "Shoulder shrug and head tilt; asking location.",
  HOW: "Palms up and head tilt; asking method or manner.",
};

const LIVE_SUPPORTED = new Set(["HELLO", "YES", "NO", "HELP", "YOU"]);

export const GESTURE_CATALOG: GestureEntry[] = Object.keys(PROCEDURAL_SIGNS).map((id) => ({
  id,
  label: id
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ")
    .replace(/\b(?:Of|And|The)\b/g, (word) => word.toLowerCase()),
  description: DESCRIPTION_BY_ID[id] ?? "Procedural sign preview from the project fallback motion catalog.",
  status: LIVE_SUPPORTED.has(id) ? "live-supported" : "browse-only",
}));

export function filterGestureEntries(query: string, entries: GestureEntry[] = GESTURE_CATALOG) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return entries;

  return entries.filter((entry) => {
    const haystack = [
      entry.id,
      entry.label,
      entry.description,
      entry.status,
    ]
      .join(" ")
      .toLowerCase();

    return haystack.includes(normalized);
  });
}

export function buildGestureGroups(entries: GestureEntry[] = GESTURE_CATALOG) {
  const sorted = [...entries].sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: "base" }));
  const groups = new Map<string, GestureEntry[]>();

  for (const entry of sorted) {
    const letter = entry.label.trim().charAt(0).toUpperCase();
    const key = /[A-Z]/.test(letter) ? letter : "#";
    const list = groups.get(key) ?? [];
    list.push(entry);
    groups.set(key, list);
  }

  return Array.from(groups.entries())
    .sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true }))
    .map(([letter, groupEntries]) => ({ letter, entries: groupEntries }));
}

export function BrowseSignsSheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (next: boolean) => void;
}) {
  const [query, setQuery] = useState("");
  const filteredEntries = useMemo(
    () => filterGestureEntries(query, GESTURE_CATALOG),
    [query],
  );
  const groups = useMemo(() => buildGestureGroups(filteredEntries), [filteredEntries]);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex h-full w-full max-w-md flex-col border-l border-border bg-popover p-0 sm:max-w-md">
        <SheetHeader className="border-b border-foreground/10 p-4 pb-3">
          <SheetTitle className="text-left text-lg">Browse Signs</SheetTitle>
          <SheetDescription className="text-left text-sm text-foreground/60">
            Preview the project vocabulary. Live recognition remains limited to a small subset.
          </SheetDescription>
        </SheetHeader>

        <div className="border-b border-foreground/10 px-4 py-3">
          <label className="relative block">
            <span className="sr-only">Search signs</span>
            <Search className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-foreground/50" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              placeholder="Search signs"
              className="pl-9"
            />
          </label>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3">
          {groups.length === 0 ? (
            <div className="flex h-full min-h-48 flex-col items-center justify-center rounded-xl border border-dashed border-foreground/15 bg-card p-6 text-center">
              <Sparkles className="mb-3 h-8 w-8 text-foreground/40" />
              <p className="text-base font-medium text-foreground">No signs match</p>
              <p className="mt-1 text-sm text-foreground/60">Try a different word or clear the search.</p>
            </div>
          ) : (
            <Accordion multiple className="space-y-1.5" defaultValue={groups.map(({ letter }) => letter)}>
              {groups.map(({ letter, entries }) => (
                <AccordionItem key={letter} value={letter} className="rounded-lg border border-foreground/10 bg-card/80">
                  <AccordionTrigger className="px-3 py-2 text-left text-sm font-medium text-foreground/85 hover:no-underline">
                    {letter}
                  </AccordionTrigger>
                  <AccordionContent className="px-2 pb-2">
                    <div className="space-y-2">
                      {entries.map((entry) => (
                        <div key={entry.id} className="rounded-lg border border-foreground/10 bg-background/60 px-3 py-2.5">
                          <div className="flex items-start justify-between gap-2">
                            <p className="text-sm font-medium text-foreground">{entry.label}</p>
                            <span className={
                              entry.status === "live-supported"
                                ? "rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] text-emerald-600 dark:text-emerald-400"
                                : "rounded-full border border-foreground/10 bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] text-foreground/70"
                            }>
                              {entry.status === "live-supported" ? "Live" : "Browse only"}
                            </span>
                          </div>
                          <p className="mt-1 text-xs leading-5 text-foreground/60">{entry.description}</p>
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
