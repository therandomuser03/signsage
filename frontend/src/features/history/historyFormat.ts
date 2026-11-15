/** Shared formatters for the history feature (components only export components). */

export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

export function directionLabel(direction: string): string {
  return direction === "sign_to_text" ? "Sign → Text" : "Text → Sign";
}
