import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import type { HistoryItem } from "@/types/api";

export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

export function HistoryDetailDialog({
  item,
  onClose,
}: {
  item: HistoryItem | null;
  onClose: () => void;
}) {
  return (
    <Dialog open={item !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Translation detail</DialogTitle>
          <DialogDescription>
            {item && formatDateTime(item.created_at)}
          </DialogDescription>
        </DialogHeader>
        {item && (
          <div className="space-y-4 text-sm">
            {item.input_text && (
              <div>
                <p className="mb-1 text-xs font-medium uppercase tracking-wider text-foreground/40">
                  Input
                </p>
                <p className="leading-6">{item.input_text}</p>
              </div>
            )}
            <div>
              <p className="mb-1 text-xs font-medium uppercase tracking-wider text-foreground/40">
                Interpretation
              </p>
              <p className="text-xl font-semibold tracking-tight">{item.output_text}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">
                {item.direction === "sign_to_text" ? "Sign → Text" : "Text → Sign"}
              </Badge>
              <Badge variant="outline">{item.result_type}</Badge>
              {item.confidence != null && (
                <Badge variant="outline">
                  {Math.round(item.confidence * 100)}% confidence
                </Badge>
              )}
            </div>
            <p className="text-xs text-foreground/50">
              Model: {item.model_version}
            </p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
