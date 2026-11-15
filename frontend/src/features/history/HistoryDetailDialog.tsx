import { useEffect, useState } from "react";
import { Loader2, MessageSquareText, Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/services/api/instance";
import { ApiError } from "@/services/api/errors";
import type { HistorySessionSummary, HistorySessionDetail } from "@/types/api";
import { directionLabel, formatDateTime } from "./historyFormat";

type LoadState = "idle" | "loading" | "ready" | "error";

export function HistoryDetailDialog({
  item,
  onClose,
  onDelete,
  deleting = false,
}: {
  /** Session to open (summary). Sentences are fetched when the dialog opens. */
  item: HistorySessionSummary | null;
  onClose: () => void;
  /** Called with the session id; the parent shows the confirmation dialog. */
  onDelete?: (id: string) => void;
  deleting?: boolean;
}) {
  // Result tagged with the id it belongs to; loading/error/ready are derived,
  // so the effect only talks to the network and never sets state synchronously.
  const [fetched, setFetched] = useState<{
    id: string;
    data?: HistorySessionDetail;
    error?: string;
  } | null>(null);

  const itemId = item?.id ?? null;

  useEffect(() => {
    if (!itemId) return;
    let cancelled = false;
    apiClient
      .getHistorySession(itemId)
      .then((res) => {
        if (!cancelled) setFetched({ id: itemId, data: res });
      })
      .catch((err) => {
        if (!cancelled) {
          setFetched({
            id: itemId,
            error: err instanceof ApiError ? err.message : "Couldn't load this session.",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [itemId]);

  const state: LoadState = !itemId
    ? "idle"
    : fetched?.id !== itemId
      ? "loading"
      : fetched.error
        ? "error"
        : "ready";
  const detail = state === "ready" ? (fetched?.data ?? null) : null;
  const error = state === "error" ? (fetched?.error ?? null) : null;

  const sentenceCount = detail?.sentence_count ?? item?.sentence_count ?? 0;

  return (
    <Dialog open={item !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Translation session</DialogTitle>
          <DialogDescription>
            {item && (
              <>
                {formatDateTime(item.created_at)}
                {" · "}
                {directionLabel(item.direction)}
                {" · "}
                {sentenceCount} {sentenceCount === 1 ? "sentence" : "sentences"}
              </>
            )}
          </DialogDescription>
        </DialogHeader>

        {item && (
          <div className="space-y-4 text-sm">
            {/* Session-level meta */}
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">{directionLabel(item.direction)}</Badge>
              <Badge variant="outline">{item.status}</Badge>
              {item.model_version && <Badge variant="outline">{item.model_version}</Badge>}
            </div>

            {state === "loading" && (
              <div className="space-y-3" aria-label="Loading session sentences">
                {[0, 1].map((i) => (
                  <Skeleton key={i} className="h-20 w-full" />
                ))}
              </div>
            )}

            {state === "error" && (
              <p className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {error ?? "Couldn't load this session."}
              </p>
            )}

            {state === "ready" && detail && (
              <ol className="space-y-3">
                {detail.sentences.map((sentence, index) => (
                  <li
                    key={`${sentence.sequence_number}-${index}`}
                    className="rounded-lg border border-foreground/10 bg-muted/30 p-3"
                  >
                    <div className="flex items-center gap-2">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-muted text-xs font-semibold text-foreground/60">
                        {index + 1}
                      </span>
                      <p className="text-xs font-medium uppercase tracking-wider text-foreground/40">
                        Sentence {index + 1}
                      </p>
                      <span className="ml-auto text-xs text-foreground/45">
                        {formatDateTime(sentence.created_at)}
                      </span>
                    </div>
                    {sentence.gloss && (
                      <div className="mt-2">
                        <p className="text-[10px] font-medium uppercase tracking-wider text-foreground/40">
                          Input{sentence.input_text && sentence.input_text !== sentence.gloss ? " (gloss)" : ""}
                        </p>
                        <p className="mt-0.5 leading-6 text-foreground/85">
                          {sentence.gloss}
                        </p>
                      </div>
                    )}
                    {sentence.input_text && sentence.input_text !== sentence.gloss && (
                      <div className="mt-2">
                        <p className="text-[10px] font-medium uppercase tracking-wider text-foreground/40">
                          Input
                        </p>
                        <p className="mt-0.5 leading-6">{sentence.input_text}</p>
                      </div>
                    )}
                    <div className="mt-2">
                      <p className="text-[10px] font-medium uppercase tracking-wider text-foreground/40">
                        Interpretation
                      </p>
                      <p className="mt-0.5 text-lg font-semibold leading-snug tracking-tight">
                        {sentence.output_text}
                      </p>
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-foreground/55">
                      <span className="inline-flex items-center gap-1">
                        <MessageSquareText aria-hidden="true" className="h-3.5 w-3.5" />
                        {sentence.result_type}
                      </span>
                      {sentence.model_version && <span>· {sentence.model_version}</span>}
                      {sentence.confidence != null && (
                        <span>· {Math.round(sentence.confidence * 100)}% confidence</span>
                      )}
                    </div>
                  </li>
                ))}
                {detail.sentences.length === 0 && (
                  <li className="rounded-lg border border-dashed border-foreground/15 px-3 py-4 text-center text-foreground/50">
                    No completed sentences in this session yet.
                  </li>
                )}
              </ol>
            )}

            {onDelete && (
              <div className="flex justify-end border-t border-foreground/10 pt-3">
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={deleting}
                  onClick={() => onDelete(item.id)}
                >
                  {deleting ? (
                    <Loader2 aria-hidden="true" className="animate-spin" />
                  ) : (
                    <Trash2 aria-hidden="true" />
                  )}
                  {deleting ? "Deleting…" : "Delete session"}
                </Button>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
