import { useCallback, useEffect, useMemo, useState } from "react";
import { Bookmark, Eye, Search, Trash2 } from "lucide-react";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/services/api/instance";
import { ApiError } from "@/services/api/errors";
import { onAuthChange } from "@/services/supabase/auth";
import type { HistoryItem } from "@/types/api";
import { HistoryDetailDialog, formatDateTime } from "@/features/history/HistoryDetailDialog";

type LoadState = "loading" | "ready" | "error";

export default function SavedSignsPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<HistoryItem | null>(null);
  const [removing, setRemoving] = useState(false);

  const load = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const res = await apiClient.listHistory(100, 0);
      setItems(res.items);
      setState("ready");
    } catch (e) {
      setState("error");
      setError(
        e instanceof ApiError
          ? e.code === "UNAUTHORIZED"
            ? "Sign in to view your saved signs."
            : e.message
          : "The backend could not be reached."
      );
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Sign-out must not leave the previous account's entries on screen: mirror
  // the signed-out (401) state. Signing in here reloads instead of sitting
  // on the guest prompt.
  useEffect(() => {
    return onAuthChange((signedIn) => {
      if (!signedIn) {
        setItems([]);
        setSelected(null);
        setState("error");
        setError("Sign in to view your saved signs.");
      } else {
        void load();
      }
    });
  }, [load]);

  const removeItem = useCallback(
    async (id: string) => {
      setRemoving(true);
      try {
        await apiClient.deleteHistoryItem(id);
        setSelected(null);
        await load();
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Remove failed.");
      } finally {
        setRemoving(false);
      }
    },
    [load]
  );

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (item) =>
        item.output_text.toLowerCase().includes(q) ||
        (item.input_text ?? "").toLowerCase().includes(q)
    );
  }, [items, query]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Saved Signs</h1>
        <p className="mt-2 leading-7 text-foreground/55">
          Keep useful signs close at hand. Saved from your translation
          sessions — text results only, no recordings.
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Couldn’t load saved signs</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="relative max-w-md">
        <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/40" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search saved signs…"
          aria-label="Search saved signs"
          className="pl-9"
        />
      </div>

      {state === "loading" ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-label="Loading saved signs">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-44 w-full" />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <div className="py-8">
          <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <Bookmark aria-hidden="true" className="h-6 w-6" />
                </EmptyMedia>
                <EmptyTitle>No saved signs yet</EmptyTitle>
                <EmptyDescription>
                  {items.length === 0
                    ? "Save signs you want to revisit later."
                    : "Try a different search."}
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {visible.map((item) => (
            <div key={item.id} className="flex flex-col rounded-xl border border-foreground/10 bg-card p-5">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">
                  {item.direction === "sign_to_text" ? "Sign → Text" : "Text → Sign"}
                </Badge>
                <Badge variant="outline">{item.result_type}</Badge>
              </div>
              <p className="mt-3 line-clamp-2 text-xl font-semibold tracking-tight">
                {item.output_text}
              </p>
              <p className="mt-1 text-xs text-foreground/50">
                Saved {formatDateTime(item.created_at)}
              </p>
              <div className="mt-auto flex gap-2 pt-4">
                <Button variant="outline" size="sm" onClick={() => setSelected(item)}>
                  <Eye aria-hidden="true" />
                  Details
                </Button>
                <AlertDialog>
                  <AlertDialogTrigger
                    render={
                      <Button variant="ghost" size="sm" aria-label={`Remove ${item.output_text}`} />
                    }
                  >
                    <Trash2 aria-hidden="true" />
                    Remove
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Remove this saved sign?</AlertDialogTitle>
                      <AlertDialogDescription>
                        “{item.output_text}” will be permanently deleted
                        from your saved translations. This cannot be undone.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Keep</AlertDialogCancel>
                      <AlertDialogAction
                        disabled={removing}
                        onClick={() => void removeItem(item.id)}
                      >
                        Remove
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          ))}
        </div>
      )}

      <HistoryDetailDialog item={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
