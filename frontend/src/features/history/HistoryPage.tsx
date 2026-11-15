import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowUpDown,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Hand,
  History as HistoryIcon,
  Languages,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react";
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
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "@/components/ui/toast";
import { apiClient } from "@/services/api/instance";
import { ApiError } from "@/services/api/errors";
import { onAuthChange } from "@/services/supabase/auth";
import type { Direction, HistorySessionSummary } from "@/types/api";
import { HistoryDetailDialog } from "./HistoryDetailDialog";
import { directionLabel, formatDateTime } from "./historyFormat";

const PAGE_SIZE = 10;

type LoadState = "loading" | "ready" | "error";
type DirectionFilter = "all" | Direction;
type SortOrder = "newest" | "oldest";

function sessionModelsWordLike(item: HistorySessionSummary): boolean {
  return (
    (item.model_version?.startsWith("word_v3") ?? false) ||
    (item.title?.startsWith("Word recognition") ?? false)
  );
}

export default function HistoryPage() {
  const [items, setItems] = useState<HistorySessionSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [direction, setDirection] = useState<DirectionFilter>("all");
  const [sort, setSort] = useState<SortOrder>("newest");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<HistorySessionSummary | null>(null);
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(new Set());
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);

  // Debounce the search box so we don't request on every keystroke.
  useEffect(() => {
    const handle = window.setTimeout(() => setDebouncedQuery(query.trim()), 300);
    return () => window.clearTimeout(handle);
  }, [query]);

  const load = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const res = await apiClient.listHistory(PAGE_SIZE, safePage * PAGE_SIZE, {
        direction: direction === "all" ? undefined : direction,
        q: debouncedQuery || undefined,
        sort,
      });
      setItems(res.items);
      setTotal(res.total);
      setState("ready");
    } catch (e) {
      setItems([]);
      setState("error");
      setError(
        e instanceof ApiError
          ? e.message
          : "Couldn't load your history. Check your connection and try again.",
      );
    }
  }, [safePage, debouncedQuery, direction, sort]);

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
        setTotal(0);
        setPage(0);
        setSelected(null);
        setSelectedIds(new Set());
        setState("error");
        setError("Sign in to view your translation history.");
      } else {
        void load();
      }
    });
  }, [load]);

  const filtering =
    debouncedQuery.length > 0 || direction !== "all" || sort !== "newest";

  // ------------------------------------------------------------- selection
  const visibleIds = useMemo(() => items.map((item) => item.id), [items]);
  const selectedOnPage = visibleIds.filter((id) => selectedIds.has(id));
  const allOnPageSelected = items.length > 0 && selectedOnPage.length === items.length;
  const someOnPageSelected = selectedOnPage.length > 0;

  const restartPaging = useCallback(() => {
    setPage(0);
    setSelectedIds(new Set());
  }, []);

  const toggleOne = useCallback((id: string, checked: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });
  }, []);

  const toggleAllOnPage = useCallback(
    (checked: boolean) => {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        for (const id of visibleIds) {
          if (checked) next.add(id);
          else next.delete(id);
        }
        return next;
      });
    },
    [visibleIds],
  );

  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  // ------------------------------------------------------------- deletion
  const removeSessions = useCallback(
    async (ids: string[], mode: "one" | "bulk" | "clear-all") => {
      setDeleting(true);
      try {
        if (mode === "clear-all") {
          await apiClient.clearHistory();
        } else if (mode === "bulk") {
          await apiClient.deleteHistoryItems(ids);
        } else {
          await apiClient.deleteHistoryItem(ids[0]);
        }
        // UI updates only after the backend confirmed the deletion.
        if (mode === "clear-all") {
          setItems([]);
          setTotal(0);
          setPage(0);
          setSelectedIds(new Set());
          setSelected(null);
          toast.add({
            title: "History cleared",
            description: "All translation sessions were deleted from your account.",
            type: "success",
          });
        } else {
          setSelectedIds((prev) => {
            const next = new Set(prev);
            for (const id of ids) next.delete(id);
            return next;
          });
          setSelected((current) => (current && ids.includes(current.id) ? null : current));
          toast.add({
            title: ids.length === 1 ? "Session deleted" : `${ids.length} sessions deleted`,
            description:
              ids.length === 1
                ? "The session and its sentences were permanently deleted."
                : "The selected sessions and their sentences were permanently deleted.",
            type: "success",
          });
          // Reload from the server so pagination and counts stay exact.
          await load();
        }
        setConfirmDeleteId(null);
        setConfirmBulk(false);
        setConfirmClearAll(false);
      } catch (err) {
        // History is unchanged on failure — keep the list and selection intact.
        toast.add({
          title: "Deletion failed",
          description:
            err instanceof ApiError
              ? err.message
              : "Something went wrong. Nothing was deleted.",
          type: "error",
        });
        setConfirmDeleteId(null);
        setConfirmBulk(false);
        setConfirmClearAll(false);
      } finally {
        setDeleting(false);
      }
    },
    [load],
  );

  const directionLabelFor = (value: DirectionFilter) =>
    value === "all" ? "All directions" : directionLabel(value);
  const sortLabelFor = (value: SortOrder) =>
    value === "newest" ? "Newest first" : "Oldest first";

  const selectedCount = selectedIds.size;
  const confirmBulkName = confirmBulk ? [...selectedIds] : [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">History</h1>
          <p className="mt-2 leading-7 text-foreground/55">
            Review your previous SignSage sessions. Each session holds every sentence
            you signed in one interaction.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSelectedIds(new Set());
              void load();
            }}
            disabled={state === "loading"}
          >
            <RefreshCw aria-hidden="true" />
            Refresh
          </Button>
          {items.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              disabled={deleting}
              onClick={() => setConfirmClearAll(true)}
            >
              <Trash2 aria-hidden="true" />
              Clear all
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <p className="text-sm text-foreground/55">
          {state === "loading" && total === 0
            ? "Loading your sessions…"
            : `${total} ${total === 1 ? "session" : "sessions"}`}
          {selectedCount > 0 && ` · ${selectedCount} selected`}
        </p>

        <div className="flex flex-col gap-2 sm:flex-row">
          <div className="relative flex-1">
            <Search
              aria-hidden="true"
              className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/40"
            />
            <Input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                restartPaging();
              }}
              placeholder="Search sentences or sessions…"
              aria-label="Search history"
              className="pl-9"
            />
          </div>
          <div className="flex gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger
                render={<Button variant="outline" size="sm" className="h-9 gap-1" />}
              >
                <Languages aria-hidden="true" />
                {directionLabelFor(direction)}
                <ChevronDown aria-hidden="true" className="h-4 w-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {(["all", "sign_to_text", "text_to_sign"] as const).map((value) => (
                  <DropdownMenuItem key={value} onClick={() => {
                    setDirection(value);
                    restartPaging();
                  }}>
                    {directionLabelFor(value)}
                    {direction === value && <Check aria-hidden="true" className="ml-auto h-4 w-4" />}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <DropdownMenu>
              <DropdownMenuTrigger
                render={<Button variant="outline" size="sm" className="h-9 gap-1" />}
              >
                <ArrowUpDown aria-hidden="true" />
                {sortLabelFor(sort)}
                <ChevronDown aria-hidden="true" className="h-4 w-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {(["newest", "oldest"] as const).map((value) => (
                  <DropdownMenuItem key={value} onClick={() => {
                    setSort(value);
                    restartPaging();
                  }}>
                    {sortLabelFor(value)}
                    {sort === value && <Check aria-hidden="true" className="ml-auto h-4 w-4" />}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Bulk-action toolbar — visible only while something is selected */}
        {selectedCount > 0 && (
          <div
            role="toolbar"
            aria-label="Bulk history actions"
            className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-foreground/10 bg-muted/40 px-3 py-2"
          >
            <div className="flex items-center gap-3">
              <Checkbox
                id="history-select-all-toolbar"
                checked={allOnPageSelected}
                indeterminate={!allOnPageSelected && someOnPageSelected}
                onCheckedChange={(checked) => toggleAllOnPage(checked === true)}
                aria-label="Select all sessions on this page"
              />
              <label
                htmlFor="history-select-all-toolbar"
                className="cursor-pointer text-sm font-medium"
              >
                {selectedCount} selected
              </label>
              <Button variant="ghost" size="sm" onClick={clearSelection}>
                Clear selection
              </Button>
            </div>
            <Button
              variant="destructive"
              size="sm"
              disabled={deleting}
              onClick={() => setConfirmBulk(true)}
            >
              <Trash2 aria-hidden="true" />
              Delete Selected
            </Button>
          </div>
        )}

        {state === "error" ? (
          <Alert variant="destructive">
            <AlertTitle>Couldn't load history</AlertTitle>
            <AlertDescription>
              {error}
              <Button variant="outline" size="sm" className="mt-2" onClick={() => void load()}>
                <RefreshCw aria-hidden="true" />
                Try again
              </Button>
            </AlertDescription>
          </Alert>
        ) : state === "loading" ? (
          <div className="space-y-3" aria-label="Loading history">
            {Array.from({ length: 4 }, (_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : items.length === 0 ? (
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HistoryIcon aria-hidden="true" className="h-6 w-6" />
              </EmptyMedia>
              <EmptyTitle>
                {filtering ? "No sessions match" : "No translation history yet"}
              </EmptyTitle>
              <EmptyDescription>
                {filtering
                  ? "Try a different search or filter."
                  : "Start a translation session and completed sentences will appear here, grouped into one session per interaction."}
              </EmptyDescription>
            </EmptyHeader>
            {!filtering && (
              <EmptyContent>
                <Button render={<a href="/app/interpreter" />}>Start interpreting</Button>
              </EmptyContent>
            )}
          </Empty>
        ) : (
          <>
            {/* Header row: select-all for the current page + count */}
            <div className="flex items-center gap-3 border-b border-foreground/10 pb-2">
              <Checkbox
                id="history-select-all-page"
                checked={allOnPageSelected}
                indeterminate={!allOnPageSelected && someOnPageSelected}
                onCheckedChange={(checked) => toggleAllOnPage(checked === true)}
                aria-label="Select all sessions on this page"
              />
              <label
                htmlFor="history-select-all-page"
                className="cursor-pointer text-xs text-foreground/55"
              >
                {allOnPageSelected ? "Deselect all on this page" : "Select all on this page"}
              </label>
              <span className="ml-auto text-xs text-foreground/45">
                {items.length} of {total} {total === 1 ? "session" : "sessions"}
              </span>
            </div>

            <div>
              {items.map((item) => {
                const DirectionIcon = item.direction === "sign_to_text" ? Languages : Hand;
                const checked = selectedIds.has(item.id);
                return (
                  <div
                    key={item.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelected(item)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSelected(item);
                      }
                    }}
                    className="group flex cursor-pointer items-center gap-3 border-t border-foreground/10 py-4 transition-colors last:border-b hover:bg-muted/40 sm:px-3"
                  >
                    <span
                      className="flex shrink-0 items-center"
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => e.stopPropagation()}
                    >
                      <Checkbox
                        checked={checked}
                        onCheckedChange={(value) => toggleOne(item.id, value === true)}
                        aria-label={`Select session from ${formatDateTime(item.created_at)}`}
                      />
                    </span>
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-foreground/60 transition-colors group-hover:text-green-600 dark:group-hover:text-green-500">
                      <DirectionIcon aria-hidden="true" className="h-5 w-5" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium">{item.preview}</span>
                      <span className="mt-0.5 block truncate text-xs text-foreground/50">
                        {directionLabel(item.direction)}
                        {" · "}
                        {item.sentence_count} {item.sentence_count === 1 ? "sentence" : "sentences"}
                        {" · "}
                        {item.result_type}
                        {sessionModelsWordLike(item) && (
                          <Badge variant="outline" className="ml-1">
                            Word recognition
                          </Badge>
                        )}
                        {" · "}
                        {formatDateTime(item.created_at)}
                      </span>
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Delete session from ${formatDateTime(item.created_at)}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setConfirmDeleteId(item.id);
                      }}
                    >
                      <Trash2 aria-hidden="true" />
                    </Button>
                  </div>
                );
              })}
            </div>

            {pageCount > 1 && (
              <div className="flex items-center justify-between pt-2 text-sm text-foreground/55">
                <span>
                  Page {safePage + 1} of {pageCount}
                </span>
                <div className="flex gap-1">
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label="Previous page"
                    disabled={safePage === 0}
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                  >
                    <ChevronLeft aria-hidden="true" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label="Next page"
                    disabled={safePage >= pageCount - 1}
                    onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                  >
                    <ChevronRight aria-hidden="true" />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      <HistoryDetailDialog
        item={selected}
        onClose={() => setSelected(null)}
        onDelete={(id) => setConfirmDeleteId(id)}
        deleting={deleting}
      />

      {/* Individual deletion confirmation */}
      <AlertDialog
        open={confirmDeleteId !== null}
        onOpenChange={(open) => !open && !deleting && setConfirmDeleteId(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this translation session?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this session and all its associated
              sentences. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction variant="destructive" disabled={deleting} onClick={() => confirmDeleteId && void removeSessions([confirmDeleteId], "one")}>
              {deleting ? "Deleting…" : "Delete Session"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Bulk deletion confirmation */}
      <AlertDialog
        open={confirmBulk}
        onOpenChange={(open) => !open && !deleting && setConfirmBulk(false)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete selected sessions?</AlertDialogTitle>
            <AlertDialogDescription>
              You are about to permanently delete {confirmBulkName.length}{" "}
              {confirmBulkName.length === 1 ? "selected translation session" : "selected translation sessions"}{" "}
              and their associated sentences. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction variant="destructive" disabled={deleting} onClick={() => void removeSessions([...selectedIds], "bulk")}>
              {deleting ? "Deleting…" : "Delete Selected"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Clear-all confirmation */}
      <AlertDialog
        open={confirmClearAll}
        onOpenChange={(open) => !open && !deleting && setConfirmClearAll(false)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear all translation history?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete all translation sessions and their associated
              sentences from your account. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction variant="destructive" disabled={deleting} onClick={() => void removeSessions([], "clear-all")}>
              {deleting ? "Clearing…" : "Clear All"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
