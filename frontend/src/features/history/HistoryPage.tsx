import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowUpDown,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  History as HistoryIcon,
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
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiClient } from "@/services/api/instance";
import { ApiError } from "@/services/api/errors";
import type { Direction, HistoryItem } from "@/types/api";
import { HistoryDetailDialog, formatDateTime } from "./HistoryDetailDialog";

type LoadState = "loading" | "ready" | "error";
type DirectionFilter = "all" | Direction;
type SortOrder = "newest" | "oldest";

const PAGE_SIZE = 15;
const FETCH_LIMIT = 100;

const DIRECTION_LABEL: Record<DirectionFilter, string> = {
  all: "All directions",
  sign_to_text: "Sign → Text",
  text_to_sign: "Text → Sign",
};

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [truncated, setTruncated] = useState(false);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const [query, setQuery] = useState("");
  const [direction, setDirection] = useState<DirectionFilter>("all");
  const [sort, setSort] = useState<SortOrder>("newest");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<HistoryItem | null>(null);

  const load = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const res = await apiClient.listHistory(FETCH_LIMIT, 0);
      setItems(res.items);
      setTotal(res.total);
      setTruncated(res.total > res.items.length);
      setPage(0);
      setState("ready");
    } catch (e) {
      setState("error");
      setError(
        e instanceof ApiError
          ? e.code === "UNAUTHORIZED"
            ? "Sign in to view your translation history."
            : e.message
          : "The backend could not be reached."
      );
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const removeItem = useCallback(
    async (id: string) => {
      try {
        await apiClient.deleteHistoryItem(id);
        await load();
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Delete failed.");
      }
    },
    [load]
  );

  const clearAll = useCallback(async () => {
    setClearing(true);
    try {
      await apiClient.clearHistory();
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Clear failed.");
    } finally {
      setClearing(false);
    }
  }, [load]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const kept = items.filter((item) => {
      if (direction !== "all" && item.direction !== direction) return false;
      if (!q) return true;
      return (
        item.output_text.toLowerCase().includes(q) ||
        (item.input_text ?? "").toLowerCase().includes(q)
      );
    });
    return [...kept].sort((a, b) =>
      sort === "newest"
        ? b.created_at.localeCompare(a.created_at)
        : a.created_at.localeCompare(b.created_at)
    );
  }, [items, query, direction, sort]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const visible = filtered.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE);
  const filtering = query.trim() !== "" || direction !== "all";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Translation History</h1>
          <p className="mt-2 leading-7 text-foreground/55">
            Review your previous SignSage sessions.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => void load()} disabled={state === "loading"}>
            <RefreshCw aria-hidden="true" />
            Refresh
          </Button>
          {items.length > 0 && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" disabled={clearing}>
                  <Trash2 aria-hidden="true" />
                  Clear all
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete all history?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This permanently removes every saved translation. This
                    cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Keep</AlertDialogCancel>
                  <AlertDialogAction onClick={() => void clearAll()}>
                    Delete all
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Couldn’t load history</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Sessions</CardTitle>
          <CardDescription>
            {state === "ready"
              ? `${total} saved translation${total === 1 ? "" : "s"}${truncated ? ` — showing the latest ${items.length}` : ""}. Only text results are stored; no camera recordings.`
              : "Stored server-side, per account."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row">
            <div className="relative flex-1">
              <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/40" />
              <Input
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setPage(0);
                }}
                placeholder="Search translations…"
                aria-label="Search translations"
                className="pl-9"
              />
            </div>
            <div className="flex gap-2">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" aria-label="Filter by direction">
                    {DIRECTION_LABEL[direction]}
                    <ChevronDown aria-hidden="true" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {(Object.keys(DIRECTION_LABEL) as DirectionFilter[]).map((d) => (
                    <DropdownMenuItem
                      key={d}
                      onSelect={() => {
                        setDirection(d);
                        setPage(0);
                      }}
                    >
                      {direction === d && <Check aria-hidden="true" />}
                      {DIRECTION_LABEL[d]}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
              <Button
                variant="outline"
                aria-label={sort === "newest" ? "Sorted newest first. Switch to oldest first." : "Sorted oldest first. Switch to newest first."}
                onClick={() => setSort((s) => (s === "newest" ? "oldest" : "newest"))}
              >
                <ArrowUpDown aria-hidden="true" />
                {sort === "newest" ? "Newest" : "Oldest"}
              </Button>
            </div>
          </div>

          {state === "loading" ? (
            <div className="space-y-3" aria-label="Loading history">
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <HistoryIcon aria-hidden="true" className="h-6 w-6" />
                </EmptyMedia>
                <EmptyTitle>
                  {items.length === 0 ? "No translation sessions yet" : "No matches"}
                </EmptyTitle>
                <EmptyDescription>
                  {items.length === 0
                    ? "Your future translation sessions will appear here."
                    : "Try a different search or filter."}
                </EmptyDescription>
              </EmptyHeader>
              {items.length === 0 && !filtering && (
                <EmptyContent>
                  <Button asChild>
                    <a href="/app/translate">Start translating</a>
                  </Button>
                </EmptyContent>
              )}
            </Empty>
          ) : (
            <>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Result</TableHead>
                      <TableHead>Direction</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>When</TableHead>
                      <TableHead>
                        <span className="sr-only">Actions</span>
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {visible.map((item) => (
                      <TableRow
                        key={item.id}
                        className="cursor-pointer"
                        onClick={() => setSelected(item)}
                      >
                        <TableCell className="max-w-64 font-medium">
                          <span className="block truncate">{item.output_text}</span>
                          {item.confidence != null && (
                            <span className="text-xs font-normal text-foreground/50">
                              {Math.round(item.confidence * 100)}% confidence
                            </span>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant="secondary">
                            {item.direction === "sign_to_text" ? "Sign → Text" : "Text → Sign"}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-foreground/60">
                          {item.result_type}
                        </TableCell>
                        <TableCell className="whitespace-nowrap text-foreground/60">
                          {formatDateTime(item.created_at)}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="icon"
                            aria-label={`Delete ${item.output_text}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              void removeItem(item.id);
                            }}
                          >
                            <Trash2 aria-hidden="true" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
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
        </CardContent>
      </Card>

      <HistoryDetailDialog item={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
