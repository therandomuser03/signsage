import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FlaskConical } from "lucide-react";
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
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { config } from "@/app/config";
import { ModeToggle } from "@/components/mode-toggle";
import { apiClient } from "@/services/api/instance";
import { ApiError } from "@/services/api/errors";
import { getAccessToken, signOut } from "@/services/supabase/auth";
import type { HistorySelfcheckResponse, MeResponse } from "@/types/api";
import { loadPrefs, savePrefs, type TranslationPrefs } from "./preferences";

export default function SettingsPage() {
  const [backend, setBackend] = useState<"checking" | "online" | "offline">("checking");
  const [me, setMe] = useState<MeResponse | null>(null);
  const [accountState, setAccountState] = useState<"checking" | "ready">("checking");
  const [prefs, setPrefs] = useState<TranslationPrefs>(loadPrefs);
  const [clearing, setClearing] = useState(false);
  const [cleared, setCleared] = useState(false);
  const [clearError, setClearError] = useState<string | null>(null);
  const [selfcheckRunning, setSelfcheckRunning] = useState(false);
  const [selfcheckResult, setSelfcheckResult] = useState<HistorySelfcheckResponse | null>(null);
  const [selfcheckError, setSelfcheckError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    apiClient
      .health()
      .then(() => live && setBackend("online"))
      .catch(() => live && setBackend("offline"));
    getAccessToken()
      .then((token) => {
        if (!live || !token) {
          if (live) setAccountState("ready");
          return null;
        }
        return apiClient.me().catch(() => null);
      })
      .then((profile) => {
        if (live) {
          setMe(profile);
          setAccountState("ready");
        }
      });
    return () => {
      live = false;
    };
  }, []);

  const updatePrefs = (next: TranslationPrefs) => {
    setPrefs(next);
    savePrefs(next);
  };

  const clearHistory = async () => {
    setClearing(true);
    setClearError(null);
    setCleared(false);
    try {
      await apiClient.clearHistory();
      setCleared(true);
    } catch (e) {
      setClearError(e instanceof ApiError ? e.message : "Clear failed.");
    } finally {
      setClearing(false);
    }
  };

  const runHistoryCheck = async () => {
    setSelfcheckRunning(true);
    setSelfcheckError(null);
    setSelfcheckResult(null);
    try {
      const res = await apiClient.historySelfcheck();
      setSelfcheckResult(res);
    } catch (e) {
      setSelfcheckError(e instanceof ApiError ? e.message : "History check failed.");
    } finally {
      setSelfcheckRunning(false);
    }
  };

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-2 leading-7 text-foreground/55">
          Every setting here does something. Nothing decorative.
        </p>
      </div>

      <section>
        <h2 className="text-base font-semibold tracking-tight">Account</h2>
        <p className="mt-0.5 text-sm text-foreground/55">Supabase authentication.</p>
        <div className="mt-3 rounded-xl border border-foreground/10 bg-card px-5">
          {accountState === "checking" ? (
            <div className="py-4">
              <Skeleton className="h-10 w-full" aria-label="Loading account" />
            </div>
          ) : me ? (
            <div className="divide-y divide-foreground/10 text-sm">
              <div className="flex items-center justify-between gap-2 py-3.5">
                <span className="text-foreground/60">Email</span>
                <span className="font-medium">{me.email ?? "Not shared"}</span>
              </div>
              <div className="flex items-center justify-between gap-2 py-3.5">
                <span className="text-foreground/60">User ID</span>
                <code className="max-w-48 truncate text-xs">{me.id}</code>
              </div>
              <div className="flex flex-col gap-2 py-3.5">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <span className="font-medium text-foreground">History diagnostic</span>
                    <p className="text-xs text-foreground/55">
                      Verify translation history session, message, read, and delete permissions.
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={selfcheckRunning}
                    onClick={() => void runHistoryCheck()}
                  >
                    {selfcheckRunning ? "Running…" : "Run history check"}
                  </Button>
                </div>
                {selfcheckResult && (
                  <div className="mt-2 rounded-md bg-foreground/5 p-2.5 font-mono text-xs">
                    <p className={selfcheckResult.ok ? "font-semibold text-green-600" : "font-semibold text-destructive"}>
                      {selfcheckResult.ok ? "History check passed" : "History check failed"}
                    </p>
                    <div className="mt-1 space-y-0.5">
                      {selfcheckResult.steps.map((s, idx) => (
                        <p key={idx} className="text-foreground/70">
                          {s.step}: {s.status ?? "unknown"} {s.code ? `(code: ${s.code})` : ""}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
                {selfcheckError && (
                  <p className="mt-1 text-xs text-destructive">{selfcheckError}</p>
                )}
              </div>
              <div className="py-3.5">
                <Button
                  variant="outline"
                  onClick={() => void signOut().then(() => setMe(null))}
                >
                  Sign out
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-3 py-3.5">
              <p className="text-sm text-foreground/55">
                You’re browsing as a guest. Sign in to sync history.
              </p>
              <Button nativeButton={false} render={<Link to="/login" />}>Sign in</Button>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold tracking-tight">Translation</h2>
        <p className="mt-0.5 text-sm text-foreground/55">
          Applied immediately to Translate and Text → Sign.
        </p>
        <div className="mt-3 divide-y divide-foreground/10 rounded-xl border border-foreground/10 bg-card px-5">
          <div className="flex items-center justify-between gap-4 py-4">
            <div className="space-y-0.5">
              <Label htmlFor="prefs-save">Save translations to history</Label>
              <p className="text-sm text-foreground/55">
                Sent as save_history with each translation. Off means
                results stay in the session only.
              </p>
            </div>
            <Switch
              id="prefs-save"
              checked={prefs.saveHistory}
              onCheckedChange={(v) => updatePrefs({ ...prefs, saveHistory: v })}
            />
          </div>
          <div className="flex items-center justify-between gap-4 py-4">
            <div className="space-y-0.5">
              <Label htmlFor="prefs-mirror">Mirror camera preview</Label>
              <p className="text-sm text-foreground/55">
                Selfie view. Off shows the camera exactly as captured.
              </p>
            </div>
            <Switch
              id="prefs-mirror"
              checked={prefs.mirrorPreview}
              onCheckedChange={(v) => updatePrefs({ ...prefs, mirrorPreview: v })}
            />
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold tracking-tight">Appearance</h2>
        <p className="mt-0.5 text-sm text-foreground/55">Light or dark interface.</p>
        <div className="mt-3 rounded-xl border border-foreground/10 bg-card px-5 py-4">
          <div className="flex items-center gap-3">
            <ModeToggle />
            <p className="text-sm text-foreground/55">Toggle the interface theme.</p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold tracking-tight">Privacy & data</h2>
        <p className="mt-0.5 text-sm text-foreground/55">What SignSage actually stores.</p>
        <div className="mt-3 space-y-3 rounded-xl border border-foreground/10 bg-card px-5 py-4 text-sm leading-6 text-foreground/60">
          <p>
            Translation history stores <span className="font-medium text-foreground">text results only</span> —
            input text, output text, direction, and model version, per
            account. Raw camera frames never leave the browser; the
            backend receives numerical landmark representations.
          </p>
          <p>
            Saved translations are history records. Deleting them here removes
            them everywhere.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold tracking-tight">Application</h2>
        <p className="mt-0.5 text-sm text-foreground/55">Read-only environment info.</p>
        <div className="mt-3 divide-y divide-foreground/10 rounded-xl border border-foreground/10 bg-card px-5 text-sm">
          <div className="flex items-center justify-between gap-2 py-3.5">
            <span className="text-foreground/60">Status</span>
            <Badge variant={backend === "online" ? "default" : backend === "offline" ? "destructive" : "secondary"}>
              {backend === "checking" ? "Checking…" : backend === "online" ? "Online" : "Offline"}
            </Badge>
          </div>
          <div className="flex items-center justify-between gap-2 py-3.5">
            <span className="text-foreground/60">API</span>
            <code className="truncate text-xs">{config.apiUrl}</code>
          </div>
          <div className="flex items-center justify-between gap-2 py-3.5">
            <span className="text-foreground/60">Live socket</span>
            <code className="truncate text-xs">{config.wsUrl}</code>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold tracking-tight text-destructive">Danger zone</h2>
        <p className="mt-0.5 text-sm text-foreground/55">
          Irreversible actions. Account deletion is not offered because
          the backend does not support it.
        </p>
        <div className="mt-3 rounded-xl border border-destructive/40 bg-card px-5 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium">Delete all history</p>
              <p className="text-sm text-foreground/55">
                Removes every saved translation.
              </p>
            </div>
            <AlertDialog>
              <AlertDialogTrigger render={<Button variant="destructive" disabled={clearing} />}>
                Delete all
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete all history?</AlertDialogTitle>
                  <AlertDialogDescription>
                    Every saved translation will be
                    permanently removed. This cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Keep</AlertDialogCancel>
                  <AlertDialogAction onClick={() => void clearHistory()}>
                    Delete everything
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
          {cleared && (
            <p className="mt-3 text-sm text-green-700 dark:text-green-400">
              History cleared.
            </p>
          )}
          {clearError && <p className="mt-3 text-sm text-destructive">{clearError}</p>}
        </div>
      </section>

      <Alert>
        <FlaskConical aria-hidden="true" />
        <AlertTitle>Use with care</AlertTitle>
        <AlertDescription>
          Interpretation results are not
          certified for medical, legal, emergency, or other high-stakes
          use.
        </AlertDescription>
      </Alert>
    </div>
  );
}
