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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { config } from "@/app/config";
import { ModeToggle } from "@/components/mode-toggle";
import { apiClient } from "@/services/api/instance";
import { ApiError } from "@/services/api/errors";
import { getAccessToken, signOut } from "@/services/supabase/auth";
import type { MeResponse } from "@/types/api";
import { loadPrefs, savePrefs, type TranslationPrefs } from "./preferences";

export default function SettingsPage() {
  const [backend, setBackend] = useState<"checking" | "online" | "offline">("checking");
  const [me, setMe] = useState<MeResponse | null>(null);
  const [accountState, setAccountState] = useState<"checking" | "ready">("checking");
  const [prefs, setPrefs] = useState<TranslationPrefs>(loadPrefs);
  const [clearing, setClearing] = useState(false);
  const [cleared, setCleared] = useState(false);
  const [clearError, setClearError] = useState<string | null>(null);

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

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-2 leading-7 text-foreground/55">
          Every setting here does something. Nothing decorative.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Account</CardTitle>
          <CardDescription>Supabase authentication.</CardDescription>
        </CardHeader>
        <CardContent>
          {accountState === "checking" ? (
            <Skeleton className="h-10 w-full" aria-label="Loading account" />
          ) : me ? (
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="text-foreground/60">Email</span>
                <span className="font-medium">{me.email ?? "Not shared"}</span>
              </div>
              <Separator />
              <div className="flex items-center justify-between gap-2">
                <span className="text-foreground/60">User ID</span>
                <code className="max-w-48 truncate text-xs">{me.id}</code>
              </div>
              <Separator />
              <Button
                variant="outline"
                onClick={() => void signOut().then(() => setMe(null))}
              >
                Sign out
              </Button>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-3">
              <p className="text-sm text-foreground/55">
                You’re browsing as a guest. Sign in to sync history.
              </p>
              <Button asChild>
                <Link to="/login">Sign in</Link>
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Translation</CardTitle>
          <CardDescription>
            Applied immediately to Translate and Text → Sign.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between gap-4">
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
          <Separator />
          <div className="flex items-center justify-between gap-4">
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
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Appearance</CardTitle>
          <CardDescription>Light or dark interface.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            <ModeToggle />
            <p className="text-sm text-foreground/55">Toggle the interface theme.</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Privacy & data</CardTitle>
          <CardDescription>What SignSage actually stores.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-6 text-foreground/60">
          <p>
            Translation history stores <span className="font-medium text-foreground">text results only</span> —
            input text, output text, direction, and model version, per
            account. Raw camera frames never leave the browser; the
            backend receives numerical landmark representations.
          </p>
          <p>
            Saved signs are history records. Deleting them here removes
            them everywhere.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Application</CardTitle>
          <CardDescription>Read-only environment info.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className="flex items-center justify-between gap-2">
            <span className="text-foreground/60">Status</span>
            <Badge variant={backend === "online" ? "default" : backend === "offline" ? "destructive" : "secondary"}>
              {backend === "checking" ? "Checking…" : backend === "online" ? "Online" : "Offline"}
            </Badge>
          </div>
          <Separator />
          <div className="flex items-center justify-between gap-2">
            <span className="text-foreground/60">API</span>
            <code className="truncate text-xs">{config.apiUrl}</code>
          </div>
          <Separator />
          <div className="flex items-center justify-between gap-2">
            <span className="text-foreground/60">Live socket</span>
            <code className="truncate text-xs">{config.wsUrl}</code>
          </div>
        </CardContent>
      </Card>

      <Card className="border-destructive/40">
        <CardHeader>
          <CardTitle className="text-destructive">Danger zone</CardTitle>
          <CardDescription>
            Irreversible actions. Account deletion is not offered because
            the backend does not support it.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium">Delete all history</p>
              <p className="text-sm text-foreground/55">
                Removes every saved translation, including saved signs.
              </p>
            </div>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="destructive" disabled={clearing}>
                  Delete all
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete all history?</AlertDialogTitle>
                  <AlertDialogDescription>
                    Every saved translation and saved sign will be
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
            <p className="text-sm text-green-700 dark:text-green-400">
              History cleared.
            </p>
          )}
          {clearError && <p className="text-sm text-destructive">{clearError}</p>}
        </CardContent>
      </Card>

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
