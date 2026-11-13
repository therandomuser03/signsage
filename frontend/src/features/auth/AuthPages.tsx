import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { FlickeringGrid } from "@/components/ui/flickering-grid";
import { Input } from "@/components/ui/input";
import { ModeToggle } from "@/components/mode-toggle";
import { Spinner } from "@/components/ui/spinner";
import { toast } from "@/components/ui/toast";
import { isSupabaseConfigured } from "@/services/supabase/client";
import {
  getAccessToken,
  resetPassword,
  signIn,
  signInWithGoogle,
  signInWithMicrosoft,
  signUp,
} from "@/services/supabase/auth";
import { SignSageLogo } from "@/components/logo";
import { isGoogleOAuthEnabled, isMicrosoftOAuthEnabled } from "./googleOAuth";


const credentialsSchema = z.object({
  email: z.string().email("Enter a valid email address."),
  password: z
    .string()
    .min(8, "Password needs at least 8 characters.")
    .regex(/[A-Z]/, "Password must contain at least one uppercase letter.")
    .regex(/[a-z]/, "Password must contain at least one lowercase letter.")
    .regex(/[0-9]/, "Password must contain at least one digit.")
    .regex(/[^A-Za-z0-9]/, "Password must contain at least one special character."),
});

type Credentials = z.infer<typeof credentialsSchema>;

// Login deliberately does NOT apply the signup complexity rules: existing
// accounts with older (simpler) passwords must still be able to sign in.
// Supabase/Auth validates the password against whatever was set at signup.
const loginSchema = z.object({
  email: z.string().email("Enter a valid email address."),
  password: z.string().min(1, "Enter your password."),
});

type LoginCredentials = z.infer<typeof loginSchema>;

function useCredentialsForm() {
  return useForm<Credentials>();
}

/** Internal redirect target. Only same-origin paths — never //host or scheme. */
function safeNext(value: string | null, fallback: string): string {
  return value && value.startsWith("/") && !value.startsWith("//") ? value : fallback;
}

function useNext(fallback = "/app") {
  const [params] = useSearchParams();
  const next = safeNext(params.get("next"), fallback);
  const q = `?next=${encodeURIComponent(next)}`;
  return { next, loginTo: `/login${q}`, signupTo: `/signup${q}` };
}


function TermsDialog({ onAgree }: { onAgree: () => void }) {
  return (
    <DialogContent className="max-h-[80svh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>Terms and Conditions</DialogTitle>
        <DialogDescription>
          The short version of the rules for using SignSage.
        </DialogDescription>
      </DialogHeader>
      <div className="space-y-3 text-sm leading-6 text-foreground/70">
        <p>
          <span className="font-medium text-foreground">Acceptable use. </span>
          Use SignSage for lawful communication and practice. Don’t abuse,
          overload, or probe the service for weaknesses.
        </p>
        <p>
          <span className="font-medium text-foreground">Your account. </span>
          You’re responsible for keeping your credentials safe and for
          activity under your account.
        </p>
        <p>
          <span className="font-medium text-foreground">Limitations. </span>
          Interpretation covers a bounded everyday vocabulary and can be
          wrong. Results are not certified for medical, legal, emergency,
          or other high-stakes use — verify before acting on them.
        </p>
        <p>
          <span className="font-medium text-foreground">Your content. </span>
          Text you translate and save remains yours. Only text results
          are stored — never camera video. You can delete saved
          translations at any time.
        </p>
        <p>
          <span className="font-medium text-foreground">Availability. </span>
          The service may change, pause, or go offline without notice,
          including when the backend or sign-in provider is unreachable.
        </p>
        <p>
          <span className="font-medium text-foreground">Termination. </span>
          You can stop using SignSage at any time by signing out and
          deleting your history. Accounts breaching these terms may lose
          access.
        </p>
      </div>
      <DialogFooter>
        <DialogClose render={<Button variant="outline" />}>Close</DialogClose>
        <DialogClose render={<Button onClick={onAgree} />}>Agree</DialogClose>
      </DialogFooter>
    </DialogContent>
  );
}

function DataDialog({ onAgree }: { onAgree: () => void }) {
  return (
    <DialogContent className="max-h-[80svh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>What data is involved</DialogTitle>
        <DialogDescription>
          Exactly what SignSage handles — and what it never touches.
        </DialogDescription>
      </DialogHeader>
      <div className="space-y-3 text-sm leading-6 text-foreground/70">
        <p>
          <span className="font-medium text-foreground">Camera. </span>
          Raw camera frames are processed in the browser to find
          landmarks. They never leave your device.
        </p>
        <p>
          <span className="font-medium text-foreground">Landmarks. </span>
          Numerical landmark representations may be sent to the backend
          for interpretation. They describe movement geometry, not
          appearance.
        </p>
        <p>
          <span className="font-medium text-foreground">Account. </span>
          Authentication and account information is handled through
          Supabase under its own terms.
        </p>
        <p>
          <span className="font-medium text-foreground">History. </span>
          Translation history — text results only — is stored only when
          saving is used or enabled. Switch it off in Settings and
          delete entries any time.
        </p>
      </div>
      <DialogFooter>
        <DialogClose render={<Button variant="outline" />}>Close</DialogClose>
        <DialogClose render={<Button onClick={onAgree} />}>Agree</DialogClose>
      </DialogFooter>
    </DialogContent>
  );
}

function LoginForm() {
  const navigate = useNavigate();
  const { next, signupTo } = useNext();
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [googleBusy, setGoogleBusy] = useState(false);
  const [microsoftBusy, setMicrosoftBusy] = useState(false);
  // Google/Microsoft are only offered when the host has explicitly enabled and configured them.
  const googleEnabled = isGoogleOAuthEnabled(import.meta.env?.VITE_GOOGLE_OAUTH_ENABLED);
  const microsoftEnabled = isMicrosoftOAuthEnabled(
    import.meta.env?.VITE_MICROSOFT_OAUTH_ENABLED,
  );
  const { register, handleSubmit, getValues, formState: { errors } } = useForm<LoginCredentials>();

  const onSubmit = (values: LoginCredentials) => {
    const parsed = loginSchema.safeParse(values);
    if (!parsed.success) return;
    setServerError(null);
    setBusy(true);
    signIn(parsed.data.email, parsed.data.password)
      .then(() => {
        toast.add({ title: "Signed in", description: "Welcome back.", type: "success" });
        navigate(next);
      })
      .catch((e: unknown) => {
        const message = e instanceof Error ? e.message : "Authentication failed.";
        setServerError(message);
        toast.add({ title: "Couldn’t sign you in", description: message, type: "error" });
      })
      .finally(() => setBusy(false));
  };

  const onForgotPassword = () => {
    const parsed = z.string().email("Enter a valid email address.").safeParse(getValues("email"));
    if (!parsed.success) {
      toast.add({ title: "Enter your email first", description: "Type the account email above, then try again.", type: "warning" });
      return;
    }
    resetPassword(parsed.data)
      .then(() => toast.add({ title: "Reset email sent", description: "Check your inbox for the password reset link.", type: "success" }))
      .catch((e: unknown) =>
        toast.add({ title: "Couldn’t send reset email", description: e instanceof Error ? e.message : "Request failed.", type: "error" })
      );
  };

  const onGoogle = () => {
    setGoogleBusy(true);
    signInWithGoogle().catch((e: unknown) => {
      // OAuth redirects away on success, so only failures land here.
      toast.add({ title: "Google sign-in failed", description: e instanceof Error ? e.message : "Request failed.", type: "error" });
      setGoogleBusy(false);
    });
  };

  const onMicrosoft = () => {
    setMicrosoftBusy(true);
    signInWithMicrosoft().catch((e: unknown) => {
      // OAuth redirects away on success, so only failures land here.
      toast.add({ title: "Microsoft sign-in failed", description: e instanceof Error ? e.message : "Request failed.", type: "error" });
      setMicrosoftBusy(false);
    });
  };

  return (
    <form onSubmit={(e) => void handleSubmit(onSubmit)(e)}>
      <FieldGroup>
        <Field>
          <FieldLabel htmlFor="login-email">Email</FieldLabel>
          <Input
            id="login-email"
            type="email"
            autoComplete="email"
            placeholder="m@example.com"
            aria-invalid={Boolean(errors.email)}
            {...register("email")}
          />
          {errors.email && (
            <p className="text-sm text-destructive">{errors.email.message}</p>
          )}
        </Field>
        <Field>
          <div className="flex items-center">
            <FieldLabel htmlFor="login-password">Password</FieldLabel>
            <button
              type="button"
              onClick={onForgotPassword}
              className="ml-auto inline-block text-sm underline-offset-4 hover:underline"
            >
              Forgot your password?
            </button>
          </div>
          <Input
            id="login-password"
            type="password"
            autoComplete="current-password"
            aria-invalid={Boolean(errors.password)}
            {...register("password")}
          />
          {errors.password && (
            <p className="text-sm text-destructive">{errors.password.message}</p>
          )}
        </Field>
        {serverError && (
          <Alert variant="destructive">
            <AlertTitle>Couldn’t sign you in</AlertTitle>
            <AlertDescription>{serverError}</AlertDescription>
          </Alert>
        )}
        <Field>
          <Button type="submit" className="w-full" disabled={busy || googleBusy || microsoftBusy}>
            {busy && <Spinner aria-hidden="true" />}
            Login
          </Button>
          <OAuthButtons
            googleEnabled={googleEnabled}
            microsoftEnabled={microsoftEnabled}
            googleBusy={googleBusy}
            microsoftBusy={microsoftBusy}
            onGoogle={onGoogle}
            onMicrosoft={onMicrosoft}
            disabled={busy || googleBusy || microsoftBusy}
            verb="Login"
          />
          <FieldDescription className="text-center">
            Don&apos;t have an account?{" "}
            <Link to={signupTo} className="underline underline-offset-4">
              Sign up
            </Link>
          </FieldDescription>
        </Field>
      </FieldGroup>
    </form>
  );
}

function SignupForm() {
  const navigate = useNavigate();
  const { next, loginTo } = useNext();
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [googleBusy, setGoogleBusy] = useState(false);
  const [microsoftBusy, setMicrosoftBusy] = useState(false);
  const [terms, setTerms] = useState(false);
  const [data, setData] = useState(false);
  const [consentError, setConsentError] = useState<string | null>(null);
  // Google/Microsoft are only offered when the host has explicitly enabled and configured them.
  const googleEnabled = isGoogleOAuthEnabled(import.meta.env?.VITE_GOOGLE_OAUTH_ENABLED);
  const microsoftEnabled = isMicrosoftOAuthEnabled(
    import.meta.env?.VITE_MICROSOFT_OAUTH_ENABLED,
  );
  const { register, handleSubmit, formState: { errors } } = useCredentialsForm();

  const onSubmit = (values: Credentials) => {
    const parsed = credentialsSchema.safeParse(values);
    if (!parsed.success) return;
    const missing: string[] = [];
    if (!terms) missing.push("Terms and Conditions");
    if (!data) missing.push("data collection and processing");
    if (missing.length > 0) {
      setConsentError(`Please agree to the ${missing.join(" and ")} to continue.`);
      return;
    }
    setConsentError(null);
    setServerError(null);
    setBusy(true);
    signUp(parsed.data.email, parsed.data.password)
      .then(() => {
        toast.add({ title: "Account created", description: "Welcome to SignSage.", type: "success" });
        navigate(next);
      })
      .catch((e: unknown) => {
        const message = e instanceof Error ? e.message : "Authentication failed.";
        setServerError(message);
        toast.add({ title: "Couldn’t create your account", description: message, type: "error" });
      })
      .finally(() => setBusy(false));
  };

  const onGoogle = () => {
    setGoogleBusy(true);
    signInWithGoogle().catch((e: unknown) => {
      toast.add({ title: "Google sign-in failed", description: e instanceof Error ? e.message : "Request failed.", type: "error" });
      setGoogleBusy(false);
    });
  };

  const onMicrosoft = () => {
    setMicrosoftBusy(true);
    signInWithMicrosoft().catch((e: unknown) => {
      toast.add({ title: "Microsoft sign-in failed", description: e instanceof Error ? e.message : "Request failed.", type: "error" });
      setMicrosoftBusy(false);
    });
  };

  return (
    <form onSubmit={(e) => void handleSubmit(onSubmit)(e)}>
      <FieldGroup>
        <Field>
          <FieldLabel htmlFor="signup-email">Email</FieldLabel>
          <Input
            id="signup-email"
            type="email"
            autoComplete="email"
            placeholder="m@example.com"
            aria-invalid={Boolean(errors.email)}
            {...register("email")}
          />
          {errors.email && (
            <p className="text-sm text-destructive">{errors.email.message}</p>
          )}
        </Field>
        <Field>
          <FieldLabel htmlFor="signup-password">Password</FieldLabel>
          <Input
            id="signup-password"
            type="password"
            autoComplete="new-password"
            placeholder="••••••••"
            aria-invalid={Boolean(errors.password)}
            {...register("password")}
          />
          {errors.password && (
            <p className="text-sm text-destructive">{errors.password.message}</p>
          )}
        </Field>
        <div className="space-y-3 rounded-lg border border-border/80 bg-muted/20 p-3">
          <div className="flex items-start gap-2.5">
            <Checkbox
              id="consent-terms"
              checked={terms}
              onCheckedChange={(v) => setTerms(v === true)}
              aria-describedby="consent-error"
              className="mt-0.5 shrink-0"
            />
            <label
              htmlFor="consent-terms"
              className="text-xs leading-relaxed text-muted-foreground select-none cursor-pointer"
            >
              I agree to the{" "}
              <Dialog>
                <DialogTrigger
                  render={
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      className="font-medium text-foreground underline underline-offset-4 hover:text-primary"
                    />
                  }
                >
                  Terms and Conditions
                </DialogTrigger>
                <TermsDialog onAgree={() => setTerms(true)} />
              </Dialog>
              .
            </label>
          </div>
          <div className="flex items-start gap-2.5">
            <Checkbox
              id="consent-data"
              checked={data}
              onCheckedChange={(v) => setData(v === true)}
              aria-describedby="consent-error"
              className="mt-0.5 shrink-0"
            />
            <label
              htmlFor="consent-data"
              className="text-xs leading-relaxed text-muted-foreground select-none cursor-pointer"
            >
              I understand the{" "}
              <Dialog>
                <DialogTrigger
                  render={
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      className="font-medium text-foreground underline underline-offset-4 hover:text-primary"
                    />
                  }
                >
                  data collection and processing
                </DialogTrigger>
                <DataDialog onAgree={() => setData(true)} />
              </Dialog>{" "}
              required to use SignSage.
            </label>
          </div>
          {consentError && (
            <p id="consent-error" role="alert" className="text-xs text-destructive">
              {consentError}
            </p>
          )}
        </div>
        {serverError && (
          <Alert variant="destructive">
            <AlertTitle>Couldn’t create your account</AlertTitle>
            <AlertDescription>{serverError}</AlertDescription>
          </Alert>
        )}
        <Field>
          <Button type="submit" className="w-full" disabled={busy || googleBusy || microsoftBusy}>
            {busy && <Spinner aria-hidden="true" />}
            Create account
          </Button>
          <OAuthButtons
            googleEnabled={googleEnabled}
            microsoftEnabled={microsoftEnabled}
            googleBusy={googleBusy}
            microsoftBusy={microsoftBusy}
            onGoogle={onGoogle}
            onMicrosoft={onMicrosoft}
            disabled={busy || googleBusy || microsoftBusy}
            verb="Sign up"
          />
          <FieldDescription className="text-center">
            Already have an account?{" "}
            <Link to={loginTo} className="underline underline-offset-4">
              Sign in
            </Link>
          </FieldDescription>
        </Field>
      </FieldGroup>
    </form>
  );
}

/** Google brand mark (downloaded SVG served from /public). */
function GoogleIcon({ className }: { className?: string }) {
  return (
    <img
      src="/google-icon-logo-svgrepo-com.svg"
      alt="Google"
      aria-hidden="true"
      className={`w-5 h-5 shrink-0 ${className ?? ""}`}
      decoding="async"
    />
  );
}

/** Microsoft brand mark (downloaded SVG served from /public). */
function MicrosoftIcon({ className }: { className?: string }) {
  return (
    <img
      src="/microsoft-svgrepo-com.svg"
      alt="Microsoft"
      aria-hidden="true"
      className={`w-5 h-5 shrink-0 ${className ?? ""}`}
      decoding="async"
    />
  );
}

/** Social sign-in buttons, rendered only when the host has enabled the provider. */
function OAuthButtons({
  googleEnabled,
  microsoftEnabled,
  googleBusy,
  microsoftBusy,
  onGoogle,
  onMicrosoft,
  disabled,
  verb,
}: {
  googleEnabled: boolean;
  microsoftEnabled: boolean;
  googleBusy: boolean;
  microsoftBusy: boolean;
  onGoogle: () => void;
  onMicrosoft: () => void;
  disabled: boolean;
  verb: "Login" | "Sign up";
}) {
  if (!googleEnabled && !microsoftEnabled) return null;
  const quick = verb === "Login" ? "Sign in" : "Sign up";
  return (
    <div className="space-y-3">
      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <span className="w-full border-t border-border/60" />
        </div>
        <span className="relative flex items-center justify-center px-2 text-xs text-muted-foreground">
          Or
        </span>
      </div>
      <p className="text-xs text-center text-muted-foreground">
        Want to make it quick? {quick} with:
      </p>
      <div className="flex flex-col gap-2">
        {googleEnabled && (
          <Button
            type="button"
            variant="outline"
            className="w-full"
            disabled={disabled}
            onClick={onGoogle}
          >
            {googleBusy && <Spinner aria-hidden="true" />}
            <GoogleIcon aria-hidden="true" className="mr-2" />
            {quick} with Google
          </Button>
        )}
        {microsoftEnabled && (
          <Button
            type="button"
            variant="outline"
            className="w-full"
            disabled={disabled}
            onClick={onMicrosoft}
          >
            {microsoftBusy && <Spinner aria-hidden="true" />}
            <MicrosoftIcon aria-hidden="true" className="mr-2" />
            {quick} with Microsoft
          </Button>
        )}
      </div>
    </div>
  );
}

function AuthShell({
  title,
  description,
  mode,
}: {
  title: string;
  description: string;
  mode: "login" | "signup";
}) {
  return (
    <div className="relative grid min-h-[calc(100svh-4rem)] place-items-center overflow-hidden px-4 py-12">
      <FlickeringGrid
        aria-hidden="true"
        className="absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,black,transparent_75%)]"
        color="rgb(34, 197, 94)"
        squareSize={3}
        gridGap={8}
        flickerChance={0.15}
        maxOpacity={0.35}
      />
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between p-4 sm:p-6">
        <Button variant="ghost" size="sm" className="pointer-events-auto" render={<Link to="/" />}>
          <ArrowLeft aria-hidden="true" />
          Back to Home
        </Button>
        <span className="pointer-events-auto">
          <ModeToggle />
        </span>
      </div>
      <Card className="relative mx-auto w-full max-w-sm">
        <CardHeader className="items-center justify-items-center text-center">
          <SignSageLogo className="mb-2 h-12 w-12" />
          <CardTitle className="text-xl font-semibold tracking-tight">{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>
          {!isSupabaseConfigured() ? (
            <Alert>
              <AlertTitle>Sign-in is not configured</AlertTitle>
              <AlertDescription>
                No Supabase credentials are set for this build, so accounts
                are unavailable. You can still explore the app as a guest.
              </AlertDescription>
            </Alert>
          ) : mode === "login" ? (
            <LoginForm />
          ) : (
            <SignupForm />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export function LoginPage() {
  return (
    <AuthShell
      title="Login to your account"
      description="Enter your email below to login to your account"
      mode="login"
    />
  );
}

export function SignupPage() {
  return (
    <AuthShell
      title="Create your account"
      description="Enter your email below to create your account"
      mode="signup"
    />
  );
}

/** Auth-checking page: signed in → the app, otherwise → login (which returns here). */
export function StartPage() {
  const navigate = useNavigate();
  const { next, loginTo } = useNext("/app/interpreter");

  useEffect(() => {
    let live = true;
    void getAccessToken().then((token) => {
      if (live) navigate(token ? next : loginTo, { replace: true });
    });
    return () => {
      live = false;
    };
  }, [navigate, next, loginTo]);

  return (
    <div className="grid min-h-[calc(100svh-4rem)] place-items-center px-4">
      <Spinner aria-label="Checking sign-in status" />
    </div>
  );
}
