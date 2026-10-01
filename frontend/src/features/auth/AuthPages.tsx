import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { isSupabaseConfigured } from "@/services/supabase/client";
import { signIn, signUp } from "@/services/supabase/auth";

const credentialsSchema = z.object({
  email: z.string().email("Enter a valid email address."),
  password: z.string().min(8, "Password needs at least 8 characters."),
});

type Credentials = z.infer<typeof credentialsSchema>;

function useCredentialsForm() {
  return useForm<Credentials>();
}

function CredentialFields({
  mode,
  register,
  errors,
}: {
  mode: "login" | "signup";
  register: ReturnType<typeof useCredentialsForm>["register"];
  errors: ReturnType<typeof useCredentialsForm>["formState"]["errors"];
}) {
  return (
    <>
      <div className="space-y-2">
        <Label htmlFor={`${mode}-email`}>Email</Label>
        <Input
          id={`${mode}-email`}
          type="email"
          autoComplete="email"
          placeholder="you@example.com"
          aria-invalid={Boolean(errors.email)}
          {...register("email")}
        />
        {errors.email && (
          <p className="text-sm text-destructive">{errors.email.message}</p>
        )}
      </div>
      <div className="space-y-2">
        <Label htmlFor={`${mode}-password`}>Password</Label>
        <Input
          id={`${mode}-password`}
          type="password"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          placeholder="••••••••"
          aria-invalid={Boolean(errors.password)}
          {...register("password")}
        />
        {errors.password && (
          <p className="text-sm text-destructive">{errors.password.message}</p>
        )}
      </div>
    </>
  );
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
        <DialogClose asChild>
          <Button variant="outline">Close</Button>
        </DialogClose>
        <DialogClose asChild>
          <Button onClick={onAgree}>Agree</Button>
        </DialogClose>
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
        <DialogClose asChild>
          <Button variant="outline">Close</Button>
        </DialogClose>
        <DialogClose asChild>
          <Button onClick={onAgree}>Agree</Button>
        </DialogClose>
      </DialogFooter>
    </DialogContent>
  );
}

function LoginForm() {
  const navigate = useNavigate();
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { register, handleSubmit, formState: { errors } } = useCredentialsForm();

  const onSubmit = (values: Credentials) => {
    const parsed = credentialsSchema.safeParse(values);
    if (!parsed.success) return;
    setServerError(null);
    setBusy(true);
    signIn(parsed.data.email, parsed.data.password)
      .then(() => navigate("/app"))
      .catch((e: unknown) =>
        setServerError(e instanceof Error ? e.message : "Authentication failed.")
      )
      .finally(() => setBusy(false));
  };

  return (
    <form onSubmit={(e) => void handleSubmit(onSubmit)(e)} className="space-y-4">
      <CredentialFields mode="login" register={register} errors={errors} />
      {serverError && (
        <Alert variant="destructive">
          <AlertTitle>Couldn’t sign you in</AlertTitle>
          <AlertDescription>{serverError}</AlertDescription>
        </Alert>
      )}
      <Button type="submit" className="w-full" disabled={busy}>
        {busy && <Spinner aria-hidden="true" />}
        Sign in
      </Button>
    </form>
  );
}

function SignupForm() {
  const navigate = useNavigate();
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [terms, setTerms] = useState(false);
  const [data, setData] = useState(false);
  const [consentError, setConsentError] = useState<string | null>(null);
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
      .then(() => navigate("/app"))
      .catch((e: unknown) =>
        setServerError(e instanceof Error ? e.message : "Authentication failed.")
      )
      .finally(() => setBusy(false));
  };

  return (
    <form onSubmit={(e) => void handleSubmit(onSubmit)(e)} className="space-y-4">
      <CredentialFields mode="signup" register={register} errors={errors} />
      <div className="space-y-3 rounded-xl border border-foreground/10 p-4">
        <div className="flex items-start gap-3">
          <Checkbox
            id="consent-terms"
            checked={terms}
            onCheckedChange={(v) => setTerms(v === true)}
            aria-describedby="consent-error"
          />
          <Label htmlFor="consent-terms" className="text-sm font-normal leading-6">
            I agree to the{" "}
            <Dialog>
              <DialogTrigger asChild>
                <button type="button" className="font-medium underline underline-offset-4">
                  Terms and Conditions
                </button>
              </DialogTrigger>
              <TermsDialog onAgree={() => setTerms(true)} />
            </Dialog>
            .
          </Label>
        </div>
        <div className="flex items-start gap-3">
          <Checkbox
            id="consent-data"
            checked={data}
            onCheckedChange={(v) => setData(v === true)}
            aria-describedby="consent-error"
          />
          <Label htmlFor="consent-data" className="text-sm font-normal leading-6">
            I understand and agree to the{" "}
            <Dialog>
              <DialogTrigger asChild>
                <button type="button" className="font-medium underline underline-offset-4">
                  collection and processing of data
                </button>
              </DialogTrigger>
              <DataDialog onAgree={() => setData(true)} />
            </Dialog>{" "}
            required to use SignSage.
          </Label>
        </div>
        {consentError && (
          <p id="consent-error" role="alert" className="text-sm text-destructive">
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
      <Button type="submit" className="w-full" disabled={busy}>
        {busy && <Spinner aria-hidden="true" />}
        Create account
      </Button>
    </form>
  );
}

function AuthShell({
  title,
  description,
  mode,
  switchTo,
  switchLabel,
}: {
  title: string;
  description: string;
  mode: "login" | "signup";
  switchTo: string;
  switchLabel: string;
}) {
  return (
    <div className="mx-auto grid min-h-[calc(100svh-4rem)] w-full max-w-md place-items-center px-4 py-12">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-2xl">{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
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
          <p className="text-center text-sm text-foreground/55">
            {switchLabel}{" "}
            <Link to={switchTo} className="font-medium text-foreground underline underline-offset-4">
              {mode === "login" ? "Create one" : "Sign in"}
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export function LoginPage() {
  return (
    <AuthShell
      title="Welcome back"
      description="Sign in to sync history across devices."
      mode="login"
      switchTo="/signup"
      switchLabel="No account yet?"
    />
  );
}

export function SignupPage() {
  return (
    <AuthShell
      title="Create your account"
      description="One account for history and saved data."
      mode="signup"
      switchTo="/login"
      switchLabel="Already have an account?"
    />
  );
}
