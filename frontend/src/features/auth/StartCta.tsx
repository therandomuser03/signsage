import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { getAccessToken } from "@/services/supabase/auth";
import { cn } from "@/lib/utils";

type StartCtaProps = {
  children?: React.ReactNode;
  className?: string;
  variant?: "default" | "outline" | "ghost" | "link";
};

/**
 * "Start interpreting" entry point. Signed in → the app.
 * Otherwise → the login flow. No fake auth state anywhere.
 */
export function StartCta({
  children = "Start interpreting",
  className,
  variant = "default",
}: StartCtaProps) {
  const navigate = useNavigate();
  const [checking, setChecking] = useState(false);

  const go = async () => {
    setChecking(true);
    try {
      const token = await getAccessToken();
      navigate(token ? "/app/translate" : "/login");
    } finally {
      setChecking(false);
    }
  };

  return (
    <Button
      variant={variant}
      className={cn(className)}
      disabled={checking}
      onClick={() => void go()}
    >
      {checking && <Spinner aria-hidden="true" />}
      {children}
    </Button>
  );
}
