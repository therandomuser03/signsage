import { useEffect, useState, type ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { Spinner } from "@/components/ui/spinner";
import { getAccessToken } from "@/services/supabase/auth";

/** Guests-only route. A signed-in session is sent straight to the app instead
 * of the marketing landing page; logged-out users browse it freely. */
export function RequireGuest({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null | undefined>(undefined);

  useEffect(() => {
    let live = true;
    void getAccessToken().then((t) => {
      if (live) setToken(t);
    });
    return () => {
      live = false;
    };
  }, []);

  if (token === undefined) {
    return (
      <div className="grid min-h-svh place-items-center px-4">
        <Spinner aria-label="Checking sign-in status" />
      </div>
    );
  }

  if (token) return <Navigate to="/app/interpreter" replace />;
  return <>{children}</>;
}
