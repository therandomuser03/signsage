// Development-only route element for the avatar validation page.
// Kept in its own file so src/app/routes.tsx stays a pure router module
// (react-refresh lint: a file that exports a component may only export components).
import { lazy, Suspense } from "react";

const AvatarTestPage = lazy(() => import("@/features/avatar-test/AvatarTestPage"));

export function DevAvatarRoute() {
  return (
    <Suspense fallback={null}>
      <AvatarTestPage />
    </Suspense>
  );
}
