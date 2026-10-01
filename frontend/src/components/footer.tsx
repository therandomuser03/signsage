import { Link } from "react-router-dom";
import { SignSageLogo } from "@/components/logo";
import { StartCta } from "@/features/auth/StartCta";
import { cn } from "@/lib/utils";

const PRODUCT = [
  { label: "Sign → Text", to: "/app/translate" },
  { label: "Text → Sign", to: "/app/text-to-sign" },
  { label: "History", to: "/app/history" },
  { label: "Saved Signs", to: "/app/saved-signs" },
];

const EXPLORE = [
  { label: "How it works", to: "/how-it-works" },
  { label: "Capabilities", to: "/capabilities" },
  { label: "About", to: "/about" },
  { label: "Privacy", to: "/privacy" },
];

function LinkColumn({
  title,
  links,
}: {
  title: string;
  links: { label: string; to: string }[];
}) {
  return (
    <nav aria-label={title}>
      <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
      <ul className="mt-4 space-y-3">
        {links.map((link) => (
          <li key={link.label}>
            <Link
              to={link.to}
              className="text-sm text-foreground/55 transition-colors hover:text-foreground"
            >
              {link.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** Site-wide footer. Single instance — reuse, don't duplicate. */
export function Footer() {
  return (
    <footer className="border-t border-transparent px-6 py-16 md:py-20">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-12 md:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div>
            <p className="flex items-center gap-2.5">
              <SignSageLogo className="h-9 w-9" />
              <span className="text-lg font-semibold tracking-tight">SignSage</span>
            </p>
            <p className="mt-5 max-w-xs text-sm leading-6 text-foreground/55">
              Communication between signs, text, and visual
              representations.
            </p>
          </div>
          <LinkColumn title="Product" links={PRODUCT} />
          <LinkColumn title="Explore" links={EXPLORE} />
          <nav aria-label="Account">
            <h3 className="text-sm font-semibold tracking-tight">Account</h3>
            <ul className="mt-4 space-y-3">
              <li>
                <Link
                  to="/login"
                  className="text-sm text-foreground/55 transition-colors hover:text-foreground"
                >
                  Login
                </Link>
              </li>
              <li>
                <StartCta
                  variant="link"
                  className={cn("h-auto p-0 text-sm text-foreground/55 hover:text-foreground")}
                />
              </li>
            </ul>
          </nav>
        </div>
        <div className="mt-12 flex flex-col justify-between gap-2 text-sm text-foreground/40 md:flex-row">
          <span>© 2026 SignSage</span>
          <span>Indian Sign Language communication platform</span>
        </div>
      </div>
    </footer>
  );
}
