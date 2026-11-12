import { SignSageLogo } from "@/components/logo";

function LinkColumn({
  title,
  links,
}: {
  title: string;
  links: { label: string; href: string }[];
}) {
  return (
    <nav aria-label={title}>
      <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
      <ul className="mt-4 space-y-3">
        {links.map((link) => (
          <li key={link.label}>
            <a
              href={link.href}
              className="text-sm text-foreground/55 transition-colors hover:text-foreground"
            >
              {link.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

const PLATFORM = [
  { label: "How it works", href: "/how-it-works" },
  { label: "Capabilities", href: "/capabilities" },
];

const SUPPORT = [
  { label: "Help", href: "/help" },
];

const ABOUT = [
  { label: "About", href: "/about" },
  { label: "Translation Log", href: "/app/translation-log" },
];

const SECURITY = [
  { label: "Privacy", href: "/privacy" },
];

/** Site-wide footer. Single instance — reuse, don't duplicate. */
export function Footer() {
  return (
    <footer className="border-t border-transparent px-6 py-16 md:py-20">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-12 md:grid-cols-2 lg:grid-cols-4">
          {/* Left Section: Brand */}
          <div className="lg:col-span-1">
            <div className="flex items-center gap-2.5 mb-5">
              <SignSageLogo className="h-9 w-9" />
              <span className="text-lg font-semibold tracking-tight">SignSage</span>
            </div>
            <p className="max-w-xs text-sm leading-6 text-foreground/55">
              Research prototype for Indian Sign Language communication. Not
              for medical, legal or emergency use.
            </p>
          </div>

          {/* Column 1: Platform */}
          <LinkColumn title="Platform" links={PLATFORM} />

          {/* Column 2: Support */}
          <LinkColumn title="Support" links={SUPPORT} />

          {/* Column 3: About */}
          <LinkColumn title="About" links={ABOUT} />

          {/* Column 4: Security & Trust */}
          <LinkColumn title="Security & Trust" links={SECURITY} />
        </div>

        <div className="mt-12 flex flex-col justify-between gap-2 text-sm text-foreground/40 md:flex-row">
          <span>© 2026 SignSage.</span>
        </div>
      </div>
    </footer>
  );
}
