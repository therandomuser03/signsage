import { SignSageLogo } from "@/components/logo";
import {
  Link,
  X,
} from "lucide-react";

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
  { label: "Case studies", href: "#" },
  { label: "Safety & AI models", href: "#" },
];

const SUPPORT = [
  { label: "Help & support", href: "/help" },
  { label: "Resource Center", href: "#" },
  { label: "Status", href: "#" },
  { label: "Developer Docs", href: "#" },
];

const ABOUT = [
  { label: "Our mission", href: "/about" },
  { label: "Medical Advisory", href: "#" },
  { label: "Research", href: "#" },
  { label: "Accessibility commitment", href: "#" },
];

const SECURITY = [
  { label: "HIPAA & privacy", href: "/privacy" },
  { label: "Security posture", href: "#" },
  { label: "Terms of service", href: "#" },
  { label: "GDPR compliance", href: "#" },
];

const SOCIALS = [
  { icon: Link, label: "LinkedIn", href: "#" },
  { icon: X, label: "Twitter/X", href: "#" },
  { icon: Link, label: "GitHub", href: "#" },
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
              Clinical-grade AI bridging sign language, text, and visual
              representations.
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
          <span>© 2026 SignSage. All rights reserved.</span>
          <div className="flex items-center gap-4">
            <span className="text-foreground/55">Clinical communication platform for sign language.</span>
            <div className="flex items-center gap-3">
              {SOCIALS.map(({ icon: Icon, label, href }) => (
                <a
                  key={label}
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={label}
                  className="text-foreground/55 hover:text-foreground transition-colors"
                >
                  <Icon className="h-4 w-4" />
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
