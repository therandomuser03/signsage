import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { SignSageLogo } from "@/components/logo";
import { ModeToggle } from "@/components/mode-toggle";

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 40);
    };

    handleScroll();

    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <header
      className={`
        fixed inset-x-0 top-0 z-50
        px-4
        transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]
        ${
          scrolled
            ? "top-4 md:top-5"
            : "top-0"
        }
      `}
    >
      <nav
        className={`
          mx-auto flex items-center justify-between

          transition-all duration-500
          ease-[cubic-bezier(0.22,1,0.36,1)]

          ${
            scrolled
              ? `
                h-14
                max-w-5xl
                rounded-full
                border border-transparent
                bg-card/70
                px-4
                shadow-[0_8px_40px_rgba(0,0,0,0.08)]
                backdrop-blur-2xl
                backdrop-saturate-150
              `
              : `
                h-20
                max-w-7xl
                rounded-none
                border-transparent
                bg-transparent
                px-2
                shadow-none
                backdrop-blur-0
              `
          }
        `}
      >
        {/* Logo */}
        <Link
          to="/"
          className={`
            flex items-center gap-2
            font-semibold tracking-tight
            transition-all duration-500
            ${
              scrolled
                ? "text-base"
                : "text-lg"
            }
          `}
        >
          <SignSageLogo
            className={`
              transition-all duration-500
              ${
                scrolled
                  ? "h-8 w-8"
                  : "h-9 w-9"
              }
            `}
          />

          <span>SignSage</span>
        </Link>

        {/* Navigation */}
        <div
          className={`
            hidden items-center gap-7
            text-sm md:flex
            transition-all duration-500
            ${
              scrolled
                ? "gap-6"
                : "gap-8"
            }
          `}
        >
          <Link
            to="/how-it-works"
            className="font-medium text-foreground/60 transition-colors hover:text-foreground"
          >
            How it works
          </Link>

          <Link
            to="/capabilities"
            className="font-medium text-foreground/60 transition-colors hover:text-foreground"
          >
            Capabilities
          </Link>

          <Link
            to="/privacy"
            className="font-medium text-foreground/60 transition-colors hover:text-foreground"
          >
            Privacy
          </Link>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <Link
            to="/login"
            className={`
              hidden rounded-full
              text-sm font-medium
              transition-all duration-300
              hover:bg-primary/5
              sm:block
              ${
                scrolled
                  ? "px-3 py-2"
                  : "px-4 py-2.5"
              }
            `}
          >
            Log in
          </Link>

          <ModeToggle />
        </div>
      </nav>
    </header>
  );
}