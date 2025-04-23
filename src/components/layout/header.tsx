"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { 
  BookOpen, 
  LogOut,
  Settings
} from "lucide-react"

// import { Button } from "@/components/ui/button"
// import { ThemeToggle } from "@/components/layout/theme-toggle"
// import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
// import { Separator } from "@/components/ui/separator"
// import { supabase } from "@/lib/supabase/client"
import { useRouter } from "next/navigation"
import { supabase } from "lib/supabase/client"
import { ThemeToggle } from "./theme-toggle"
import { Button } from "components/ui/button"

export function Header() {
  const router = useRouter()
  const pathname = usePathname()

  const handleSignOut = async () => {
    await supabase.auth.signOut()
    router.push("/")
    router.refresh()
  }

  if (pathname === "/" || pathname.startsWith("/auth")) {
    return null
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 md:hidden">
      <div className="container flex h-14 items-center justify-between">
        <div className="flex items-center gap-2">
          <Link href="/dashboard" className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-primary" />
            <span className="font-bold">Summr</span>
          </Link>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <Button variant="ghost" size="icon" onClick={handleSignOut}>
            <LogOut className="h-5 w-5" />
            <span className="sr-only">Sign out</span>
          </Button>
          <Button variant="ghost" size="icon" asChild>
            <Link href="/dashboard/settings">
              <Settings className="h-5 w-5" />
              <span className="sr-only">Settings</span>
            </Link>
          </Button>
        </div>
      </div>
    </header>
  )
}