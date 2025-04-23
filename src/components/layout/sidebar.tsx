"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { 
  BookOpen, 
  FileText, 
  LogOut, 
  Menu, 
  Plus, 
  Settings,
  X
} from "lucide-react"
import { useNotes } from "hooks/use-notes"
import { supabase } from "lib/supabase/client"
import { Sheet, SheetContent, SheetTrigger } from "components/ui/sheet"
import { Button } from "components/ui/button"
import { ThemeToggle } from "./theme-toggle"
import { ScrollArea } from "components/ui/scroll-area"


export function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const [isOpen, setIsOpen] = useState(false)

  const handleCreateNote = () => {
    router.push("/dashboard/notes/new")
    setIsOpen(false)
  }

  const handleSignOut = async () => {
    await supabase.auth.signOut()
    router.push("/")
    router.refresh()
  }

  const MobileNav = () => (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Toggle menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[280px] p-0">
        <nav className="flex flex-col h-full">
          <div className="px-6 py-6 flex items-center justify-between border-b">
            <Link href="/dashboard" className="flex items-center gap-2">
              <BookOpen className="h-6 w-6 text-primary" />
              <span className="text-xl font-bold">Summr</span>
            </Link>
            <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)}>
              <X className="h-5 w-5" />
            </Button>
          </div>
          <div className="px-6 py-4">
            <Button 
              onClick={handleCreateNote} 
              className="w-full justify-start gap-2"
            >
              <Plus className="h-4 w-4" />
              New Note
            </Button>
          </div>
          <SidebarContent pathname={pathname} setIsOpen={setIsOpen} />
        </nav>
      </SheetContent>
    </Sheet>
  )

  return (
    <>
      <MobileNav />
      <div className="hidden md:flex h-screen w-[280px] flex-col border-r bg-card">
        <div className="px-6 py-6 flex items-center gap-2 border-b">
          <Link href="/dashboard" className="flex items-center gap-2">
            <BookOpen className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold">Summr</span>
          </Link>
        </div>
        <div className="px-6 py-4">
          <Button 
            onClick={handleCreateNote} 
            className="w-full justify-start gap-2"
          >
            <Plus className="h-4 w-4" />
            New Note
          </Button>
        </div>
        <SidebarContent pathname={pathname} />
        <div className="mt-auto px-6 py-4 border-t">
          <div className="flex items-center justify-between">
            <Button variant="ghost" size="icon" onClick={handleSignOut}>
              <LogOut className="h-5 w-5" />
              <span className="sr-only">Sign out</span>
            </Button>
            <ThemeToggle />
            <Button variant="ghost" size="icon" asChild>
              <Link href="/dashboard/settings">
                <Settings className="h-5 w-5" />
                <span className="sr-only">Settings</span>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}

interface SidebarContentProps {
  pathname: string
  setIsOpen?: (open: boolean) => void
}

function SidebarContent({ pathname, setIsOpen }: SidebarContentProps) {
  const { data: notes = [], isLoading } = useNotes()

  return (
    <ScrollArea className="flex-1 py-2">
      <div className="px-6 py-2">
        <p className="text-sm font-semibold text-muted-foreground mb-2">All Notes</p>
        {isLoading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-10 bg-muted rounded animate-pulse" />
            ))}
          </div>
        ) : notes.length > 0 ? (
          <div className="flex flex-col gap-1">
            {notes.map((note) => (
              <NoteLink 
                key={note.id} 
                note={note} 
                active={pathname === `/dashboard/notes/${note.id}`} 
                onClick={() => setIsOpen?.(false)}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-4 text-muted-foreground text-sm">
            No notes yet. Create your first note!
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

interface NoteLinkProps {
  note: { id: string; title: string }
  active: boolean
  onClick?: () => void
}

function NoteLink({ note, active, onClick }: NoteLinkProps) {
  return (
    <Link
      href={`/dashboard/notes/${note.id}`}
      onClick={onClick}
      className={`
        flex items-center gap-2 px-3 py-2 rounded-md text-sm
        ${active 
          ? 'bg-primary/10 text-primary font-medium' 
          : 'text-foreground hover:bg-muted transition-colors'}
      `}
    >
      <FileText className="h-4 w-4" />
      <span className="truncate">{note.title || "Untitled"}</span>
    </Link>
  )
}