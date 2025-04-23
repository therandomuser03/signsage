"use client"

import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { Sparkles, Save, Trash2, ArrowLeft, AlertCircle, Keyboard } from "lucide-react"
import { useHotkeys } from "react-hotkeys-hook"
import debounce from "lodash/debounce"
import { Database } from "../../../types/supabase"
import { useToast } from "hooks/use-toast"
import { useCreateNote, useDeleteNote, useSummarizeNote, useUpdateNote } from "hooks/use-notes"
import { Button } from "components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "components/ui/dialog"
import { Alert, AlertDescription } from "components/ui/alert"
import { Input } from "components/ui/input"
import { Textarea } from "components/ui/textarea"
import { Separator } from "components/ui/separator"
import { Card } from "components/ui/card"
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "components/ui/alert-dialog"

type Note = Database["public"]["Tables"]["notes"]["Row"]

interface NoteEditorProps {
  note?: Note
  isNew?: boolean
}

export function NoteEditor({ note, isNew = false }: NoteEditorProps) {
  const [title, setTitle] = useState(note?.title || "")
  const [content, setContent] = useState(note?.content || "")
  const [isSaving, setIsSaving] = useState(false)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const initialLoad = useRef(true)
  const lastSavedContent = useRef({ title: note?.title || "", content: note?.content || "" })
  const contentTextareaRef = useRef<HTMLTextAreaElement>(null)
  
  const router = useRouter()
  const { toast } = useToast()
  const createNoteMutation = useCreateNote()
  const updateNoteMutation = useUpdateNote()
  const deleteNoteMutation = useDeleteNote()
  const summarizeNoteMutation = useSummarizeNote()

  const debouncedSaveRef = useRef<ReturnType<typeof debounce> | null>(null)

  useEffect(() => {
    if (!note || isNew) return

    debouncedSaveRef.current = debounce(async (title: string, content: string) => {
      try {
        await updateNoteMutation.mutateAsync({
          id: note.id,
          note: {
            title: title.trim() || "Untitled",
            content: content.trim(),
          },
        })
        lastSavedContent.current = { title, content }
        setHasChanges(false)
      } catch (error) {
        console.error("Auto-save failed:", error)
      }
    }, 2000)

    return () => {
      debouncedSaveRef.current?.cancel()
    }
  }, [note, isNew, updateNoteMutation])

  useEffect(() => {
    if (initialLoad.current) {
      initialLoad.current = false
      return
    }

    const hasUnsavedChanges = 
      title !== lastSavedContent.current.title || 
      content !== lastSavedContent.current.content

    setHasChanges(hasUnsavedChanges)

    if (hasUnsavedChanges && !isNew) {
      debouncedSaveRef.current?.(title, content)
    }
  }, [title, content, isNew])

  useHotkeys('mod+s', (e) => {
    e.preventDefault()
    if (!isSaving && (hasChanges || isNew)) {
      handleSave()
    }
  }, [hasChanges, isNew, isSaving])

  useHotkeys('mod+/', (e) => {
    e.preventDefault()
    setShowKeyboardShortcuts(true)
  })

  useHotkeys('mod+shift+s', (e) => {
    e.preventDefault()
    if (!isNew && note && !isSummarizing) {
      handleSummarize()
    }
  }, [isNew, note, isSummarizing])

  useHotkeys('alt+n', (e) => {
    e.preventDefault()
    if (!hasChanges) {
      router.push('/dashboard/notes/new')
    }
  }, [hasChanges])

  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasChanges) {
        e.preventDefault()
        e.returnValue = ''
      }
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [hasChanges])

  const handleSave = async () => {
    if (!title.trim() && !content.trim()) {
      setError("Please add a title or some content to your note")
      return
    }

    setError(null)
    setIsSaving(true)
    try {
      if (isNew) {
        await createNoteMutation.mutateAsync({
          title: title.trim() || "Untitled",
          content: content.trim(),
        })
        toast({
          title: "Note created",
          description: "Your note has been created successfully.",
        })
      } else if (note) {
        await updateNoteMutation.mutateAsync({
          id: note.id,
          note: {
            title: title.trim() || "Untitled",
            content: content.trim(),
          },
        })
        toast({
          title: "Note saved",
          description: "Your changes have been saved successfully.",
        })
        lastSavedContent.current = { title, content }
      }
      setHasChanges(false)
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to save note")
    } finally {
      setIsSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!note) return

    try {
      await deleteNoteMutation.mutateAsync(note.id)
      toast({
        title: "Note deleted",
        description: "Your note has been deleted successfully.",
      })
      setShowDeleteDialog(false)
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to delete note")
    }
  }

  const handleSummarize = async () => {
    if (!content.trim() || !note) {
      setError("Please add some content to your note first")
      return
    }

    if (content.trim().length < 50) {
      setError("Your note is too short to summarize. Add more content")
      return
    }

    setError(null)
    setIsSummarizing(true)
    try {
      await summarizeNoteMutation.mutateAsync({
        id: note.id,
        content: content.trim(),
      })
      toast({
        title: "Note summarized",
        description: "Your note has been summarized successfully.",
      })
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to summarize note")
    } finally {
      setIsSummarizing(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <Button 
            variant="ghost" 
            size="icon"
            onClick={() => {
              if (hasChanges) {
                if (confirm("You have unsaved changes. Are you sure you want to leave?")) {
                  router.push("/dashboard")
                }
              } else {
                router.push("/dashboard")
              }
            }}
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <h1 className="text-xl font-semibold">
            {isNew ? "New Note" : "Edit Note"}
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <Dialog open={showKeyboardShortcuts} onOpenChange={setShowKeyboardShortcuts}>
            <DialogTrigger asChild>
              <Button variant="ghost" size="icon">
                <Keyboard className="h-5 w-5" />
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Keyboard Shortcuts</DialogTitle>
                <DialogDescription>
                  <div className="space-y-4 pt-4">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="font-medium">Save note</div>
                      <div className="font-mono">⌘/Ctrl + S</div>
                      <div className="font-medium">Show shortcuts</div>
                      <div className="font-mono">⌘/Ctrl + /</div>
                      <div className="font-medium">Summarize note</div>
                      <div className="font-mono">⌘/Ctrl + ⇧ + S</div>
                      <div className="font-medium">New note</div>
                      <div className="font-mono">Alt + N</div>
                    </div>
                  </div>
                </DialogDescription>
              </DialogHeader>
            </DialogContent>
          </Dialog>

          {!isNew && note && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleSummarize}
              disabled={isSummarizing || isSaving}
              className="gap-1"
            >
              <Sparkles className="h-4 w-4" />
              {isSummarizing ? "Summarizing..." : "Summarize"}
            </Button>
          )}
          <Button
            variant={hasChanges ? "default" : "outline"}
            size="sm"
            onClick={handleSave}
            disabled={isSaving || (!hasChanges && !isNew)}
            className="gap-1"
          >
            <Save className="h-4 w-4" />
            {isSaving ? "Saving..." : "Save"}
          </Button>
          {!isNew && note && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowDeleteDialog(true)}
              disabled={deleteNoteMutation.isPending}
            >
              <Trash2 className="h-5 w-5 text-destructive" />
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Note title"
          className="text-2xl font-semibold border-0 border-b rounded-none px-0 h-auto py-2 focus-visible:ring-0"
        />

        <Textarea
          ref={contentTextareaRef}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Write your note content here..."
          className="min-h-[300px] resize-none border-0 focus-visible:ring-0 text-base"
        />

        {note?.summary && (
          <>
            <Separator />
            <div className="space-y-2">
              <h3 className="text-sm font-medium flex items-center gap-1 text-primary">
                <Sparkles className="h-4 w-4" />
                AI Summary
              </h3>
              <Card className="p-3 bg-muted/50">
                <p className="text-sm">{note.summary}</p>
              </Card>
            </div>
          </>
        )}
      </div>

      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete your note.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
