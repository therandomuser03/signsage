"use client"

import { Sidebar } from "components/layout/sidebar"
import { NoteEditor } from "components/notes/note-editor"
import { Skeleton } from "components/ui/skeleton"
import { useNote } from "hooks/use-notes"
import { useParams } from "next/navigation"

export default function NotePage() {
  const params = useParams()
  const id = typeof params.id === "string" ? params.id : ""
  const isNew = id === "new"
  const { data: note, isLoading, error } = useNote(id)

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        {isLoading && !isNew ? (
          <div className="flex flex-col h-full p-4">
            <div className="h-14 flex items-center justify-between border-b mb-4">
              <Skeleton className="h-6 w-48" />
              <div className="flex items-center gap-2">
                <Skeleton className="h-9 w-24" />
                <Skeleton className="h-9 w-20" />
                <Skeleton className="h-9 w-9 rounded-full" />
              </div>
            </div>
            <Skeleton className="h-10 w-full mb-4" />
            <Skeleton className="h-[300px] w-full" />
          </div>
        ) : error && !isNew ? (
          <div className="flex flex-col items-center justify-center h-full">
            <h2 className="text-2xl font-bold">Error loading note</h2>
            <p className="text-muted-foreground">
              {error instanceof Error ? error.message : "Failed to load the note"}
            </p>
          </div>
        ) : (
          <NoteEditor note={note} isNew={isNew} />
        )}
      </main>
    </div>
  )
}