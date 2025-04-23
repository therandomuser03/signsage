"use client"

import { formatDistanceToNow } from "date-fns"
import { FileText, Sparkles } from "lucide-react"
import Link from "next/link"




import { Database } from "../../../types/supabase"
import { Badge } from "components/ui/badge"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "components/ui/card"

type Note = Database["public"]["Tables"]["notes"]["Row"]

interface NoteCardProps {
  note: Note
}

export function NoteCard({ note }: NoteCardProps) {
  const timeSince = formatDistanceToNow(new Date(note.updated_at), { 
    addSuffix: true,
    includeSeconds: true 
  })
  
  // Truncate content for preview while preserving word boundaries
  const contentPreview = note.content.length > 150 
    ? note.content.substring(0, 150).split(' ').slice(0, -1).join(' ') + '...'
    : note.content

  return (
    <Link href={`/dashboard/notes/${note.id}`}>
      <Card className="h-full cursor-pointer transition-all hover:shadow-md hover:border-primary/20">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              <CardTitle className="text-xl truncate">
                {note.title || "Untitled"}
              </CardTitle>
            </div>
            {note.summary && (
              <Badge variant="secondary" className="ml-2 gap-1">
                <Sparkles className="h-3 w-3" />
                AI Summary
              </Badge>
            )}
          </div>
          <CardDescription className="text-xs">
            {timeSince}
          </CardDescription>
        </CardHeader>
        <CardContent className="pb-3">
          {note.summary ? (
            <p className="text-sm text-muted-foreground line-clamp-3">
              {note.summary}
            </p>
          ) : (
            <p className="text-sm text-muted-foreground line-clamp-3">
              {contentPreview}
            </p>
          )}
        </CardContent>
        {note.tags && note.tags.length > 0 && (
          <CardFooter className="pt-0">
            <div className="flex flex-wrap gap-1">
              {note.tags.slice(0, 3).map((tag, i) => (
                <Badge key={i} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
              {note.tags.length > 3 && (
                <Badge variant="outline" className="text-xs">
                  +{note.tags.length - 3} more
                </Badge>
              )}
            </div>
          </CardFooter>
        )}
      </Card>
    </Link>
  )
}