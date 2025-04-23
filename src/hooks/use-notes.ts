"use client"

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useRouter } from "next/navigation"
import { Database } from "../../types/supabase"
import { supabase } from "lib/supabase/client"
import { summarizeText } from "lib/api/summarize"

type Note = Database["public"]["Tables"]["notes"]["Row"]
type InsertNote = Database["public"]["Tables"]["notes"]["Insert"]
type UpdateNote = Database["public"]["Tables"]["notes"]["Update"]

export function useNotes() {
  return useQuery({
    queryKey: ["notes"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("notes")
        .select("*")
        .eq("is_deleted", false)
        .order("updated_at", { ascending: false })

      if (error) throw error
      return data as Note[]
    },
  })
}

export function useNote(noteId: string) {
  return useQuery({
    queryKey: ["notes", noteId],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("notes")
        .select("*")
        .eq("id", noteId)
        .single()

      if (error) throw error
      return data as Note
    },
    enabled: !!noteId && noteId !== "new",
  })
}

export function useCreateNote() {
  const queryClient = useQueryClient()
  const router = useRouter()

  return useMutation({
    mutationFn: async (note: Omit<InsertNote, "user_id">) => {
      const { data: userData, error: userError } = await supabase.auth.getUser()
      if (userError) throw userError
      
      const { data, error } = await supabase
        .from("notes")
        .insert({
          ...note,
          user_id: userData.user.id,
        })
        .select()
        .single()

      if (error) throw error
      return data as Note
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["notes"] })
      router.push(`/dashboard/notes/${data.id}`)
    },
  })
}

export function useUpdateNote() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ id, note }: { id: string; note: UpdateNote }) => {
      const { data, error } = await supabase
        .from("notes")
        .update(note)
        .eq("id", id)
        .select()
        .single()

      if (error) throw error
      return data as Note
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["notes"] })
      queryClient.invalidateQueries({ queryKey: ["notes", data.id] })
    },
  })
}

export function useDeleteNote() {
  const queryClient = useQueryClient()
  const router = useRouter()

  return useMutation({
    mutationFn: async (id: string) => {
      // Soft delete by setting is_deleted to true
      const { data, error } = await supabase
        .from("notes")
        .update({ is_deleted: true })
        .eq("id", id)
        .select()
        .single()

      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notes"] })
      router.push("/dashboard")
    },
  })
}

export function useSummarizeNote() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ id, content }: { id: string; content: string }) => {
      const { summary, error } = await summarizeText(content)
      
      if (error) throw new Error(error)
      
      const { data, error: dbError } = await supabase
        .from("notes")
        .update({ summary })
        .eq("id", id)
        .select()
        .single()

      if (dbError) throw dbError
      return data as Note
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["notes", data.id] })
    },
  })
}