import { summarizeText } from "lib/api/summarize"
import { NextResponse } from "next/server"


export async function POST(request: Request) {
  try {
    const { text } = await request.json()
    
    if (!text || typeof text !== "string") {
      return NextResponse.json(
        { error: "Text is required and must be a string" },
        { status: 400 }
      )
    }
    
    const result = await summarizeText(text)
    
    if (result.error) {
      return NextResponse.json(
        { error: result.error },
        { status: 500 }
      )
    }
    
    return NextResponse.json({ summary: result.summary })
  } catch (error) {
    console.error("Error in summarize API:", error)
    return NextResponse.json(
      { error: "Failed to summarize text" },
      { status: 500 }
    )
  }
}