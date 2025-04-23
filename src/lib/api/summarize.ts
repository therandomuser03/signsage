import { z } from "zod"

const summaryResponseSchema = z.object({
  summary: z.string(),
  error: z.string().optional(),
})

export type SummarizeResponse = z.infer<typeof summaryResponseSchema>

export async function summarizeText(text: string): Promise<SummarizeResponse> {
  try {
    // DeepSeek API integration
    const apiUrl = 'https://api.deepseek.ai/v1/summarize'
    const apiKey = process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY

    if (!apiKey) {
      console.warn('No DeepSeek API key found, using mock summarization')
      return mockSummarize(text)
    }

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        text: text,
        max_length: 150,
        temperature: 0.7,
        top_p: 0.95
      })
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const data = await response.json()
    return summaryResponseSchema.parse(data)
  } catch (error) {
    console.error('Error summarizing text:', error)
    return { 
      summary: '', 
      error: error instanceof Error ? error.message : 'Unknown error occurred' 
    }
  }
}

function mockSummarize(text: string): SummarizeResponse {
  // Enhanced mock summarization that better handles different text lengths
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0)
  
  if (sentences.length === 0) {
    return { summary: 'No content to summarize' }
  }

  // For very short text, just return it as is
  if (text.length < 100) {
    return { summary: text.trim() }
  }

  // Take first 2-3 sentences depending on length
  const numSentences = sentences.length > 5 ? 3 : 2
  let summary = sentences.slice(0, numSentences).join('. ').trim()

  // Ensure we don't exceed max length while keeping complete sentences
  if (summary.length > 150) {
    summary = summary.substring(0, 147) + '...'
  } else {
    summary += '.'
  }

  return { summary }
}