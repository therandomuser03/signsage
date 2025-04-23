
import { Metadata } from "next"
import Link from "next/link"
import { BookOpen, ArrowLeft } from "lucide-react"
import { AuthForm } from "components/auth/auth-form"

export const metadata: Metadata = {
  title: "Sign Up - Summr",
  description: "Create your Summr account",
}

export default function SignUpPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1 flex items-center justify-center py-12">
        <div className="mx-auto max-w-md space-y-6 px-4 sm:px-0">
          <div className="flex flex-col space-y-2 text-center">
            <Link href="/" className="mx-auto flex items-center gap-2 mb-4">
              <BookOpen className="h-8 w-8 text-primary" />
              <span className="text-2xl font-bold">Summr</span>
            </Link>
          </div>
          <AuthForm />
          <div className="text-center">
            <Link 
              href="/" 
              className="text-sm text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
            >
              <ArrowLeft className="h-3 w-3" />
              Back to home
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}