import { createMiddlewareClient } from '@supabase/auth-helpers-nextjs'
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export async function middleware(req: NextRequest) {
  const res = NextResponse.next()
  const supabase = createMiddlewareClient({ req, res })
  const { data: { session } } = await supabase.auth.getSession()

  // Check auth state
  const isLoggedIn = !!session
  const isAuthPage = req.nextUrl.pathname.startsWith('/auth')
  const isDashboardPage = req.nextUrl.pathname.startsWith('/dashboard')

  // Redirect rules
  if (isLoggedIn && isAuthPage) {
    // Logged in users shouldn't access auth pages
    return NextResponse.redirect(new URL('/dashboard', req.url))
  }

  if (!isLoggedIn && isDashboardPage) {
    // Unauthenticated users trying to access dashboard
    return NextResponse.redirect(new URL('/auth/login', req.url))
  }

  return res
}

export const config = {
  matcher: [
    '/dashboard/:path*',
    '/auth/:path*',
  ],
}