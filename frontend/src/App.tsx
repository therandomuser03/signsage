import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/toast"
import type { ReactNode } from "react"

function App({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <Toaster />
      {children}
    </ThemeProvider>
  )
}

export default App