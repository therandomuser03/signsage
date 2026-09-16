export function NotFoundPage({ navigate }: { navigate: (hash: string) => void }) {
  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center px-4 py-16 text-center">
      <span className="font-technical-data text-xs uppercase tracking-widest text-primary mb-2">404 · Precision Instrument</span>
      <h1 className="font-headline-xl text-headline-xl text-on-surface">Signal not found.</h1>
      <p className="font-body-md text-body-md text-on-surface-variant max-w-md mt-2">The requested workspace coordinate does not exist in this build. Return to the operational viewport.</p>
      <div className="flex gap-3 mt-6">
        <button onClick={() => navigate("#/")} className="px-5 py-2.5 rounded-lg bg-primary text-on-primary font-label-md text-sm">Back to Home</button>
        <button onClick={() => navigate("#/translate/sign")} className="px-5 py-2.5 rounded-lg bg-surface-container border border-outline-variant text-on-surface font-label-md text-sm">Open Translator</button>
      </div>
    </div>
  )
}
