import * as React from "react"
import { Drawer, DrawerTrigger, DrawerContent, DrawerHeader, DrawerTitle, DrawerDescription, DrawerFooter } from "@/components/ui/drawer"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PHRASE_INSTRUCTIONS, PHRASE_SVGS } from "@/lib/phrase-data"

export function NeedHelpDrawer() {
  return (
    <>
      <DrawerTrigger as-child>
        <button
          onClick={() => {}}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 hover:text-primary transition-colors font-medium text-sm"
        >
          <span className="material-symbols-outlined text-[14px]">help_outline</span>
          Need help?
        </button>
      </DrawerTrigger>

      <Drawer modal={false} swipeDirection="up" className="w-full max-w-md">
        <DrawerContent className="p-4">
          <DrawerHeader>
            <DrawerTitle>Supported Phrases</DrawerTitle>
            <DrawerDescription>
              The following phrases are supported by the demo lexicon fallback. Select a phrase
              to see how to sign it in ISL.
            </DrawerDescription>
          </DrawerHeader>

          <ScrollArea className="h-[400px] overflow-y-auto">
            {PHRASE_INSTRUCTIONS.map((phrase) => (
              <div
                key={phrase.id}
                className="py-3 px-2 rounded bg-surface-container-lowest/50 border border-outline-variant/20 hover:bg-surface-container-lowest/80 transition-colors cursor-pointer group"
                onClick={() => window.open(`/sign/${phrase.id}`, "_blank")}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`
                      flex-shrink-0 w-12 h-12 rounded bg-surface-container/50 flex items-center justify-center
                      group-hover:w-16 group-hover:h-16 group-hover:rounded-md
                      flex items-center justify-center shrink-0
                    `}
                  >
                    {PHRASE_SVGS[phrase.id]}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-primary truncate">{phrase.name}</p>
                    <p className="text-sm text-muted-foreground truncate">{phrase.instruction}</p>
                  </div>
                </div>
              </div>
            ))}
          </ScrollArea>

          {PHRASE_INSTRUCTIONS.some((p) => p.pending_verification)}
          {PHRASE_INSTRUCTIONS.some((p) => p.pending_verification) && (
            <DrawerFooter className="pt-2">
              <p className="text-xs text-muted-foreground">
                Illustration pending verification for some phrases — see project maintainer
              </p>
            </DrawerFooter>
          )}
        </DrawerContent>
      </Drawer>
    </>
  )
}