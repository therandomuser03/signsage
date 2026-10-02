// SignSage — useSignAnimation
// Drives an AvatarViewer ref from raw English text via a pluggable gloss mapper.

import { useCallback, useRef, useState } from "react";
import type { AvatarViewerHandle } from "../components/AvatarViewer";
import { localGlossMapper } from "../services/glossMapper";
import type { GlossSequence, SignManifest, SignPlaybackState } from "../types";

type GlossMapper = (text: string, manifest: SignManifest) => GlossSequence | Promise<GlossSequence>;

export function useSignAnimation(
  avatarRef: React.RefObject<AvatarViewerHandle | null>,
  manifest: SignManifest,
  mapper: GlossMapper = localGlossMapper
) {
  const [state, setState] = useState<SignPlaybackState>({
    isPlaying: false,
    currentIndex: -1,
    currentGlossId: null,
    queue: [],
  });
  const [unmatchedWords, setUnmatchedWords] = useState<string[]>([]);
  const runId = useRef(0);

  const perform = useCallback(
    async (text: string) => {
      const thisRun = ++runId.current;
      const { glossIds, unmatchedWords } = await mapper(text, manifest);
      setUnmatchedWords(unmatchedWords);

      if (!avatarRef.current || glossIds.length === 0) return;

      setState({ isPlaying: true, currentIndex: -1, currentGlossId: null, queue: glossIds });

      await avatarRef.current.playSequence(glossIds, (id, index) => {
        if (runId.current !== thisRun) return; // a newer call superseded this one
        setState((s) => ({ ...s, currentIndex: index, currentGlossId: id }));
      });

      if (runId.current === thisRun) {
        setState((s) => ({ ...s, isPlaying: false, currentGlossId: null }));
      }
    },
    [avatarRef, manifest, mapper]
  );

  return { state, unmatchedWords, perform };
}
