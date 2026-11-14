// SignSage — useSignAnimation
// Drives an avatar ref from raw English text via a pluggable gloss mapper.
// Works with the VRM <AvatarViewer /> or the procedural fallback: both
// expose { playSequence, stop }.

import { useCallback, useRef, useState } from "react";
import { localGlossMapper } from "../services/glossMapper";
import type { GlossSequence, SignManifest, SignPlaybackState } from "../types";

type AvatarHandle = {
  playSequence: (
    glossIds: string[],
    onStep?: (id: string, index: number) => void,
  ) => Promise<string[]>;
  stop: () => void;
};

type GlossMapper = (text: string, manifest: SignManifest) => GlossSequence | Promise<GlossSequence>;

export function useSignAnimation(
  avatarRef: React.RefObject<AvatarHandle | null>,
  manifest: SignManifest,
  mapper: GlossMapper = localGlossMapper,
) {
  const [state, setState] = useState<SignPlaybackState>({
    isPlaying: false,
    currentIndex: -1,
    currentGlossId: null,
    queue: [],
  });
  const [unmatchedWords, setUnmatchedWords] = useState<string[]>([]);
  /** Gloss IDs whose animation assets were unavailable during the last playback. */
  const [unavailableGlosses, setUnavailableGlosses] = useState<string[]>([]);
  const runId = useRef(0);

  const perform = useCallback(
    async (text: string) => {
      const thisRun = ++runId.current;
      const { glossIds, unmatchedWords } = await mapper(text, manifest);
      if (runId.current !== thisRun) return; // superseded while mapping
      setUnmatchedWords(unmatchedWords);
      setUnavailableGlosses([]);

      if (!avatarRef.current || glossIds.length === 0) {
        setState({ isPlaying: false, currentIndex: -1, currentGlossId: null, queue: glossIds });
        return;
      }

      setState({ isPlaying: true, currentIndex: -1, currentGlossId: null, queue: glossIds });

      const unavailable = await avatarRef.current.playSequence(glossIds, (id, index) => {
        if (runId.current !== thisRun) return; // a newer call superseded this one
        setState((s) => ({ ...s, currentIndex: index, currentGlossId: id }));
      });
      if (runId.current === thisRun) setUnavailableGlosses(unavailable);

      if (runId.current === thisRun) {
        setState((s) => ({ ...s, isPlaying: false, currentGlossId: null }));
      }
    },
    [avatarRef, manifest, mapper],
  );

  /** Cancel playback immediately; safe to call when idle. */
  const stop = useCallback(() => {
    runId.current += 1;
    avatarRef.current?.stop();
    setState((s) => ({ ...s, isPlaying: false, currentGlossId: null, currentIndex: -1 }));
  }, [avatarRef]);

  /** Re-play the last queue without re-running the mapper. */
  const replay = useCallback(async () => {
    const queue = state.queue;
    if (!avatarRef.current || queue.length === 0 || state.isPlaying) return;
    setUnavailableGlosses([]);
    const thisRun = ++runId.current;
    setState((s) => ({ ...s, isPlaying: true, currentIndex: -1, currentGlossId: null }));
    const unavailable = await avatarRef.current.playSequence(queue, (id, index) => {
      if (runId.current !== thisRun) return;
      setState((s) => ({ ...s, currentIndex: index, currentGlossId: id }));
    });
    if (runId.current === thisRun) setUnavailableGlosses(unavailable);
    if (runId.current === thisRun) {
      setState((s) => ({ ...s, isPlaying: false, currentGlossId: null }));
    }
  }, [avatarRef, state.queue, state.isPlaying]);

  return { state, unmatchedWords, unavailableGlosses, perform, stop, replay };
}
