"use client";

import { lazy, Suspense, useEffect, useState } from "react";
import type { BotAvatarState, BotAvatarType } from "@/vendor/bot-avatars/index.es.js";

// The avatars are the app's own: the same MIT-licensed library (bot-avatars by Libraries.dev, Jakub Antalik)
// that serve/ui vendors, copied unchanged into vendor/bot-avatars with its licence. It is loaded only when needed.
const Bot = lazy(() => import("@/vendor/bot-avatars/index.es.js").then((m) => ({ default: m.BotAvatar })));

export type { BotAvatarState, BotAvatarType };

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const q = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(q.matches);
    on();
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, []);
  return reduced;
}

/** A size that follows the screen: `big` normally, `small` on narrow screens. */
export function useSize(big: number, small: number, query = "(max-width: 600px)") {
  const [n, setN] = useState(big);
  useEffect(() => {
    const q = window.matchMedia(query);
    const on = () => setN(q.matches ? small : big);
    on();
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, [big, small, query]);
  return n;
}

export function Avatar({
  type,
  state = "default",
  size = 160,
  interactive = true,
  label,
  className = "",
}: {
  type: BotAvatarType;
  state?: BotAvatarState;
  size?: number;
  interactive?: boolean;
  label?: string;
  className?: string;
}) {
  const [mounted, setMounted] = useState(false);
  const reduced = usePrefersReducedMotion();
  useEffect(() => setMounted(true), []);
  return (
    <span className={"avatar " + className} style={{ width: size, height: size }} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      {mounted ? (
        <Suspense fallback={<span className="avatar__ph" aria-hidden="true" />}>
          <Bot type={type} state={state} size={size} shading="plastic" interactive={interactive && !reduced} paused={reduced} aria-hidden="true" role="presentation" />
        </Suspense>
      ) : (
        <span className="avatar__ph" aria-hidden="true" />
      )}
    </span>
  );
}
