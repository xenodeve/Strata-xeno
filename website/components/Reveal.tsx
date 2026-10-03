"use client";

import { useEffect, useRef, useState, type CSSProperties, type ElementType, type ReactNode } from "react";

/** True once the element has come into view (and stays true). */
export function useInView<T extends Element>(margin = "0px 0px -8% 0px") {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    if (typeof IntersectionObserver === "undefined") {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: margin, threshold: 0.01 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [margin, seen]);
  return [ref, seen] as const;
}

/**
 * Content settles in as it arrives: opacity and a short rise, one easing, a 38 ms stagger by `i`.
 * It is applied only when scripts run (html.js) and never for reduced motion, so nothing is ever hidden for good.
 */
export function Reveal({
  children,
  i = 0,
  as: Tag = "div",
  className = "",
}: {
  children: ReactNode;
  i?: number;
  as?: ElementType;
  className?: string;
}) {
  const [ref, seen] = useInView<HTMLElement>();
  return (
    <Tag ref={ref} className={"reveal " + (seen ? "is-in " : "") + className} style={{ "--i": i } as CSSProperties}>
      {children}
    </Tag>
  );
}
