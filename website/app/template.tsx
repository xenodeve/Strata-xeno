import type { ReactNode } from "react";

/** A template (unlike a layout) mounts again on every navigation, so each page settles in the same way. */
export default function Template({ children }: { children: ReactNode }) {
  return <div className="page">{children}</div>;
}
