import type { ReactNode } from "react";

/** Small stroke icons, drawn for this site (24×24, 1.6 stroke). */
const P: Record<string, ReactNode> = {
  terminal: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="3" />
      <path d="m7.5 10 3 2-3 2M12.5 15h4" />
    </>
  ),
  shield: <path d="M12 3.5 5 6v5.5c0 4.2 2.9 7.3 7 9 4.1-1.7 7-4.8 7-9V6l-7-2.5Z" />,
  key: (
    <>
      <circle cx="8" cy="15.5" r="3.3" />
      <path d="m10.5 13 7-7m-2.5 2.5L17 10.5M17.5 6l1.5 1.5" />
    </>
  ),
  folder: <path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2.2h7a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2V7.5Z" />,
  git: (
    <>
      <circle cx="7" cy="6" r="2" />
      <circle cx="7" cy="18" r="2" />
      <circle cx="17" cy="9" r="2" />
      <path d="M7 8v8M17 11c0 3-5 2.5-9.4 5.2" />
    </>
  ),
  keys: (
    <>
      <rect x="3.5" y="6.5" width="17" height="11" rx="2.5" />
      <path d="M7 10h.01M10 10h.01M13 10h.01M16 10h.01M8 14h8" />
    </>
  ),
  plug: (
    <>
      <path d="M9 4v4M15 4v4M6.5 8h11v3.5a5.5 5.5 0 0 1-11 0V8Z" />
      <path d="M12 17v3.5" />
    </>
  ),
  pulse: <path d="M3 12h4l2.2-6 4 12 2.3-6H21" />,
  face: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="6" />
      <path d="M9 11v1.5M15 11v1.5" />
    </>
  ),
  layers: (
    <>
      <path d="m12 4 8.5 4.5L12 13 3.5 8.5 12 4Z" />
      <path d="m3.5 12.5 8.5 4.5 8.5-4.5M3.5 16.5 12 21l8.5-4.5" />
    </>
  ),
  gauge: (
    <>
      <path d="M4.5 16.5a8 8 0 1 1 15 0" />
      <path d="m12 13 3.5-4" />
    </>
  ),
  globe: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M3.5 12h17M12 3.5c2.6 2.4 3.6 5.2 3.6 8.5S14.6 18 12 20.5C9.4 18 8.4 15.2 8.4 12S9.4 5.9 12 3.5Z" />
    </>
  ),
  th: (
    <>
      <path d="M5 6.5h7M8.5 6.5V4.5M6 9.5c1 3 3.2 5 6 6M11 9.5c-.6 2.4-2.6 4.8-6 6.2" />
      <path d="m13 19.5 3.5-9 3.5 9M14.2 16.8h4.6" />
    </>
  ),
  link: (
    <>
      <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" />
      <path d="M14 10a4 4 0 0 0-5.7 0l-3 3A4 4 0 0 0 11 18.7l1-1" />
    </>
  ),
  chip: (
    <>
      <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
      <path d="M9.5 3.5v3M14.5 3.5v3M9.5 17.5v3M14.5 17.5v3M3.5 9.5h3M3.5 14.5h3M17.5 9.5h3M17.5 14.5h3" />
    </>
  ),
  drive: (
    <>
      <rect x="3.5" y="9" width="17" height="7" rx="2.5" />
      <path d="M7 12.5h.01M10 12.5h.01" />
    </>
  ),
  split: (
    <>
      <path d="M4 12h5l3-6h8M9 12l3 6h8" />
    </>
  ),
  timer: (
    <>
      <circle cx="12" cy="13" r="7" />
      <path d="M12 9.5V13l2.5 1.5M9.5 3.5h5" />
    </>
  ),
  bolt: <path d="M13 3.5 5.5 13.5H11L10 20.5l7.5-10H12l1-7Z" />,
  bars: <path d="M5 19.5v-6M10 19.5v-11M15 19.5v-8M20 19.5V5" />,
  list: <path d="M8.5 6.5H20M8.5 12H20M8.5 17.5H20M4.5 6.5h.01M4.5 12h.01M4.5 17.5h.01" />,
  book: (
    <>
      <path d="M4.5 5.5a2 2 0 0 1 2-2H12v16H6.5a2 2 0 0 0-2 2v-16Z" />
      <path d="M19.5 5.5a2 2 0 0 0-2-2H12v16h5.5a2 2 0 0 1 2 2v-16Z" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5M12 8h.01" />
    </>
  ),
  brain: (
    <>
      <path d="M9 4.5a3 3 0 0 0-3 3v.2A3.2 3.2 0 0 0 4.5 11a3 3 0 0 0 1 2.3A3.2 3.2 0 0 0 8 18.5a3 3 0 0 0 4 1.5V5.5A3 3 0 0 0 9 4.5Z" />
      <path d="M15 4.5a3 3 0 0 1 3 3v.2a3.2 3.2 0 0 1 1.5 3.3 3 3 0 0 1-1 2.3A3.2 3.2 0 0 1 16 18.5a3 3 0 0 1-4 1.5" />
    </>
  ),
};

export type IconName = keyof typeof P;

export function Icon({ name }: { name: IconName }) {
  return (
    <svg className="ico" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {P[name]}
    </svg>
  );
}
