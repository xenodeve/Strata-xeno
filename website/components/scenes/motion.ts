/**
 * The motion vocabulary of the app, as numbers. Each value here is read from the app's own stylesheet and components
 * (serve/ui/src/styles.css, thought.css, prompt-bar.css, components/*.tsx), not chosen for this site: a scene that moves
 * should move for as long, with the same curve, as the app does. The comment on each says where it comes from.
 */

/** A CSS cubic-bezier(x1, y1, x2, y2) as a function of progress 0..1 (Newton steps, then bisection). */
export function bezier(x1: number, y1: number, x2: number, y2: number): (t: number) => number {
  const cx = 3 * x1;
  const bx = 3 * (x2 - x1) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * y1;
  const by = 3 * (y2 - y1) - cy;
  const ay = 1 - cy - by;
  const X = (s: number) => ((ax * s + bx) * s + cx) * s;
  const Y = (s: number) => ((ay * s + by) * s + cy) * s;
  const dX = (s: number) => (3 * ax * s + 2 * bx) * s + cx;
  return (t) => {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    let s = t;
    for (let i = 0; i < 6; i++) {
      const err = X(s) - t;
      if (Math.abs(err) < 1e-5) return Y(s);
      const d = dX(s);
      if (Math.abs(d) < 1e-6) break;
      s -= err / d;
    }
    let lo = 0;
    let hi = 1;
    s = t;
    for (let i = 0; i < 24; i++) {
      const x = X(s);
      if (Math.abs(x - t) < 1e-5) break;
      if (x < t) lo = s;
      else hi = s;
      s = (lo + hi) / 2;
    }
    return Y(s);
  };
}

/** `--ease` of the app: every ordinary move. */
export const EASE = bezier(0.23, 1, 0.32, 1);
/** The even stretch of a big section (`.collapse-soft`, kind-pane, fit/glide): cubic-bezier(0.4, 0, 0.2, 1). */
export const EASE_SOFT = bezier(0.4, 0, 0.2, 1);
/** A digit that pops in (`--digit-ease`): it overshoots a little. */
export const EASE_DIGIT = bezier(0.34, 1.45, 0.64, 1);
/** A reel that turns (`--reel-ease`). */
export const EASE_REEL = bezier(0.16, 1, 0.3, 1);
/** The plus-to-menu morph (`--morph-ease` and `--morph-close-ease`). */
export const EASE_MORPH = bezier(0.34, 1.25, 0.64, 1);
export const EASE_MORPH_CLOSE = bezier(0.22, 1, 0.36, 1);
/** The lattice's own curve and the thought timer's (cubic-bezier(0.77, 0, 0.175, 1)). */
export const EASE_LATTICE = bezier(0.77, 0, 0.175, 1);

/** Durations, in ms. */
export const DUR = {
  /** `.collapse-grid`: the rows; its opacity is 240. */
  collapse: 320,
  collapseOpacity: 240,
  /** `.collapse-soft` */
  collapseSoft: 520,
  collapseSoftOpacity: 360,
  /** `.msg-in`: a message arrives. */
  msg: 300,
  /** `.stagger > *` and the page's own content: rise 420 with 50 ms between the children. */
  rise: 420,
  stagger: 50,
  /** `.row-in`: a row that is added to a list. */
  row: 360,
  /** `.page-in` (260) and `.swap-in` (280): a page or a label that changes sharpens in from a blur. */
  page: 260,
  swap: 280,
  /** `.orb-in` / `.orb-out`: one orb dissolves into another. */
  orb: 450,
  /** `.fade-swap`: a line whose words are replaced. */
  fadeSwap: 250,
  /** `.panel-in` */
  panel: 220,
  /** `.toast-in` */
  toast: 240,
  /** `.skill-tip` */
  tip: 150,
  /** `.handover-in` 380 (after 90), `.handover-out` 300 */
  handoverIn: 380,
  handoverDelay: 90,
  handoverOut: 300,
  /** `t-digit-pop-in`, with 38 ms between the last digits */
  digit: 320,
  digitStagger: 38,
  /** `.t-reel-strip` and the 45 ms between reels (`--reel-stagger`; the app uses `dur * 0.08` per digit) */
  reel: 450,
  /** `--reason-hold` and `--reason-step` */
  reasonHold: 840,
  reasonStep: 500,
  /** `--morph-open-dur` / `--morph-close-dur` */
  morphOpen: 350,
  morphClose: 250,
  /** `.thought`'s `--th-settle` */
  thoughtSettle: 450,
  /** `.thought-shimmer` and `.t-shimmer` cycles */
  thoughtShimmer: 1800,
  shimmer: 2000,
  /** `.run-dot` pulse */
  runDot: 1400,
  /** `.streaming ... ::after` caret */
  caret: 1000,
  /** `--matrix-cycle` */
  matrix: 1200,
  /** `.fit-box` */
  fit: 300,
  /** `stretch()` in lib/glide.ts */
  glide: 420,
} as const;
