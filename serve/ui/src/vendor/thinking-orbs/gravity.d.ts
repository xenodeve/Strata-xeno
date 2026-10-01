/**
 * Gravity — the orb pulls the pointer in.
 *
 * An optional effect: as the pointer nears an orb, its body bends toward
 * it and the side facing it streaks toward the orb's centre, like
 * something caught in a well. The pointer itself does not move — its tip
 * stays exactly where the OS has it, and it is never rotated or scaled as
 * a whole. A page cannot move or reshape the OS cursor, so the
 * trick is the one cursorjoy (and metal-fx v2's cursor light) uses: while
 * the pointer is within reach and the element under it shows the plain
 * arrow, that element gets `cursor: none` and a raster of the platform's
 * pointer is drawn in its place, warped by the well. The raster has to be the platform's real pointer or the swap
 * shows; consumers supply it (`setGravitySprite`, or `gravity.sprite`).
 * Without one the effect stays off. Over buttons and text (hand, I-beam)
 * the OS cursor is left alone.
 *
 * Runs only on `(pointer: fine)` devices. Tracking starts when the first
 * orb attaches and stops with the last; the per-frame loop runs only while
 * the pointer is within reach of some orb (plus the fade-out).
 *
 * Fail-safes for the cursor swap (the only part that can hurt someone):
 *   • off under `prefers-reduced-motion`, `forced-colors`, coarse/no-hover
 *     pointers, and pen/touch input;
 *   • off while the page is zoomed (DPR differs from when the sprite was
 *     registered, or pinch-zoomed) — the sprite would scale, the OS cursor
 *     wouldn't;
 *   • the OS cursor is hidden by one class on <html> for exactly as long
 *     as the pointer is in a well, lifted on every leave/blur/hide/keydown,
 *     over anything with a cursor of its own, and on any exception (which
 *     also disables the effect for the session);
 *   • a frame-time watchdog disables it if it ever becomes expensive;
 *   • it never runs where the pointer is anything but the plain arrow.
 * Not detectable: Accessibility › Pointer size/colour on macOS. A user with
 * an enlarged pointer sees it swap to the stock one — ship the sprite only
 * where that trade-off is acceptable, and give them a way to turn it off.
 */
/** A raster of the platform's real pointer. `width`/`height` in CSS px,
 *  `hotX`/`hotY` the click point. */
export interface CursorSprite {
    src: string;
    width: number;
    height: number;
    hotX: number;
    hotY: number;
}
export interface GravityOptions {
    /** How far outside the orb's edge the pull is felt, CSS px. */
    reach?: number;
    /** How far the pointer trails toward the orb on contact, CSS px. The
     *  pointer itself never moves; this is the length of its tail. */
    strength?: number;
    /** How far the pointer's body is bent toward the orb on contact, CSS
     *  px. The tip stays pinned; the rest of the body is drawn displaced
     *  toward the orb, more the farther a pixel sits from the tip. */
    deform?: number;
    /** How the bend is spread along the body (1..4): 1 bends evenly from
     *  the tip outward, higher keeps the tip end firm and bends the far end
     *  hardest. */
    taper?: number;
    /** How late the pull builds (1..4): the proximity weight is raised to
     *  this power, so higher keeps the pointer whole until it is close. */
    curve?: number;
    /** Width of the band across the pointer, CSS px, over which the tail
     *  fades from the side facing the orb to the side away from it. Small
     *  keeps the tail to the facing edge; large lets the whole body trail. */
    falloff?: number;
    /** Inertia (0..1): how much the deformation lags the pointer. 0 follows
     *  instantly; higher reads heavier. */
    smoothing?: number;
    /** Handover (0..1): how slowly the pull swings from one orb to the next
     *  when the nearest changes. 0 flips at once; higher glides. */
    handover?: number;
    /** Squash (0..3): extra bend and tail when the orb lies off the tip's
     *  side, where the pull shortens the body instead of stretching it. A
     *  shortening reads far weaker than a stretch of the same size, so this
     *  evens the two — 0 leaves them equal in pixels, not in feel. */
    squash?: number;
    /** Progressive blur on the tail, CSS px: none at the pointer, this much
     *  at the tail's far end, growing along it. 0 keeps the tail crisp. */
    blur?: number;
    /** Envelope on enter/leave, ms (~95% settled). */
    fadeMs?: number;
    /** The pointer raster, if not already set with `setGravitySprite`. */
    sprite?: CursorSprite;
}
export declare const GRAVITY_DEFAULTS: Readonly<Required<Omit<GravityOptions, 'sprite'>>>;
export declare function setGravityConfig(patch: Partial<Required<Omit<GravityOptions, 'sprite'>>> | null): void;
export declare function getGravityConfig(): Readonly<Required<Omit<GravityOptions, 'sprite'>>>;
/** Clear the fail-safe off-switch (a dev panel's "re-arm"). */
export declare function resetGravity(): void;
/** Supply the pointer raster (or null to turn the effect off). Must match
 *  the OS pointer pixel-for-pixel, or the swap is visible. */
export declare function setGravitySprite(next: CursorSprite | null): void;
/** Register an orb's element. Returns the detach function. */
export declare function attachGravity(el: HTMLElement, options?: GravityOptions | true): () => void;
/** For dev tooling: whether the effect can run, and why not if it cannot. */
export declare function getGravityStatus(): {
    orbs: number;
    tracking: boolean;
    active: boolean;
    blockedBy: string | null;
};
