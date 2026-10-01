export interface ModeOpts {
    [key: string]: number | undefined;
}
export declare function scaleCounts(opts: ModeOpts, scale: number): ModeOpts;
export declare function scaleRadii(opts: ModeOpts, scale: number): ModeOpts;
/**
 * How many dots a resolved profile draws.
 *
 * There is no single count knob: a mode either lays its dots out on a grid
 * (a lat/lon sphere, a lane/segment ribbon) or carries standalone counts,
 * and several modes do both. This sums exactly the knobs `scaleCounts`
 * treats as counts — grid pairs as their product, the rest as themselves —
 * so the number tracks what `dots` actually scales. Density-driven layers
 * have no fixed count and are left out.
 */
export declare function countDots(opts: ModeOpts): number;
/** Base (fine) profiles per mode, before preset multipliers. */
export declare const BASE_PROFILES: Record<string, ModeOpts>;
