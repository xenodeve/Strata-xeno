import { Lattice } from "./thought"
import { Matrix } from "./matrix"
import { loaderFor } from "../lib/avatar"
import type { OrbDesign } from "../lib/orbs"

/** The loading style for one design: the lattice of dots (React Bits LatticeLoader) or the matrix of dots (transitions.dev), with the
 *  pattern of its own, in the slot an orb would fill. Still when paused. */
export function LoaderMark({ design, px, moving, color }: { design: OrbDesign; px: number; moving: boolean; color?: string }) {
  const l = loaderFor(design)
  return (
    <span className="inline-grid place-items-center" style={{ width: px, height: px, ...(color ? { ["--ink-2" as string]: color, ["--matrix-active" as string]: color } : null) }} aria-hidden>
      {l.family === "lattice"
        ? <Lattice status="working" pattern={l.pattern} cell={Math.max(2, px * 0.2)} gap={Math.max(1, px * 0.07)} paused={!moving} />
        : <Matrix variant={l.variant} px={px * 0.9} paused={!moving} />}
    </span>
  )
}
