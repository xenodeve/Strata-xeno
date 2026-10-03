import type { Metrics } from "./metrics"

// The last two minutes of each disk's read speed, write speed and time per read, as the server's readings came. They are kept here, not in the disk's page: the page is taken away when
// another one is opened, and its graphs started again from nothing every time it was opened. The app feeds them from every reading, whichever page is open.
export interface Trail { read: (number | null)[]; write: (number | null)[]; lat: (number | null)[] }
const MAX = 120
const trails = new Map<number, Trail>()
const NONE: Trail = { read: [], write: [], lat: [] }
let last = -1

interface Rate { index: number; read_mb?: number | null; write_mb?: number | null; read_ms_op?: number | null }

export function feedTrails(m: Metrics) {
  if (m.time === last) return                                  // one reading is counted once, however many parts look at it
  last = m.time
  const disks = (m.hardware as { disks?: Rate[] | null }).disks
  if (!disks) return
  for (const d of disks) {
    let tr = trails.get(d.index)
    if (!tr) trails.set(d.index, (tr = { read: [], write: [], lat: [] }))
    const push = (a: (number | null)[], v: number | null | undefined) => { a.push(v ?? null); if (a.length > MAX) a.shift() }
    push(tr.read, d.read_mb); push(tr.write, d.write_mb); push(tr.lat, d.read_ms_op)
  }
}

export const trailOf = (index: number): Trail => trails.get(index) ?? NONE
