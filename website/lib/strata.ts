/**
 * The schematic behind the core sample.
 *
 * It models one principle only: an expert that a GPU owns has no copy in system RAM.
 * Experts are ranked by how often they are used (0 = hottest). The more VRAM is free,
 * the more of the hottest experts the GPU strata own, and the thinner the RAM stratum gets.
 * With capacity mode on, RAM is a bounded cache and the rest sits on NVMe.
 *
 * Nothing here is a measurement. The numbers are shares of a schematic, never GiB.
 */

export const EXPERTS = 2400;

export type StratumId = "gpu1" | "gpu2" | "ram" | "nvme";
export const STRATA: StratumId[] = ["gpu1", "gpu2", "ram", "nvme"];

export type Alloc = Record<StratumId, number>;

/** Share of all experts each GPU stratum can hold at full VRAM. */
const GPU1_FULL = 0.3;
const GPU2_FULL = 0.18;
/** In capacity mode the RAM stratum is a bounded cache of this share. */
const RAM_CAP = 0.3;

/** vram: 0..1 free VRAM. Returns how many experts sit in each stratum. */
export function allocate(vram: number, capacity: boolean): Alloc {
  const v = Math.min(1, Math.max(0, vram));
  const gpu1 = Math.round(EXPERTS * GPU1_FULL * v);
  const gpu2 = Math.round(EXPERTS * GPU2_FULL * v);
  const rest = EXPERTS - gpu1 - gpu2;
  const ram = capacity ? Math.min(rest, Math.round(EXPERTS * RAM_CAP)) : rest;
  const nvme = rest - ram;
  return { gpu1, gpu2, ram, nvme };
}

/** Which stratum expert #i (0 = hottest) belongs to. */
export function stratumOf(i: number, a: Alloc): StratumId {
  if (i < a.gpu1) return "gpu1";
  if (i < a.gpu1 + a.gpu2) return "gpu2";
  if (i < a.gpu1 + a.gpu2 + a.ram) return "ram";
  return "nvme";
}

export const share = (n: number) => Math.round((n / EXPERTS) * 100);

/** Deterministic pseudo-random so the sample looks the same on every load. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
