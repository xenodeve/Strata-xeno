// INFO gpu_arch from the engine: "sm120@NVIDIA_GeForce_RTX_5060_Ti,sm89@NVIDIA_GeForce_RTX_4070_SUPER" - the cards in
// CUDA's order. NVML (the Hardware page's cards) numbers them by PCI bus, so a card is matched by its name, never by
// its position.
export interface Arch { sm: string; name: string }

export function parseArch(s: string): Arch[] {
  return s.split(",").map((p) => {
    const [sm, name = ""] = p.split("@")
    return { sm, name: name.replaceAll("_", " ") }
  }).filter((a) => /^sm\d+$/.test(a.sm))
}

export function archOf(list: Arch[], cardName?: string): string | null {
  if (!cardName) return null
  return list.find((a) => a.name === cardName)?.sm ?? null
}
