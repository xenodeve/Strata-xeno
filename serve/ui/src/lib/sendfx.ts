// A prompt that has just been sent rises out of the composer to its place in the conversation. The composer notes where it was
// when the prompt was sent; the message made for that send takes the note once (a message from a reload, or an old one, takes none).
let pending: { rect: DOMRect; at: number } | null = null

export function noteSend(rect: DOMRect | null | undefined, now = Date.now()) {
  pending = rect ? { rect, at: now } : null
}

export function takeSend(messageTime: number): DOMRect | null {
  const p = pending
  if (!p || Math.abs(messageTime - p.at) > 800) return null
  pending = null
  return p.rect
}
