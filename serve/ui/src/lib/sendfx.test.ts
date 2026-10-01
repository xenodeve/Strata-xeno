import { describe, expect, test } from "bun:test"
import { noteSend, takeSend } from "./sendfx"

// Where the composer was when a prompt was sent, handed to the message that is created for it (once), so it can rise from there.
describe("send effect", () => {
  const rect = { top: 600, left: 40, width: 700, height: 40 } as DOMRect
  test("the message made for a send takes the composer's place, once", () => {
    noteSend(rect, 1000)
    expect(takeSend(1100)).toBe(rect)
    expect(takeSend(1100)).toBeNull()
  })
  test("a message from long after (a reload, an old one) takes nothing", () => {
    noteSend(rect, 1000)
    expect(takeSend(9000)).toBeNull()
  })
  test("nothing noted, nothing taken", () => {
    noteSend(null, 1000)
    expect(takeSend(1000)).toBeNull()
  })
})
