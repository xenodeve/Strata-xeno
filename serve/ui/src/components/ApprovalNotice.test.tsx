import { describe, expect, test } from "bun:test"
import { renderToStaticMarkup } from "react-dom/server"
import { ApprovalBanners } from "./ApprovalNotice"

// The approval banner (#310) must be in view on every page whatever the page's scroll: on the chat page the composer's focus scrolls the
// page, and a banner in the page's flow went off the top (verify-report, defect 1). So it is a layer out of the page's flow, hung under the
// sticky header (absolute, top-full), never taller than part of the screen (it scrolls inside itself), so it cannot cover the composer.
const rows = [
  { key: "guest1", device: "dev-aaaaaaaaaaaaaaaaaaaa", label: "Chrome on Windows", addr: "26.33.142.160", code: "QF2Y", first_seen: 1 },
  { key: "guest2", device: "dev-bbbbbbbbbbbbbbbbbbbb", label: "", addr: "", code: "AB12", first_seen: 2 },
]
const html = (busy: Record<string, "approve" | "refuse"> = {}) => renderToStaticMarkup(<ApprovalBanners rows={rows} busy={busy} onDecide={() => {}} />)
const layer = (h: string) => {
  const m = h.match(/<div data-approval-notice="[^"]*" class="([^"]*)"/)
  expect(m).not.toBeNull()
  return m![1].split(" ")
}

describe("ApprovalBanners", () => {
  test("a layer out of the page's flow, under the header: the page's scroll cannot take it away", () => {
    const c = layer(html())
    expect(c).toContain("absolute")
    expect(c).toContain("top-full")
    expect(c).toContain("inset-x-0")
    expect(c).not.toContain("fixed")                      // it hangs from the sticky header, so it moves with nothing
  })
  test("never taller than part of the screen, scrolling inside itself: the composer at the bottom stays free", () => {
    const h = html()
    expect(h).toMatch(/class="[^"]*max-h-\[45dvh\][^"]*overflow-y-auto/)
  })
  test("a banner per device, with its key, label, address, code, and Approve / Refuse", () => {
    const h = html()
    expect(h.match(/data-pending-device=/g)?.length).toBe(2)
    expect(h).toContain("A new device asks to use the key guest1")
    expect(h).toContain("Chrome on Windows at 26.33.142.160 · code QF2Y")
    expect(h).toContain("(no name) at an unknown address · code AB12")
    expect(h.match(/>Approve</g)?.length).toBe(2)
    expect(h.match(/>Refuse</g)?.length).toBe(2)
  })
  test("while a decision is sent, its buttons say so and wait", () => {
    const h = html({ "guest1/dev-aaaaaaaaaaaaaaaaaaaa": "approve" })
    expect(h).toContain("Approving…")
    expect(h.match(/>Approve</g)?.length).toBe(1)
  })
  test("no device: nothing", () => {
    expect(renderToStaticMarkup(<ApprovalBanners rows={[]} busy={{}} onDecide={() => {}} />)).toBe("")
  })
})
