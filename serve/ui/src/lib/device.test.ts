import { beforeEach, describe, expect, test } from "bun:test"

// The device this browser is (#310): an id made once and kept, a name the owner sees when it asks to be let in, both sent on
// every request; and what a 403 says about the device (waiting for approval with a code, or refused).
const data = new Map<string, string>()
;(globalThis as Record<string, unknown>).localStorage = {
  getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
  setItem: (k: string, v: string) => { data.set(k, String(v)) },
  removeItem: (k: string) => { data.delete(k) },
}

const { deviceId, deviceName, setDeviceName, uaSummary, headerSafe, deviceGate, isDevicePending, _resetDevice } = await import("./device")
const { apiHeaders } = await import("./api")

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })

beforeEach(() => { data.clear(); _resetDevice() })

describe("deviceId", () => {
  test("22 characters of [A-Za-z0-9_-], the same on every call, kept as strata.device", () => {
    const a = deviceId()
    expect(a).toMatch(/^[A-Za-z0-9_-]{22}$/)
    expect(deviceId()).toBe(a)
    expect(JSON.parse(data.get("strata.device")!)).toBe(a)
  })
  test("read back from storage after a reload", () => {
    const a = deviceId()
    _resetDevice()
    expect(deviceId()).toBe(a)
  })
  test("a stored value that is not an id is replaced", () => {
    data.set("strata.device", JSON.stringify("bad id!"))
    expect(deviceId()).toMatch(/^[A-Za-z0-9_-]{22}$/)
  })
  test("stays the same when the browser refuses to store it", () => {
    const set = (globalThis as { localStorage: { setItem: unknown } }).localStorage.setItem
    ;(globalThis as { localStorage: { setItem: unknown } }).localStorage.setItem = () => { throw new Error("full") }
    try {
      const a = deviceId()
      expect(deviceId()).toBe(a)
    } finally { (globalThis as { localStorage: { setItem: unknown } }).localStorage.setItem = set }
  })
})

describe("deviceName", () => {
  test("a summary of the browser and the system when none was given", () => {
    expect(uaSummary("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36")).toBe("Chrome on Windows")
    expect(uaSummary("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36 Edg/131.0")).toBe("Edge on Windows")
    expect(uaSummary("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1")).toBe("Safari on iPhone")
    expect(uaSummary("Mozilla/5.0 (Linux; Android 15) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Mobile Safari/537.36")).toBe("Chrome on Android")
    expect(uaSummary("Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0")).toBe("Firefox on Linux")
    expect(uaSummary("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15")).toBe("Safari on macOS")
    expect(uaSummary("")).toBe("A browser")
  })
  test("the name given in Settings is kept as strata.deviceName", () => {
    setDeviceName("Ann's laptop")
    expect(deviceName()).toBe("Ann's laptop")
    expect(JSON.parse(data.get("strata.deviceName")!)).toBe("Ann's laptop")
    setDeviceName("  ")
    expect(deviceName()).toBe(uaSummary(navigator.userAgent))
  })
  test("only what a header can carry: printable ASCII, at most 80 characters", () => {
    expect(headerSafe("โน้ตบุ๊ก ของ Ann")).toBe("Ann")
    expect(headerSafe("a\tb\nc")).toBe("a b c")
    expect(headerSafe("x".repeat(100))).toHaveLength(80)
    setDeviceName("โน้ตบุ๊ก")                                 // nothing left: the browser's summary goes instead
    expect(apiHeaders()["X-Strata-Device-Name"]).toBe(headerSafe(uaSummary(navigator.userAgent)))
  })
})

describe("apiHeaders", () => {
  test("carries the device id and the device name on every request", () => {
    setDeviceName("Ann's laptop")
    const h = apiHeaders()
    expect(h["X-Strata-Device"]).toBe(deviceId())
    expect(h["X-Strata-Device-Name"]).toBe("Ann's laptop")
    expect(apiHeaders(true)["Content-Type"]).toBe("application/json")
    expect(apiHeaders(true)["X-Strata-Device"]).toBe(deviceId())
  })
})

describe("isDevicePending / deviceGate", () => {
  test("the code of a 403 device_pending", async () => {
    expect(await isDevicePending(json(403, { error: { type: "device_pending", code: "ABCD", message: "waits" } }))).toBe("ABCD")
    expect(await isDevicePending(json(403, { error: { type: "device_pending", code: "----", message: "keys file" } }))).toBe("----")
  })
  test("null for every other answer", async () => {
    expect(await isDevicePending(json(403, { error: { type: "forbidden", message: "owner only" } }))).toBeNull()
    expect(await isDevicePending(json(403, { error: { type: "device_refused", message: "no" } }))).toBeNull()
    expect(await isDevicePending(json(403, { error: { message: "not this PC" } }))).toBeNull()
    expect(await isDevicePending(json(401, { error: { type: "device_pending", code: "ABCD" } }))).toBeNull()
    expect(await isDevicePending(json(200, { ok: true }))).toBeNull()
    expect(await isDevicePending(new Response("not json", { status: 403 }))).toBeNull()
  })
  test("the body is still there for the caller to read", async () => {
    const r = json(403, { error: { type: "device_pending", code: "ABCD", message: "waits" } })
    await isDevicePending(r)
    expect(((await r.json()) as { error: { code: string } }).error.code).toBe("ABCD")
  })
  test("deviceGate: pending with its code, refused, or nothing", async () => {
    expect(await deviceGate(json(403, { error: { type: "device_pending", code: "WXYZ" } }))).toEqual({ kind: "pending", code: "WXYZ" })
    expect(await deviceGate(json(403, { error: { type: "device_refused", message: "no" } }))).toEqual({ kind: "refused" })
    expect(await deviceGate(json(403, { error: { type: "forbidden" } }))).toBeNull()
    expect(await deviceGate(json(200, {}))).toBeNull()
  })
})
