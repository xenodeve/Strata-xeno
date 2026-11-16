import { afterEach, describe, expect, test } from "bun:test"
import { setLang } from "./i18n"
import { aboutWhere, chatWhere, isLoopbackHost } from "./where"

// Where a chat's data goes (#310 final review I-2): "Nothing leaves it." only for a page opened on this PC itself and a chat on
// the Server PC; a remote browser and every My PC chat are told plainly where the prompt and what the tools read go.
afterEach(() => setLang("en"))

describe("isLoopbackHost", () => {
  test("this PC by name or address", () => {
    for (const h of ["localhost", "LOCALHOST", "app.localhost", "127.0.0.1", "127.1.2.3", "[::1]", "::1"]) expect(isLoopbackHost(h)).toBe(true)
  })
  test("anything else is another PC", () => {
    for (const h of ["26.33.142.160", "192.168.1.5", "100.64.0.1", "strata.example.com", "localhost.example.com", "128.0.0.1", "127.0.0", "[::2]", "0.0.0.0", ""])
      expect(isLoopbackHost(h)).toBe(false)
  })
})

describe("chatWhere", () => {
  test("this PC, a Server PC chat: nothing leaves it", () => {
    expect(chatWhere("Qwen", "localhost", "server")).toBe("Qwen runs on this PC. Nothing leaves it.")
    expect(chatWhere("Qwen", "127.0.0.1", "server")).toBe("Qwen runs on this PC. Nothing leaves it.")
  })
  test("another PC's browser, a Server PC chat: the prompt and what the tools read go to the server PC", () => {
    expect(chatWhere("Qwen", "26.33.142.160", "server")).toBe("Qwen runs on the server PC (26.33.142.160). Your prompts, and what the tools read for them, go there.")
  })
  test("a My PC chat from another PC: the files the tools read go to the server PC, the commands run on your PC", () => {
    expect(chatWhere("Qwen", "26.33.142.160", "my")).toBe("Qwen runs on the server PC (26.33.142.160). Your prompts, and the files the tools read on your PC, go there. Commands run on your PC.")
  })
  test("a My PC chat on this PC never says nothing leaves", () => {
    const s = chatWhere("Qwen", "localhost", "my")
    expect(s).toBe("Qwen runs on this PC. The files the tools read on your PC come here. Commands run on your PC.")
    expect(s).not.toContain("Nothing leaves it")
  })
  test("Thai", () => {
    setLang("th")
    expect(chatWhere("Qwen", "localhost", "server")).toBe("Qwen ทำงานบนเครื่องนี้ ไม่มีข้อมูลออกจากเครื่อง")
    expect(chatWhere("Qwen", "26.1.2.3", "my")).toContain("26.1.2.3")
    expect(chatWhere("Qwen", "26.1.2.3", "my")).not.toBe(chatWhere("Qwen", "26.1.2.3", "server"))
  })
})

describe("aboutWhere", () => {
  test("this PC: nothing leaves it; another PC: where it goes", () => {
    expect(aboutWhere("localhost")).toBe("Strata runs a large mixture-of-experts model on this PC. Nothing leaves it.")
    expect(aboutWhere("strata.example.com")).toBe("Strata runs a large mixture-of-experts model on the server PC (strata.example.com). Your prompts, and what the tools read for them, go there.")
  })
})
