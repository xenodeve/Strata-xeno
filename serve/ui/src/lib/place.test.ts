import { afterEach, describe, expect, test } from "bun:test"
import { setLang } from "./i18n"
import { placeBadge, placeHeading, placeOf } from "./place"

describe("#338: where an MCP server runs, in one set of words", () => {
  afterEach(() => setLang("en"))
  test("a server's own group is the central PC, a runner group my_<name> this PC", () => {
    setLang("en")
    expect(placeBadge(placeOf({ name: "files" } as { machine?: string }))).toBe("Central PC")
    expect(placeBadge(placeOf({ machine: "my" }))).toBe("This PC")
    expect(placeHeading("my", "ann-pc")).toBe("On this PC (ann-pc)")
    expect(placeHeading("server")).toBe("On the central PC (the server)")
  })
  test("Thai: บนเครื่องนี้ / บนเครื่องกลาง", () => {
    setLang("th")
    expect(placeHeading("my", "ann-pc")).toBe("บนเครื่องนี้ (ann-pc)")
    expect(placeHeading("server")).toBe("บนเครื่องกลาง (เซิร์ฟเวอร์)")
    expect(placeBadge("my")).toBe("เครื่องนี้")
    expect(placeBadge("server")).toBe("เครื่องกลาง")
  })
})
