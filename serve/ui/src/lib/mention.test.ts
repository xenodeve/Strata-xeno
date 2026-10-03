import { describe, expect, test } from "bun:test"
import { atQuery, mentionsOf, pickMention } from "./mention"

describe("atQuery: the @name the caret is in", () => {
  test("at the start, and after a space", () => {
    expect(atQuery("@src/ap", 7)).toEqual({ start: 0, query: "src/ap" })
    expect(atQuery("look at @src/ap", 15)).toEqual({ start: 8, query: "src/ap" })
    expect(atQuery("see @", 5)).toEqual({ start: 4, query: "" })
  })
  test("only what is before the caret counts", () => { expect(atQuery("see @src/app.py now", 9)).toEqual({ start: 4, query: "src/" }) })
  test("not an e-mail address, not a word that has gone on, not an @ that is not at a word", () => {
    expect(atQuery("me@example.com", 14)).toBeNull()
    expect(atQuery("@src/app.py and", 15)).toBeNull()
    expect(atQuery("plain text", 5)).toBeNull()
    expect(atQuery("a@@b", 4)).toBeNull()
  })
  test("a new line is a word start", () => { expect(atQuery("first\n@re", 9)).toEqual({ start: 6, query: "re" }) })
})

describe("pickMention", () => {
  test("the typed name becomes @path and a space, the caret after it", () => {
    const r = pickMention("look at @ap", { start: 8, query: "ap" }, 11, "src/app.py")
    expect(r).toEqual({ text: "look at @src/app.py ", caret: 20 })
  })
  test("what follows is kept, and what was left of the word after the caret goes", () => {
    const r = pickMention("fix @ap please", { start: 4, query: "a" }, 6, "src/app.py")
    expect(r.text).toBe("fix @src/app.py please")
    const mid = pickMention("fix @apXYZ please", { start: 4, query: "ap" }, 7, "a.py")
    expect(mid.text).toBe("fix @a.py please")
  })
})

describe("mentionsOf: the files a prompt mentions", () => {
  test("each at the start of a word, once, in order", () => {
    expect(mentionsOf("compare @src/a.py with @docs/b.md and @src/a.py")).toEqual(["src/a.py", "docs/b.md"])
  })
  test("the punctuation that ends a sentence is not part of the name", () => {
    expect(mentionsOf("see @README.md. Also (@src/x.ts), and @a.py?")).toEqual(["README.md", "src/x.ts", "a.py"])
  })
  test("not an e-mail address, not a way out of the project, not a drive or an absolute path", () => {
    expect(mentionsOf("mail me@example.com")).toEqual([])
    expect(mentionsOf("@../secret.txt @sub/../x @/etc/passwd @C:/x.txt @..")).toEqual([])
    expect(mentionsOf("@..\\x")).toEqual([])
  })
  test("no more than eight", () => { expect(mentionsOf(Array.from({ length: 12 }, (_, i) => `@f${i}.txt`).join(" "))).toHaveLength(8) })
  test("nothing: nothing", () => { expect(mentionsOf("")).toEqual([]) })
})
