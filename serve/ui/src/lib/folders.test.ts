import { describe, expect, test } from "bun:test"
import { matchParts, rank, sepOf, splitTyped } from "./folders"

describe("splitTyped: the folder to list and the part of a name typed after it", () => {
  test("a path typed up to a name", () => {
    expect(splitTyped("C:/work/app-w")).toEqual({ base: "C:/work/", partial: "app-w" })
    expect(splitTyped("C:\\Users\\xe")).toEqual({ base: "C:\\Users\\", partial: "xe" })
    expect(splitTyped("/home/me/pro")).toEqual({ base: "/home/me/", partial: "pro" })
    expect(splitTyped("~/src/st")).toEqual({ base: "~/src/", partial: "st" })
  })
  test("a path that ends with a separator lists that folder", () => {
    expect(splitTyped("C:/work/")).toEqual({ base: "C:/work/", partial: "" })
    expect(splitTyped("/")).toEqual({ base: "/", partial: "" })
  })
  test("a drive and the home folder are folders to list", () => {
    expect(splitTyped("C:")).toEqual({ base: "C:\\", partial: "" })
    expect(splitTyped("~")).toEqual({ base: "~/", partial: "" })
  })
  test("nothing to complete: empty, blank, a name with no separator", () => {
    expect(splitTyped("")).toBeNull()
    expect(splitTyped("   ")).toBeNull()
    expect(splitTyped("work")).toBeNull()
  })
  test("spaces before the path are not part of it", () => { expect(splitTyped("  C:/w")).toEqual({ base: "C:/", partial: "w" }) })
})

describe("sepOf", () => {
  test("the kind of separator the path was typed with", () => {
    expect(sepOf("C:\\Users\\")).toBe("\\")
    expect(sepOf("C:/work/")).toBe("/")
    expect(sepOf("/")).toBe("/")
  })
})

describe("rank: the folders that go with what was typed", () => {
  const names = ["docs", "app", "app-wt-fix", "my-app", "lib", "Application"]
  test("those that start with it first, then those that contain it, in any case", () => {
    expect(rank(names, "app")).toEqual(["app", "app-wt-fix", "Application", "my-app"])
    expect(rank(names, "APP-W")).toEqual(["app-wt-fix"])
  })
  test("nothing typed after the separator: all of them, as they are", () => { expect(rank(names, "")).toEqual(names) })
  test("nothing goes with it: none", () => { expect(rank(names, "zzz")).toEqual([]) })
})

describe("matchParts", () => {
  test("before, the match, after", () => {
    expect(matchParts("my-app-fix", "app")).toEqual(["my-", "app", "-fix"])
    expect(matchParts("App", "app")).toEqual(["", "App", ""])
  })
  test("no match or nothing typed: the name whole", () => {
    expect(matchParts("lib", "app")).toEqual(["lib", "", ""])
    expect(matchParts("lib", "")).toEqual(["lib", "", ""])
  })
})
