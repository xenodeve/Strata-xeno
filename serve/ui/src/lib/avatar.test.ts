import { describe, expect, test } from "bun:test"
import { botFor, BOT_TYPES, getAvatar, getBotChoice, nextAvatar, plainLook, randomBot, setAvatar, setBotChoice } from "./avatar"
import { ORB_DESIGNS } from "./orbs"

// The avatar choice: orbs (the nine forms) or bots (Libraries.dev bot-avatars), the same states told by a shape and a mood.
describe("bot for an orb design", () => {
  test("each of the nine forms has a bot shape of its own", () => {
    const types = ORB_DESIGNS.map((d) => botFor(d, { moving: true }).type)
    expect(new Set(types).size).toBe(9)
  })
  test("something at work works; something at rest looks around; a dormant one sleeps", () => {
    expect(botFor("working", { moving: true }).state).toBe("working")
    expect(botFor("composing", { moving: true }).state).toBe("working")
    expect(botFor("searching", { moving: true, rest: true }).state).toBe("default")       // an idle server
    expect(botFor("breathing", { moving: true }).state).toBe("default")                  // waiting
    expect(botFor("shaping", { moving: true, rest: true }).state).toBe("sleeping")       // an unloaded model
    expect(botFor("composing", { moving: false }).state).toBe("sleeping")                // paused
  })
})

describe("the choice", () => {
  test("orbs unless bots were chosen, and the choice is kept", () => {
    expect(getAvatar()).toBe("orbs")
    setAvatar("bots")
    expect(getAvatar()).toBe("bots")
    setAvatar("orbs")
    expect(getAvatar()).toBe("orbs")
  })
})

describe("the three choices", () => {
  test("they go round: orbs, a plain loading mark, bots", () => {
    expect(nextAvatar("orbs")).toBe("loading")
    expect(nextAvatar("loading")).toBe("bots")
    expect(nextAvatar("bots")).toBe("orbs")
  })
  test("the plain loading mark is kept as a choice too", () => {
    setAvatar("loading")
    expect(getAvatar()).toBe("loading")
    setAvatar("orbs")
  })
  test("the plain mark turns at work, breathes at rest or while waiting, and stands still when paused", () => {
    expect(plainLook("composing", { moving: true })).toBe("spin")
    expect(plainLook("connecting", { moving: true })).toBe("spin")
    expect(plainLook("searching", { moving: true, rest: true })).toBe("pulse")     // an idle server
    expect(plainLook("breathing", { moving: true })).toBe("pulse")                 // waiting
    expect(plainLook("shaping", { moving: true, rest: true })).toBe("still")       // an unloaded model
    expect(plainLook("composing", { moving: false })).toBe("still")
  })
})

describe("which avatar", () => {
  test("by default each status has its own shape, as before", () => {
    expect(getBotChoice()).toBe("auto")
    expect(botFor("solving", { moving: true }, "auto").type).toBe("cat")
  })
  test("a chosen avatar is the one for every status, and its mood still follows the status", () => {
    for (const d of ["working", "solving", "composing", "shaping"] as const) expect(botFor(d, { moving: true }, "star").type).toBe("star")
    expect(botFor("working", { moving: true }, "ghost").state).toBe("working")
    expect(botFor("searching", { moving: true, rest: true }, "ghost").state).toBe("default")
    expect(botFor("composing", { moving: false }, "ghost").state).toBe("sleeping")
  })
  test("there are eighteen to choose from, and the choice is kept", () => {
    expect(BOT_TYPES).toHaveLength(18)
    setBotChoice("cat")
    expect(getBotChoice()).toBe("cat")
    setBotChoice("auto")
    expect(getBotChoice()).toBe("auto")
  })
})

describe("a random avatar", () => {
  test("one of the eighteen is drawn by a number in [0, 1)", () => {
    expect(randomBot(0)).toBe("clover")
    expect(randomBot(0.999999)).toBe("puddle")
    expect(new Set(Array.from({ length: 180 }, (_, i) => randomBot(i / 180))).size).toBe(18)
  })
  test("with random chosen, the avatar is the one that was drawn, whatever the status; its mood still follows the status", () => {
    expect(botFor("working", { moving: true }, "random", "alien").type).toBe("alien")
    expect(botFor("shaping", { moving: true }, "random", "alien").type).toBe("alien")
    expect(botFor("working", { moving: true }, "random", "alien").state).toBe("working")
    expect(botFor("working", { moving: true }, "random").type).toBe("mech")      // nothing drawn yet: by status
  })
  test("random is a choice that is kept", () => {
    setBotChoice("random")
    expect(getBotChoice()).toBe("random")
    setBotChoice("auto")
  })
})
