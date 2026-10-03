import { describe, expect, test } from "bun:test"
import { effortChoices, settleEffort } from "./effort"

// The thinking levels a model accepts come from the server (what its chat template renders without an error). The chat offers
// those, in that order, and a level saved for another model is mapped onto one that exists.
describe("effort choices", () => {
  test("the levels the server lists, each with its name; the highest is the last", () => {
    expect(effortChoices(["none", "low", "medium", "xhigh"]).map((c) => c.value)).toEqual(["none", "low", "medium", "xhigh"])
    expect(effortChoices(["none", "low", "medium", "xhigh"]).map((c) => c.label)).toEqual(["Off", "Low", "Medium", "XHigh"])
    expect(effortChoices(["low", "high"]).map((c) => c.label)).toEqual(["Low", "High"])
  })
  test("a server that does not say (an older one) gets the four the app always had", () => {
    expect(effortChoices(undefined).map((c) => c.value)).toEqual(["none", "low", "medium", "high"])
    expect(effortChoices([]).map((c) => c.value)).toEqual(["none", "low", "medium", "high"])
  })
  test("an unknown level shows as its own name", () => {
    expect(effortChoices(["none", "ultra"]).map((c) => c.label)).toEqual(["Off", "ultra"])
  })
})

describe("a saved level for the model in use", () => {
  const qwen = ["none", "low", "medium", "xhigh"]
  test("a level the model has stays", () => {
    expect(settleEffort("medium", qwen, "xhigh")).toBe("medium")
    expect(settleEffort("none", qwen, "xhigh")).toBe("none")
  })
  test("high and xhigh are one level to a model that has the other name", () => {
    expect(settleEffort("high", qwen, "xhigh")).toBe("xhigh")
    expect(settleEffort("xhigh", ["none", "low", "medium", "high"], "high")).toBe("high")
  })
  test("a level the model lacks becomes its default, or its highest", () => {
    expect(settleEffort("medium", ["none", "low", "xhigh"], "xhigh")).toBe("xhigh")
    expect(settleEffort("none", ["low", "medium", "xhigh"], null)).toBe("xhigh")
    expect(settleEffort("low", [], null)).toBe("low")                 // nothing known: leave it
  })
})
