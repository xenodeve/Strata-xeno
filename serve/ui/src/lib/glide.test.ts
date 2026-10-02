import { describe, expect, test } from "bun:test"
import { chooseStretching, type Seen } from "./glide"

// GlidePanel finds where a panel really grew and stretches that part, not the whole box. These are the choices it makes from what it saw: for
// every element its height before (none for an element that was not there) and now. A tree is {name, kids}; `up(a, b)` says a is an ancestor of b.
interface N { name: string; kids: N[] }
const tree = (name: string, ...kids: N[]): N => ({ name, kids })
const within = (a: N, b: N): boolean => a !== b && a.kids.some((k) => k === b || within(k, b))
const seen = (node: N, was: number | undefined, now: number): Seen<N> => ({ node, was, now })
const names = (xs: Seen<N>[]) => xs.map((x) => x.node.name).sort()

describe("chooseStretching: which parts stretch", () => {
  test("a paragraph that got longer, not the boxes round it", () => {
    const p = tree("p"), box = tree("box", p), root = tree("root", box)
    expect(names(chooseStretching([seen(root, 100, 122), seen(box, 100, 122), seen(p, 44, 66)], within))).toEqual(["p"])
  })
  test("two parts that changed each stretch", () => {
    const a = tree("a"), b = tree("b"), root = tree("root", a, b)
    expect(names(chooseStretching([seen(root, 100, 130), seen(a, 20, 30), seen(b, 20, 40)], within))).toEqual(["a", "b"])
  })
  test("a part that was not there is the new part: its own top, not what is inside it", () => {
    const leaf1 = tree("leaf1"), leaf2 = tree("leaf2"), row = tree("row", leaf1, leaf2), root = tree("root", row)
    expect(names(chooseStretching([seen(root, 50, 80), seen(row, undefined, 30), seen(leaf1, undefined, 15), seen(leaf2, undefined, 15)], within))).toEqual(["row"])
  })
  test("a part that is new and an old part that grew each stretch", () => {
    const p = tree("p"), row = tree("row"), root = tree("root", p, row)
    expect(names(chooseStretching([seen(root, 50, 100), seen(p, 20, 40), seen(row, undefined, 30)], within))).toEqual(["p", "row"])
  })
  test("a change of a pixel or two is not a stretch (rounding, a border)", () => {
    const p = tree("p"), root = tree("root", p)
    expect(chooseStretching([seen(root, 100, 101), seen(p, 44, 45)], within)).toEqual([])
  })
  test("a new part that is too small to see is not one either", () => {
    expect(chooseStretching([seen(tree("x"), undefined, 1)], within)).toEqual([])
  })
  test("a part that got shorter stretches too (it shrinks smoothly)", () => {
    const p = tree("p"), root = tree("root", p)
    expect(names(chooseStretching([seen(root, 122, 100), seen(p, 66, 44)], within))).toEqual(["p"])
  })
  test("nothing changed: nothing", () => {
    const p = tree("p")
    expect(chooseStretching([seen(p, 44, 44)], within)).toEqual([])
  })
})
