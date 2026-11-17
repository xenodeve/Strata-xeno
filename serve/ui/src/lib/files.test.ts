import { describe, expect, test } from "bun:test"
import { fileKind, MAX_DOC_FILE } from "./files"

// #334: what the attach button takes, by the file's name and type (a browser often gives no type for a document).
describe("fileKind", () => {
  test("documents by their name or the PDF type", () => {
    for (const name of ["a.pdf", "B.DOCX", "c.xlsx", "d.pptx", "e.odt", "f.ods", "g.odp", "h.epub", "i.rtf"]) expect(fileKind({ name, type: "" })).toBe("doc")
    expect(fileKind({ name: "scan", type: "application/pdf" })).toBe("doc")
    expect(fileKind({ name: "notes.rtf", type: "text/rtf" })).toBe("doc")                // not read as a text file
    expect(fileKind({ name: "r.docx", type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" })).toBe("doc")
  })
  test("old binary Office files, pictures, text and the rest", () => {
    for (const name of ["a.doc", "b.xls", "c.ppt"]) expect(fileKind({ name, type: "application/msword" })).toBe("legacy")
    expect(fileKind({ name: "p.png", type: "image/png" })).toBe("image")
    expect(fileKind({ name: "a.py", type: "" })).toBe("text")
    expect(fileKind({ name: "a.zip", type: "application/zip" })).toBe("other")
    expect(MAX_DOC_FILE).toBe(30e6)
  })
})
