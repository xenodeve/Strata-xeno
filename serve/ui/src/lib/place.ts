import { msg, t } from "./i18n"

// #338: where an MCP server or a tool runs, in one set of words for Settings, the + menu and the call cards, so they cannot drift: "this PC" is
// the user's own PC (through this browser's runner: a call's `machine: "my"`, a runner group `my_<name>` the page marked), the "central PC" the
// server. Only these functions say it.
export type Place = "my" | "server"

/** The place of a server in a list or of a call: `machine: "my"` is this PC; anything else is the central PC. */
export const placeOf = (x: { machine?: string } | null | undefined): Place => (x?.machine === "my" ? "my" : "server")

const BADGE: Record<Place, string> = { my: msg("This PC"), server: msg("Central PC") }

/** The small badge: "This PC" / "Central PC". */
export const placeBadge = (p: Place): string => t(BADGE[p])

/** A heading for a group of servers: "On this PC (<machine>)" / "On the central PC (the server)". */
export function placeHeading(p: Place, machine?: string | null): string {
  if (p === "server") return t("On the central PC (the server)")
  return machine ? t("On this PC ({name})", { name: machine }) : t("On this PC")
}
