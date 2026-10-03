import { part as core } from "./th-core"
import { part as chat } from "./th-chat"
import { part as overview } from "./th-overview"
import { part as requests } from "./th-requests"
import { part as trace } from "./th-trace"

/** English text -> Thai, merged from the parts (one per area, so they can be written side by side). */
export const th: Record<string, string> = { ...core, ...chat, ...overview, ...requests, ...trace }
