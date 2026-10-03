// The Requests page keeps its list when it is left, so opening it again shows the list at once, as far down as it had been opened. These put what the server sends into it.
export interface Row { id: string }

/** The newest page, read again, put in front of the list that was kept. When it reaches into the kept list (its oldest row is in it) what is older stays; when it does not (many requests came since)
 *  the list starts again from it, and `reset` says so (the page counts from the start again). */
export function mergeNewest<T extends Row>(kept: T[], newest: T[]): { rows: T[]; reset: boolean } {
  const oldest = newest[newest.length - 1]
  const at = oldest ? kept.findIndex((r) => r.id === oldest.id) : -1
  if (at >= 0) return { rows: [...newest, ...kept.slice(at + 1)], reset: false }
  return { rows: newest, reset: kept.length > 0 }
}

/** A page further down, added to the end: a row that is already there (requests came meanwhile, and moved the pages) is not shown twice. */
export function appendPage<T extends Row>(kept: T[], more: T[]): T[] {
  const have = new Set(kept.map((r) => r.id))
  return [...kept, ...more.filter((r) => !have.has(r.id))]
}
