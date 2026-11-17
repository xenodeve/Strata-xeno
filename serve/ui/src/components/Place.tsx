import { placeBadge, type Place } from "../lib/place"

/** #338: the small badge of where a server or a call runs ("This PC" / "Central PC"), the same in Settings, the + menu and the call cards. */
export function PlaceBadge({ place }: { place: Place }) {
  return <span data-machine={place} data-place={place} className="shrink-0 rounded-sm bg-fill px-1.5 text-[11px] text-ink-2">{placeBadge(place)}</span>
}
