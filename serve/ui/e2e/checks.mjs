// Each check: { name, run({ browser, fast, long, t, errors }) }. `t.ok(name, pass, detail)` records a result; a console error or a page
// error pushed to `errors` fails the check. `fast` and `long` are mock servers ({ base }): `long` streams a long answer.

/** A page in a context of its own, with the language and width set, and its errors collected. */
async function open(browser, errors, { lang = "en", width = 1100, height = 760 } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height } })
  await ctx.addInitScript((l) => { try { if (!localStorage.getItem("strata.lang")) localStorage.setItem("strata.lang", JSON.stringify(l)) } catch { /* private window */ } }, lang)      // only the first time: a reload must keep what the page chose
  const pg = await ctx.newPage()
  pg.on("pageerror", (e) => errors.push(`pageerror: ${e}`))
  pg.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`))
  return pg
}

const finished = (pg) => pg.waitForFunction(() => !document.querySelector("button[aria-label='Stop'], button[aria-label='หยุด']") && document.querySelector(".prose-chat"), null, { timeout: 60000 })
const send = async (pg, text) => { await pg.fill("textarea", text); await pg.keyboard.press("Enter"); await finished(pg); await pg.waitForTimeout(400) }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

export const checks = [
  {
    // every page, both languages, a wide and a narrow screen: nothing wider than the screen, no error, the language set
    name: "pages: no overflow and no error in English and Thai at 1280 and 390 px",
    async run({ browser, fast, t, errors }) {
      const rows = await (await fetch(fast.base + "/metrics/requests?page=1&size=1")).json()
      const id = rows.items?.[0]?.id
      const pages = ["#/chat", "#/dashboard", "#/live", "#/requests", id ? `#/requests/${id}` : null, "#/hardware", "#/settings", "#/settings/mcp-servers", "#/about", "#/requests/trace"].filter(Boolean)
      for (const lang of ["en", "th"]) for (const width of [1280, 390]) {
        const pg = await open(browser, errors, { lang, width, height: 800 })
        for (const hash of pages) {
          await pg.goto(`${fast.base}/${hash}`)
          await pg.waitForTimeout(1500)
          const r = await pg.evaluate(() => ({ over: document.documentElement.scrollWidth - innerWidth, lang: document.documentElement.lang }))
          t.ok(`${lang} ${width}px ${hash} fits the screen`, r.over <= 1 && r.lang === lang, `overflow ${r.over}px, lang ${r.lang}`)
        }
        await pg.context().close()
      }
    },
  },
  {
    // the answer is followed while the reader is at the end, left alone when they scroll up, and picked up again from the button
    name: "follow: the chat follows a streaming answer, lets go when the reader scrolls up, and comes back",
    async run({ browser, long, t, errors }) {
      const pg = await open(browser, errors, { width: 900, height: 700 })
      await pg.goto(long.base + "/#/chat")
      await pg.waitForSelector("textarea")
      const gap = () => pg.evaluate(() => Math.round(document.documentElement.scrollHeight - (scrollY + innerHeight)))
      await pg.fill("textarea", "tell me a long story")
      await pg.keyboard.press("Enter")
      const gaps = []
      for (let i = 0; i < 6; i++) { await pg.waitForTimeout(450); gaps.push(await gap()) }
      t.ok("while it streams the end stays in view", gaps.every((g) => g <= 4), gaps.join(" "))
      await pg.waitForFunction(() => document.documentElement.scrollHeight > 1000, null, { timeout: 30000 })
      await pg.mouse.move(450, 300)
      await pg.mouse.wheel(0, -400)
      await pg.waitForTimeout(300)
      const y1 = await pg.evaluate(() => scrollY)
      await pg.waitForTimeout(900)
      const y2 = await pg.evaluate(() => scrollY)
      t.ok("scrolling up while it streams is not undone", Math.abs(y2 - y1) <= 2, `${y1} -> ${y2}`)
      const jump = pg.locator("button[aria-label='Jump to the latest']")
      t.ok("a button to jump to the latest appears", (await jump.count()) === 1)
      await jump.click()
      await pg.waitForTimeout(900)
      t.ok("the button takes the reader to the end and goes away", (await gap()) <= 4 && (await jump.count()) === 0)
      await pg.waitForTimeout(600)
      t.ok("and the chat follows again", (await gap()) <= 4)
      await pg.context().close()
    },
  },
  {
    // opening the thinking must expand downward: the page must not scroll with it
    name: "open: opening the thinking does not scroll the page",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await send(pg, "think about it")
      const top = () => pg.evaluate(() => ({ head: Math.round(document.querySelector(".thought-head").getBoundingClientRect().top), max: document.documentElement.scrollHeight - innerHeight }))
      const a = await top()
      await pg.click(".thought-head")
      await pg.waitForTimeout(700)
      const b = await top()
      t.ok("the head stays where it is while the thought opens below it", Math.abs(b.head - a.head) <= 2 && b.max > a.max, `${a.head} -> ${b.head}`)
      await pg.click(".thought-head")
      await pg.waitForTimeout(700)
      t.ok("and closing it does not move the head either", Math.abs((await top()).head - a.head) <= 2)
      await pg.context().close()
    },
  },
  {
    // rewrite and take back a prompt: the edit box stretches open, an undone exchange closes up
    name: "edit: a prompt can be rewritten (the box stretches) and taken back (the exchange closes up)",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await send(pg, "first question")
      await send(pg, "second question")
      const users = () => pg.evaluate(() => [...document.querySelectorAll('.msg-in.group .fit-box [class*="whitespace-pre-wrap"]')].map((e) => e.textContent))
      const fit = () => pg.evaluate(() => Math.round(document.querySelector(".msg-in.group .fit-box").getBoundingClientRect().height))
      t.ok("two prompts are there", (await users()).join("|") === "first question|second question")
      await pg.locator(".msg-in.group").first().hover()
      const h0 = await fit()
      await pg.click("button[aria-label='Edit this prompt']")
      const open_ = [h0]
      for (let i = 0; i < 8; i++) { await pg.waitForTimeout(45); open_.push(await fit()) }
      t.ok("the edit box stretches through in-between heights", new Set(open_).size > 2 && open_[open_.length - 1] > h0, open_.join(" "))
      t.ok("it holds the prompt", (await pg.inputValue("textarea[aria-label='Edit the prompt']")) === "first question")
      await pg.fill("textarea[aria-label='Edit the prompt']", "changed my mind")
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(400)
      t.ok("Escape keeps the prompt as it was", (await users()).join("|") === "first question|second question")
      await pg.locator(".msg-in.group").first().hover()
      await pg.click("button[aria-label='Edit this prompt']")
      await pg.fill("textarea[aria-label='Edit the prompt']", "first, reworded")
      await pg.keyboard.press("Enter")
      await finished(pg)
      await pg.waitForTimeout(1000)
      t.ok("Enter replaces the prompt and what followed it with a new exchange", (await users()).join("|") === "first, reworded")
      await send(pg, "one more")
      await pg.locator(".msg-in.group").last().hover()
      const ghost = []
      await pg.click("button[aria-label='Take this prompt back']")
      for (let i = 0; i < 14; i++) { ghost.push(await pg.evaluate(() => { const g = document.querySelector(".ghost"); return g ? Math.round(g.getBoundingClientRect().height) : -1 })); await pg.waitForTimeout(40) }
      const live = ghost.filter((h) => h >= 0)
      t.ok("the taken-back exchange closes up gradually", live.length >= 3 && live[0] > live[live.length - 1] && new Set(live).size >= 3, ghost.join(" "))
      await pg.waitForTimeout(600)
      t.ok("and is gone, its prompt back in the composer", (await pg.locator(".ghost").count()) === 0 && (await pg.inputValue("textarea")) === "one more")
      await pg.context().close()
    },
  },
  {
    // sending: the prompt rises out of the composer to its place, and the empty chat's heading closes up instead of vanishing
    name: "send: the sent prompt glides out of the composer and the empty-chat heading closes up",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(800)
      const hero = () => pg.evaluate(() => { const e = document.querySelector("h1.display"); const g = e && e.closest(".collapse-grid"); return g ? Math.round(g.getBoundingClientRect().height) : -1 })
      const bubble = () => pg.evaluate(() => { const b = document.querySelector(".msg-in.group"); if (!b) return null; const c = getComputedStyle(b); return { tf: c.transform, op: +c.opacity, top: Math.round(b.getBoundingClientRect().top) } })
      const h0 = await hero()
      await pg.fill("textarea", "a question that is sent")
      await pg.keyboard.press("Enter")
      const rows = []
      for (let i = 0; i < 12; i++) { rows.push({ b: await bubble(), h: await hero() }); await pg.waitForTimeout(40) }
      const seen = rows.map((r) => r.b).filter(Boolean)
      t.ok("the prompt appears and starts away from its place (a transform)", seen.length > 0 && seen[0].tf !== "none", seen[0] ? seen[0].tf : "none")
      t.ok("it fades in on the way", seen.some((s) => s.op < 1))
      t.ok("and settles with no transform", seen[seen.length - 1].tf === "none" && seen[seen.length - 1].op === 1)
      const hs = rows.map((r) => r.h).filter((h) => h >= 0)
      t.ok("the heading closes up through in-between heights", h0 > 0 && hs.length >= 2 && Math.min(...hs) < h0 && new Set(hs).size >= 3, [h0, ...hs].join(" "))
      await pg.waitForTimeout(600)
      t.ok("and is gone", (await hero()) === -1)
      await pg.context().close()
    },
  },
  {
    // numbers that change pop in (transitions.dev "Number pop-in"), and the ones that were there from the start do not
    name: "pop: a number that changes pops in, and a page does not start with every figure popping",
    async run({ browser, long, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(long.base + "/#/live")
      await pg.waitForTimeout(2200)
      const popping = () => pg.evaluate(() => [...document.querySelectorAll(".t-digit:not([data-still])")].filter((e) => getComputedStyle(e).animationName === "t-digit-pop-in").length)
      t.ok("at rest no figure is popping", (await popping()) === 0)
      await pg.evaluate(() => { fetch("/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: "m", max_tokens: 800, messages: [{ role: "user", content: "go" }] }) }) })
      let most = 0, digits = 0, reels = 0, moving = 0, popMs = 0, reelMs = 0
      const lengths = () => pg.evaluate(() => ({
        pop: Math.max(0, ...[...document.querySelectorAll(".t-digit:not([data-still])")].filter((e) => getComputedStyle(e).animationName === "t-digit-pop-in").map((e) => parseFloat(getComputedStyle(e).animationDuration) * 1000)),
        reel: Math.max(0, ...[...document.querySelectorAll(".t-reel-strip[data-moving]")].map((e) => parseFloat(getComputedStyle(e).transitionDuration) * 1000)),
      }))
      for (let i = 0; i < 60; i++) { await pg.waitForTimeout(100); const l = await lengths(); popMs = Math.max(popMs, l.pop); reelMs = Math.max(reelMs, l.reel); most = Math.max(most, await popping()); digits = Math.max(digits, await pg.locator(".t-digit").count()); reels = Math.max(reels, await pg.locator(".t-reel-col").count()); moving = Math.max(moving, await pg.locator(".t-reel-strip[data-moving]").count()) }
      t.ok("a speed that goes up and down is made of reels", reels > 0, `${reels} reels`)
      t.ok("and a reel turns when its digit changes", moving > 0, `${moving} moving at once`)
      t.ok("a request makes figures out of digits", digits > 0, `${digits} digit elements`)
      t.ok("and the digits that change pop in", most > 0, `at most ${most} at once`)
      t.ok("a figure that changes about twice a second moves for a short part of that time (pop at most 330 ms, reel at most 460 ms), so it can be read", popMs <= 330 && reelMs <= 460 && (popMs > 0 || reelMs > 0), `pop ${Math.round(popMs)} ms, reel ${Math.round(reelMs)} ms`)
      await pg.context().close()
    },
  },
  {
    // the thinking is a window of a fixed height: a long thought cannot push the heading that closes it out of reach
    name: "reason: the thinking is a bounded window that follows the end while it is written and can always be closed",
    async run({ browser, long, t, errors }) {
      const pg = await open(browser, errors, { width: 900, height: 700 })
      await pg.addInitScript(() => { try { localStorage.setItem("strata.avatar", JSON.stringify("orbs")) } catch { /* private window */ } })      // orbs only: the thinking is marked by an orb
      await pg.goto(long.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.fill("textarea", "think about it at length")
      await pg.keyboard.press("Enter")
      await pg.waitForSelector(".t-reason-viewport", { timeout: 20000 })
      await pg.waitForFunction(() => { const v = document.querySelector(".t-reason-viewport"); return v && v.scrollHeight > v.clientHeight + 60 }, null, { timeout: 20000 })
      await pg.waitForTimeout(1500)
      const info = () => pg.evaluate(() => { const v = document.querySelector(".t-reason-viewport"); const h = document.querySelector(".thought-head").getBoundingClientRect(); return { h: Math.round(v.getBoundingClientRect().height), more: v.scrollHeight - v.clientHeight, fromEnd: Math.round(v.scrollHeight - v.clientHeight - v.scrollTop), above: v.hasAttribute("data-above"), below: v.hasAttribute("data-below"), headTop: Math.round(h.top), vh: innerHeight } })
      const a = await info()
      t.ok("the window has a fixed height though the thought is longer", a.h <= 180 && a.more > 60, `window ${a.h}px, ${a.more}px more`)
      t.ok("while it is written the window stays at the end", a.fromEnd <= 40, `${a.fromEnd}px from the end`)
      t.ok("the edge fades say there is more above", a.above)
      t.ok("the heading that closes it is on screen", a.headTop >= 0 && a.headTop < a.vh - 40, `y ${a.headTop}`)
      const glyph = () => pg.evaluate(() => { const g = document.querySelector(".thought-glyph"); const orb = g?.querySelector("canvas"); const lat = g?.querySelector(".lat"); return { orb: !!orb, orbShown: orb ? +getComputedStyle(orb.closest("span.absolute")).opacity : 0, tick: lat ? lat.getAttribute("data-status") : null } })
      const g1 = await glyph()
      t.ok("while it thinks the mark is an orb (Solving), not the lattice", g1.orb && g1.orbShown === 1 && g1.tick === "working", JSON.stringify(g1))
      await pg.click(".thought-head")
      await pg.waitForTimeout(600)
      t.ok("and it closes", (await pg.locator(".thought-head").getAttribute("aria-expanded")) === "false" && (await pg.locator(".t-reason-viewport").count()) === 0)
      await pg.click(".thought-head")
      await pg.waitForTimeout(600)
      t.ok("and opens again, still at the end while it is written", (await pg.locator(".t-reason-viewport").count()) === 1 && (await info()).fromEnd <= 60)
      await pg.waitForFunction(() => document.querySelector(".thought:not([data-working])"), null, { timeout: 30000 })
      await pg.waitForTimeout(700)
      const g2 = await glyph()
      t.ok("when it is done the orb gives way to the lattice's tick", g2.orbShown === 0 && g2.tick === "done", JSON.stringify(g2))
      await pg.context().close()
    },
  },
  {
    // the + button becomes the menu it opens (transitions.dev "Plus to menu morph") and the menu closes back into it
    name: "plus: the + button morphs into its menu and back",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(600)
      const box = () => pg.evaluate(() => { const m = document.querySelector(".t-morph"); if (!m) return null; const r = m.getBoundingClientRect(); return { w: Math.round(r.width), h: Math.round(r.height), br: parseFloat(getComputedStyle(m).borderTopLeftRadius) } })
      const b0 = await box()
      t.ok("the button is a small box", !!b0 && b0.w <= 40 && b0.h <= 40, JSON.stringify(b0))
      await pg.click("button[aria-label='Photos, files, new chat, save']")
      const grow = []
      for (let i = 0; i < 9; i++) { await pg.waitForTimeout(40); grow.push(await box()) }
      t.ok("it grows into the menu through in-between sizes", new Set(grow.map((g) => g.h)).size > 3 && grow[grow.length - 1].h > b0.h + 80 && grow[grow.length - 1].w > b0.w + 150, grow.map((g) => `${g.w}x${g.h}`).join(" "))
      await pg.waitForTimeout(400)
      t.ok("the six actions are there to use", (await pg.locator(".t-morph-menu [role=option]").count()) === 6 && (await pg.locator(".t-morph-menu [role=option]").first().isVisible()))
      await pg.keyboard.press("Escape")
      const shrink = []
      for (let i = 0; i < 9; i++) { await pg.waitForTimeout(40); shrink.push(await box()) }
      t.ok("Escape closes it back into the button through in-between sizes", new Set(shrink.map((g) => g.h)).size > 3 && shrink[shrink.length - 1].h < 60, shrink.map((g) => `${g.w}x${g.h}`).join(" "))
      await pg.waitForTimeout(400)
      const b1 = await box()
      t.ok("and it is the small button again", b1.w <= 40 && b1.h <= 40)
      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(500)
      await pg.locator(".t-morph-menu [role=option]").nth(2).click({ trial: true })
      t.ok("a row can be clicked while it is open", true)
      await pg.context().close()
    },
  },
  {
    // what the server is doing is named: reading, thinking, answering are different statuses (thinking is not "writing")
    name: "status: reading the prompt, thinking and answering are named apart",
    async run({ browser, long, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(long.base + "/#/live")
      await pg.waitForTimeout(1800)
      await pg.evaluate(() => { fetch("/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: "m", max_tokens: 2000, messages: [{ role: "user", content: "go" }] }) }) })
      const seen = new Set()
      for (let i = 0; i < 90; i++) { await pg.waitForTimeout(150); const h = await pg.evaluate(() => document.querySelector(".page-title")?.textContent || ""); seen.add(h.replace(/[0-9,]+/g, "N")) }
      const all = [...seen].join(" | ")
      t.ok("while the thought is written the status says Thinking, not Writing", [...seen].some((h) => /^Thinking · N tokens/.test(h)), all)
      t.ok("then it says Answering", [...seen].some((h) => /^Answering · N tokens/.test(h)), all)
      t.ok("and it never calls thinking or the answer Writing", ![...seen].some((h) => /^Writing · /.test(h)), all)
      await pg.context().close()
    },
  },
  {
    // the status marks: one list to read and choose from (a menu from the header, the same list in Settings)
    name: "avatar: the status marks are chosen from a list that shows each way, and it is remembered",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      const scripts = []
      pg.on("request", (r) => { if (/\.js(\?|$)/.test(r.url())) scripts.push(r.url()) })
      await pg.goto(fast.base + "/#/dashboard")
      await pg.waitForTimeout(2000)
      const before = scripts.length
      const btn = pg.locator("button[aria-label^='Status marks']")
      const menu = pg.locator("[role=dialog][aria-label='Status marks']")
      const radios = menu.locator("[role=radio]")
      const checked = () => pg.evaluate(() => [...document.querySelectorAll("[role=dialog] [role=radio]")].map((r) => r.getAttribute("aria-checked") === "true").indexOf(true))
      const painted = () => pg.evaluate(() => [...document.querySelectorAll(".orb-slot canvas")].map((c) => { try { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n } catch { return -1 } }))
      t.ok("the button names the way in use and opens a list", (await btn.getAttribute("aria-label")).includes("Orbs + Loading") && (await menu.count()) === 0)
      await btn.click()
      await pg.waitForTimeout(500)
      t.ok("four ways are listed, each with its name and a line on what it is", (await radios.count()) === 4 && (await menu.innerText()).includes("A dotted ball") && (await menu.innerText()).includes("Loaders of dots"))
      t.ok("the one in use is marked, and it says which is the default", (await checked()) === 1 && (await menu.innerText()).includes("Default"))
      t.ok("each way shows three of its marks, and the avatars are not loaded for it yet", (await radios.nth(0).locator("canvas").count()) === 3 && (await radios.nth(2).locator(".lat, .t-matrix").count()) === 3 && scripts.length === before)
      await radios.nth(1).focus()
      await pg.keyboard.press("ArrowDown")
      await pg.waitForTimeout(700)
      const m = await pg.evaluate(() => ({ lat: document.querySelectorAll(".orb-slot .lat").length, matrix: document.querySelectorAll(".orb-slot .t-matrix").length, canvas: document.querySelectorAll(".orb-slot canvas").length }))
      t.ok("the arrow keys move through them: Loading only puts loaders of dots in place of the orbs", (await checked()) === 2 && m.lat + m.matrix >= 2 && scripts.length === before, JSON.stringify(m))
      const panel = () => pg.evaluate(() => { const b = document.querySelector("[role=dialog][aria-label='Status marks']").getBoundingClientRect(); return { w: Math.round(b.width), h: Math.round(b.height) } })
      const p0 = await panel()
      await pg.keyboard.press("ArrowDown")
      const grow = []
      for (let i = 0; i < 12; i++) { await pg.waitForTimeout(45); grow.push(await panel()) }
      await pg.waitForTimeout(2200)
      t.ok("choosing Avatar stretches the list open through in-between heights", new Set(grow.map((g) => g.h)).size >= 4 && grow[grow.length - 1].h > p0.h + 100, grow.map((g) => g.h).join(" "))
      t.ok("and it does not widen: the scrollbar's room is kept, so nothing jumps sideways", grow.every((g) => Math.abs(g.w - p0.w) <= 1) && Math.abs((await panel()).w - p0.w) <= 1, `${p0.w} -> ${grow.map((g) => g.w).join(" ")}`)
      t.ok("Avatar loads the avatars (one more script) and shows which avatar to use", (await checked()) === 3 && scripts.length > before && (await menu.locator("[role=group][aria-label='Which avatar'] button").count()) === 20, `${before} -> ${scripts.length}`)
      await menu.locator("button[aria-label='Cat']").click()
      await pg.waitForTimeout(500)
      t.ok("choosing Cat is kept", (await pg.evaluate(() => localStorage.getItem("strata.avatar.type"))) === '"cat"' && (await menu.locator("button[aria-label='Cat']").getAttribute("aria-pressed")) === "true")
      await menu.getByRole("button", { name: "Random", exact: true }).click()
      await pg.waitForTimeout(500)
      t.ok("and so is Random", (await pg.evaluate(() => localStorage.getItem("strata.avatar.type"))) === '"random"')
      t.ok("an avatar is drawn in each place", (await painted()).filter((n) => n > 0).length >= 2)
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(300)
      t.ok("Escape closes the list", (await menu.count()) === 0)
      await btn.click()
      await pg.waitForTimeout(400)
      await pg.mouse.click(300, 650)
      await pg.waitForTimeout(300)
      t.ok("so does a click elsewhere", (await menu.count()) === 0)
      await pg.reload()
      await pg.waitForTimeout(2500)
      t.ok("the choice is remembered after a reload", (await btn.getAttribute("aria-label")).includes("Avatar"))
      await pg.goto(fast.base + "/#/settings")
      await pg.waitForTimeout(2000)
      const aboutRadios = pg.locator("main [role=radiogroup][aria-label='Status marks'] [role=radio]")
      t.ok("Settings has the same list", (await aboutRadios.count()) === 4 && (await aboutRadios.nth(3).getAttribute("aria-checked")) === "true")
      const section = () => pg.evaluate(() => Math.round(document.querySelector("main [role=radiogroup][aria-label='Status marks']").closest("section").getBoundingClientRect().height))
      const s0 = await section()
      await aboutRadios.nth(0).click()
      const shrink = []
      for (let i = 0; i < 14; i++) { await pg.waitForTimeout(45); shrink.push(await section()) }
      t.ok("in Settings, leaving Avatar shrinks the section smoothly through in-between heights", new Set(shrink).size >= 5 && shrink[shrink.length - 1] < s0 - 100, `${s0} -> ${shrink.join(" ")}`)
      await pg.waitForTimeout(500)
      await aboutRadios.nth(3).click()
      const stretch = []
      for (let i = 0; i < 14; i++) { await pg.waitForTimeout(45); stretch.push(await section()) }
      t.ok("and choosing Avatar stretches it open smoothly", new Set(stretch).size >= 5 && stretch[stretch.length - 1] > stretch[0] + 60, stretch.join(" "))
      await pg.waitForTimeout(700)
      await aboutRadios.nth(0).click()
      await pg.waitForTimeout(800)
      const label = await btn.getAttribute("aria-label")
      t.ok("and choosing Orbs there changes the page at once", label.endsWith("Orbs") && (await painted()).some((n) => n > 0), label)
      await pg.context().close()
    },
  },
  {
    // on a phone the list stays on the screen
    name: "marks: on a phone the list of status marks fits the screen",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors, { width: 390, height: 780 })
      await pg.goto(fast.base + "/#/dashboard")
      await pg.waitForTimeout(1800)
      await pg.click("button[aria-label^='Status marks']")
      await pg.waitForTimeout(600)
      const r = await pg.evaluate(() => { const b = document.querySelector("[role=dialog][aria-label='Status marks']").getBoundingClientRect(); return { l: Math.round(b.left), r: Math.round(b.right), w: innerWidth, over: document.documentElement.scrollWidth - innerWidth } })
      t.ok("the list is inside the screen, on both sides", r.l >= 0 && r.r <= r.w && r.over <= 1, JSON.stringify(r))
      await pg.context().close()
    },
  },
  {
    // "orbs with loading": by the thinking the loader of dots (as before), not an orb
    name: "mixed: by default the thinking is marked by the lattice of dots, not an orb",
    async run({ browser, long, t, errors }) {
      const pg = await open(browser, errors, { width: 900, height: 700 })
      await pg.goto(long.base + "/#/chat")       // nothing chosen: orbs with loading is how it starts
      await pg.waitForSelector("textarea")
      await pg.fill("textarea", "think about it")
      await pg.keyboard.press("Enter")
      await pg.waitForSelector(".thought-glyph .lat", { timeout: 20000 })
      const g = await pg.evaluate(() => ({ lat: document.querySelector(".thought-glyph .lat")?.getAttribute("data-status"), canvas: document.querySelectorAll(".thought-glyph canvas").length }))
      t.ok("while it thinks: the lattice runs and there is no orb in the mark", g.lat === "working" && g.canvas === 0, JSON.stringify(g))
      await pg.waitForFunction(() => document.querySelector(".thought:not([data-working]) .lat[data-status='done']"), null, { timeout: 40000 })
      t.ok("when it is done: its tick", true)
      await pg.context().close()
    },
  },
  {
    // the + menu's MCP row opens the list of servers: each with its tools and its own switch; what is switched off is named in the request
    name: "mcp: the MCP row opens a list of servers with their tools, and each one can be switched off",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      const bodies = []
      const servers = [
        { name: "fs", transport: "stdio", status: "ready", tools: [{ tool: "read", description: "Read a file" }, { tool: "list" }, { tool: "search" }] },
        { name: "web", transport: "http", status: "ready", tools: [{ tool: "fetch" }, { tool: "screenshot" }] },
        { name: "down", transport: "stdio", status: "failed", error: "could not start", tools: [] },
      ]
      await pg.route("**/mcp", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ servers, tools: 5 }) }))
      await pg.route("**/v1/chat/completions", async (r) => { try { bodies.push(JSON.parse(r.request().postData() || "{}")) } catch { /* not json */ } await r.continue() })
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(1500)
      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(600)
      const row = pg.locator(".t-morph-menu [role=option]", { hasText: "MCP tools" })
      const rowText = () => row.innerText().then((x) => x.split(String.fromCharCode(10)).join(" | "))
      t.ok("the row says how many servers are on and how many tools", (await row.count()) === 1 && (await rowText()).includes("2 of 3 servers on · 5 tools"), await rowText())
      t.ok("it is an ordinary row now, not a switch", (await row.getAttribute("aria-checked")) !== null && (await row.isDisabled()) === false)
      await row.click()
      await pg.waitForTimeout(500)
      const panel = pg.locator("[role=dialog][aria-label='MCP tools']")
      t.ok("clicking it opens the list and closes the menu", (await panel.count()) === 1 && (await pg.locator(".t-morph[data-open='true']").count()) === 0)
      const fs = panel.locator("li[data-server='fs']")
      t.ok("each server shows its state and its tools by name", (await fs.innerText()).includes("Connected") && (await fs.locator(".prompt-bar__mcp-tools li").allInnerTexts()).join(",") === "read,list,search" && (await panel.locator("li[data-server='web'] .prompt-bar__mcp-tools li").allInnerTexts()).join(",") === "fetch,screenshot")
      t.ok("a tool's description is its tooltip", (await fs.locator("li[title='Read a file']").count()) === 1)
      t.ok("a server that failed says why, has no tools, and cannot be switched", (await panel.locator("li[data-server='down']").innerText()).includes("could not start") && (await panel.locator("li[data-server='down'] [role=switch]").isDisabled()))
      const sw = (n) => panel.locator(`[role=switch][aria-label='Use ${n}']`)
      t.ok("both working servers are on", (await sw("fs").getAttribute("aria-checked")) === "true" && (await sw("web").getAttribute("aria-checked")) === "true")
      await sw("web").click()
      await pg.waitForTimeout(300)
      t.ok("switching web off is kept, and the others stay on", (await sw("web").getAttribute("aria-checked")) === "false" && (await sw("fs").getAttribute("aria-checked")) === "true" && JSON.stringify(await pg.evaluate(() => JSON.parse(localStorage.getItem("strata.sampling")).mcpOff)) === '["web"]')
      t.ok("and its tools are dimmed", (await panel.locator("li[data-server='web']").getAttribute("data-off")) !== null && (await panel.locator("li[data-server='fs']").getAttribute("data-off")) === null)
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(300)
      t.ok("Escape closes the list", (await panel.count()) === 0)
      await pg.fill("textarea", "hi")
      await pg.keyboard.press("Enter")
      await pg.waitForFunction(() => document.querySelector(".prose-chat, .thought"), null, { timeout: 30000 })
      t.ok("the request asks for MCP and names the server that is off", bodies[0]?.strata_mcp === true && JSON.stringify(bodies[0]?.strata_mcp_off) === '["web"]', JSON.stringify([bodies[0]?.strata_mcp, bodies[0]?.strata_mcp_off]))
      await finished(pg)

      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(500)
      t.ok("the row now says one of the two is on", (await rowText()).includes("1 of 3 servers on"), await rowText())
      await row.click()
      await pg.waitForTimeout(400)
      await sw("fs").click()
      await pg.waitForTimeout(300)
      await pg.keyboard.press("Escape")
      await pg.fill("textarea", "again")
      await pg.keyboard.press("Enter")
      await finished(pg)
      t.ok("with every server off the request does not ask for MCP", bodies[1] && bodies[1].strata_mcp !== true && bodies[1].strata_mcp_off === undefined, JSON.stringify([bodies[1]?.strata_mcp, bodies[1]?.strata_mcp_off]))

      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(500)
      await row.click()
      await pg.waitForTimeout(400)
      await panel.locator("[role=switch][aria-label='All MCP tools']").click()
      await pg.waitForTimeout(300)
      t.ok("the master switch turns the tools off and the servers can no longer be switched", (await panel.locator("[role=switch][aria-label='All MCP tools']").getAttribute("aria-checked")) === "false" && (await sw("fs").isDisabled()) && (await pg.evaluate(() => JSON.parse(localStorage.getItem("strata.sampling")).mcp)) === false)
      await pg.click("body", { position: { x: 5, y: 300 } })
      await pg.waitForTimeout(300)
      t.ok("a click elsewhere closes it", (await panel.count()) === 0)
      await pg.context().close()
    },
  },
  {
    // with no MCP server the row is there, opens, and says where to add them
    name: "mcpnone: with no MCP server the list says so and points to Settings",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(1200)
      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(600)
      const row = pg.locator(".t-morph-menu [role=option]", { hasText: "MCP tools" })
      t.ok("the row is there and says no server is set up", (await row.count()) === 1 && (await row.innerText()).includes("No server is set up"))
      await row.click()
      await pg.waitForTimeout(400)
      const panel = pg.locator("[role=dialog][aria-label='MCP tools']")
      t.ok("it opens a list that says so, with no switch", (await panel.innerText()).includes("No server is set up") && (await panel.locator("[role=switch]").count()) === 0)
      await panel.getByRole("link", { name: "Set up servers" }).click()
      await pg.waitForTimeout(800)
      t.ok("and the link goes to Settings", pg.url().endsWith("#/settings/mcp-servers") && (await panel.count()) === 0)
      await pg.context().close()
    },
  },
  {
    // a prompt that is only a file has no text, so it has no (empty) bubble; it can still be rewritten
    name: "file: a prompt of only a file shows the file and no empty bubble",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.setInputFiles("input[type=file]", { name: "notes.md", mimeType: "text/markdown", buffer: Buffer.from("# notes") })
      await pg.waitForTimeout(500)
      await pg.keyboard.press("Enter")
      await finished(pg)
      await pg.waitForTimeout(500)
      t.ok("the file is shown", (await pg.locator(".msg-in.group", { hasText: "notes.md" }).count()) === 1)
      t.ok("there is no empty bubble under it", (await pg.locator(".msg-in.group .fit-box").count()) === 0)
      await pg.locator(".msg-in.group").first().hover()
      await pg.click("button[aria-label='Edit this prompt']")
      await pg.waitForTimeout(400)
      t.ok("Edit still opens a box to add words", (await pg.locator("textarea[aria-label='Edit the prompt']").count()) === 1)
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(500)
      t.ok("and closing it leaves no empty bubble", (await pg.locator(".msg-in.group .fit-box").count()) === 0)
      await pg.context().close()
    },
  },
  {
    // the idle orbs are alive: a still picture would mean the animation stopped
    name: "orbs: every orb on the Dashboard redraws",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/dashboard")
      await pg.waitForTimeout(2000)
      const sample = () => pg.evaluate(() => [...document.querySelectorAll(".orb-layer canvas")].map((c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 3; i < d.length; i += 4) h = (h * 31 + d[i]) | 0; return h }))
      const a = await sample()
      await sleep(1300)
      const b = await sample()
      t.ok("there are orbs", a.length >= 2, String(a.length))
      t.ok("none of them is a frozen picture", a.length > 0 && a.every((h, i) => h !== b[i]))
      await pg.context().close()
    },
  },
  {
    // the language switch changes the page at once, and is remembered
    name: "language: the switch changes the page without a reload and is remembered",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.click("button[aria-label^='Language']")
      await pg.waitForTimeout(400)
      t.ok("the page is Thai", (await pg.evaluate(() => document.documentElement.lang)) === "th" && (await pg.locator("nav a[aria-label='แชท']").count()) === 1)
      await pg.reload()
      await pg.waitForSelector("textarea")
      t.ok("and it stays Thai after a reload", (await pg.evaluate(() => document.documentElement.lang)) === "th")
      await pg.context().close()
    },
  },
  {
    // the thinking levels are the model's: what the server lists, and what the request carries
    name: "effort: the composer offers the model's own thinking levels and the request carries the chosen one",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      const bodies = []
      await pg.route("**/v1/chat/completions", async (r) => { try { bodies.push(JSON.parse(r.request().postData() || "{}")) } catch { /* not json */ } await r.continue() })
      await pg.addInitScript(() => { try { localStorage.setItem("strata.sampling", JSON.stringify({ thinking: "high" })) } catch { /* private window */ } })
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(2000)
      await pg.click("button[aria-label='Thinking effort']")
      await pg.waitForTimeout(400)
      t.ok("four steps, the top one named XHigh", (await pg.locator(".prompt-bar__effort-dot").count()) === 4 && (await pg.locator(".prompt-bar__effort-level").textContent()) === "XHigh")
      await pg.keyboard.press("Escape")
      await send(pg, "hi")
      t.ok("a level saved as high is sent as xhigh", bodies[0]?.reasoning_effort === "xhigh", String(bodies[0]?.reasoning_effort))
      await pg.context().close()
    },
  },
  {
    // MCP servers are set up from Settings, against a real run config file and a real (fixture) MCP server: add, edit (a secret is
    // kept), turn off, delete, paste a Claude Desktop block, the limits; the secret never reaches the page
    name: "mcpset: MCP servers are set up from Settings",
    async run({ browser, admin, t, errors }) {
      const fs = await import("node:fs")
      const disk = () => JSON.parse(fs.readFileSync(admin.config, "utf8"))
      const pg = await open(browser, errors)
      await pg.goto(admin.base + "/#/settings/mcp-servers")
      const row = (n) => pg.locator(`li[data-server='${n}']`)
      const connected = (n) => pg.waitForFunction((x) => document.querySelector(`li[data-server='${x}']`)?.innerText.includes("Connected"), n, { timeout: 40000 })
      await row("fake").waitFor({ timeout: 15000 })
      await connected("fake")
      const first = await row("fake").innerText()
      t.ok("the server of the run config is listed as connected, with its tools", first.includes("6 tools") && first.includes("echo"), first.split(String.fromCharCode(10)).join(" | "))
      const raw = await pg.evaluate(async () => (await fetch("mcp/config")).text())
      t.ok("the secret is neither in what the server sends nor on the page", !raw.includes(admin.secret) && !(await pg.content()).includes(admin.secret))

      await pg.getByRole("button", { name: "Add a server", exact: true }).click()
      const form = pg.locator("form[aria-label='Add a server']")
      await form.waitFor()
      // Program <-> Address: the form stretches and shrinks to the other kind's fields instead of jumping, and the fields slide
      // sideways (the program ones sit to the left, the address ones to the right) while it does
      const swap = (to) => pg.evaluate(async () => {
        const f = document.querySelector("form[aria-label='Add a server']")
        const h = [], x = { left: [], right: [] }, pill = []
        const mark = f.querySelector("[role=radiogroup] [data-pill]")
        f.querySelector("[role=radio][aria-checked=false]").click()
        const t0 = performance.now()
        while (performance.now() - t0 < 900) {
          h.push(Math.round(f.getBoundingClientRect().height))
          for (const p of f.querySelectorAll(".kind-pane")) x[p.dataset.side].push(Math.round(p.getBoundingClientRect().left))      // where each pane is, sideways, every frame
          if (mark) pill.push(Math.round(mark.getBoundingClientRect().left))      // the active-button marker, sideways, every frame
          await new Promise((r) => requestAnimationFrame(r))
        }
        return { h, x, pill }
      })
      const moves = (a) => new Set(a).size > 3
      await pg.waitForTimeout(700)
      const toAddress = await swap()
      t.ok("switching to an address changes the form's height through in-between sizes", new Set(toAddress.h).size > 4 && toAddress.h[0] !== toAddress.h.at(-1) && toAddress.h.slice(1, -1).some((v) => v !== toAddress.h[0] && v !== toAddress.h.at(-1)), JSON.stringify([...new Set(toAddress.h)]))
      t.ok("and the address fields slide in from the right while the program fields slide out to the left", moves(toAddress.x.right) && toAddress.x.right[0] > toAddress.x.right.at(-1) && moves(toAddress.x.left) && toAddress.x.left[0] > toAddress.x.left.at(-1), JSON.stringify([toAddress.x.right.slice(0, 40), toAddress.x.left.slice(0, 40)]))
      t.ok("the active-button marker glides from the program button to the address button, as the top navigation's does", new Set(toAddress.pill).size > 4 && toAddress.pill[0] < toAddress.pill.at(-1), JSON.stringify([...new Set(toAddress.pill)]))
      t.ok("and the address fields are what is left", (await form.locator("input[aria-label='Address']").count()) === 1 && (await form.locator("input[aria-label='Program']").count()) === 0)
      const toProgram = await swap()
      t.ok("switching back stretches it again, and the program fields are back", new Set(toProgram.h).size > 4 && (await form.locator("input[aria-label='Program']").count()) === 1 && (await form.locator("input[aria-label='Address']").count()) === 0, JSON.stringify([...new Set(toProgram.h)]))
      t.ok("and it glides back", new Set(toProgram.pill).size > 4 && toProgram.pill[0] > toProgram.pill.at(-1), JSON.stringify([...new Set(toProgram.pill)]))
      t.ok("and this time the program fields slide in from the left", moves(toProgram.x.left) && toProgram.x.left[0] < toProgram.x.left.at(-1), JSON.stringify(toProgram.x.left.slice(0, 40)))
      await form.locator("input[aria-label='Name']").fill("a b")
      await form.locator("input[aria-label='Program']").fill(admin.py)
      await form.getByRole("button", { name: "Save", exact: true }).click()
      await pg.waitForTimeout(400)
      t.ok("a name with a space is refused beside the field, and nothing is written", (await form.innerText()).includes("no spaces") && !("a b" in disk().mcp_servers))
      await form.locator("input[aria-label='Name']").fill("second")
      await form.locator("textarea[aria-label='Arguments']").fill(admin.fakeMcp)
      await form.getByRole("button", { name: "Save", exact: true }).click()
      await row("second").waitFor({ timeout: 15000 })
      await connected("second")
      t.ok("a new server is written to the run config, starts without a restart and lists its tools", JSON.stringify(disk().mcp_servers.second.args) === JSON.stringify([admin.fakeMcp]) && (await row("second").innerText()).includes("6 tools"))

      await pg.click("button[aria-label='Edit fake']")
      const edit = pg.locator("form[aria-label='Edit']")
      await edit.waitFor()
      t.ok("editing shows the secret as a mask, and the name cannot change", (await edit.locator("textarea[aria-label='Environment']").inputValue()) === "TOKEN=********" && (await edit.locator("input[aria-label='Name']").isDisabled()))
      await edit.locator("input[aria-label='Folder']").fill(".")
      await edit.getByRole("button", { name: "Save", exact: true }).click()
      await pg.waitForFunction(() => !document.querySelector("form[aria-label='Edit']"), null, { timeout: 15000 })
      await connected("fake")
      t.ok("saving an edit keeps the stored secret and starts the server again", disk().mcp_servers.fake.env.TOKEN === admin.secret && disk().mcp_servers.fake.cwd === ".")

      await pg.click("button[aria-label='Turn off second']")
      await pg.waitForFunction(() => document.querySelector("li[data-server='second']")?.innerText.includes("Disabled"), null, { timeout: 15000 })
      t.ok("turning one off keeps it in the file, disabled", disk().mcp_servers.second.disabled === true)
      await pg.click("button[aria-label='Delete second']")
      t.ok("Delete asks once more before it deletes", (await pg.locator("button[aria-label='Delete second']").innerText()).includes("Sure"))
      await pg.click("button[aria-label='Delete second']")
      await row("second").waitFor({ state: "detached", timeout: 15000 })
      t.ok("and then it is gone from the file", !("second" in disk().mcp_servers))

      await pg.getByRole("button", { name: "Paste from Claude Desktop", exact: true }).click()
      const box = pg.locator("textarea[aria-label='The block to paste']")
      await box.waitFor()
      await box.fill("{ nope")
      await pg.getByRole("button", { name: "Add", exact: true }).click()
      await pg.waitForTimeout(300)
      t.ok("a block that is not JSON says so", (await pg.locator("[role=alert]").first().innerText()).includes("not valid JSON"))
      await box.fill(JSON.stringify({ mcpServers: { pasted: { command: admin.py, args: [admin.fakeMcp] } } }))
      await pg.getByRole("button", { name: "Add", exact: true }).click()
      await row("pasted").waitFor({ timeout: 15000 })
      t.ok("a Claude Desktop block adds its servers", disk().mcp_servers.pasted.command === admin.py)

      await pg.locator(".branched-menu__item", { hasText: "Limits" }).click()
      await pg.waitForSelector("input[aria-label='Tool rounds in one answer']")
      const rounds = pg.locator("input[aria-label='Tool rounds in one answer']")
      await rounds.fill("99")
      await pg.getByRole("button", { name: "Save the limits" }).click()
      await pg.waitForTimeout(600)
      t.ok("a limit out of range is refused with its name and nothing is written", (await pg.locator("[role=alert]").first().innerText()).includes("max_rounds") && !(disk().mcp && disk().mcp.max_rounds))
      for (let i = errors.length - 1; i >= 0; i--) if (/status of 400/.test(errors[i])) errors.splice(i, 1)      // the browser logs the refusal it was just asked to provoke
      await rounds.fill("3")
      await pg.getByRole("button", { name: "Save the limits" }).click()
      await pg.waitForFunction(() => !document.querySelector("[role=alert]"), null, { timeout: 15000 })
      await pg.waitForTimeout(300)
      t.ok("a limit in range is saved", disk().mcp.max_rounds === 3)
      t.ok("the other keys of the run config are still there, and the original is kept", disk().model === "m" && fs.existsSync(admin.config + ".bak-mcp"))
      await pg.context().close()

      const ph = await open(browser, errors, { width: 390, height: 800 })
      await ph.goto(admin.base + "/#/settings/mcp-servers")
      await ph.getByRole("button", { name: "Add a server", exact: true }).click()
      await ph.waitForTimeout(900)
      t.ok("on a phone the open form fits the screen", (await ph.evaluate(() => document.documentElement.scrollWidth - innerWidth)) <= 1)
      await ph.context().close()

      const th = await open(browser, errors, { lang: "th" })
      await th.goto(admin.base + "/#/settings/mcp-servers")
      await th.getByRole("button", { name: "เพิ่มเซิร์ฟเวอร์", exact: true }).waitFor({ timeout: 15000 })
      await th.waitForFunction(() => document.querySelector("li[data-server='fake']")?.innerText.includes("เชื่อมต่อแล้ว"), null, { timeout: 40000 })
      t.ok("in Thai the button and the state are Thai", true)
      await th.context().close()
    },
  },
  {
    // where the servers cannot be changed (not this PC and no key, or no run config file) the section is a list that says why
    name: "mcpro: where MCP servers cannot be changed the section is read-only and says why",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/settings/mcp-servers")
      await pg.waitForTimeout(1500)
      t.ok("with no run config file it says there is nowhere to save, and offers no Add", (await pg.locator("[role=note]").innerText()).includes("without a run config file") && (await pg.getByRole("button", { name: "Add a server", exact: true }).count()) === 0)
      await pg.context().close()
      const other = await open(browser, errors)
      const server = { name: "files", kind: "address", disabled: false, source: "config", editable: true, status: "ready", error: null, tools: [{ tool: "read" }], info: {} }
      await other.route("**/mcp/config", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ servers: [server], editable: false, reason: "x", settings: { timeout_s: 60, max_result_chars: 20000, max_rounds: 8 }, config_file: "run.json", tools: 1 }) }))
      await other.goto(fast.base + "/#/settings/mcp-servers")
      await other.waitForSelector("li[data-server='files']")
      t.ok("from another address it lists the server and says only this PC or an API key may change it", (await other.locator("[role=note]").innerText()).includes("only from this PC") && (await other.locator("li[data-server='files']").innerText()).includes("Address"))
      t.ok("and has no Edit, Turn off, Delete or Add", (await other.locator("li[data-server='files'] button").count()) === 0 && (await other.getByRole("button", { name: "Add a server", exact: true }).count()) === 0)
      await other.locator(".branched-menu__item", { hasText: "Limits" }).click()
      await other.waitForSelector("input[aria-label='Tool rounds in one answer']")
      t.ok("the limits are shown, cannot be changed, and say why", (await other.getByRole("button", { name: "Save the limits" }).count()) === 0 && (await other.locator("input[aria-label='Tool rounds in one answer']").isDisabled()) && (await other.locator("[role=note]").innerText()).includes("only from this PC"))
      await other.context().close()
    },
  },
  {
    // whatever stands for a status sits in the middle of its slot, in every way it can be shown (the loaders once sat on the text baseline, 9 px low)
    name: "slot: a status mark is centered in its slot in every way, in the header and in the chat",
    async run({ browser, fast, t, errors }) {
      for (const kind of ["orbs", "mixed", "loading", "bots"]) {
        const ctx = await browser.newContext({ viewport: { width: 1100, height: 700 } })
        await ctx.addInitScript((k) => { try { localStorage.setItem("strata.lang", JSON.stringify("en")); localStorage.setItem("strata.avatar", JSON.stringify(k)) } catch { /* private window */ } }, kind)
        const pg = await ctx.newPage()
        pg.on("pageerror", (e) => errors.push(`pageerror: ${e}`))
        await pg.goto(fast.base + "/#/chat")
        await pg.waitForSelector("main .orb-slot")
        await pg.waitForTimeout(kind === "bots" ? 3000 : 1500)
        const off = await pg.evaluate(() => [...document.querySelectorAll(".orb-slot")].map((s) => {
          const b = s.getBoundingClientRect()
          const inner = [...s.querySelectorAll(".lat, .t-matrix, canvas, svg")].filter((e) => e.getBoundingClientRect().width > 1)[0]
          if (!inner) return null
          const c = inner.getBoundingClientRect()
          const m = getComputedStyle(inner)                    // the avatars draw a canvas bigger than their box and say so with a negative margin: the box is the margin box
          const [t, r, bo, l] = ["Top", "Right", "Bottom", "Left"].map((x) => parseFloat(m["margin" + x]) || 0)
          const x0 = c.x - l, y0 = c.y - t, w = c.width + l + r, h = c.height + t + bo
          return { dx: Math.round(x0 + w / 2 - (b.x + b.width / 2)), dy: Math.round(y0 + h / 2 - (b.y + b.height / 2)), size: Math.round(b.width) }
        }).filter(Boolean))
        t.ok(`${kind}: every mark is centered in its slot (${off.length} marks)`, off.length >= 2 && off.every((o) => Math.abs(o.dx) <= 1 && Math.abs(o.dy) <= 1), JSON.stringify(off))
        await ctx.close()
      }
    },
  },
  {
    // Settings is a page of its own, with a branched menu of sections and topics; the route says which topic shows; About keeps what it tells
    name: "settings: the settings are a page of their own, in a branched menu, apart from About",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      const menu = pg.locator("nav[aria-label='Settings sections']")
      const heads = () => menu.locator(".branched-menu__head").allInnerTexts()
      const topic = (name) => menu.locator(".branched-menu__item", { hasText: new RegExp("^" + name + "$") })          // exact: "Servers" is not "MCP servers"
      const h2 = () => pg.locator("main h2").allInnerTexts()
      await pg.goto(fast.base + "/#/settings")
      await pg.waitForSelector("nav[aria-label='Settings sections']")
      await pg.waitForTimeout(1200)
      t.ok("the menu has three sections, General, MCP tools and Import", (await heads()).join(",") === "General,MCP tools,Import", (await heads()).join(","))
      const expanded = async () => (await Promise.all([0, 1, 2].map((i) => menu.locator(".branched-menu__head").nth(i).getAttribute("aria-expanded")))).join(",")
      t.ok("it opens on Status marks with every section open, so no topic has to be looked for, and the topic is marked", (await expanded()) === "true,true,true" && (await topic("Status marks").getAttribute("aria-current")) === "true" && (await topic("Servers").isVisible()) && (await topic("Limits").isVisible()) && (await topic("Skills").isVisible()), await expanded())
      t.ok("only that topic is shown, with its content", (await h2()).join(",") === "Status marks" && (await pg.locator("main [role=radiogroup][aria-label='Status marks']").count()) === 1, (await h2()).join(","))
      t.ok("its title is Settings and the top navigation marks it", (await pg.locator("main h1").innerText()) === "Settings" && (await pg.locator("nav a[aria-current='page']").getAttribute("aria-label")) === "Settings")
      await menu.locator(".branched-menu__head", { hasText: "MCP tools" }).click()
      await pg.waitForTimeout(600)
      t.ok("a heading still folds its section, and the others stay open", (await expanded()) === "true,false,true" && (await topic("Servers").evaluate((e) => e.closest(".branched-menu__fold").inert)) === true && (await topic("Skills").evaluate((e) => e.closest(".branched-menu__fold").inert)) === false, await expanded())
      await menu.locator(".branched-menu__head", { hasText: "MCP tools" }).click()
      await pg.waitForTimeout(600)
      t.ok("and opens it again", (await expanded()) === "true,true,true" && (await topic("Servers").isVisible()) && (await topic("Limits").isVisible()))
      await topic("Servers").click()
      await pg.waitForTimeout(600)
      t.ok("choosing Servers puts it in the address and shows it", pg.url().endsWith("#/settings/mcp-servers") && (await h2()).join(",") === "MCP servers" && (await topic("Servers").getAttribute("aria-current")) === "true" && (await topic("Status marks").getAttribute("aria-current")) === null, pg.url())
      await topic("Limits").click()
      await pg.waitForTimeout(600)
      t.ok("and Limits shows the limits and not the servers", pg.url().endsWith("#/settings/mcp-limits") && (await h2()).join(",") === "Limits" && (await pg.locator("input[aria-label='Tool rounds in one answer']").count()) === 1 && (await pg.getByRole("button", { name: "Add a server", exact: true }).count()) === 0, (await h2()).join(","))
      const line = await pg.evaluate(() => { const p = document.querySelector(".branched-menu__base"); const s = p ? getComputedStyle(p) : null; return s ? { stroke: s.stroke, w: s.strokeWidth } : null })
      t.ok("the branch lines are drawn in a colour", line && line.stroke !== "none" && !/rgba?\(0, 0, 0(, 0)?\)$/.test(line.stroke), JSON.stringify(line))

      await pg.goto(fast.base + "/#/settings/api-key")
      await pg.reload()
      await pg.waitForSelector("nav[aria-label='Settings sections']")
      await pg.waitForTimeout(1000)
      t.ok("a link to a topic opens its section and shows it", (await menu.locator(".branched-menu__head").nth(0).getAttribute("aria-expanded")) === "true" && (await topic("API key").getAttribute("aria-current")) === "true" && (await h2()).join(",") === "API key" && (await pg.locator("input[type=password]").count()) === 1)
      await pg.goto(fast.base + "/#/settings/mcp-limits")
      await pg.reload()
      await pg.waitForSelector("nav[aria-label='Settings sections']")
      await pg.waitForTimeout(1000)
      t.ok("a link into the other section opens that one", (await menu.locator(".branched-menu__head").nth(1).getAttribute("aria-expanded")) === "true" && (await h2()).join(",") === "Limits")
      await pg.goto(fast.base + "/#/settings/nope")
      await pg.waitForTimeout(800)
      t.ok("a topic that does not exist shows Status marks", (await h2()).join(",") === "Status marks")

      await pg.goto(fast.base + "/#/about")
      await pg.waitForTimeout(1200)
      const a = await h2()
      t.ok("About keeps the model and the server, and none of the settings", a.includes("The model") && a.includes("This server") && !["Status marks", "API key", "MCP servers", "Limits"].some((x) => a.includes(x)), a.join(", "))
      await pg.context().close()

      const th = await open(browser, errors, { lang: "th" })
      await th.goto(fast.base + "/#/settings")
      await th.waitForSelector("nav[aria-label='หัวข้อในหน้าตั้งค่า']")
      await th.waitForTimeout(1000)
      t.ok("in Thai the page and its sections are Thai", (await th.locator("main h1").innerText()) === "ตั้งค่า" && (await th.locator(".branched-menu__head").allInnerTexts()).join(",") === "ทั่วไป,เครื่องมือ MCP,นำเข้า")
      await th.context().close()

      const ph = await open(browser, errors, { width: 390, height: 800 })
      await ph.goto(fast.base + "/#/settings/mcp-servers")
      await ph.waitForSelector("nav[aria-label='Settings sections']")
      await ph.waitForTimeout(1000)
      const box = await ph.evaluate(() => { const m = document.querySelector("nav[aria-label='Settings sections']").getBoundingClientRect(); const c = document.querySelector("main h2").getBoundingClientRect(); return { over: document.documentElement.scrollWidth - innerWidth, menuBottom: m.bottom, contentTop: c.top } })
      t.ok("on a phone the menu is above the content and nothing is wider than the screen", box.over <= 1 && box.menuBottom <= box.contentTop + 1, JSON.stringify(box))
      await ph.context().close()
    },
  },
  {
    // when a status ends it gives way smoothly: "Thinking…" to "Thought for 2.4s" (the shimmer goes on while it fades, no pop), and
    // "Answering…" to the figures under the answer (the status leaves as they arrive); measured frame by frame
    name: "handover: the live status gives way to what it becomes without a jump",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(1200)
      await pg.evaluate(() => {
        window.__f = []
        const num = (e) => (e ? +getComputedStyle(e).opacity : null)
        const tick = () => {
          const w = document.querySelector(".thought-text:not(.thought-text--done)"), d = document.querySelector(".thought-text--done")
          const sh = document.querySelector(".thought-shimmer"), o = document.querySelector(".handover-out"), i = document.querySelector(".handover-in")
          window.__f.push({ w: num(w), d: num(d), fill: sh ? getComputedStyle(sh).webkitTextFillColor : null, wy: w ? getComputedStyle(w).transform : null, o: num(o), i: num(i), it: i ? i.textContent : null })
          window.__raf = requestAnimationFrame(tick)
        }
        tick()
      })
      await pg.fill("textarea", "hi")
      await pg.keyboard.press("Enter")
      await finished(pg)
      await pg.waitForTimeout(900)
      const f = await pg.evaluate(() => { cancelAnimationFrame(window.__raf); return window.__f })
      const mid = (k) => f.filter((x) => x[k] != null && x[k] > 0.05 && x[k] < 0.95)
      const transparent = (s) => s === "rgba(0, 0, 0, 0)"
      t.ok("Thinking… fades over several frames while Thought for… arrives", mid("w").length >= 4 && f.filter((x) => x.d != null && x.d > 0.05 && x.d < 0.7).length >= 4, `work ${mid("w").length}, done ${f.filter((x) => x.d != null && x.d > 0.05 && x.d < 0.7).length} frames`)
      t.ok("the shimmer goes on while Thinking… fades: it does not turn into a solid word first", mid("w").length > 0 && mid("w").every((x) => transparent(x.fill)), JSON.stringify(mid("w").slice(0, 2).map((x) => x.fill)))
      t.ok("Thinking… rises a little as it leaves", mid("w").some((x) => x.wy && x.wy !== "none" && x.wy !== "matrix(1, 0, 0, 1, 0, 0)"), JSON.stringify(mid("w")[0]?.wy))
      t.ok("Answering… leaves over several frames while the figures arrive", mid("o").length >= 4 && mid("i").length >= 4, `status ${mid("o").length}, figures ${mid("i").length} frames`)
      t.ok("for a moment both are on screen (a handover, not a gap)", f.some((x) => x.o != null && x.o > 0.05 && x.i != null && x.i > 0.05))
      const last = f[f.length - 1]
      t.ok("in the end the figures are there, with the speed, and the status is gone", last.o == null && last.i === 1 && /tok\/s/.test(last.it || ""), JSON.stringify([last.o, last.i, last.it]))
      await pg.context().close()
    },
  },
  {
    // Recents and Projects in a sidebar of the Chat page (#92): many conversations, kept in the browser, grouped in folders
    name: "history: the Chat page lists conversations in Recents and groups them in projects",
    async run({ browser, fast, long, t, errors }) {
      const pg = await open(browser, errors)
      const side = pg.locator("aside[aria-label='Conversations']")
      const row = (title) => side.locator("[data-topic]", { hasText: title })
      const menu = pg.locator("[role=menu]")
      const user = () => pg.locator(".msg-in.group").allInnerTexts().then((a) => a.map((x) => x.split(String.fromCharCode(10))[0]))
      const sendIn = async (text) => { await pg.fill("textarea", text); await pg.keyboard.press("Enter"); await finished(pg); await pg.waitForTimeout(500) }
      await pg.goto(fast.base + "/#/chat")
      await pg.evaluate(() => localStorage.clear())
      await pg.reload()
      await pg.waitForSelector("aside[aria-label='Conversations']")
      t.ok("the page has Projects and Recents, empty to begin with", (await side.locator("h2").allInnerTexts()).join(",") === "Projects" && (await side.locator("section[data-recents] .branched-menu__head").innerText()) === "Recents" && (await side.innerText()).includes("No conversations yet"))

      await sendIn("hello one")
      const idOne = pg.url().split("/chat/")[1]
      t.ok("the first prompt adds a conversation, titled from it, on the line of Recents, and the address names it", (await row("hello one").count()) === 1 && (await side.locator("section[data-recents] [data-topic]", { hasText: "hello one" }).count()) === 1 && !!idOne && (await row("hello one").locator("button").first().getAttribute("aria-current")) === "true", pg.url())

      await side.getByRole("button", { name: "New chat", exact: true }).click()
      await pg.waitForTimeout(500)
      t.ok("New chat starts an empty conversation and the old one stays in the list", (await user()).length === 0 && (await row("hello one").count()) === 1 && pg.url().endsWith("#/chat"), pg.url())
      await sendIn("second one")
      const titles = await side.locator("[data-topic]").allInnerTexts()
      t.ok("the new one is first, newest on top", titles.length === 2 && titles[0].includes("second one"), JSON.stringify(titles))
      t.ok("the conversations are one list on one line from Recents, not split by day", (await side.locator("section[data-recents] .branched-menu__section").count()) === 1 && !/Today|Yesterday|Earlier/.test(await side.innerText()))

      await row("hello one").locator("button").first().click()
      await pg.waitForTimeout(600)
      t.ok("opening the old one brings its messages back, and the address", (await user()).some((x) => x.includes("hello one")) && !(await user()).some((x) => x.includes("second one")) && pg.url().endsWith("/chat/" + idOne), pg.url())
      const drawn = await pg.evaluate(() => [...document.querySelectorAll("aside .branched-menu__reach")].filter((e) => getComputedStyle(e).strokeDashoffset === "0px").length)
      t.ok("the branch line is drawn to the conversation that is open, and to no other", drawn === 1 && (await pg.locator("aside .branched-menu__marker[data-on]").count()) === 1, `${drawn} lines`)
      await pg.reload()
      await pg.waitForSelector("aside[aria-label='Conversations']")
      await pg.waitForTimeout(800)
      t.ok("after a reload the same conversation is open and the list is there", (await user()).some((x) => x.includes("hello one")) && (await side.locator("[data-topic]").count()) === 2)

      await row("hello one").getByRole("button", { name: /Options for/ }).click()
      await menu.getByRole("menuitem", { name: "Rename" }).click()
      const rename = side.locator("input[aria-label='Rename']")
      await rename.fill("Renamed one")
      await rename.press("Enter")
      await pg.waitForTimeout(300)
      t.ok("a conversation can be renamed", (await row("Renamed one").count()) === 1)

      const fs = await import("node:fs")
      const os = await import("node:os")
      const path = await import("node:path")
      const work = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "strata-proj-")))
      await side.getByRole("button", { name: "New project", exact: true }).click()
      await pg.locator("[data-new-project] input[aria-label='Project name']").fill("Work")
      await pg.locator("[data-new-project] input[aria-label='Add a folder']").fill(work)
      await pg.locator("[data-new-project]").getByRole("button", { name: "Create project", exact: true }).click()
      await pg.waitForTimeout(500)
      const proj = side.locator("section[data-projects] [data-section]", { hasText: "Work" })
      t.ok("a project is created and shows as a folder", (await proj.count()) === 1)
      await row("Renamed one").getByRole("button", { name: /Options for/ }).click()
      await menu.getByRole("menuitem", { name: "Move to project" }).click()
      await menu.getByRole("menuitem", { name: "Work" }).click()
      await pg.waitForTimeout(400)
      t.ok("moving a conversation puts it in the project, and not in Recents", (await proj.locator("[data-topic]", { hasText: "Renamed one" }).count()) === 1 && (await side.locator("section[data-recents] [data-topic]", { hasText: "Renamed one" }).count()) === 0)
      await proj.getByRole("button", { name: /Options for project/ }).click()
      await menu.getByRole("menuitem", { name: "Delete project" }).click()
      await menu.getByRole("menuitem", { name: "Sure? Delete project" }).click()
      await pg.waitForTimeout(400)
      t.ok("deleting a project keeps its conversations, back in Recents", (await side.locator("section[data-projects] [data-section]").count()) === 0 && (await side.locator("section[data-recents] [data-topic]", { hasText: "Renamed one" }).count()) === 1)

      await pg.goto(fast.base + "/#/chat/nope")
      await pg.waitForTimeout(800)
      t.ok("an address with an id that does not exist shows a new chat", (await user()).length === 0 && pg.url().endsWith("#/chat"), pg.url())
      await row("second one").getByRole("button", { name: /Options for/ }).click()
      await menu.getByRole("menuitem", { name: "Delete", exact: true }).click()
      t.ok("Delete asks once more", (await menu.getByRole("menuitem", { name: "Sure? Delete" }).count()) === 1)
      await menu.getByRole("menuitem", { name: "Sure? Delete" }).click()
      await pg.waitForTimeout(400)
      t.ok("and then the conversation is gone from the list", (await row("second one").count()) === 0 && (await side.locator("[data-topic]").count()) === 1)

      const widths = () => pg.evaluate(() => new Promise((res) => { const w = []; const a = document.querySelector("aside[aria-label='Conversations']"); const t0 = performance.now(); const tick = () => { w.push(Math.round(a.getBoundingClientRect().width)); if (performance.now() - t0 < 700) requestAnimationFrame(tick); else res(w) }; tick() }))
      const [closing] = await Promise.all([widths(), (async () => { await pg.waitForTimeout(60); await side.getByRole("button", { name: "Hide the list" }).click() })()])
      const between = (w) => w.filter((x) => x > 45 && x < 240).length
      t.ok("hiding the list shrinks it smoothly through in-between widths", between(closing) >= 4 && closing[0] >= 240 && closing[closing.length - 1] <= 40, JSON.stringify(closing.filter((_, i) => i % 3 === 0)))
      await pg.waitForTimeout(100)
      t.ok("the list can be hidden, and stays hidden after a reload", (await side.locator("[data-topic]").count()) === 0 && (await pg.evaluate(() => localStorage.getItem("strata.sidebar"))) === '"closed"')
      await pg.reload()
      await pg.waitForSelector("aside[aria-label='Conversations']")
      await pg.waitForTimeout(500)
      t.ok("still hidden", (await side.locator("[data-topic]").count()) === 0)
      const [opening] = await Promise.all([widths(), (async () => { await pg.waitForTimeout(60); await side.getByRole("button", { name: "Show the list" }).click() })()])
      t.ok("and showing it stretches it back out", between(opening) >= 4 && opening[0] <= 40 && opening[opening.length - 1] >= 240, JSON.stringify(opening.filter((_, i) => i % 3 === 0)))
      await pg.waitForTimeout(100)
      t.ok("and shown again", (await side.locator("[data-topic]").count()) === 1)
      await pg.context().close()

      // a conversation the app had before the list existed is the first item
      const old = await open(browser, errors)
      await old.addInitScript(() => { if (!localStorage.getItem("strata.chat")) localStorage.setItem("strata.chat", JSON.stringify([{ role: "user", text: "from before", time: 1 }, { role: "assistant", text: "yes", time: 2 }])) })
      await old.goto(fast.base + "/#/chat")
      await old.waitForSelector("aside[aria-label='Conversations']")
      await old.waitForTimeout(800)
      t.ok("an existing conversation becomes the first item, and is open", (await old.locator("aside[aria-label='Conversations'] [data-topic]", { hasText: "from before" }).count()) === 1 && (await old.locator(".msg-in.group").count()) === 1)
      await old.context().close()

      // while an answer is written the others cannot be opened
      const busy = await open(browser, errors)
      await busy.addInitScript(() => { if (!localStorage.getItem("strata.chats")) { localStorage.setItem("strata.chats", JSON.stringify({ active: null, items: [{ id: "x1", title: "an older one", time: 1 }], projects: [] })); localStorage.setItem("strata.chat.x1", JSON.stringify([{ role: "user", text: "an older one", time: 1 }])) } })
      await busy.goto(long.base + "/#/chat")
      await busy.waitForSelector("textarea")
      await busy.fill("textarea", "write a lot")
      await busy.keyboard.press("Enter")
      await busy.waitForSelector("button[aria-label='Stop']")
      const side2 = busy.locator("aside[aria-label='Conversations']")
      const older = side2.locator("[data-topic]", { hasText: "an older one" }).locator("button").first()
      t.ok("while an answer is being written the other conversations can be opened", (await older.getAttribute("aria-disabled")) !== "true")
      t.ok("and the conversation that is answering is marked in the list", (await side2.locator("[data-running='answering']").count()) === 1)
      await older.click()
      await busy.waitForTimeout(700)
      t.ok("opening another shows it, with a composer that is free to send, while the first goes on", (await busy.locator(".msg-in.group").allInnerTexts()).join().includes("an older one") && (await busy.locator("button[aria-label='Stop']").count()) === 0 && (await busy.locator("button[aria-label='Send']").count()) === 1)
      t.ok("the marked one is still answering", (await side2.locator("[data-running='answering']").count()) === 1)
      await side2.getByRole("button", { name: "New chat", exact: true }).click()
      await busy.waitForTimeout(500)
      t.ok("a new chat can be started too, empty and free", (await busy.locator(".msg-in").count()) === 0 && (await busy.locator("button[aria-label='Stop']").count()) === 0)
      await side2.locator("[data-topic]", { hasText: "write a lot" }).locator("button").first().click()
      await busy.waitForTimeout(700)
      t.ok("coming back to the one that is answering shows it still answering, with Stop", (await busy.locator("button[aria-label='Stop']").count()) === 1)
      await busy.click("button[aria-label='Stop']")
      await busy.waitForTimeout(500)
      t.ok("when it is stopped nothing is marked", (await side2.locator("[data-running]").count()) === 0)
      await busy.context().close()

      const ph = await open(browser, errors, { width: 390, height: 800 })
      await ph.addInitScript(() => { if (!localStorage.getItem("strata.chats")) { localStorage.setItem("strata.chats", JSON.stringify({ active: null, items: [{ id: "x1", title: "an older one", time: 1 }], projects: [] })); localStorage.setItem("strata.chat.x1", JSON.stringify([{ role: "user", text: "an older one", time: 1 }])) } })
      await ph.goto(fast.base + "/#/chat")
      await ph.waitForSelector("textarea")
      await ph.waitForTimeout(800)
      const over = () => ph.evaluate(() => document.documentElement.scrollWidth - innerWidth)
      t.ok("on a phone the list is closed, and nothing is wider than the screen", (await ph.locator("aside[aria-label='Conversations']:visible").count()) === 0 && (await over()) <= 1)
      await ph.getByRole("button", { name: "Recents", exact: true }).click()
      await ph.waitForTimeout(500)
      t.ok("a button opens it as a drawer that fits", (await ph.locator("aside[aria-label='Conversations']:visible [data-topic]").count()) === 1 && (await over()) <= 1)
      await ph.locator("aside[aria-label='Conversations']:visible [data-topic] button").first().click()
      await ph.waitForTimeout(600)
      t.ok("choosing one opens it and closes the drawer", (await ph.locator(".msg-in.group").count()) === 1 && (await ph.locator("aside[aria-label='Conversations']:visible").count()) === 0)
      await ph.context().close()

      const th = await open(browser, errors, { lang: "th" })
      await th.goto(fast.base + "/#/chat")
      await th.waitForSelector("aside[aria-label='การสนทนา']")
      t.ok("in Thai the sections are Thai", (await th.locator("aside[aria-label='การสนทนา'] h2").allInnerTexts()).join(",") === "โปรเจกต์" && (await th.locator("aside[aria-label='การสนทนา'] section[data-recents] .branched-menu__head").innerText()) === "ล่าสุด")
      await th.context().close()
    },
  },
  {
    // A project is one or more folders (issue #96): it cannot be made without one, the folders are typed or chosen from the folders of this PC (the server lists them and
    // checks what is typed), the first is the main one, and the coding tools of the project's chats work in all of them.
    name: "projectfolder: a project needs a folder, can have several (worktrees), typed or chosen, and the coding tools of its chats work in them",
    async run({ browser, fast, t, errors }) {
      const fs = await import("node:fs")
      const os = await import("node:os")
      const path = await import("node:path")
      const NL = String.fromCharCode(10)
      const work = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "strata-proj-")))
      const alpha = path.join(work, "alpha"), beta = path.join(work, "beta")
      fs.mkdirSync(alpha)
      fs.mkdirSync(beta)
      fs.mkdirSync(path.join(work, ".hidden"))
      fs.writeFileSync(path.join(work, "a-file.txt"), "x")
      const info = { available: true, allowed: true, shell: "bash", tools: ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "TodoWrite"] }
      const tail = `data: ${JSON.stringify({ choices: [{ delta: { content: "ok" } }] })}${NL}${NL}` + `data: ${JSON.stringify({ choices: [], usage: { completion_tokens: 1 } })}${NL}${NL}` + `data: [DONE]${NL}${NL}`
      const pg = await open(browser, errors)
      const sent = []
      await pg.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(info) }))
      await pg.route("**/v1/chat/completions", (r) => { sent.push(JSON.parse(r.request().postData() || "{}")); return r.fulfill({ status: 200, contentType: "text/event-stream", body: tail }) })
      await pg.goto(fast.base + "/#/chat")
      await pg.evaluate(() => localStorage.clear())
      await pg.reload()
      await pg.waitForSelector("aside[aria-label='Conversations']")
      await pg.waitForTimeout(600)
      const side = pg.locator("aside[aria-label='Conversations']")
      const dlg = pg.locator("[data-new-project]")
      const nameIn = dlg.locator("input[aria-label='Project name']")
      const folderIn = dlg.locator("input[aria-label='Add a folder']")
      const create = dlg.getByRole("button", { name: "Create project", exact: true })
      const addBtn = dlg.getByRole("button", { name: "Add", exact: true })
      const rows = dlg.locator("[data-folders] [data-folder]")
      const list = dlg.locator("[data-folder-list]")
      const projects = () => pg.evaluate(() => (JSON.parse(localStorage.getItem("strata.chats") || "{}").projects ?? []))
      const heights = (ms = 900) => pg.evaluate(async (ms) => {                 // the dialog's height at every frame
        const el = document.querySelector("[data-new-project]")
        const seen = []
        const t0 = performance.now()
        while (performance.now() - t0 < ms) { seen.push(el.offsetHeight); await new Promise((r) => requestAnimationFrame(r)) }
        return seen
      }, ms)
      const glides = (hs, dir) => new Set(hs).size > 4 && (dir > 0 ? hs.at(-1) > hs[0] : hs.at(-1) < hs[0]) && hs.slice(1, -1).some((h) => (dir > 0 ? h > hs[0] && h < hs.at(-1) : h < hs[0] && h > hs.at(-1))) && hs.every((h, i) => i === 0 || (dir > 0 ? h >= hs[i - 1] - 1 : h <= hs[i - 1] + 1))

      await side.getByRole("button", { name: "New project", exact: true }).click()
      await pg.waitForTimeout(400)
      t.ok("the + of Projects opens a dialog that asks for a name and folders, and says what they are for", (await dlg.count()) === 1 && (await nameIn.count()) === 1 && (await folderIn.count()) === 1 && (await dlg.innerText()).includes("coding tools") && (await dlg.innerText()).includes("worktrees"))
      t.ok("the name has the focus, and Create waits", (await nameIn.evaluate((e) => e === document.activeElement)) && (await create.isDisabled()))
      await nameIn.fill("Work")
      t.ok("a name alone is not enough: a project needs a folder", await create.isDisabled())
      await nameIn.press("Enter")
      t.ok("Enter in the name goes on to the folder", await folderIn.evaluate((e) => e === document.activeElement))
      t.ok("Add waits for something to add", await addBtn.isDisabled())
      await folderIn.fill(path.join(work, "nope"))
      await create.click()
      await pg.waitForTimeout(500)
      t.ok("a folder that is not there is refused with a reason, and nothing is made", (await dlg.locator("[role=alert]").innerText()).includes("not a folder") && (await dlg.count()) === 1 && (await projects()).length === 0)
      await folderIn.fill(path.join(work, "a-file.txt"))
      await addBtn.click()
      await pg.waitForTimeout(500)
      t.ok("a file is not a folder either, and is not added", (await dlg.locator("[role=alert]").count()) === 1 && (await rows.count()) === 0)

      await folderIn.fill(work)
      t.ok("typing again takes the reason away", (await dlg.locator("[role=alert]").count()) === 0)
      await dlg.locator("h2").click()                                             // the field is left: what is offered while typing closes up first, so the list of folders opens on its own
      await pg.waitForTimeout(800)
      const opening = heights()
      await dlg.getByRole("button", { name: "Browse", exact: true }).click()
      const hs1 = await opening
      t.ok("the list of folders stretches the dialog open through in-between heights, not in one jump", glides(hs1, 1), JSON.stringify([...new Set(hs1)]))
      t.ok("Browse lists the folders in it, by name, and no files and no hidden ones", (await list.locator("[data-dir]").allInnerTexts()).join(",") === "alpha,beta", (await list.innerText()).split(NL).join(" | "))
      await list.locator("[data-dir]", { hasText: "alpha" }).click()
      await pg.waitForTimeout(500)
      t.ok("choosing one goes into it, and the field follows", (await folderIn.inputValue()) === alpha && (await list.locator("[data-dir]").count()) === 0 && (await list.innerText()).includes("No folders in here"), await folderIn.inputValue())
      await list.getByRole("button", { name: "Up one folder" }).click()
      await pg.waitForTimeout(500)
      t.ok("Up goes back out", (await folderIn.inputValue()) === work && (await list.locator("[data-dir]").count()) === 2)
      await list.locator("[data-dir]", { hasText: "alpha" }).click()
      await pg.waitForTimeout(400)
      await addBtn.click()
      await pg.waitForTimeout(500)
      t.ok("Add makes it a folder of the project and empties the field", (await rows.count()) === 1 && (await rows.first().getAttribute("data-folder")) === "main" && (await rows.first().innerText()).includes("alpha") && (await folderIn.inputValue()) === "")
      t.ok("one folder is not labelled main (there is nothing to tell it from)", (await rows.first().innerText()).replace(/\s+/g, "") === alpha)
      t.ok("Create is ready now: the name and a folder", await create.isEnabled())

      // a second folder: Browse starts next to the first one
      const closing = heights()
      await dlg.getByRole("button", { name: "Hide", exact: true }).click()
      const hs2 = await closing
      t.ok("hiding the list closes the dialog up through in-between heights", glides(hs2, -1), JSON.stringify([...new Set(hs2)]))
      await dlg.getByRole("button", { name: "Browse", exact: true }).click()
      await pg.waitForTimeout(700)
      t.ok("Browse starts in the folder above the last one, to pick one beside it", (await list.locator("[data-dir]").allInnerTexts()).join(",") === "alpha,beta")
      await list.locator("[data-dir]", { hasText: "beta" }).click()
      await pg.waitForTimeout(400)
      await pg.waitForTimeout(500)                                               // the list that closed up when a folder was chosen has settled
      const adding = heights()
      await folderIn.press("Enter")
      const hs3 = await adding
      t.ok("a folder that is added stretches the dialog by its row", glides(hs3, 1), JSON.stringify([...new Set(hs3)]))
      t.ok("Enter in the field adds it too: the project has two folders, the first is the main one", (await rows.count()) === 2 && (await rows.nth(0).getAttribute("data-folder")) === "main" && (await rows.nth(1).getAttribute("data-folder")) === "other" && (await rows.nth(0).innerText()).includes("Main"))
      await folderIn.fill(alpha)
      await addBtn.click()
      await pg.waitForTimeout(500)
      t.ok("the same folder twice is one folder", (await rows.count()) === 2)
      await rows.nth(1).getByRole("button", { name: /Make .* the main folder/ }).click()
      await pg.waitForTimeout(300)
      t.ok("another folder can be made the main one", (await rows.nth(0).innerText()).includes("beta") && (await rows.nth(0).innerText()).includes("Main") && (await rows.nth(1).innerText()).includes("alpha"))
      const box = await dlg.boundingBox()
      t.ok("the dialog fits the screen with two folders and the list open", box.y >= 0 && box.y + box.height <= 760 && box.x >= 0, JSON.stringify(box))
      await create.click()
      await pg.waitForTimeout(600)
      const made = await projects()
      t.ok("Create makes the project with the folders in that order, and the dialog closes", (await dlg.count()) === 0 && made.length === 1 && made[0].name === "Work" && JSON.stringify(made[0].folders) === JSON.stringify([beta, alpha]), JSON.stringify(made))
      t.ok("the project is in the list", (await side.locator("section[data-projects] [data-section]", { hasText: "Work" }).count()) === 1)

      // what is typed and not added counts when Create is pressed
      await side.getByRole("button", { name: "New project", exact: true }).click()
      await pg.waitForTimeout(400)
      await nameIn.fill("Typed")
      await folderIn.fill(work)
      t.ok("a folder that is typed makes Create ready without Add", await create.isEnabled())

      // a rough path and one click: while a path is typed the folders that go with it are offered
      const sug = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "strata-sug-")))
      for (const d of ["app", "app-wt-fix", "my-app", "lib", ".hid"]) fs.mkdirSync(path.join(sug, d))
      const offered = dlg.locator("[data-suggestion]")
      const typeSlowly = async (text) => { await folderIn.fill(""); await folderIn.pressSequentially(text, { delay: 15 }); await pg.waitForTimeout(450) }

      // two ways to choose, one list at a time: typing while the list of folders is open replaces it with what goes with what is typed
      await dlg.getByRole("button", { name: "Browse", exact: true }).click()
      await pg.waitForTimeout(600)
      t.ok("the list of folders is open", (await list.count()) === 1 && (await offered.count()) === 0)
      await typeSlowly(sug + path.sep + "ap")
      t.ok("typing a path then leaves one list, the one that goes with what is typed, and not both", (await list.count()) === 0 && (await offered.count()) === 3 && (await dlg.locator("[data-folder-list], [data-suggestions]").count()) === 1, `${await list.count()} browse, ${await offered.count()} offered`)
      t.ok("and Browse offers the list again", (await dlg.getByRole("button", { name: "Browse", exact: true }).count()) === 1)
      await dlg.getByRole("button", { name: "Browse", exact: true }).click()
      await pg.waitForTimeout(600)
      t.ok("choosing in the list of folders gives the list and not what was offered", (await list.count()) === 1 && (await offered.count()) === 0)
      await dlg.getByRole("button", { name: "Hide", exact: true }).click()
      await pg.waitForTimeout(500)
      await folderIn.fill("")
      await pg.waitForTimeout(400)
      const offering = heights()
      await folderIn.fill(sug + path.sep + "ap")
      const hs4 = await offering
      t.ok("what is offered while a path is typed stretches the dialog open", glides(hs4, 1), JSON.stringify([...new Set(hs4)]))
      await typeSlowly(sug + path.sep + "ap")
      t.ok("typing a rough path offers the folders that go with it, those that start with it first, then those that contain it", (await offered.allInnerTexts()).join(",") === "app,app-wt-fix,my-app", (await offered.allInnerTexts()).join(","))
      t.ok("the part that matches is in bold", (await offered.first().locator("b").innerText()) === "ap")
      t.ok("the field is a combobox that says it has a list", (await folderIn.getAttribute("role")) === "combobox" && (await folderIn.getAttribute("aria-expanded")) === "true" && (await dlg.locator("[role=listbox]").count()) === 1)
      await offered.filter({ hasText: "app-wt-fix" }).click()
      await pg.waitForTimeout(500)
      t.ok("a click finishes the path with a separator and the field keeps the focus", (await folderIn.inputValue()) === sug + path.sep + "app-wt-fix" + path.sep && (await folderIn.evaluate((e) => e === document.activeElement)), await folderIn.inputValue())
      t.ok("and what is inside it is offered next (here nothing)", (await offered.count()) === 0)
      await addBtn.click()
      await pg.waitForTimeout(500)
      t.ok("Add takes the path with the separator as the folder", (await rows.count()) === 1 && (await rows.first().innerText()).replace(/\s+/g, "") === path.join(sug, "app-wt-fix"))

      await typeSlowly(sug + path.sep)
      t.ok("a path that ends with a separator offers all its folders, hidden ones left out", (await offered.allInnerTexts()).join(",") === "app,app-wt-fix,lib,my-app", (await offered.allInnerTexts()).join(","))
      await folderIn.press("ArrowDown")
      await folderIn.press("ArrowDown")
      t.ok("the arrow keys move through them", (await offered.nth(1).getAttribute("aria-selected")) === "true" && (await offered.nth(0).getAttribute("aria-selected")) === "false")
      await folderIn.press("Enter")
      await pg.waitForTimeout(400)
      t.ok("Enter takes the one chosen; it does not add it yet", (await folderIn.inputValue()) === sug + path.sep + "app-wt-fix" + path.sep && (await rows.count()) === 1)
      await typeSlowly(sug.split(path.sep).join("/") + "/li")
      await offered.first().click()
      await pg.waitForTimeout(400)
      t.ok("a path typed with / stays with /", (await folderIn.inputValue()) === sug.split(path.sep).join("/") + "/lib/", await folderIn.inputValue())
      await typeSlowly(sug + path.sep + "ap")
      const withdrawing = heights()
      await folderIn.fill(sug + path.sep + "zzz")
      const hs5 = await withdrawing
      t.ok("and closes up when nothing goes with what is typed", glides(hs5, -1), JSON.stringify([...new Set(hs5)]))
      t.ok("nothing goes with it: nothing is offered", (await offered.count()) === 0)
      await typeSlowly(sug + path.sep + "ap")
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(300)
      t.ok("Escape closes the list first, not the dialog", (await offered.count()) === 0 && (await dlg.count()) === 1)
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(400)
      t.ok("Escape closes it and makes nothing", (await dlg.count()) === 0 && (await projects()).length === 1)
      fs.rmSync(sug, { recursive: true, force: true })

      // the project's chats: the coding tools work in its folders
      await pg.fill("textarea[aria-label='Message']", "hello")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(900)
      const first = sent.at(-1)?.strata_agent
      t.ok("a chat that is in no project has no folder unless one was set as the default", first !== undefined && !("cwd" in first) && !("dirs" in first), JSON.stringify(first))
      await side.locator("[data-topic]", { hasText: "hello" }).getByRole("button", { name: /Options for/ }).click()
      await pg.getByRole("menuitem", { name: "Move to project" }).click()
      await pg.getByRole("menuitem", { name: "Work" }).click()
      await pg.waitForTimeout(400)
      await pg.fill("textarea[aria-label='Message']", "again")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(900)
      const second = sent.at(-1)?.strata_agent
      t.ok("once it is in the project the request names the main folder and the other one", second?.cwd === beta && JSON.stringify(second?.dirs) === JSON.stringify([alpha]), JSON.stringify(second))

      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(500)
      const prow = pg.locator(".t-morph-menu [role=option]", { hasText: "Coding tools" })
      t.ok("the + menu row names the main folder and says there is one more", (await prow.innerText()).includes(beta) && (await prow.innerText()).includes("+1"), (await prow.innerText()).split(NL).join(" | "))
      await prow.click()
      await pg.waitForTimeout(600)
      const panel = pg.locator("[role=dialog][aria-label='Coding tools']")
      const prows = panel.locator("[data-folders] [data-folder]")
      const padd = panel.locator("input[aria-label='Add a folder']")
      t.ok("the panel lists the project's folders, the main one first, and whose they are", (await prows.count()) === 2 && (await prows.nth(0).innerText()).includes("beta") && (await panel.innerText()).includes('project "Work"'))
      await prows.nth(1).getByRole("button", { name: /^Remove / }).click()
      await pg.waitForTimeout(300)
      t.ok("a folder can be taken away", (await prows.count()) === 1 && JSON.stringify((await projects())[0].folders) === JSON.stringify([beta]))
      t.ok("but not the last one: a project always has a folder", await prows.first().getByRole("button", { name: /^Remove / }).isDisabled())
      await padd.fill(alpha)
      await panel.getByRole("button", { name: "Add", exact: true }).click()
      await pg.waitForTimeout(400)
      t.ok("and one can be added, and it is kept with the project", (await prows.count()) === 2 && JSON.stringify((await projects())[0].folders) === JSON.stringify([beta, alpha]))

      // a new chat inside the project
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(400)
      const newIn = side.getByRole("button", { name: "New chat in project Work" })
      t.ok("a project has a button for a new chat in it", (await newIn.count()) === 1)
      await newIn.click()
      await pg.waitForTimeout(500)
      t.ok("the empty chat says which project it will be in", (await pg.locator("[data-in-project]").innerText()).includes("Work"))
      await pg.fill("textarea[aria-label='Message']", "inside the project")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(900)
      t.ok("its first prompt puts it in the project, in the sidebar", (await side.locator("section[data-projects] [data-section]", { hasText: "Work" }).locator("[data-topic]", { hasText: "inside the project" }).count()) === 1)
      t.ok("and its tools work in the project's folders from that first request", sent.at(-1)?.strata_agent?.cwd === beta && JSON.stringify(sent.at(-1)?.strata_agent?.dirs) === JSON.stringify([alpha]), JSON.stringify(sent.at(-1)?.strata_agent))

      // the project can be chosen on any new chat
      await side.getByRole("button", { name: "New chat", exact: true }).click()
      await pg.waitForTimeout(500)
      const picker = pg.locator("[data-project-picker] button[aria-haspopup='menu']")
      t.ok("a new chat has a project to choose, and begins with none", (await picker.count()) === 1 && (await picker.innerText()).trim() === "No project" && (await picker.getAttribute("data-in-project")) === null)
      await picker.click()
      await pg.waitForTimeout(300)
      t.ok("the menu lists the projects with their folders", (await pg.locator("[role=menu][aria-label='Choose the project this chat works in'] [role=menuitemradio]").count()) === 2 && (await pg.locator("[data-project-option='Work']").innerText()).includes("beta +1"))
      await pg.locator("[data-project-option='Work']").click()
      await pg.waitForTimeout(400)
      t.ok("choosing one shows it on the new chat", (await picker.getAttribute("data-in-project")) === "Work" && (await picker.innerText()).trim() === "Work")
      await pg.fill("textarea[aria-label='Message']", "chosen project")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(900)
      t.ok("the chat is in that project from its first prompt, and its tools work in the project's folders", (await side.locator("section[data-projects] [data-section]", { hasText: "Work" }).locator("[data-topic]", { hasText: "chosen project" }).count()) === 1 && sent.at(-1)?.strata_agent?.cwd === beta)
      await side.getByRole("button", { name: "New chat", exact: true }).click()
      await pg.waitForTimeout(400)
      await picker.click()
      await pg.locator("[data-project-option='Work']").click()
      await picker.click()
      await pg.getByRole("menuitemradio", { name: /No project/ }).click()
      await pg.waitForTimeout(300)
      t.ok("and it can be taken away again before the first prompt", (await picker.getAttribute("data-in-project")) === null)
      await pg.context().close()
      fs.rmSync(work, { recursive: true, force: true })
    },
  },
  {
    // Compacting (issue #96), as Claude Code does: /compact asks the model for a summary that takes the place of the messages; a conversation that nears the end of the
    // context is summarised by itself before the next prompt is sent. The server's answers are scripted (health with a small context, usage that says how much is used).
    name: "compact: /compact summarises the conversation, and one that nears the end of the context is summarised before the next prompt",
    async run({ browser, fast, t, errors }) {
      const NL = String.fromCharCode(10)
      const chunk = (o) => `data: ${JSON.stringify(o)}${NL}${NL}`
      const reply = (text, usage) => chunk({ choices: [{ delta: { content: text } }] }) + chunk({ choices: [], usage }) + `data: [DONE]${NL}${NL}`
      const pg = await open(browser, errors)
      const bodies = []
      let used = 100
      await pg.route("**/health", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ model: "m", images: false, max_context: 1000 }) }))
      await pg.route("**/v1/chat/completions", async (r) => {
        const b = JSON.parse(r.request().postData() || "{}")
        bodies.push(b)
        const asked = String(b.messages.at(-1)?.content ?? "").includes("Primary request and intent")
        if (asked) await new Promise((res) => setTimeout(res, 700))
        return r.fulfill({ status: 200, contentType: "text/event-stream", body: asked ? reply("<analysis>n</analysis><summary>1. Primary request: the e2e thing</summary>", { prompt_tokens: 900, completion_tokens: 30 }) : reply("answer " + bodies.length, { prompt_tokens: used, completion_tokens: 20 }) })
      })
      await pg.goto(fast.base + "/#/chat")
      await pg.evaluate(() => localStorage.clear())
      await pg.reload()
      const box = pg.locator("textarea[aria-label='Message']")
      await box.waitFor()
      await pg.waitForTimeout(700)
      const prompts = pg.locator(".msg-in.group")
      const notice = pg.locator("[data-compact]")
      const send = async (text) => { await box.fill(text); await box.press("Enter"); await pg.waitForTimeout(900) }

      // nothing to compact yet
      await send("/compact")
      t.ok("with nothing said yet /compact says there is nothing to compact, and sends nothing", (await notice.count()) === 0 && bodies.length === 0 && (await pg.locator("[role=status], [role=alert]").allInnerTexts()).join(" ").includes("Nothing to compact"), (await pg.locator("[role=status], [role=alert]").allInnerTexts()).join(" | "))

      await send("hello one")
      t.ok("a prompt and its answer", (await prompts.count()) === 1 && bodies.length === 1)

      // the context window: the share used, what it holds, where it is compacted
      const chip = pg.locator("button.prompt-bar__ctx")
      const panel = pg.locator("[role=dialog][aria-label='Context window']")
      t.ok("the prompt bar shows the share of the context window that is used, from what the server reported", (await chip.innerText()).trim() === "12%" && (await chip.getAttribute("data-level")) === "ok", await chip.innerText())
      await chip.hover()
      await pg.waitForTimeout(300)
      const tip = pg.locator("[data-context-tip]")
      t.ok("pointing at the chip says at once how much of the window is used, without opening anything", (await tip.count()) === 1 && (await tip.innerText()).includes("120 / 1,000 tokens") && (await tip.innerText()).includes("12%") && (await panel.count()) === 0, await tip.innerText())
      await pg.mouse.move(5, 5)
      await pg.waitForTimeout(300)
      t.ok("and it goes when the pointer does", (await tip.count()) === 0)
      await chip.click()
      await pg.waitForTimeout(500)
      t.ok("clicking it opens the panel, and the tip is not shown over it", (await tip.count()) === 0 && (await panel.count()) === 1)
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(300)
      await chip.click()
      await pg.waitForTimeout(500)
      t.ok("it opens to how much is used of how much, and what the window holds", (await panel.count()) === 1 && (await panel.locator("[data-context-figures]").innerText()).includes("120 of 1,000 tokens") && (await panel.locator("li[data-part='conversation']").count()) === 1 && (await panel.locator("li[data-part='system']").count()) === 1 && (await panel.locator("li[data-part='free']").innerText()).includes("880"), await panel.innerText())
      t.ok("it says where the conversation is compacted by itself", (await panel.locator("[data-context-auto]").innerText()).includes("950 tokens (95%)"))
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(300)
      t.ok("Escape closes it", (await panel.count()) === 0)
      const sentBefore = bodies.length
      await box.fill("/context")
      await box.press("Enter")
      await pg.waitForTimeout(500)
      t.ok("/context opens the same panel and sends nothing", (await panel.count()) === 1 && bodies.length === sentBefore && (await box.inputValue()) === "")
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(300)

      await box.fill("/")
      const list = pg.locator("[role=listbox][aria-label='Skills']")
      await list.waitFor({ timeout: 5000 })
      const opt = list.locator("[role=option][data-skill='compact']")
      t.ok("a / lists /compact, a command of Strata, with what it does", (await opt.count()) === 1 && (await opt.innerText()).includes("Summarise the conversation"), (await list.innerText()).split(NL).join(" | "))
      await box.fill("/compact focus on the tests")
      await pg.waitForTimeout(200)
      await box.press("Enter")
      await pg.waitForTimeout(250)
      t.ok("while the model writes the summary the chat says so, and the composer is empty", (await pg.getByText("Compacting the conversation…").count()) >= 1 && (await box.inputValue()) === "")
      t.ok("and it is shown with the orb that is made for compacting, an SVG that packs and springs back", (await pg.locator("[data-compacting-status] svg").count()) >= 1)
      t.ok("and nothing else can be sent meanwhile (Stop is offered)", (await pg.locator("button[aria-label='Stop']").count()) === 1)
      await pg.waitForTimeout(1200)
      const last = bodies.at(-1)
      t.ok("the request is for the summary: the conversation, then the request with what to focus on, and no tools", bodies.length === 2 && String(last.messages.at(-1).content).includes("focus on the tests") && last.messages.length === 3 && !("strata_mcp" in last) && !("strata_agent" in last), JSON.stringify(last.messages.map((m) => m.role)))
      t.ok("the messages are replaced by a line that says so, with the sizes", (await notice.count()) === 1 && (await prompts.count()) === 0 && (await notice.innerText()).includes("Conversation compacted") && (await notice.innerText()).includes("tokens"), await notice.innerText())
      t.ok("and what was asked of /compact is not a message of the conversation", !(await pg.locator("body").innerText()).includes("/compact focus"))
      await notice.getByRole("button").click()
      await pg.waitForTimeout(600)
      t.ok("the line opens to the summary the model reads, and what it was asked to focus on", (await notice.innerText()).includes("Primary request: the e2e thing") && (await notice.innerText()).includes("focus on the tests"))
      await pg.reload()
      await pg.waitForSelector("textarea[aria-label='Message']")
      await pg.waitForTimeout(800)
      t.ok("the summary is kept: after a reload it is still there", (await notice.count()) === 1)

      // the next prompt goes after the summary
      used = 930
      await send("hello two")
      t.ok("near the end of the window the chip says so", (await chip.innerText()).trim() === "95%" && (await chip.getAttribute("data-level")) === "full", await chip.innerText())
      const next = bodies.at(-1)
      t.ok("the next prompt is sent after the summary, which the model reads as the earlier part", next.messages.length === 2 && String(next.messages[0].content).includes("the e2e thing") && next.messages[1].content === "hello two", JSON.stringify(next.messages.map((m) => m.role)))

      // near the end of the context (950 of 1000): the next prompt is preceded by a summary
      const before = bodies.length
      used = 120
      await send("hello three")
      await pg.waitForTimeout(900)
      t.ok("near the end of the context the conversation is summarised before the next prompt, without it", bodies.length === before + 2 && String(bodies.at(-2).messages.at(-1).content).includes("Primary request and intent") && !JSON.stringify(bodies.at(-2).messages).includes("hello three"), String(bodies.length - before))
      t.ok("the prompt is shown with its answer after the line that says it was compacted by itself", (await notice.count()) === 1 && (await notice.innerText()).includes("automatically") && (await prompts.count()) === 1 && (await prompts.first().innerText()).includes("hello three") && (await pg.locator("body").innerText()).includes("answer "))
      t.ok("the request that carried the prompt starts with the summary and has the prompt, not the old messages", bodies.at(-1).messages.length === 2 && bodies.at(-1).messages[1].content === "hello three")

      // switched off
      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(500)
      await pg.locator(".t-morph-menu [role=option]", { hasText: "Sampling" }).click()
      await pg.waitForTimeout(600)
      const sw = pg.getByRole("switch", { name: "Compact the conversation by itself" })
      t.ok("a switch in the sampling settings says it is on", (await sw.getAttribute("aria-checked")) === "true")
      await sw.click()
      await pg.waitForTimeout(300)
      t.ok("it can be turned off", (await sw.getAttribute("aria-checked")) === "false")
      await pg.getByRole("button", { name: "Apply", exact: true }).click()
      await pg.waitForTimeout(500)
      t.ok("which is kept when the settings are applied", (await pg.evaluate(() => JSON.parse(localStorage.getItem("strata.sampling") || "{}").autoCompact)) === false)
      used = 950
      await send("hello four")
      const n4 = bodies.length
      await send("hello five")
      t.ok("with it off a conversation that nears the end is not summarised by itself", bodies.length === n4 + 1 && !String(bodies.at(-1).messages.at(-1).content).includes("Primary request and intent"))
      await pg.context().close()
    },
  },
  {
    // The Chat's right panel (issue #99): Git (the branch, what changed, the diff of a file, branches, worktrees, commits; read only, from the server), the plan, the skills used, the context.
    // The repository is a real one made in a temporary folder; the model's answers are scripted.
    name: "panel: the right panel shows the Git state of the project's folders, the plan, the skills used and the context",
    async run({ browser, fast, t, errors }) {
      const fs = await import("node:fs")
      const os = await import("node:os")
      const path = await import("node:path")
      const cp = await import("node:child_process")
      const NL = String.fromCharCode(10)
      const git = (cwd, ...args) => cp.execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", ...args], { cwd, stdio: "pipe" })
      const base = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "strata-panel-")))
      const repo = path.join(base, "app"), wt = path.join(base, "app-wt"), plain = path.join(base, "plain")
      fs.mkdirSync(repo); fs.mkdirSync(plain)
      git(repo, "init", "-q", "-b", "main")
      fs.writeFileSync(path.join(repo, "a.txt"), "one" + NL + "two" + NL + "three" + NL)
      fs.writeFileSync(path.join(repo, "b.txt"), "bee" + NL)
      git(repo, "add", "."); git(repo, "commit", "-q", "-m", "first commit")
      git(repo, "branch", "feature/x")
      git(repo, "worktree", "add", "-q", wt, "feature/x")
      fs.writeFileSync(path.join(repo, "a.txt"), "one" + NL + "TWO" + NL + "three" + NL)
      fs.writeFileSync(path.join(repo, "b.txt"), "bee" + NL + "buzz" + NL)
      git(repo, "add", "b.txt")
      fs.writeFileSync(path.join(repo, "new.txt"), "fresh" + NL)

      const chunk = (o) => `data: ${JSON.stringify(o)}${NL}${NL}`
      const ev = (e) => chunk({ choices: [{ delta: {} }], strata_mcp: e })
      const tail = (text) => chunk({ choices: [{ delta: { content: text } }] }) + chunk({ choices: [], usage: { prompt_tokens: 300, completion_tokens: 20 } }) + `data: [DONE]${NL}${NL}`
      const rich = ev({ event: "start", id: "s1", name: "skills__use_skill" }) + ev({ event: "call", id: "s1", name: "skills__use_skill", server: "skills", tool: "use_skill", arguments: { name: "git-flow" }, round: 1 })
        + ev({ event: "result", id: "s1", ok: true, text: "instructions", chars: 12, truncated: false, ms: 5 })
        + ev({ event: "start", id: "p1", name: "ExitPlanMode" }) + ev({ event: "call", id: "p1", name: "ExitPlanMode", server: "agent", tool: "ExitPlanMode", arguments: { plan: "1. Read the code" + NL + "2. Write the fix" }, round: 1 })
        + ev({ event: "result", id: "p1", ok: true, text: "approved", chars: 8, truncated: false, ms: 5 })
        + ev({ event: "todos", call_id: "t1", todos: [{ content: "Read the code", status: "completed", activeForm: "Reading the code" }, { content: "Write the fix", status: "in_progress", activeForm: "Writing the fix" }, { content: "Run the tests", status: "pending", activeForm: "Running the tests" }] })
        + tail("done")
      const info = { available: true, allowed: true, shell: "bash", tools: ["Read"] }

      const pg = await open(browser, errors, { width: 1400, height: 900 })
      await pg.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(info) }))
      await pg.route("**/health", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ model: "m", images: false, max_context: 1000 }) }))
      let n = 0
      await pg.route("**/v1/chat/completions", (r) => r.fulfill({ status: 200, contentType: "text/event-stream", body: n++ === 0 ? rich : tail("again") }))
      await pg.addInitScript(([r, w, p]) => {
        if (!localStorage.getItem("strata.chats")) localStorage.setItem("strata.chats", JSON.stringify({ active: null, items: [], projects: [{ id: "p1", name: "Work", folders: [r, w] }, { id: "p2", name: "Plain", folders: [p] }] }))
      }, [repo, wt, plain])
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("aside[aria-label='Conversations']")
      await pg.waitForTimeout(800)
      const side = pg.locator("aside[aria-label='Conversations']")
      const toggle = pg.locator("[data-panel-toggle]")
      const dock = pg.locator("aside[aria-label='Session panel']")
      const view = dock.locator("[data-git-view]")

      await side.getByRole("button", { name: "New chat in project Work" }).click()
      await pg.waitForTimeout(400)
      t.ok("the panel is closed to begin with, and a button opens it", (await dock.getAttribute("data-panel")) === "closed" && (await toggle.getAttribute("aria-pressed")) === "false")
      const widths = pg.evaluate(async () => { const el = document.querySelector("aside[aria-label='Session panel']"); const seen = []; const t0 = performance.now(); while (performance.now() - t0 < 700) { seen.push(Math.round(el.getBoundingClientRect().width)); await new Promise((r) => requestAnimationFrame(r)) } return seen })
      await toggle.click()
      const w = await widths
      t.ok("it stretches open through in-between widths", new Set(w).size > 4 && w[0] <= 5 && w.at(-1) >= 330 && w.slice(1, -1).some((x) => x > 10 && x < 330), JSON.stringify([...new Set(w)]))
      await view.waitFor({ timeout: 8000 })
      t.ok("the Git tab is the first, with a tab for each folder of the project", (await dock.locator("[role=tab][data-tab='git']").getAttribute("aria-selected")) === "true" && (await dock.locator("[role=tablist][aria-label='Folders of the project'] [role=tab]").allInnerTexts()).join(",") === "app,app-wt")
      t.ok("the branch of the main folder", (await view.locator("[data-branch]").innerText()).includes("main"))
      const group = (name) => view.locator(`[data-group-title='${name}']`)
      t.ok("what changed, in groups: staged, not staged and new", (await group("Staged").locator("[data-file]").allInnerTexts()).join().includes("b.txt") && (await group("Not staged").locator("[data-file]").allInnerTexts()).join().includes("a.txt") && (await group("New files").locator("[data-file]").allInnerTexts()).join().includes("new.txt"))
      t.ok("each with a letter for its change", (await group("Not staged").locator("[data-file='a.txt'] button span").first().innerText()) === "M" && (await group("New files").locator("[data-file='new.txt'] button span").first().innerText()) === "?")
      await group("Not staged").locator("[data-file='a.txt'] button").click()
      await pg.waitForTimeout(900)
      t.ok("a file opens to its diff, with the lines that went and the lines that came", (await view.locator("[data-line='del']").innerText()).includes("two") && (await view.locator("[data-line='add']").innerText()).includes("TWO"))
      await group("New files").locator("[data-file='new.txt'] button").click()
      await pg.waitForTimeout(900)
      t.ok("a new file is shown as all added", (await group("New files").locator("[data-line='add']").innerText()).includes("fresh"))
      await dock.locator("[data-section-title='Branches'] > button").click()
      await pg.waitForTimeout(500)
      t.ok("the branches are listed with the current one marked", (await view.locator("[data-branch-row]").count()) === 2 && (await view.locator("[data-branch-row][data-current]").getAttribute("data-branch-row")) === "main")
      await dock.locator("[data-section-title='Worktrees'] > button").click()
      await pg.waitForTimeout(500)
      t.ok("the worktrees of the repository are listed", (await view.locator("[data-worktree]").count()) === 2)
      t.ok("the last commits", (await view.locator("[data-commit]").first().innerText()).includes("first commit"))
      await dock.locator("[role=tablist][aria-label='Folders of the project'] [role=tab]", { hasText: "app-wt" }).click()
      await pg.waitForTimeout(1500)
      t.ok("the other folder has its own branch", (await view.locator("[data-branch]").innerText()).includes("feature/x") && (await view.locator("[data-clean]").count()) === 1)

      // an answer ends: the view is read again
      fs.writeFileSync(path.join(wt, "later.txt"), "x")
      await dock.locator("[role=tablist][aria-label='Folders of the project'] [role=tab]", { hasText: "app" }).first().click()
      await pg.waitForTimeout(1200)
      await pg.fill("textarea[aria-label='Message']", "go")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(1500)
      fs.writeFileSync(path.join(repo, "after.txt"), "y")
      await pg.fill("textarea[aria-label='Message']", "once more")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(1800)
      t.ok("when an answer ends the Git view is read again, so what the model changed is there", (await view.locator("[data-file='after.txt']").count()) === 1)

      // the plan
      await dock.locator("[data-tab='plan']").click()
      await pg.waitForTimeout(500)
      t.ok("the plan tab has the to-do list with how far it is", (await dock.locator("[data-progress]").innerText()).includes("1 of 3 done") && (await dock.getByText("Write the fix").count()) >= 1)
      t.ok("and the plan that was sent for approval", (await dock.locator("[data-plan-text]").innerText()).includes("Write the fix"))
      // the skills
      await dock.locator("[data-tab='skills']").click()
      await pg.waitForTimeout(500)
      t.ok("the skills tab says which skill the model loaded", (await dock.locator("[data-skill-used='git-flow']").innerText()).includes("the model loaded"))
      t.ok("and the chat says so where it happened", (await pg.locator("[data-skill-call='use']").innerText()).includes("Used skill: git-flow"))
      // the context
      await dock.locator("[data-tab='context']").click()
      await pg.waitForTimeout(500)
      t.ok("the context tab has what the window holds", (await dock.locator("[data-context-panel]").innerText()).includes("Context window") && (await dock.locator("[data-context-figures]").innerText()).includes("of 1,000 tokens"))

      // kept
      await pg.reload()
      await pg.waitForSelector("aside[aria-label='Session panel']")
      await pg.waitForTimeout(900)
      t.ok("after a reload the panel is open on the tab it was left on", (await dock.getAttribute("data-panel")) === "open" && (await dock.locator("[data-tab='context'][aria-selected='true']").count()) === 1)
      await toggle.click()
      await pg.waitForTimeout(600)
      t.ok("the button closes it", (await dock.getAttribute("data-panel")) === "closed")

      // a folder that is not a repository
      await toggle.click()
      await dock.locator("[data-tab='git']").click()
      await side.getByRole("button", { name: "New chat in project Plain" }).click()
      await pg.waitForTimeout(1500)
      t.ok("a folder that is not in a repository says so", (await dock.locator("[data-not-repo]").count()) === 1)
      await pg.context().close()

      // a narrow screen: a sheet over the chat
      const ph = await open(browser, errors, { width: 390, height: 800 })
      await ph.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(info) }))
      await ph.addInitScript(([r]) => { if (!localStorage.getItem("strata.chats")) localStorage.setItem("strata.chats", JSON.stringify({ active: null, items: [], projects: [{ id: "p1", name: "Work", folders: [r] }] })) }, [repo])
      await ph.goto(fast.base + "/#/chat")
      await ph.waitForSelector("textarea[aria-label='Message']")
      await ph.waitForTimeout(800)
      await ph.locator("[data-panel-toggle]").click()
      await ph.waitForTimeout(700)
      const sheet = ph.locator("[data-panel='sheet']")
      t.ok("on a narrow screen it opens as a sheet over the chat, and nothing is wider than the screen", (await sheet.count()) === 1 && (await ph.evaluate(() => document.documentElement.scrollWidth - innerWidth)) <= 0)
      await ph.keyboard.press("Escape")
      await ph.waitForTimeout(600)
      t.ok("Escape closes the sheet", (await sheet.count()) === 0)
      await ph.context().close()
      fs.rmSync(base, { recursive: true, force: true })
    },
  },
  {
    // The memory and instruction files the chat reads (issue #99): the project's own are always on; what other apps wrote down is off until switched on, from the panel or from
    // Settings > Import > Memory. A fake home folder is read (the importer mock), never the real one.
    name: "memory: the notes the chat reads are listed for the project, switched on from the panel and from Settings, and /memory, /init and /clear work",
    async run({ browser, importer, t, errors }) {
      const fs = await import("node:fs")
      const os = await import("node:os")
      const path = await import("node:path")
      const NL = String.fromCharCode(10)
      const proj = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "strata-mem-")))
      fs.writeFileSync(path.join(proj, "CLAUDE.md"), "Project: use tabs." + NL)
      const slug = proj.replace(/[^A-Za-z0-9]/g, "-")
      const files = [
        [path.join(importer.home, ".claude", "CLAUDE.md"), "Global: I like haiku." + NL],
        [path.join(importer.home, ".claude", "projects", slug, "memory", "MEMORY.md"), "- [style](style.md) the user likes short answers" + NL],
        [path.join(importer.home, ".claude", "projects", slug, "memory", "style.md"), "Short answers please." + NL],
      ]
      for (const [f, text] of files) { fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, text) }
      const saved = () => JSON.parse(fs.readFileSync(importer.config, "utf8"))

      const pg = await open(browser, errors, { width: 1400, height: 900 })
      const bodies = []
      await pg.route("**/v1/chat/completions", (r) => { bodies.push(JSON.parse(r.request().postData() || "{}")); return r.fulfill({ status: 200, contentType: "text/event-stream", body: `data: ${JSON.stringify({ choices: [{ delta: { content: "ok" } }] })}${NL}${NL}data: [DONE]${NL}${NL}` }) })
      await pg.addInitScript((p) => { if (!localStorage.getItem("strata.chats")) localStorage.setItem("strata.chats", JSON.stringify({ active: null, items: [], projects: [{ id: "p1", name: "Notes", folders: [p] }] })) }, proj)
      await pg.goto(importer.base + "/#/chat")
      await pg.waitForSelector("aside[aria-label='Conversations']")
      await pg.waitForTimeout(800)
      const side = pg.locator("aside[aria-label='Conversations']")
      const dock = pg.locator("aside[aria-label='Session panel']")
      const box = pg.locator("textarea[aria-label='Message']")
      await side.getByRole("button", { name: "New chat in project Notes" }).click()
      await pg.waitForTimeout(400)

      // /memory opens the panel on the notes
      await box.fill("/memory")
      await box.press("Enter")
      await pg.waitForTimeout(900)
      t.ok("/memory opens the panel on the Memory tab and sends nothing", (await dock.getAttribute("data-panel")) === "open" && (await dock.locator("[data-tab='memory'][aria-selected='true']").count()) === 1 && bodies.length === 0 && (await box.inputValue()) === "")
      const tab = dock.locator("[data-memory-tab]")
      await tab.waitFor({ timeout: 8000 })
      t.ok("the project's own file is listed and always on", (await tab.locator("[data-memory-source='project:0']").innerText()).includes("always on") && (await tab.locator("[data-memory-source='project:0'] [role=switch]").count()) === 0)
      t.ok("what other apps wrote down is listed and off", (await tab.locator("[data-memory-source='claude:instructions']").getAttribute("data-on")) === null && (await tab.locator("[data-memory-source='claude:memory']").getAttribute("data-on")) === null)
      await tab.locator("[data-memory-file='CLAUDE.md'] button").click()
      await pg.waitForTimeout(700)
      t.ok("a file opens to its text", (await tab.locator("[data-memory-text]").first().innerText()).includes("Project: use tabs."))
      await tab.getByRole("button", { name: "Ask the model to change it" }).click()
      t.ok("and the model can be asked to change it: the composer has the start of the request", (await box.inputValue()).includes("CLAUDE.md"))
      await box.fill("")
      await tab.locator("[data-memory-source='claude:instructions'] [role=switch]").click()
      await pg.waitForTimeout(900)
      t.ok("switching another app's instructions on is saved in the run config", JSON.stringify(saved().import.memory.on) === '["claude:instructions"]' && (await tab.locator("[data-memory-source='claude:instructions']").getAttribute("data-on")) !== null, JSON.stringify(saved().import))
      await tab.locator("[data-memory-source='claude:memory'] [role=switch]").click()
      await pg.waitForTimeout(900)
      t.ok("and the memory of the project too", JSON.stringify(saved().import.memory.on.sort()) === '["claude:instructions","claude:memory"]')
      await tab.locator("[data-memory-source='claude:memory'] [data-memory-file$='MEMORY.md'] button").click()
      await pg.waitForTimeout(700)
      t.ok("its index opens to its text", (await tab.locator("[data-memory-source='claude:memory'] [data-memory-text]").first().innerText()).includes("short answers"))

      // Settings
      await pg.goto(importer.base + "/#/settings/import-memory")
      await pg.waitForSelector("[data-memory-source]")
      await pg.waitForTimeout(500)
      const row = (id) => pg.locator(`[data-memory-source='${id}']`)
      t.ok("Settings > Import > Memory lists the same, with their switches as saved", (await row("claude:instructions").locator("[role=switch]").getAttribute("aria-checked")) === "true" && (await row("claude:memory").locator("[role=switch]").getAttribute("aria-checked")) === "true")
      await row("claude:instructions").locator("[role=switch]").click()
      await pg.waitForTimeout(800)
      t.ok("a switch is turned off from there and saved", JSON.stringify(saved().import.memory.on) === '["claude:memory"]')
      await row("claude:memory").locator("[role=switch]").click()
      await pg.waitForTimeout(800)
      t.ok("and all are off again", JSON.stringify(saved().import.memory.on) === "[]")

      // /init and /clear
      await pg.goto(importer.base + "/#/chat")
      await pg.waitForSelector("textarea[aria-label='Message']")
      await pg.waitForTimeout(800)
      await side.getByRole("button", { name: "New chat in project Notes" }).click()
      await pg.waitForTimeout(300)
      await box.fill("/init")
      await box.press("Enter")
      await pg.waitForTimeout(1200)
      t.ok("/init sends the prompt that has the model write the project's CLAUDE.md", bodies.length === 1 && String(bodies[0].messages.at(-1).content).includes("CLAUDE.md") && String(bodies[0].messages.at(-1).content).includes("improve it"))
      await box.fill("/clear")
      await box.press("Enter")
      await pg.waitForTimeout(600)
      t.ok("/clear starts a new chat, and the one before stays in Recents", (await pg.locator(".msg-in").count()) === 0 && (await side.locator("[data-topic]").count()) >= 1)
      await pg.context().close()
      for (const [f] of files) fs.rmSync(f, { force: true })
      fs.rmSync(path.join(importer.home, ".claude", "projects", slug), { recursive: true, force: true })
      fs.rmSync(proj, { recursive: true, force: true })
    },
  },
  {
    // Rules that last (issue #99): a card that asks can keep the answer for a project or everywhere (More choices); Settings > Permissions lists, adds and removes the rules; the
    // requests carry them. The server's answers are scripted.
    name: "perms: an answer can be kept as a rule for good, and the rules are listed, added and removed in Settings",
    async run({ browser, fast, t, errors }) {
      const NL = String.fromCharCode(10)
      const chunk = (o) => `data: ${JSON.stringify(o)}${NL}${NL}`
      const ev = (e) => chunk({ choices: [{ delta: {} }], strata_mcp: e })
      const done = chunk({ choices: [{ delta: { content: "ok" } }] }) + chunk({ choices: [], usage: { completion_tokens: 1 } }) + `data: [DONE]${NL}${NL}`
      const ask = (id, qid, command, rule) => ev({ event: "start", id, name: "Bash" }) + ev({ event: "call", id, name: "Bash", server: "agent", tool: "Bash", arguments: { command }, round: 1 })
        + ev({ event: "permission", id: qid, call_id: id, tool: "Bash", arguments: { command }, why: "a command asks every time", danger: false, rule }) + done
      const info = { available: true, allowed: true, shell: "bash", tools: ["Bash"] }
      const pg = await open(browser, errors)
      const sent = [], answers = []
      const streams = [ask("c1", "q1", "npm test", "Bash(npm test:*)"), ask("c2", "q2", "curl x.example", "Bash(curl:*)"), done]
      let n = 0
      await pg.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(info) }))
      await pg.route("**/agent/permission", (r) => { answers.push(JSON.parse(r.request().postData() || "{}")); return r.fulfill({ status: 200, contentType: "application/json", body: "{\"ok\":true}" }) })
      await pg.route("**/v1/chat/completions", (r) => { sent.push(JSON.parse(r.request().postData() || "{}")); return r.fulfill({ status: 200, contentType: "text/event-stream", body: streams[Math.min(n++, streams.length - 1)] }) })
      await pg.addInitScript(() => { if (!localStorage.getItem("strata.chats")) localStorage.setItem("strata.chats", JSON.stringify({ active: null, items: [], projects: [{ id: "p1", name: "Work", folders: ["/work/app"] }] })) })
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea[aria-label='Message']")
      await pg.waitForTimeout(900)
      const perms = () => pg.evaluate(() => JSON.parse(localStorage.getItem("strata.agent.perms") || "{}"))
      const card = pg.locator("[data-agent-ask]").first()

      // a chat in no project: only "everywhere" is offered
      await pg.fill("textarea[aria-label='Message']", "run the tests")
      await pg.keyboard.press("Enter")
      await card.waitFor({ timeout: 8000 })
      t.ok("a card that asks has More choices", (await card.locator("[data-ask-more]").count()) === 1 && (await card.locator("[data-ask-keep]").count()) === 0)
      await card.locator("[data-ask-more]").click()
      await pg.waitForTimeout(300)
      const keep = (await card.locator("[data-ask-keep] button").allInnerTexts()).join("|")
      t.ok("in a chat that is in no project the choices are for everywhere only", keep === "Always allow everywhere|Never anywhere", keep)
      await card.getByRole("button", { name: "Always allow everywhere" }).click()
      await pg.waitForTimeout(700)
      t.ok("it is sent as allowed for now, and the rule is kept for good", JSON.stringify(answers.at(-1)) === '{"id":"q1","decision":"allow_chat"}' && JSON.stringify((await perms()).everywhere?.allow) === '["Bash(npm test:*)"]', JSON.stringify(await perms()))
      t.ok("the card says what was done", (await pg.locator("[data-agent-ask-answer], .border-t").filter({ hasText: "You allowed it everywhere from now on." }).count()) >= 1)

      // a chat in a project: the project's choices too; the next request carries the rule
      await pg.getByRole("button", { name: "New chat in project Work" }).click()
      await pg.waitForTimeout(400)
      await pg.fill("textarea[aria-label='Message']", "fetch it")
      await pg.keyboard.press("Enter")
      await pg.waitForSelector("[data-agent-ask]", { timeout: 8000 })
      t.ok("the request carries what was allowed everywhere", JSON.stringify(sent.at(-1).strata_agent.allow) === '["Bash(npm test:*)"]', JSON.stringify(sent.at(-1).strata_agent))
      const card2 = pg.locator("[data-agent-ask]").last()
      await card2.locator("[data-ask-more]").click()
      await pg.waitForTimeout(300)
      const keep2 = (await card2.locator("[data-ask-keep] button").allInnerTexts()).join("|")
      t.ok("in a project the choices are for the project and for everywhere", keep2 === "Always allow in this project|Always allow everywhere|Never in this project|Never anywhere", keep2)
      await card2.getByRole("button", { name: "Never in this project" }).click()
      await pg.waitForTimeout(700)
      t.ok("\"never\" is sent as a no and kept as a deny rule of the project", answers.at(-1).decision === "deny" && JSON.stringify((await perms()).projects?.p1?.deny) === '["Bash(curl:*)"]', JSON.stringify(await perms()))
      await pg.fill("textarea[aria-label='Message']", "again")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(900)
      const last = sent.at(-1).strata_agent
      t.ok("and the next request refuses it", JSON.stringify(last.deny) === '["Bash(curl:*)"]' && last.allow.includes("Bash(npm test:*)"), JSON.stringify(last))

      // Settings > Permissions
      await pg.goto(fast.base + "/#/settings/permissions")
      await pg.waitForSelector("[data-permissions]")
      await pg.waitForTimeout(500)
      const rule = (r) => pg.locator(`[data-perm-rule='${r}']`)
      t.ok("the page lists the rules where they are: everywhere and the project", (await rule("Bash(npm test:*)").getAttribute("data-perm-scope")) === "everywhere" && (await rule("Bash(curl:*)").getAttribute("data-perm-scope")) === "p1" && (await rule("Bash(curl:*)").getAttribute("data-effect")) === "deny")
      const everywhere = pg.locator("[data-perm-block='everywhere']")
      const field = everywhere.locator("input")
      await field.fill("not a rule")
      await everywhere.getByRole("button", { name: "Add", exact: true }).click()
      t.ok("something that is not a rule is refused with a reason", (await everywhere.locator("[role=alert]").innerText()).includes("not a rule"))
      await field.fill("Read(src/**)")
      await everywhere.getByRole("button", { name: "Add", exact: true }).click()
      await pg.waitForTimeout(300)
      t.ok("a rule is added", (await rule("Read(src/**)").count()) === 1 && (await rule("Read(src/**)").getAttribute("data-effect")) === "allow")
      await everywhere.getByRole("radio", { name: "Never" }).click()
      await field.fill("Bash(rm:*)")
      await field.press("Enter")
      await pg.waitForTimeout(300)
      t.ok("and a \"never\" too", (await rule("Bash(rm:*)").getAttribute("data-effect")) === "deny")
      await field.fill("Bash(rm:*)")
      await field.press("Enter")
      t.ok("the same rule twice is said to be there already", (await everywhere.locator("[role=alert]").innerText()).includes("there already"))
      await pg.reload()
      await pg.waitForSelector("[data-permissions]")
      t.ok("the rules are kept", (await rule("Read(src/**)").count()) === 1 && (await rule("Bash(rm:*)").count()) === 1)
      await pg.getByRole("button", { name: "Remove the rule Read(src/**)" }).click()
      await pg.waitForTimeout(300)
      t.ok("a rule is removed", (await rule("Read(src/**)").count()) === 0 && !JSON.stringify(await perms()).includes("Read(src"))
      await pg.getByRole("button", { name: "Remove the rule Bash(curl:*)" }).click()
      await pg.waitForTimeout(300)
      t.ok("a project's rule is removed from the project", (await perms()).projects?.p1 === undefined)

      // /permissions
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea[aria-label='Message']")
      await pg.fill("textarea[aria-label='Message']", "/permissions")
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(600)
      t.ok("/permissions opens the page of the rules", pg.url().endsWith("#/settings/permissions") && (await pg.locator("[data-permissions]").count()) === 1)
      await pg.context().close()
    },
  },
  {
    // `@file` in the prompt and messages that wait while an answer is written (issue #99). The project's folder is a real temporary one; the model's answers are scripted.
    name: "mention: @ offers the files of the project and sends the ones mentioned; a message typed while it answers waits in line",
    async run({ browser, fast, t, errors }) {
      const fs = await import("node:fs")
      const os = await import("node:os")
      const path = await import("node:path")
      const NL = String.fromCharCode(10)
      const proj = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "strata-at-")))
      for (const [rel, text] of [["src/app.py", "print('the app')" + NL], ["src/util/application.py", "x = 1" + NL], ["docs/my-app.md", "# my app" + NL], ["README.md", "readme text" + NL], [".env", "TOKEN=hunter2" + NL]]) {
        fs.mkdirSync(path.dirname(path.join(proj, rel)), { recursive: true })
        fs.writeFileSync(path.join(proj, rel), text)
      }
      const chunk = (o) => `data: ${JSON.stringify(o)}${NL}${NL}`
      const answer = (text) => chunk({ choices: [{ delta: { content: text } }] }) + chunk({ choices: [], usage: { prompt_tokens: 50, completion_tokens: 5 } }) + `data: [DONE]${NL}${NL}`
      const info = { available: true, allowed: true, shell: "bash", tools: ["Read"] }
      const pg = await open(browser, errors)
      const bodies = []
      let slow = 0                                               // how many of the next requests are held back, to have an answer that is being written
      await pg.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(info) }))
      await pg.route("**/v1/chat/completions", async (r) => {
        bodies.push(JSON.parse(r.request().postData() || "{}"))
        if (slow > 0) { slow--; await new Promise((res) => setTimeout(res, 1800)) }
        return r.fulfill({ status: 200, contentType: "text/event-stream", body: answer("answer " + bodies.length) })
      })
      await pg.addInitScript((p) => { if (!localStorage.getItem("strata.chats")) localStorage.setItem("strata.chats", JSON.stringify({ active: null, items: [], projects: [{ id: "p1", name: "Files", folders: [p] }] })) }, proj)
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("aside[aria-label='Conversations']")
      await pg.waitForTimeout(800)
      await pg.getByRole("button", { name: "New chat in project Files" }).click()
      await pg.waitForTimeout(400)
      const box = pg.locator("textarea[aria-label='Message']")
      const list = pg.locator("[role=listbox][aria-label='Files']")
      const names = () => list.locator("[role=option]").evaluateAll((os) => os.map((o) => o.dataset.file))

      await box.click()
      await box.pressSequentially("look at @ap", { delay: 15 })
      await list.waitFor({ timeout: 5000 })
      const offered = await names()
      t.ok("typing @ and a few letters offers the files of the project that go with them, the name that starts with them first", offered[0] === "src/app.py" && offered.includes("src/util/application.py") && offered.includes("docs/my-app.md"), JSON.stringify(offered))
      t.ok("a secret is not offered", !offered.some((n) => n.includes(".env")))
      await pg.keyboard.press("ArrowDown")
      await pg.waitForTimeout(100)
      t.ok("the arrow keys move the mark", (await list.locator("[role=option][aria-selected=true]").getAttribute("data-file")) === offered[1])
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(200)
      t.ok("Escape closes the list and keeps what was typed", (await list.count()) === 0 && (await box.inputValue()) === "look at @ap")
      await box.fill("")
      await box.pressSequentially("look at @ap", { delay: 15 })
      await list.waitFor()
      await list.locator("[data-file='src/app.py']").click()
      await pg.waitForTimeout(300)
      t.ok("picking a file writes it as @path and a space", (await box.inputValue()) === "look at @src/app.py ", await box.inputValue())
      await box.pressSequentially("and @.env and @nope.txt please", { delay: 5 })
      await box.press("Enter")
      await pg.waitForTimeout(1200)
      const sentBody = bodies.at(-1)
      const content = String(sentBody.messages.at(-1).content)
      t.ok("the file that is mentioned goes with the prompt, as text", content.includes("File: src/app.py") && content.includes("print('the app')") && content.includes("@src/app.py"), content.slice(0, 200))
      t.ok("a secret that is mentioned is not sent, nor a file that is not there", !content.includes("hunter2") && !content.includes("nope.txt\n"))
      t.ok("the prompt shows the file as an attachment", (await pg.locator(".msg-in.group").first().innerText()).includes("src/app.py"))
      await box.fill("")
      await box.pressSequentially("mail me@example.com", { delay: 5 })
      await pg.waitForTimeout(400)
      t.ok("an e-mail address is not a mention", (await list.count()) === 0)
      await box.fill("")

      // a message typed while the answer is written
      slow = 1
      await box.fill("first question")
      await box.press("Enter")
      await pg.waitForSelector("button[aria-label='Stop']")
      t.ok("while it answers the composer says what typing does", (await box.getAttribute("placeholder")).includes("sent when the answer ends"))
      await box.fill("second question")
      await box.press("Enter")
      await pg.waitForTimeout(400)
      const queued = pg.locator("[data-queued]")
      t.ok("a message typed meanwhile waits in line, as a dashed bubble that says so, and the composer is free", (await queued.count()) === 1 && (await queued.innerText()).includes("second question") && (await queued.innerText()).includes("Waits for the answer to end") && (await box.inputValue()) === "")
      await box.fill("third question")
      await box.press("Enter")
      await pg.waitForTimeout(300)
      t.ok("and several can wait", (await queued.count()) === 2)
      await queued.nth(1).getByRole("button", { name: "Remove" }).click()
      await pg.waitForTimeout(300)
      t.ok("one can be dropped", (await queued.count()) === 1)
      const before = bodies.length
      await pg.waitForFunction(() => document.querySelectorAll("button[aria-label='Stop']").length === 0, null, { timeout: 15000 })
      await pg.waitForTimeout(1500)
      t.ok("when the answer ends the next goes by itself, with the answer before it in the conversation", bodies.length === before + 1 && String(bodies.at(-1).messages.at(-1).content) === "second question" && bodies.at(-1).messages.some((m) => m.content === "first question"))
      t.ok("and nothing waits any more", (await queued.count()) === 0)
      const mine = (await pg.locator(".msg-in.group").allInnerTexts()).join(" | ")
      t.ok("the conversation has them in order", mine.indexOf("first question") < mine.indexOf("second question") && !mine.includes("third question"), mine)

      // Stop: nothing is sent, and what waits can be sent or taken back
      slow = 1
      await box.fill("long one")
      await box.press("Enter")
      await pg.waitForSelector("button[aria-label='Stop']")
      await box.fill("kept for later")
      await box.press("Enter")
      await pg.waitForTimeout(300)
      const n1 = bodies.length
      await pg.click("button[aria-label='Stop']")
      await pg.waitForTimeout(1200)
      t.ok("Stop sends nothing: what waited is still there and says it was not sent", bodies.length === n1 && (await queued.count()) === 1 && (await queued.innerText()).includes("Not sent: the answer was stopped"))
      await queued.getByRole("button", { name: "Edit" }).click()
      await pg.waitForTimeout(300)
      t.ok("Edit takes it back to the composer", (await queued.count()) === 0 && (await box.inputValue()) === "kept for later")
      await box.press("Enter")
      await pg.waitForTimeout(1500)
      t.ok("and it can be sent as any prompt", String(bodies.at(-1).messages.at(-1).content) === "kept for later")
      await pg.context().close()
      fs.rmSync(proj, { recursive: true, force: true })
    },
  },
  {
    // Rewind (issue #99) against the real coding tools (the demo mock: a scripted model that makes a file and changes one, inside the folder): every prompt is a checkpoint; the dialog says
    // which files would be put back, and which were changed since; the files, the conversation or both can go back.
    name: "rewind: a prompt can be rewound: its files put back, and the conversation cut",
    async run({ browser, agentDemo, t, errors }) {
      const fs = await import("node:fs")
      const path = await import("node:path")
      const existing = path.join(agentDemo.dir, "existing.txt")
      const made = path.join(agentDemo.dir, "rewind-demo.txt")
      fs.writeFileSync(existing, "original text\n")
      fs.rmSync(made, { force: true })
      const pg = await open(browser, errors, { width: 1200, height: 900 })
      await pg.addInitScript((dir) => { try { localStorage.setItem("strata.sampling", JSON.stringify({ agentFolder: dir })) } catch { /* private window */ } }, agentDemo.dir)
      await pg.goto(agentDemo.base + "/#/chat")
      await pg.evaluate(() => localStorage.removeItem("strata.chats"))
      await pg.reload()
      const box = pg.locator("textarea[aria-label='Message']")
      await box.waitFor()
      await pg.waitForTimeout(1200)
      const prompts = pg.locator(".msg-in.group")
      const done = async () => { await pg.waitForSelector("button[aria-label='Stop']", { timeout: 5000 }).catch(() => {}); await pg.waitForFunction(() => document.querySelectorAll("button[aria-label='Stop']").length === 0, null, { timeout: 60000 }); await pg.waitForTimeout(600) }
      const dlg = pg.locator("[data-rewind]")

      await box.fill("rewind demo one")
      await box.press("Enter")
      await done()
      t.ok("the tools made a file and changed one, inside the folder, without asking", fs.existsSync(made) && fs.readFileSync(existing, "utf8").includes("changed by the model") && (await pg.locator("[data-agent-ask]").count()) === 0)
      await box.fill("and now a plain question")
      await box.press("Enter")
      await done()
      t.ok("two prompts in the chat", (await prompts.count()) === 2)
      t.ok("each prompt has a Rewind button", (await pg.locator("[data-rewind-button]").count()) === 2)

      await prompts.first().hover()
      await pg.locator("[data-rewind-button]").first().click()
      await dlg.waitFor({ timeout: 5000 })
      await pg.waitForSelector("[data-rewind-file]", { timeout: 8000 })
      const rows = await dlg.locator("[data-rewind-file]").evaluateAll((els) => els.map((e) => e.dataset.action + ":" + e.dataset.rewindFile.split(/[\\/]/).pop()).sort())
      t.ok("the dialog lists what would be put back: the file that was changed, and the file that was made, which would be deleted", JSON.stringify(rows) === JSON.stringify(["delete:rewind-demo.txt", "restore:existing.txt"]), JSON.stringify(rows))
      t.ok("it says that commands are not undone", (await dlg.innerText()).includes("What commands did to files is not undone."))
      t.ok("nothing is marked as changed since", (await dlg.locator("[data-changed]").count()) === 0)
      t.ok("nothing has been done yet", fs.existsSync(made) && fs.readFileSync(existing, "utf8").includes("changed by the model"))
      await dlg.getByRole("radio", { name: /Only the files/ }).check()
      await dlg.getByRole("button", { name: "Rewind", exact: true }).click()
      await pg.waitForTimeout(1200)
      t.ok("only the files: the changed file is back and the made file is gone", !fs.existsSync(made) && fs.readFileSync(existing, "utf8") === "original text\n")
      t.ok("and the conversation is as it was", (await prompts.count()) === 2 && (await dlg.count()) === 0)

      // again, and now someone else changes the file
      await box.fill("rewind demo two")
      await box.press("Enter")
      await done()
      t.ok("the tools changed it again", fs.existsSync(made) && fs.readFileSync(existing, "utf8").includes("changed by the model"))
      fs.writeFileSync(existing, "edited by the user afterwards\n")
      await prompts.last().hover()
      await pg.locator("[data-rewind-button]").last().click()
      await dlg.waitFor()
      await pg.waitForSelector("[data-rewind-file]", { timeout: 8000 })
      t.ok("a file that was changed since is marked, with a choice to put it back too", (await dlg.locator("[data-changed]").count()) === 1 && (await dlg.getByText("Also put back the files that were changed since").count()) === 1)
      await dlg.getByRole("button", { name: "Rewind", exact: true }).click()           // the files and the conversation
      await pg.waitForTimeout(1500)
      t.ok("the file that was made is deleted, and the one the user changed is left alone", !fs.existsSync(made) && fs.readFileSync(existing, "utf8") === "edited by the user afterwards\n")
      t.ok("the conversation is cut before that prompt, which goes back to the composer", (await prompts.count()) === 2 && (await box.inputValue()) === "rewind demo two", String(await prompts.count()))
      t.ok("and the toast says what was left", (await pg.locator("[role=status]").allInnerTexts()).join(" ").includes("left as you changed"))

      // /rewind
      await box.fill("")
      await box.fill("/rewind")
      await box.press("Enter")
      await dlg.waitFor({ timeout: 5000 })
      t.ok("/rewind opens the dialog, with a choice among the prompts", (await dlg.locator("select").count()) === 1 && (await dlg.locator("select option").count()) === 2)
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(400)
      t.ok("Escape closes it", (await dlg.count()) === 0)
      await pg.context().close()
      fs.rmSync(existing, { force: true })
      fs.rmSync(made, { force: true })
    },
  },
  {
    // The model asks the user a question with choices (AskUserQuestion, issue #99): a form on its call, answered or skipped. The server's stream is scripted.
    name: "askq: the model's question is a form with choices, answered or skipped",
    async run({ browser, fast, t, errors }) {
      const NL = String.fromCharCode(10)
      const chunk = (o) => `data: ${JSON.stringify(o)}${NL}${NL}`
      const ev = (e) => chunk({ choices: [{ delta: {} }], strata_mcp: e })
      const done = chunk({ choices: [{ delta: { content: "ok" } }] }) + chunk({ choices: [], usage: { completion_tokens: 1 } }) + `data: [DONE]${NL}${NL}`
      const questions = [
        { question: "Which library should we use?", header: "Library", multiSelect: false, options: [{ label: "requests", description: "the usual one" }, { label: "httpx", description: "async too" }] },
        { question: "Which checks should run?", header: "Checks", multiSelect: true, options: [{ label: "lint", description: "style" }, { label: "tests", description: "the suite" }, { label: "types", description: "the type checker" }] },
      ]
      const ask = (id, qid) => ev({ event: "start", id, name: "AskUserQuestion" }) + ev({ event: "call", id, name: "AskUserQuestion", server: "agent", tool: "AskUserQuestion", arguments: { questions }, round: 1 }) + ev({ event: "question", id: qid, call_id: id, questions }) + done
      const info = { available: true, allowed: true, shell: "bash", tools: ["AskUserQuestion"] }
      const pg = await open(browser, errors)
      const answers = []
      let n = 0
      await pg.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(info) }))
      await pg.route("**/agent/question", (r) => { answers.push(JSON.parse(r.request().postData() || "{}")); return r.fulfill({ status: 200, contentType: "application/json", body: "{\"ok\":true}" }) })
      await pg.route("**/v1/chat/completions", (r) => r.fulfill({ status: 200, contentType: "text/event-stream", body: [ask("c1", "q1"), ask("c2", "q2")][Math.min(n++, 1)] }))
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea[aria-label='Message']")
      await pg.waitForTimeout(900)
      await pg.fill("textarea[aria-label='Message']", "set it up")
      await pg.keyboard.press("Enter")
      const form = pg.locator("[data-agent-question]").first()
      await form.waitFor({ timeout: 8000 })
      t.ok("the question is a form with each question, its short label, and its choices with what they mean", (await form.locator("fieldset").count()) === 2 && (await form.innerText()).includes("Library") && (await form.innerText()).includes("async too") && (await form.locator("[data-option]").count()) === 5)
      t.ok("one question takes one choice (radio), the other several (checkboxes)", (await form.locator("fieldset").nth(0).locator("input[type=radio]").count()) === 2 && (await form.locator("fieldset").nth(1).locator("input[type=checkbox]").count()) === 3)
      const send = form.getByRole("button", { name: "Send", exact: true })
      t.ok("Send waits until every question has an answer", await send.isDisabled())
      await form.locator("[data-option='httpx']").click()
      await form.locator("[data-option='lint']").click()
      await form.locator("[data-option='tests']").click()
      t.ok("and is ready when they all have", await send.isEnabled())
      await form.locator("[data-option='requests']").click()
      await form.locator("[data-option='requests'] input").waitFor()
      t.ok("a radio changes its choice and does not add one", (await form.locator("fieldset").nth(0).locator("input:checked").count()) === 1)
      await form.locator("[data-option='lint']").click()
      await form.getByLabel("Something else: Checks").fill("my own: e2e")
      await send.click()
      await pg.waitForTimeout(700)
      t.ok("the answers are sent with the question's id: the choices, and what was written", JSON.stringify(answers.at(-1)) === JSON.stringify({ id: "q1", answers: { "Which library should we use?": ["requests"], "Which checks should run?": ["tests", "my own: e2e"] } }), JSON.stringify(answers.at(-1)))
      t.ok("the form becomes the answers", (await pg.locator("[data-asked-answers]").first().innerText()).includes("requests") && (await pg.locator("[data-asked-answers]").first().innerText()).includes("tests, my own: e2e") && (await pg.locator("[data-agent-question]").count()) === 0)

      // skipped
      await pg.fill("textarea[aria-label='Message']", "and again")
      await pg.keyboard.press("Enter")
      await pg.waitForSelector("[data-agent-question]", { timeout: 8000 })
      await pg.locator("[data-agent-question]").getByRole("button", { name: "Skip", exact: true }).click()
      await pg.waitForTimeout(700)
      t.ok("Skip sends no answers", JSON.stringify(answers.at(-1)) === JSON.stringify({ id: "q2", answers: null }))
      t.ok("and says so", (await pg.locator("[data-asked-answers]").last().innerText()).includes("You skipped the questions."))
      await pg.context().close()
    },
  },
  {
    // The user's hooks (issue #99): written in the run config, listed in Settings with a switch each, and shown in the chat where they acted.
    name: "hooks: Settings list them with a switch kept in the run config; the chat shows what a hook did on the call and on the answer",
    async run({ browser, importer, fast, t, errors }) {
      const fs = await import("node:fs")
      const NL = String.fromCharCode(10)
      const original = fs.readFileSync(importer.config, "utf8")
      const saved = () => JSON.parse(fs.readFileSync(importer.config, "utf8"))
      try {
        const pg = await open(browser, errors, { width: 1200, height: 900 })
        await pg.goto(importer.base + "/#/settings/hooks")
        await pg.waitForSelector("[data-hooks]")
        await pg.waitForTimeout(600)
        t.ok("with none written it says so, and shows how to write one", (await pg.locator("[data-hooks]").innerText()).includes("No hooks are set.") && (await pg.locator("[data-hooks] pre").innerText()).includes("before_tool"))

        fs.writeFileSync(importer.config, JSON.stringify({ ...saved(), hooks: [{ event: "before_tool", matcher: "Bash", command: "./check.sh", timeout: 10 }, { event: "stop", command: "echo done" }, { event: "sometime", command: "x" }] }, null, 2))
        await pg.reload()
        await pg.waitForSelector("[data-hook-list]")
        t.ok("each hook is listed with what runs it, its tools and its time limit", (await pg.locator("[data-hook-id]").count()) === 2 && (await pg.locator("[data-hook-list]").innerText()).includes("./check.sh") && (await pg.locator("[data-hook-list]").innerText()).includes("Time limit: 10 s") && (await pg.locator("[data-hook-list]").innerText()).includes("Bash"))
        t.ok("an entry that cannot be used is reported, not hidden", (await pg.locator("[data-hook-problems]").innerText()).includes("hook 3"))
        const sw = pg.getByRole("switch", { name: "Run the hook ./check.sh" })
        t.ok("they are all on", (await sw.getAttribute("aria-checked")) === "true")
        await sw.click()
        await pg.waitForFunction(() => document.querySelector("[role=switch][aria-label='Run the hook ./check.sh']")?.getAttribute("aria-checked") === "false")
        const after = saved()
        t.ok("a switch is kept in the run config and the hooks themselves are left as they were", after.hooks_off.length === 1 && after.hooks.length === 3 && after.hooks[0].command === "./check.sh")
        await pg.reload()
        await pg.waitForSelector("[data-hook-list]")
        t.ok("and is still off after a reload", (await pg.getByRole("switch", { name: "Run the hook ./check.sh" }).getAttribute("aria-checked")) === "false" && (await pg.getByRole("switch", { name: "Run the hook echo done" }).getAttribute("aria-checked")) === "true")
        await pg.context().close()
      } finally {
        fs.writeFileSync(importer.config, original)
      }

      // what a hook did, in the chat (the server's stream is scripted)
      const chunk = (o) => `data: ${JSON.stringify(o)}${NL}${NL}`
      const ev = (e) => chunk({ choices: [{ delta: {} }], strata_mcp: e })
      const note = (more) => ({ event: "hook", hook: "h0123abcd", command: "./check.sh", ok: false, code: 2, blocked: false, timeout: false, error: null, text: "", ms: 5, ...more })
      const stream = ev(note({ on: "prompt", call_id: null, ok: true, code: 0, command: "git status", text: "branch is main" })) +
        ev({ event: "start", id: "c1", name: "Bash" }) + ev({ event: "call", id: "c1", name: "Bash", server: "agent", tool: "Bash", arguments: { command: "git commit -m x" }, round: 1 }) +
        ev(note({ on: "before_tool", call_id: "c1", tool: "Bash", blocked: true, text: "no commits to main" })) +
        ev({ event: "result", id: "c1", ok: false, text: "Bash was stopped by a hook: no commits to main", chars: 44, truncated: false, ms: 9 }) +
        ev(note({ on: "stop", call_id: null, ok: false, code: 1, command: "npm test", text: "1 test failed" })) +
        chunk({ choices: [{ delta: { content: "ok" } }] }) + chunk({ choices: [], usage: { completion_tokens: 1 } }) + `data: [DONE]${NL}${NL}`
      const info = { available: true, allowed: true, shell: "bash", tools: ["Bash"] }
      const pg2 = await open(browser, errors, { width: 1200, height: 900 })
      await pg2.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(info) }))
      await pg2.route("**/v1/chat/completions", (r) => r.fulfill({ status: 200, contentType: "text/event-stream", body: stream }))
      await pg2.goto(fast.base + "/#/chat")
      await pg2.waitForSelector("textarea[aria-label='Message']")
      await pg2.waitForTimeout(900)
      await pg2.fill("textarea[aria-label='Message']", "commit it")
      await pg2.keyboard.press("Enter")
      const call = pg2.locator("[data-agent-call='Bash']").first()
      await call.waitFor({ timeout: 8000 })
      await pg2.waitForTimeout(600)
      const stopped = call.locator("[data-hook-state='blocked']")
      t.ok("a hook that stopped a call says so on that call, with its reason and its command", (await stopped.count()) === 1 && (await stopped.innerText()).includes("A hook stopped this call") && (await stopped.innerText()).includes("no commits to main") && (await stopped.innerText()).includes("./check.sh"))
      const answerNotes = pg2.locator("[data-hook-notes]").filter({ hasNot: pg2.locator("[data-agent-call]") })
      t.ok("the hooks of the prompt and of the end are on the answer, not on a call", (await pg2.locator("[data-hook='prompt']").count()) === 1 && (await pg2.locator("[data-hook='stop']").count()) === 1 && (await call.locator("[data-hook='prompt'], [data-hook='stop']").count()) === 0 && (await answerNotes.count()) >= 1)
      t.ok("one that finished badly says with which exit code, and what it printed", (await pg2.locator("[data-hook='stop']").innerText()).includes("exit code 1") && (await pg2.locator("[data-hook='stop']").innerText()).includes("1 test failed"))
      await pg2.context().close()
    },
  },
  {
    // code in an answer is coloured like an IDE, in the colours of the theme, and Copy still copies the plain text
    name: "code: code in an answer is coloured like an IDE in both themes and copies as plain text",
    async run({ browser, fast, t, errors }) {
      const want = {
        light: { builtin: "rgb(121, 94, 38)", number: "rgb(9, 134, 88)" },          // VS Code Light+: function, number
        dark: { builtin: "rgb(102, 217, 239)", number: "rgb(174, 129, 255)" },      // Monokai (the theme of Claude Code's own code): built-in cyan, number purple
      }
      const got = {}
      for (const scheme of ["light", "dark"]) {
        const ctx = await browser.newContext({ viewport: { width: 1000, height: 760 }, colorScheme: scheme })
        await ctx.grantPermissions(["clipboard-read", "clipboard-write"], { origin: fast.base })
        await ctx.addInitScript(() => { try { localStorage.setItem("strata.lang", JSON.stringify("en")) } catch { /* private window */ } })
        const pg = await ctx.newPage()
        pg.on("pageerror", (e) => errors.push(`pageerror: ${e}`))
        pg.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`))
        await pg.goto(fast.base + "/#/chat")
        await pg.waitForSelector("textarea")
        await send(pg, "show code")
        const colours = await pg.evaluate(() => {
          const c = (sel) => { const e = document.querySelector(`.code-block code ${sel}`); return e ? getComputedStyle(e).color : null }
          const base = getComputedStyle(document.querySelector(".code-block code")).color
          return { base, builtin: c(".hljs-built_in"), number: c(".hljs-number"), marked: !!document.querySelector(".code-block code.hljs"), bg: getComputedStyle(document.querySelector(".code-block pre")).backgroundColor }
        })
        got[scheme] = colours
        t.ok(`${scheme}: the code is coloured (function and number in the theme's colours, not the text colour)`, colours.marked && colours.builtin === want[scheme].builtin && colours.number === want[scheme].number && colours.builtin !== colours.base, JSON.stringify(colours))
        if (scheme === "dark") {
          await pg.click("[data-code-copy]")
          await pg.waitForTimeout(300)
          const copied = await pg.evaluate(() => navigator.clipboard.readText())
          t.ok("Copy copies the code as plain text, with no markup", copied === "print(1)", JSON.stringify(copied))
          const sel = await pg.evaluate(() => document.querySelector(".code-block code").textContent)
          t.ok("and the text of the block is the code, nothing added", sel === "print(1)", JSON.stringify(sel))
        }
        await ctx.close()
      }
      t.ok("the two themes use different colours", got.light.builtin !== got.dark.builtin && got.light.bg !== got.dark.bg)
    },
  },
  {
    // taking the first prompt back would leave the conversation empty, so it asks before deleting it; and a conversation is in the
    // list from the moment its first prompt is sent
    name: "undoall: taking back the first prompt asks before it deletes the conversation, which is in the list from the first send",
    async run({ browser, fast, long, t, errors }) {
      const pg = await open(browser, errors)
      const side = pg.locator("aside[aria-label='Conversations']")
      const dialog = pg.locator("[role=alertdialog]")
      const undo = async () => { await pg.locator(".msg-in.group").last().hover(); await pg.click("button[aria-label='Take this prompt back']") }
      await pg.goto(fast.base + "/#/chat")
      await pg.evaluate(() => localStorage.clear())
      await pg.reload()
      await pg.waitForSelector("textarea")
      await send(pg, "only prompt")
      await undo()
      await pg.waitForTimeout(400)
      t.ok("taking back the only prompt asks first, and says what happens", (await dialog.count()) === 1 && (await dialog.innerText()).includes("Delete this conversation?") && (await pg.locator(".msg-in.group").count()) === 1)
      t.ok("nothing has been taken back yet", (await side.locator("[data-topic]", { hasText: "only prompt" }).count()) === 1 && (await pg.inputValue("textarea")) === "")
      t.ok("the safe answer has the focus", (await pg.evaluate(() => document.activeElement?.textContent)) === "Cancel")
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(400)
      t.ok("Escape leaves everything as it was", (await dialog.count()) === 0 && (await pg.locator(".msg-in.group").count()) === 1 && (await side.locator("[data-topic]", { hasText: "only prompt" }).count()) === 1)
      await undo()
      await pg.waitForTimeout(300)
      await dialog.getByRole("button", { name: "Cancel", exact: true }).click()
      await pg.waitForTimeout(400)
      t.ok("Cancel does too", (await dialog.count()) === 0 && (await pg.locator(".msg-in.group").count()) === 1)
      await undo()
      await pg.waitForTimeout(300)
      await dialog.getByRole("button", { name: "Delete conversation", exact: true }).click()
      await pg.waitForTimeout(800)
      t.ok("Delete takes the prompt back to the composer and the conversation leaves the list", (await dialog.count()) === 0 && (await pg.inputValue("textarea")) === "only prompt" && (await side.locator("[data-topic]").count()) === 0 && pg.url().endsWith("#/chat"), pg.url())

      await send(pg, "first of two")
      await send(pg, "second of two")
      await undo()
      await pg.waitForFunction(() => document.querySelectorAll(".msg-in.group").length === 1 && !document.querySelector(".ghost"), null, { timeout: 8000 }).catch(() => {})      // it closes up over about 400 ms
      await pg.waitForTimeout(300)
      t.ok("with an earlier prompt there is nothing to ask: the last one is simply taken back", (await dialog.count()) === 0 && (await pg.locator(".msg-in.group").count()) === 1 && (await side.locator("[data-topic]", { hasText: "first of two" }).count()) === 1)
      await pg.context().close()

      const slow = await open(browser, errors)
      await slow.goto(long.base + "/#/chat")
      await slow.evaluate(() => localStorage.clear())
      await slow.reload()
      await slow.waitForSelector("textarea")
      await slow.fill("textarea", "listed at once")
      await slow.keyboard.press("Enter")
      await slow.waitForSelector("button[aria-label='Stop']")
      await slow.waitForTimeout(400)
      const listed = await slow.locator("aside[aria-label='Conversations'] [data-topic]", { hasText: "listed at once" }).count()
      t.ok("a conversation is in the list while its first answer is still being written", listed === 1 && /#\/chat\/.+/.test(slow.url()) && (await slow.locator("button[aria-label='Stop']").count()) === 1, slow.url())
      await slow.click("button[aria-label='Stop']")
      await slow.context().close()

      const th = await open(browser, errors, { lang: "th" })
      await th.goto(fast.base + "/#/chat")
      await th.evaluate(() => localStorage.clear())
      await th.reload()
      await th.waitForSelector("textarea")
      await th.fill("textarea", "คำถามเดียว")
      await th.keyboard.press("Enter")
      await th.waitForFunction(() => !document.querySelector("button[aria-label='หยุด']") && document.querySelector(".prose-chat"), null, { timeout: 60000 })
      await th.waitForTimeout(500)
      await th.locator(".msg-in.group").last().hover()
      await th.click("button[aria-label='เอา prompt นี้กลับมา']")
      await th.waitForTimeout(400)
      t.ok("in Thai the question is Thai", (await th.locator("[role=alertdialog]").innerText()).includes("ลบการสนทนานี้ไหม"))
      await th.context().close()
    },
  },
  {
    // The chat's coding tools (issue #96), against pages whose server answers are scripted (GET /agent, the chat stream, POST /agent/permission):
    // the + menu's row and panel (switch, mode, folder), the request, the cards (Allow, Allow for this chat, Deny, a dangerous one), the diff, the
    // steps, auto mode's verdict, and a server that has none or does not allow this page.
    name: "agent: the coding tools: a panel, cards that ask, the call shown as what it is",
    async run({ browser, fast, t, errors }) {
      const NL = String.fromCharCode(10)
      const info = { available: true, allowed: true, shell: "bash", tools: ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "TodoWrite"] }
      const chunk = (extra) => `data: ${JSON.stringify({ choices: [{ delta: {} }], ...extra })}${NL}${NL}`
      const ev = (e) => chunk({ strata_mcp: e })
      const calls = (...list) => list.map(([id, name, args]) => ev({ event: "start", id, name }) + ev({ event: "call", id, name, server: "agent", tool: name, arguments: args, round: 1 })).join("")
      const tail = `data: ${JSON.stringify({ choices: [{ delta: { content: "ok" } }] })}${NL}${NL}` + `data: ${JSON.stringify({ choices: [], usage: { completion_tokens: 1 } })}${NL}${NL}` + `data: [DONE]${NL}${NL}`
      const script = (body) => ({ status: 200, contentType: "text/event-stream", body })
      const setup = async (pg, { agent = info, stream = [] } = {}) => {
        const sent = [], answers = []
        await pg.route("**/agent", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(agent) }))
        await pg.route("**/agent/permission", (r) => { answers.push(JSON.parse(r.request().postData() || "{}")); return r.fulfill({ status: 200, contentType: "application/json", body: "{\"ok\":true}" }) })
        let n = 0
        await pg.route("**/v1/chat/completions", (r) => { sent.push(JSON.parse(r.request().postData() || "{}")); return r.fulfill(script(stream[Math.min(n++, stream.length - 1)] ?? tail)) })
        return { sent, answers }
      }
      const menu = async (pg) => { await pg.click("button[aria-label='Photos, files, new chat, save']"); await pg.waitForTimeout(500) }
      const row = (pg) => pg.locator(".t-morph-menu [role=option]", { hasText: "Coding tools" })
      const typeAndSend = async (pg, text) => { await pg.fill("textarea[aria-label='Message']", text); await pg.keyboard.press("Enter"); await pg.waitForTimeout(900) }

      const heightsNow = (pg, sel = "[role=dialog][aria-label='Coding tools']") => pg.evaluate(async (q) => {
        const el = document.querySelector(q)
        const seen = []
        const t0 = performance.now()
        while (performance.now() - t0 < 800) { seen.push(Math.round(el.getBoundingClientRect().height)); await new Promise((r) => requestAnimationFrame(r)) }
        return seen
      }, sel)
      // ---- the row and the panel
      const pg = await open(browser, errors)
      const { sent, answers } = await setup(pg, {
        stream: [
          calls(["c1", "Bash", { command: "npm test" }]) + ev({ event: "permission", id: "q1", call_id: "c1", tool: "Bash", arguments: { command: "npm test" }, why: "a command asks every time", danger: false, rule: "Bash(npm test:*)" }) + tail,
          calls(["c2", "Bash", { command: "rm -rf ~" }]) + ev({ event: "permission", id: "q2", call_id: "c2", tool: "Bash", arguments: { command: "rm -rf ~" }, why: "this command can do harm that is hard to undo", danger: true, rule: null }) + tail,
          calls(["c3", "Edit", { file_path: "src/a.py", old_string: "keep\nold\nend", new_string: "keep\nnew\nend" }]) + ev({ event: "judging", id: "q3", call_id: "c3", tool: "Edit" }) + ev({ event: "judged", id: "q3", call_id: "c3", verdict: "allow", severity: 1 })
            + ev({ event: "result", id: "c3", ok: true, text: "The file src/a.py has been updated successfully", chars: 46, truncated: false, ms: 12 })
            + ev({ event: "todos", call_id: "c4", todos: [{ content: "Write the test", status: "completed", activeForm: "Writing the test" }, { content: "Fix the bug", status: "in_progress", activeForm: "Fixing the bug" }, { content: "Run everything", status: "pending", activeForm: "Running everything" }] }) + tail,
        ],
      })
      await pg.goto(fast.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(1200)
      const chip = pg.locator("button[aria-label='Coding mode']")
      const modeMenu = pg.locator("[role=dialog][aria-label='Coding mode']")
      t.ok("the mode is on the prompt bar itself, as a chip that says which mode it is", (await chip.count()) === 1 && (await chip.innerText()).trim() === "Ask" && (await chip.getAttribute("aria-expanded")) === "false" && (await chip.getAttribute("data-agent-mode")) === "ask")
      const chipBox = await chip.boundingBox(), effortBox = await pg.locator("button[aria-label='Thinking effort']").boundingBox()
      t.ok("the chip sits in front of the thinking effort, on the same line", chipBox.x + chipBox.width <= effortBox.x + 1 && Math.abs(chipBox.y - effortBox.y) < 3, JSON.stringify([chipBox, effortBox]))
      t.ok("the sampling button is no longer on the bar", (await pg.locator(".prompt-bar__bar button[aria-label='Sampling']").count()) === 0)
      await menu(pg)
      const samplingRow = pg.locator(".t-morph-menu [role=option]", { hasText: "Sampling" })
      t.ok("it is a row of the + menu now, with what it sets", (await samplingRow.count()) === 1 && (await samplingRow.innerText()).toLowerCase().includes("temperature"), (await samplingRow.innerText()).split(NL).join(" | "))
      await samplingRow.click()
      await pg.waitForTimeout(600)
      t.ok("choosing it opens the sampling settings and closes the menu", (await pg.locator("aside[aria-label='Sampling']").getAttribute("aria-hidden")) === "false" && (await pg.locator(".t-morph[data-open='true']").count()) === 0)
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(500)
      t.ok("and Escape closes it again", (await pg.locator("aside[aria-label='Sampling']").getAttribute("aria-hidden")) === "true")
      await pg.click("textarea[aria-label='Message']")
      await chip.click()
      await pg.waitForTimeout(500)
      t.ok("it opens the three modes with what the current one means", (await modeMenu.count()) === 1 && (await modeMenu.locator("[role=radio]").allInnerTexts()).join(",") === "Ask,Plan,Auto" && (await modeMenu.locator("[data-agent-mode-text]").innerText()).includes("Files inside the folder") && (await chip.getAttribute("aria-expanded")) === "true")
      await modeMenu.locator("[role=radio]", { hasText: "Plan" }).click()
      await pg.waitForTimeout(300)
      t.ok("choosing a mode says what it means and the chip follows at once", (await modeMenu.locator("[data-agent-mode-text]").innerText()).includes("Nothing is changed") && (await chip.innerText()).trim() === "Plan" && (await chip.getAttribute("data-agent-mode")) === "plan")
      await modeMenu.locator("[role=radio]", { hasText: "Auto" }).click()
      await pg.waitForTimeout(300)
      t.ok("auto says a second check decides", (await modeMenu.locator("[data-agent-mode-text]").innerText()).includes("second check") && (await chip.innerText()).trim() === "Auto")
      await pg.keyboard.press("Escape")
      await pg.waitForTimeout(300)
      t.ok("Escape closes it and the mode stays", (await modeMenu.count()) === 0 && (await chip.innerText()).trim() === "Auto")
      await menu(pg)
      const r = row(pg)
      const rt = async () => (await r.innerText()).split(NL).join(" | ")
      t.ok("the + menu has a Coding tools row that says it is on and that no folder is set", (await r.count()) === 1 && (await rt()).includes("on") && (await rt()).includes("No folder yet"), await rt())
      await r.click()
      await pg.waitForTimeout(500)
      const panel = pg.locator("[role=dialog][aria-label='Coding tools']")
      t.ok("clicking it opens the panel with a switch and the folder, and no modes (they are on the bar)", (await panel.count()) === 1 && (await panel.locator("[role=switch][aria-label='Coding tools']").getAttribute("aria-checked")) === "true" && (await panel.locator("[role=radio]").count()) === 0 && (await panel.locator("[data-agent-mode-text]").count()) === 0)
      const saveBtn = panel.getByRole("button", { name: "Save", exact: true })
      t.ok("Save is off until the folder changes", await saveBtn.isDisabled())
      await panel.locator("input[aria-label='Folder the tools work in']").fill("C:/work/app")
      await saveBtn.click()
      await pg.waitForTimeout(300)
      const saved = await pg.evaluate(() => JSON.parse(localStorage.getItem("strata.sampling") || "{}"))
      t.ok("the folder and the mode are kept (this chat is in no project, so it is the default folder)", saved.agentFolder === "C:/work/app" && saved.agentMode === "auto", JSON.stringify([saved.agentFolder, saved.agentMode]))
      t.ok("and the panel says whose folder it is", (await panel.innerText()).includes("no project"))
      await chip.click()
      await pg.waitForTimeout(400)
      t.ok("the chip closes the panel and opens the modes instead", (await panel.count()) === 0 && (await modeMenu.count()) === 1)
      await modeMenu.locator("[role=radio]", { hasText: "Ask" }).click()
      await pg.waitForTimeout(200)
      await pg.click("textarea[aria-label='Message']")

      // ---- on a narrow screen the description of Auto is longer than Plan's: it stretches there, and what is the same stays where it was
      const pn = await open(browser, errors, { width: 360, height: 700 })
      await setup(pn)
      await pn.goto(fast.base + "/#/chat")
      await pn.waitForSelector("textarea")
      await pn.waitForTimeout(1200)
      const pchip = pn.locator("button[aria-label='Coding mode']")
      await pchip.click()
      await pn.waitForTimeout(500)
      const pmode = pn.locator("[role=dialog][aria-label='Coding mode']")
      await pmode.locator("[role=radio]", { hasText: "Plan" }).click()
      await pn.waitForTimeout(700)
      const watching = pn.evaluate(async () => {
        const box = document.querySelector("[data-agent-mode-text]")
        const menu = document.querySelector("[role=dialog][aria-label='Coding mode']")
        const top = (q) => Math.round(document.querySelector(q).getBoundingClientRect().top)
        const rows = []
        const t0 = performance.now()
        while (performance.now() - t0 < 900) {
          rows.push({ box: Math.round(box.getBoundingClientRect().height), modes: top("[role=dialog][aria-label='Coding mode'] [role=radiogroup]"), bottom: Math.round(menu.getBoundingClientRect().bottom) })
          await new Promise((r) => requestAnimationFrame(r))
        }
        return rows
      })
      await pmode.locator("[role=radio]", { hasText: "Auto" }).click()
      const w = await watching
      const boxes = w.map((x) => x.box), modes = w.map((x) => x.modes), bottoms = w.map((x) => x.bottom)
      t.ok("the description stretches where the text gets longer, through in-between heights", new Set(boxes).size > 3 && boxes.at(-1) > boxes[0] && boxes.slice(1, -1).some((h) => h > boxes[0] && h < boxes.at(-1)), JSON.stringify([...new Set(boxes)]))
      t.ok("from there upward: what is above it moves up as it grows", modes[0] - modes.at(-1) > 10 && modes.slice(1, -1).some((m) => m < modes[0] && m > modes.at(-1)), JSON.stringify([...new Set(modes)]))
      t.ok("and the bottom of the menu, with what is below it, stays exactly where it was", Math.max(...bottoms) - Math.min(...bottoms) <= 1, JSON.stringify([...new Set(bottoms)]))
      await pn.keyboard.press("Escape")
      await pn.waitForTimeout(300)
      await menu(pn)
      await row(pn).click()
      await pn.waitForTimeout(500)
      const pnl = pn.locator("[role=dialog][aria-label='Coding tools']")
      // a part that is added (here a row put in at the top of the panel) stretches open from nothing, and what is below it stays
      await pn.waitForTimeout(300)
      const adding = await pn.evaluate(async () => {
        const inner = document.querySelector("[role=dialog][aria-label='Coding tools'] [data-agent-controls]").parentElement
        const row = document.createElement("div")
        row.textContent = "a row that comes"
        row.style.height = "24px"
        const folder = document.querySelector("input[aria-label='Folder the tools work in']")
        const seen = []
        inner.insertBefore(row, inner.firstChild)
        const t0 = performance.now()
        while (performance.now() - t0 < 900) { seen.push({ h: Math.round(row.getBoundingClientRect().height), folder: Math.round(folder.getBoundingClientRect().top) }); await new Promise((r) => requestAnimationFrame(r)) }
        row.remove()
        return seen
      })
      const rowH = adding.map((x) => x.h), rowFolders = adding.map((x) => x.folder)
      t.ok("a part that is added opens from nothing to its height through in-between heights", new Set(rowH).size > 3 && rowH.at(-1) === 24 && rowH.slice(0, -1).some((h) => h > 0 && h < 24), JSON.stringify([...new Set(rowH)]))
      t.ok("and what is below it does not move", Math.max(...rowFolders) - Math.min(...rowFolders) <= 1, JSON.stringify([...new Set(rowFolders)]))

      // ---- the request, and a card that asks
      await typeAndSend(pg, "run the tests")
      t.ok("the request names the folder, the mode and the chat", sent.length === 1 && sent[0].strata_agent?.cwd === "C:/work/app" && sent[0].strata_agent?.mode === "ask" && !!sent[0].strata_agent?.session && JSON.stringify(sent[0].strata_agent?.allow) === "[]", JSON.stringify(sent[0]?.strata_agent))
      const callBox = pg.locator("[data-agent-call='Bash']").first()
      const card = callBox.locator("[data-agent-ask]")
      t.ok("the call shows as Bash with the command, and says it waits for the user", (await callBox.count()) === 1 && (await callBox.innerText()).includes("npm test") && (await callBox.innerText()).includes("Waiting for you"))
      t.ok("its card asks, shows the command and why, and has Allow, Allow for this chat and Deny", (await card.count()) === 1 && (await card.innerText()).includes("Run this command?") && (await card.innerText()).includes("A command asks every time.") && (await card.getByRole("button", { name: "Allow", exact: true }).count()) === 1 && (await card.getByRole("button", { name: "Allow for this chat" }).count()) === 1 && (await card.getByRole("button", { name: "Deny", exact: true }).count()) === 1)
      t.ok("the rule it would remember is shown", (await card.innerText()).includes("Bash(npm test:*)"))
      await card.getByRole("button", { name: "Allow for this chat" }).click()
      await pg.waitForTimeout(500)
      t.ok("the answer goes to the server with the id of the question", answers.length === 1 && answers[0].id === "q1" && answers[0].decision === "allow_chat", JSON.stringify(answers))
      t.ok("the card is gone and says what the user did", (await card.count()) === 0 && (await callBox.innerText()).includes("You allowed it for this chat."))
      await typeAndSend(pg, "again")
      t.ok("the rule goes with the next request of this chat", JSON.stringify(sent[1].strata_agent?.allow) === JSON.stringify(["Bash(npm test:*)"]), JSON.stringify(sent[1]?.strata_agent?.allow))

      // ---- a dangerous command
      const danger = pg.locator("[data-agent-call='Bash']").nth(1).locator("[data-agent-ask]")
      t.ok("a dangerous command says so, and cannot be allowed for the whole chat", (await danger.count()) === 1 && (await danger.innerText()).includes("hard to undo") && (await danger.getByRole("button", { name: "Allow for this chat" }).count()) === 0)
      await danger.getByRole("button", { name: "Deny", exact: true }).click()
      await pg.waitForTimeout(400)
      t.ok("Deny is sent as deny", answers.length === 2 && answers[1].id === "q2" && answers[1].decision === "deny")

      // ---- an edit as a diff, auto mode's verdict, the steps
      await typeAndSend(pg, "fix it")
      const edit = pg.locator("[data-agent-call='Edit']")
      t.ok("an Edit shows its file and what auto mode found", (await edit.count()) === 1 && (await edit.innerText()).includes("src/a.py") && (await edit.locator("[data-agent-judge='allow']").innerText()).includes("safe (1/5)"))
      await edit.locator("button[aria-expanded]").first().click()
      await pg.waitForTimeout(500)
      const rows = await edit.locator("[data-diff]").evaluateAll((els) => els.map((e) => [e.dataset.diff, e.textContent.trim()]))
      t.ok("opened, an Edit is a diff: what stays, what goes, what comes", JSON.stringify(rows) === JSON.stringify([["same", "keep"], ["del", "- old"], ["add", "+ new"], ["same", "end"]]), JSON.stringify(rows))
      const todos = pg.locator("[data-todos]")
      t.ok("the steps are a list: done, the one in progress by its -ing form, and what is left", (await todos.count()) === 1 && (await todos.innerText()).includes("1 of 3 steps done") && (await todos.locator("[data-todo='in_progress']").innerText()).includes("Fixing the bug") && (await todos.locator("[data-todo='completed']").count()) === 1 && (await todos.locator("[data-todo='pending']").count()) === 1)

      // ---- forgetting the rules, and the switch
      await menu(pg)
      await row(pg).click()
      await pg.waitForTimeout(400)
      t.ok("the panel counts the rules of this chat and can forget them", (await panel.innerText()).includes("1 rules allowed for this chat"))
      const sampling = heightsNow(pg)
      await panel.getByRole("button", { name: "Forget them" }).click()
      const hs = await sampling
      t.ok("a part that goes shrinks smoothly too: the panel's height takes in-between values as the rows round it follow", new Set(hs).size > 3 && hs.at(-1) < hs[0] && hs.slice(1, -1).some((h) => h < hs[0] && h > hs.at(-1)), JSON.stringify([...new Set(hs)]))
      await pg.waitForTimeout(300)
      t.ok("forgotten", !(await panel.innerText()).includes("rules allowed"))
      await panel.locator("[role=switch][aria-label='Coding tools']").click()
      await pg.waitForTimeout(300)
      t.ok("switched off, the mode chip is gone from the bar", (await chip.count()) === 0)
      await pg.click("textarea[aria-label='Message']")
      await typeAndSend(pg, "plain question")
      t.ok("switched off, the request has no strata_agent", !("strata_agent" in sent.at(-1)))

      // ---- a server that does not have them, and one that does not allow this page
      for (const [label, agent, text] of [["has none", { available: false, allowed: false, shell: null, tools: [] }, "Not on this server"], ["does not allow this page", { available: true, allowed: false, shell: "bash", tools: ["Read"], reason: "from this PC only" }, "Only from the PC that runs Strata"]]) {
        const p2 = await open(browser, errors)
        const s2 = await setup(p2, { agent })
        await p2.goto(fast.base + "/#/chat")
        await p2.waitForSelector("textarea")
        await p2.waitForTimeout(1200)
        await menu(p2)
        t.ok(`a server that ${label}: the row says so`, (await row(p2).innerText()).includes(text), (await row(p2).innerText()).split(NL).join(" | "))
        await p2.keyboard.press("Escape")
        await typeAndSend(p2, "hi")
        t.ok(`and the request has no strata_agent (${label})`, s2.sent.length === 1 && !("strata_agent" in s2.sent[0]))
      }

      // ---- Thai
      const th = await open(browser, errors, { lang: "th" })
      await setup(th)
      await th.goto(fast.base + "/#/chat")
      await th.waitForSelector("textarea")
      await th.waitForTimeout(1200)
      await th.click(".t-morph-plus")
      await th.waitForTimeout(500)
      t.ok("in Thai the row has a Thai name", (await th.locator(".t-morph-menu [role=option]", { hasText: "เครื่องมือเขียนโค้ด" }).count()) === 1)
    },
  },
  {
    // The whole path with a scripted model (the mock server with STRATA_MOCK_AGENT=1): the real server, the real tools and the real permission cards. A
    // message with "agent demo" makes the model call TodoWrite, Glob, Read, a read-only Bash, a Write outside the folder and a Bash that asks.
    name: "agentdemo: the chat uses the real coding tools: free steps run, a file outside the folder and a command ask first",
    async run({ browser, agentDemo, t, errors }) {
      const fs = await import("node:fs")
      const os = await import("node:os")
      const path = await import("node:path")
      const outside = path.join(os.tmpdir(), "strata-agent-demo.txt")
      const inside = path.join(agentDemo.dir, "strata-agent-demo.txt")
      fs.rmSync(outside, { force: true })
      fs.rmSync(inside, { force: true })
      const pg = await open(browser, errors)
      await pg.addInitScript((dir) => { try { if (!localStorage.getItem("strata.sampling")) localStorage.setItem("strata.sampling", JSON.stringify({ agentFolder: dir })) } catch { /* private window */ } }, agentDemo.dir)
      await pg.goto(agentDemo.base + "/#/chat")
      await pg.waitForSelector("textarea[aria-label='Message']")
      await pg.waitForTimeout(1200)
      await pg.fill("textarea[aria-label='Message']", "agent demo please")
      await pg.keyboard.press("Enter")
      const card = pg.locator("[data-agent-ask]")
      await card.first().waitFor({ timeout: 30000 })
      const todos = pg.locator("[data-todos]")
      t.ok("the steps the model wrote are a list, and the free steps ran before anything asked", (await todos.count()) === 1 && (await pg.locator("[data-agent-call='Glob']").count()) === 1 && (await pg.locator("[data-agent-call='Read']").count()) === 1)
      await pg.locator("[data-agent-call='Read'] button[aria-expanded]").click()
      await pg.waitForTimeout(500)
      t.ok("a Read inside the folder ran without asking and shows the file", (await pg.locator("[data-agent-call='Read']").innerText()).includes("demo project"))
      t.ok("a command that only reads ran without asking too", (await pg.locator("[data-agent-call='Bash']").first().locator("[data-agent-ask]").count()) === 0)
      t.ok("the first card is the file outside the folder, and says why", (await card.count()) === 1 && (await card.first().innerText()).includes("Write this file?") && (await card.first().innerText()).includes("outside the project folder") && (await card.first().innerText()).includes("strata-agent-demo.txt"))
      t.ok("nothing has been written yet", !fs.existsSync(outside))
      await card.first().getByRole("button", { name: "Allow", exact: true }).click()
      await pg.waitForFunction(() => document.body.innerText.includes("Make a file here"), null, { timeout: 30000 })
      t.ok("allowed, the file was written", fs.existsSync(outside) && fs.readFileSync(outside, "utf8").includes("scripted model"))
      const cmd = pg.locator("[data-agent-ask]")
      await cmd.first().waitFor({ timeout: 30000 })
      t.ok("then a command that is not a plain read asks, with the command and a rule to remember", (await cmd.first().innerText()).includes("Run this command?") && (await cmd.first().innerText()).includes("touch strata-agent-demo.txt") && (await cmd.first().innerText()).includes("Bash(touch:*)"))
      t.ok("and the file it would make is not there yet", !fs.existsSync(inside))
      await cmd.first().getByRole("button", { name: "Allow for this chat" }).click()
      await pg.waitForFunction(() => document.body.innerText.includes("That was the demo of the coding tools"), null, { timeout: 30000 })
      t.ok("allowed for this chat, the command ran", fs.existsSync(inside))
      const rules = await pg.evaluate(() => JSON.parse(localStorage.getItem("strata.agent.rules") || "{}"))
      t.ok("and the rule is kept for the chat", Object.values(rules.rules || {}).flat().includes("Bash(touch:*)"), JSON.stringify(rules))
      t.ok("no card is left waiting, and the calls ended as done", (await pg.locator("[data-agent-ask]").count()) === 0 && (await pg.locator("[data-agent-call][data-state='asking']").count()) === 0)
      fs.rmSync(outside, { force: true })
    },
  },
  {
    // "/" at the start of the prompt lists the skills in use (the importer mock has the fake home's), to be picked by name; a message that starts
    // with a skill's name asks the server to load it. Without skills (the other mocks have none) "/" is only a character.
    name: "slash: / in the prompt lists the skills in use and a message that starts with one loads it",
    async run({ browser, importer, fast, t, errors }) {
      const pg = await open(browser, errors)
      const box = pg.locator("textarea[aria-label='Message']")
      const menu = pg.locator("[role=listbox][aria-label='Skills']")
      const names = () => menu.locator("[role=option]").evaluateAll((os) => os.map((o) => o.dataset.skill))
      const bodies = []
      pg.on("request", (r) => { if (r.url().endsWith("/v1/chat/completions") && r.method() === "POST") bodies.push(JSON.parse(r.postData() || "{}")) })
      await pg.goto(importer.base + "/#/chat")
      await box.waitFor()
      await pg.waitForTimeout(700)
      await box.fill("/")
      await menu.waitFor({ timeout: 5000 })
      const all = await names()
      t.ok("a / lists the skills in use, one of each name, by name", all.includes("pdf-tools") && all.includes("agents-skill") && all.filter((n) => n === "shared-skill").length === 1 && JSON.stringify(all) === JSON.stringify([...all].sort((a, b) => a.localeCompare(b))), JSON.stringify(all.slice(0, 5)))
      t.ok("each says what it does", (await menu.locator("[data-skill='pdf-tools']").innerText()).includes("Read and write PDF files"))
      await box.fill("/pdf")
      await pg.waitForTimeout(150)
      t.ok("typing narrows the list", JSON.stringify(await names()) === '["pdf-tools"]', JSON.stringify(await names()))
      await box.fill("/zzz")
      await pg.waitForTimeout(150)
      t.ok("a name nothing matches closes it", (await menu.count()) === 0)
      await box.fill("/")
      await menu.waitFor()
      await pg.keyboard.press("Escape")
      t.ok("Escape closes it and keeps what was typed", (await menu.count()) === 0 && (await box.inputValue()) === "/")
      await box.fill("see /pdf")
      await pg.waitForTimeout(150)
      t.ok("a / that is not at the start opens nothing", (await menu.count()) === 0)
      await box.fill("/bulk-0")
      await menu.waitFor()
      await pg.keyboard.press("ArrowDown")
      await pg.waitForTimeout(100)
      t.ok("the arrow keys move the mark, which says so", (await menu.locator("[role=option][aria-selected=true]").count()) === 1 && (await menu.locator("[role=option][aria-selected=true]").getAttribute("data-skill")) === (await names())[1], (await names()).slice(0, 3).join(","))
      await pg.keyboard.press("Enter")
      await pg.waitForTimeout(150)
      t.ok("Enter picks it: its name and a space are in the prompt, nothing was sent", /^\/bulk-\d\d $/.test(await box.inputValue()) && bodies.length === 0 && (await menu.count()) === 0, await box.inputValue())
      await box.fill("/")
      await menu.waitFor()
      await menu.locator("[data-skill='pdf-tools']").click()
      await pg.waitForTimeout(150)
      t.ok("a click picks one too, and the field keeps the focus with the caret after the name", (await box.inputValue()) === "/pdf-tools " && (await pg.evaluate(() => document.activeElement?.getAttribute("aria-label"))) === "Message" && (await box.evaluate((e) => e.selectionStart)) === 11)
      await pg.keyboard.type("make a pdf")
      const mark = pg.locator(".prompt-bar__mirror [data-mark]")
      const tip = pg.locator(".prompt-bar [role=tooltip]")
      t.ok("in the prompt the command is marked in bold, and the text itself is unchanged", (await mark.count()) === 1 && (await mark.innerText()) === "/pdf-tools" && (await mark.evaluate((e) => parseFloat(getComputedStyle(e).webkitTextStrokeWidth))) > 0 && (await box.inputValue()) === "/pdf-tools make a pdf")
      t.ok("the marked command lies exactly over the text typed under it", await pg.evaluate(() => {
        const m = document.querySelector(".prompt-bar__mirror"), a = document.querySelector("textarea[aria-label='Message']")
        const r1 = m.getBoundingClientRect(), r2 = a.getBoundingClientRect()
        return Math.abs(r1.left - r2.left) < 1 && Math.abs(r1.top - r2.top) < 1 && Math.abs(r1.width - r2.width) < 1 && Math.abs(r1.height - r2.height) < 1 && getComputedStyle(m).fontSize === getComputedStyle(a).fontSize && getComputedStyle(m).lineHeight === getComputedStyle(a).lineHeight
      }))
      t.ok("no card until the pointer is on it", (await tip.count()) === 0)
      const at = await mark.boundingBox()
      await pg.mouse.move(at.x + at.width / 2, at.y + at.height / 2)
      await tip.waitFor({ timeout: 3000 })
      const said = await tip.innerText()
      t.ok("pointing at it says which skill it is, what it does and where it comes from", said.includes("/pdf-tools") && said.includes("Read and write PDF files") && said.includes("Skill from Claude Code"), said.split(String.fromCharCode(10)).join(" | "))
      await pg.mouse.move(at.x + at.width + 120, at.y + at.height / 2)
      await pg.waitForTimeout(250)
      t.ok("and the card goes when the pointer leaves it", (await tip.count()) === 0)
      await pg.keyboard.press("Enter")
      await pg.waitForFunction(() => !document.querySelector("[aria-label='Stop']"), null, { timeout: 20000 })
      t.ok("the request names the skill, and the message is what the user typed", bodies.length === 1 && bodies[0].strata_skill === "pdf-tools" && bodies[0].messages.at(-1).content === "/pdf-tools make a pdf", JSON.stringify(bodies[0] && [bodies[0].strata_skill, bodies[0].messages?.at(-1)?.content]))
      t.ok("and it is shown as typed", (await pg.locator("main").innerText()).includes("/pdf-tools make a pdf"))
      const tag = pg.locator("[data-skill-tag='pdf-tools']")
      t.ok("in the conversation the command is in bold and the rest is not", (await tag.count()) === 1 && (await tag.evaluate((e) => parseInt(getComputedStyle(e).fontWeight))) >= 600 && (await tag.evaluate((e) => parseInt(getComputedStyle(e.parentElement).fontWeight))) < 600)
      const card = pg.locator(".skill-tip")
      t.ok("its card is not there until the pointer is on it", (await card.count()) === 0)
      await tag.hover()
      await card.waitFor({ timeout: 3000 })
      await pg.waitForTimeout(350)
      const said2 = await card.innerText()
      const at2 = await card.boundingBox()
      t.ok("then it says which skill it is, what it does and where it comes from", said2.includes("/pdf-tools") && said2.includes("Read and write PDF files") && said2.includes("Skill from Claude Code"), said2.split(String.fromCharCode(10)).join(" | "))
      t.ok("and it is really on the screen (not cut off by the boxes round the bubble), inside the page", await pg.evaluate(() => { const c = document.querySelector(".skill-tip"), r = c.getBoundingClientRect(); return c.parentElement === document.body && getComputedStyle(c).opacity === "1" && r.width > 0 && r.left >= 0 && r.right <= innerWidth && r.bottom <= innerHeight }), JSON.stringify(at2))
      await pg.mouse.move(5, 400)
      await pg.waitForTimeout(150)
      t.ok("and it goes when the pointer leaves", (await card.count()) === 0)
      await box.fill("and then?")
      await pg.keyboard.press("Enter")
      await pg.waitForFunction(() => !document.querySelector("[aria-label='Stop']"), null, { timeout: 20000 })
      t.ok("a later plain message does not ask for it again", bodies.length === 2 && !("strata_skill" in bodies[1]))
      await box.fill("/nope go")
      await pg.keyboard.press("Enter")
      await pg.waitForFunction(() => !document.querySelector("[aria-label='Stop']"), null, { timeout: 20000 })
      t.ok("a name that is no skill is only text", bodies.length === 3 && !("strata_skill" in bodies[2]))
      t.ok("and it is not in bold", (await pg.locator("[data-skill-tag='nope']").count()) === 0 && (await pg.locator("[data-skill-tag]").count()) === 1)
      await box.fill("/nope go")
      t.ok("nor marked in the prompt", (await pg.locator(".prompt-bar__mirror").count()) === 0)

      const plain = await open(browser, errors)
      await plain.goto(fast.base + "/#/chat")
      await plain.locator("textarea[aria-label='Message']").waitFor()
      await plain.waitForTimeout(700)
      await plain.locator("textarea[aria-label='Message']").fill("/")
      await plain.waitForTimeout(250)
      t.ok("with no skills at all a / lists only Strata's own commands (clear, compact, context, init, memory, permissions, rewind)", JSON.stringify(await plain.locator("[role=listbox][aria-label='Skills'] [role=option]").evaluateAll((os) => os.map((o) => o.dataset.skill))) === '["clear","compact","context","init","memory","permissions","rewind"]')
    },
  },
  {
    // Settings > Import (#94), against a mock that reads a FAKE home folder: skills of other apps are on until switched off (the whole import,
    // an app, a skill); MCP servers of other apps are listed by app and imported by a click, and nothing starts before that
    name: "import: skills of other apps are on until switched off, MCP servers of other apps are imported by a click",
    async run({ browser, importer, t, errors }) {
      const fs = await import("node:fs")
      const saved = () => JSON.parse(fs.readFileSync(importer.config, "utf8"))
      const pg = await open(browser, errors)
      const sw = (label) => pg.locator(`button[role=switch][aria-label='${label}']`)
      const on = async (label) => (await sw(label).getAttribute("aria-checked")) === "true"
      const summary = () => pg.locator("main").innerText().then((x) => (x.match(/(\d+) skills in use \((\d+) found in (\d+) apps\)/) || []).slice(1).join("/"))
      const served = (path) => pg.evaluate(async (p) => (await fetch(p)).json(), path)
      await pg.goto(importer.base + "/#/settings/import-skills")
      await pg.waitForSelector("[data-harness='claude']")
      await pg.waitForTimeout(400)
      t.ok("the skills of the other apps are in use from the start, one of each name", (await summary()) === "64/65/3", await summary())
      t.ok("the whole import, and each app, is on", (await on("Import skills")) && (await on("Import from Claude Code")) && (await on("Import from Codex")) && (await on("Import from Shared agents folder")))
      t.ok("an app that has nothing is not listed as one that has", (await pg.locator("[data-harness='cursor']").count()) === 0)
      t.ok("the files it was read from are named", (await pg.locator("[data-harness='codex']").innerText()).includes("~/.codex/skills"))
      const tools = (await served("/mcp")).servers.find((s) => s.name === "skills")
      t.ok("the model has the skills as a server with three tools", !!tools && tools.tools.map((x) => x.tool).join(",") === "find_skills,use_skill,read_skill_file", JSON.stringify(tools && tools.tools.map((x) => x.tool)))

      const pageH = await pg.evaluate(() => document.documentElement.scrollHeight)
      await pg.click("[data-harness='agents'] button[aria-expanded]")
      await pg.waitForTimeout(700)
      const win = pg.locator("[data-harness='agents'] [role=region]")
      const m = await win.evaluate((e) => ({ h: Math.round(e.getBoundingClientRect().height), more: e.scrollHeight - e.clientHeight, below: e.hasAttribute("data-below"), above: e.hasAttribute("data-above"), rows: e.querySelectorAll("li[data-skill]").length }))
      t.ok("a long list of skills is a window of a fixed height with more to scroll, not a page-long list", m.h <= 380 && m.more > 200 && m.below && !m.above && (await pg.evaluate(() => document.documentElement.scrollHeight)) - pageH < 460, JSON.stringify(m))
      t.ok("every skill is in it, with no Show all to press", m.rows === 61 && (await pg.getByRole("button", { name: /Show all/ }).count()) === 0, String(m.rows))
      await win.evaluate((e) => { e.scrollTop = e.scrollHeight })
      await pg.waitForTimeout(400)
      const after = await pg.evaluate(() => { const b = document.querySelector("[data-harness='agents'] button[aria-expanded]").getBoundingClientRect(); const w = document.querySelector("[data-harness='agents'] [role=region]"); return { top: Math.round(b.top), bottom: Math.round(b.bottom), vh: innerHeight, above: w.hasAttribute("data-above"), below: w.hasAttribute("data-below"), y: scrollY } })
      t.ok("at the end of the list the heading that closes it is still on screen, and the edges say where there is more", after.top >= 0 && after.bottom <= after.vh && after.above && !after.below, JSON.stringify(after))
      await pg.click("[data-harness='agents'] button[aria-expanded]")
      await pg.waitForTimeout(700)
      t.ok("it closes from there, and the page did not move", (await pg.locator("[data-harness='agents'] [role=region]").count()) === 0 && (await pg.evaluate(() => scrollY)) === after.y)
      await pg.click("[data-harness='codex'] button[aria-expanded]")
      await pg.waitForTimeout(500)
      t.ok("a skill that another app has too is marked as the same as that one, and is not the copy in use", (await pg.locator("li[data-skill='codex:shared-skill']").innerText()).includes("Same as Claude Code") && (await pg.locator("li[data-skill='codex:codex-only']").innerText()).includes("codex-only"))

      await sw("Import from Codex").click()
      await pg.waitForTimeout(700)
      t.ok("switching an app off is saved as an off-list and takes effect", JSON.stringify(saved().import.skills.harness_off) === '["codex"]' && (await summary()) === "63/65/3" && !(await on("Import from Codex")), JSON.stringify(saved().import))
      const seenSwitch = () => pg.evaluate(() => {      // an OFF switch must still be seen: its track and its knob have a colour of their own, on any theme
        const alpha = (c) => { const slash = /\/\s*([\d.]+)\s*\)/.exec(c); if (slash) return parseFloat(slash[1]); const m = /rgba\(([^)]+)\)/.exec(c); if (m) return parseFloat(m[1].split(",")[3]); return /^(rgb|color)\(/.test(c) ? 1 : 0 }       // rgb(), rgba(), and the color(srgb ... / a) a color-mix comes back as
        const off = [...document.querySelectorAll("main [role=switch][aria-checked=false]")]
        return { n: off.length, bad: off.filter((b) => alpha(getComputedStyle(b).backgroundColor) < 0.05 || alpha(getComputedStyle(b.firstElementChild).backgroundColor) < 0.2).length }
      })
      const offSwitches = await seenSwitch()
      t.ok("a switch that is off can still be seen (on a dark screen too): its track and knob are drawn", offSwitches.n > 0 && offSwitches.bad === 0, JSON.stringify(offSwitches))
      t.ok("the rest of the run config is kept", saved().model === "m")
      await pg.click("[data-harness='claude'] button[aria-expanded]")
      await pg.waitForTimeout(500)
      await sw("Use pdf-tools").click()
      await pg.waitForTimeout(700)
      t.ok("one skill off is saved too, by app and name", JSON.stringify(saved().import.skills.off) === '{"claude":["pdf-tools"]}' && (await summary()) === "62/65/3", JSON.stringify(saved().import.skills))
      const served2 = await served("/import")
      t.ok("the server says the same", served2.skills.used === 62 && served2.skills.settings.off.claude[0] === "pdf-tools")

      await pg.fill("input[aria-label='Filter by name or what it does']", "agents-skill")
      await pg.waitForTimeout(400)
      t.ok("the filter keeps what matches, in every app", (await pg.locator("li[data-skill]").count()) === 1 && (await pg.locator("li[data-skill='agents:agents-skill']").count()) === 1)
      await pg.fill("input[aria-label='Filter by name or what it does']", "zzzz")
      await pg.waitForTimeout(300)
      t.ok("and says when nothing does", (await pg.locator("main").innerText()).includes("No skill matches."))
      await pg.fill("input[aria-label='Filter by name or what it does']", "")

      await sw("Import skills").click()
      await pg.waitForTimeout(700)
      t.ok("the whole import off: the other switches cannot be used and the model has no skills", saved().import.skills.enabled === false && (await sw("Import from Claude Code").isDisabled()) && !(await served("/mcp")).servers.some((s) => s.name === "skills"))
      await sw("Import skills").click()
      await pg.waitForTimeout(700)
      t.ok("and on again: the same choices are back", saved().import.skills.enabled === true && (await summary()) === "62/65/3" && (await served("/mcp")).servers.some((s) => s.name === "skills"))

      fs.mkdirSync(`${importer.home}/.claude/skills/later-one`, { recursive: true })
      fs.writeFileSync(`${importer.home}/.claude/skills/later-one/SKILL.md`, "---\nname: later-one\ndescription: turned up later\n---\nx\n", { flag: "w" })
      await pg.click("button:has-text('Rescan')")
      await pg.waitForFunction(() => document.querySelector("main")?.innerText.includes("63 skills in use"), null, { timeout: 15000 })
      t.ok("a rescan finds a skill added since, and it is on; the one switched off stays off", (await summary()) === "63/66/3" && saved().import.skills.off.claude[0] === "pdf-tools")

      // the chat's + menu lists the skills as a server of its own
      await pg.goto(importer.base + "/#/chat")
      await pg.waitForSelector("textarea")
      await pg.waitForTimeout(2000)
      await pg.click("button[aria-label='Photos, files, new chat, save']")
      await pg.waitForTimeout(500)
      await pg.locator(".t-morph-menu [role=option]", { hasText: "MCP tools" }).click()
      await pg.waitForTimeout(500)
      const row = pg.locator("[role=dialog][aria-label='MCP tools'] li[data-server='skills']")
      t.ok("in the chat's MCP list the skills are a server with their three tools, to switch per chat", (await row.count()) === 1 && (await row.locator(".prompt-bar__mcp-tools li").allInnerTexts()).join(",") === "find_skills,use_skill,read_skill_file")

      // MCP servers: listed by app, nothing started, imported by a click
      await pg.goto(importer.base + "/#/settings/import-mcp")
      await pg.waitForSelector("[data-harness='cursor']")
      await pg.waitForTimeout(400)
      const cand = (id) => pg.locator(`li[data-candidate='${id}']`)
      t.ok("the servers are listed by app, with what each is", (await pg.locator("[data-harness='cursor'] li[data-candidate]").count()) === 3 && (await cand("cursor:fake").innerText()).includes("Program"))
      t.ok("nothing was started: Strata runs only its own servers and the skills", (await served("/mcp")).servers.map((s) => s.name).join(",") === "skills")
      t.ok("the command line of a server is shown to who may change things, its secret is not", (await cand("cursor:fake").innerText()).includes("mcp_fake_server.py") && !(await pg.content()).includes(importer.secret))
      t.ok("one that cannot be imported says why and has no button", (await cand("codex:nothing").innerText()).includes("no command or address") && (await cand("gemini:g-sse").innerText()).includes("SSE") && (await cand("codex:nothing").locator("button").count()) === 0)
      t.ok("one that is off in its own app says so", (await cand("codex:off-here").innerText()).includes("Off in Codex"))

      await pg.click("button[aria-label='Import fake']")
      await pg.waitForFunction(() => document.querySelector("li[data-candidate='cursor:fake']")?.innerText.includes("Already in Strata as fake"), null, { timeout: 20000 })
      t.ok("importing copies it into Strata's own servers, and the row says it is there", saved().mcp_servers.fake.command === saved().mcp_servers.fake.command && saved().mcp_servers.fake.args.length === 1 && saved().mcp_servers.fake.disabled === undefined)
      let ready = false
      for (let i = 0; i < 40 && !ready; i++) { ready = ((await served("/mcp")).servers.find((s) => s.name === "fake") || {}).status === "ready"; if (!ready) await pg.waitForTimeout(500) }
      t.ok("and it runs now, like any server of Strata's", ready)
      await pg.click("button[aria-label='Import quiet']")
      await pg.waitForFunction(() => document.querySelector("li[data-candidate='cursor:quiet']")?.innerText.includes("Already in Strata"), null, { timeout: 15000 })
      t.ok("one that was off in its own app is imported off", saved().mcp_servers.quiet.disabled === true)
      await pg.click("button[aria-label='Import secret-off']")
      await pg.waitForFunction(() => document.querySelector("li[data-candidate='cursor:secret-off']")?.innerText.includes("Already in Strata"), null, { timeout: 15000 })
      t.ok("a secret is copied on the server into the run config and never shown on the page", saved().mcp_servers["secret-off"].env.TOKEN === importer.secret && !(await pg.content()).includes(importer.secret))
      t.ok("the imported servers are in MCP tools > Servers", (await (async () => { await pg.goto(importer.base + "/#/settings/mcp-servers"); await pg.waitForSelector("li[data-server='fake']", { timeout: 15000 }); return pg.locator("li[data-server='quiet']").count() })()) === 1)
      await pg.context().close()

      // where nothing can be changed, it is a list that says why
      const ro = await open(browser, errors)
      const view = { available: true, editable: false, reason: "x", config_file: "run.json",
        harnesses: [{ id: "claude", label: "Claude Code", found: true, files: ["~/.claude/skills"], errors: [], skills: 1, skills_used: 1, skills_off: false, servers: 1 }],
        skills: { settings: { enabled: true, harness_off: [], off: {} }, used: 1, total: 1, items: [{ id: "claude:a", harness: "claude", name: "a", description: "d", origin: "user", off: false, used: true, same_as: null }] },
        mcp: { harnesses: [{ id: "claude", label: "Claude Code", servers: [{ id: "claude:s", harness: "claude", name: "s", kind: "program", state: "available", reason: null, note: null, disabled_in_source: false, already_as: null }] }] } }
      await ro.route("**/import", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(view) }))
      await ro.goto(importer.base + "/#/settings/import-skills")
      await ro.waitForSelector("[data-harness='claude']")
      t.ok("without the right to change it, the skills are a list that says why, and no switch can be used", (await ro.locator("[role=note]").innerText()).includes("only from this PC") && (await ro.locator("button[role=switch]").evaluateAll((els) => els.every((e) => e.disabled))))
      await ro.goto(importer.base + "/#/settings/import-mcp")
      await ro.waitForSelector("li[data-candidate]")
      t.ok("and the servers show no command line and cannot be imported", (await ro.locator("li[data-candidate]").innerText()).trim().split(String.fromCharCode(10)).join(" ").indexOf("npx") < 0 && (await ro.locator("button[aria-label='Import s']").isDisabled()))
      await ro.context().close()

      const ph = await open(browser, errors, { width: 390, height: 800 })
      await ph.goto(importer.base + "/#/settings/import-skills")
      await ph.waitForSelector("[data-harness='claude']")
      await ph.click("[data-harness='claude'] button[aria-expanded]")
      await ph.waitForTimeout(700)
      t.ok("on a phone it fits the screen", (await ph.evaluate(() => document.documentElement.scrollWidth - innerWidth)) <= 1)
      await ph.context().close()

      const th = await open(browser, errors, { lang: "th" })
      await th.goto(importer.base + "/#/settings/import-skills")
      await th.waitForSelector("[data-harness='claude']")
      t.ok("in Thai the page is Thai", (await th.locator("main h2").innerText()) === "Skill จากแอปอื่น")
      await th.context().close()
    },
  },
]
