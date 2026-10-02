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
      t.ok("the four actions are there to use", (await pg.locator(".t-morph-menu [role=option]").count()) === 4 && (await pg.locator(".t-morph-menu [role=option]").first().isVisible()))
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
        const h = [], x = { left: [], right: [] }
        f.querySelector("[role=radio][aria-checked=false]").click()
        const t0 = performance.now()
        while (performance.now() - t0 < 900) {
          h.push(Math.round(f.getBoundingClientRect().height))
          for (const p of f.querySelectorAll(".kind-pane")) x[p.dataset.side].push(Math.round(p.getBoundingClientRect().left))      // where each pane is, sideways, every frame
          await new Promise((r) => requestAnimationFrame(r))
        }
        return { h, x }
      })
      const moves = (a) => new Set(a).size > 3
      await pg.waitForTimeout(700)
      const toAddress = await swap()
      t.ok("switching to an address changes the form's height through in-between sizes", new Set(toAddress.h).size > 4 && toAddress.h[0] !== toAddress.h.at(-1) && toAddress.h.slice(1, -1).some((v) => v !== toAddress.h[0] && v !== toAddress.h.at(-1)), JSON.stringify([...new Set(toAddress.h)]))
      t.ok("and the address fields slide in from the right while the program fields slide out to the left", moves(toAddress.x.right) && toAddress.x.right[0] > toAddress.x.right.at(-1) && moves(toAddress.x.left) && toAddress.x.left[0] > toAddress.x.left.at(-1), JSON.stringify([toAddress.x.right.slice(0, 40), toAddress.x.left.slice(0, 40)]))
      t.ok("and the address fields are what is left", (await form.locator("input[aria-label='Address']").count()) === 1 && (await form.locator("input[aria-label='Program']").count()) === 0)
      const toProgram = await swap()
      t.ok("switching back stretches it again, and the program fields are back", new Set(toProgram.h).size > 4 && (await form.locator("input[aria-label='Program']").count()) === 1 && (await form.locator("input[aria-label='Address']").count()) === 0, JSON.stringify([...new Set(toProgram.h)]))
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
      t.ok("it opens on Status marks: its section is open, the other closed, and the topic is marked", (await menu.locator(".branched-menu__head").nth(0).getAttribute("aria-expanded")) === "true" && (await menu.locator(".branched-menu__head").nth(1).getAttribute("aria-expanded")) === "false" && (await topic("Status marks").getAttribute("aria-current")) === "true")
      t.ok("only that topic is shown, with its content", (await h2()).join(",") === "Status marks" && (await pg.locator("main [role=radiogroup][aria-label='Status marks']").count()) === 1, (await h2()).join(","))
      t.ok("its title is Settings and the top navigation marks it", (await pg.locator("main h1").innerText()) === "Settings" && (await pg.locator("nav a[aria-current='page']").getAttribute("aria-label")) === "Settings")
      await menu.locator(".branched-menu__head", { hasText: "MCP tools" }).click()
      await pg.waitForTimeout(500)
      t.ok("opening a section shows its topics", (await menu.locator(".branched-menu__head").nth(1).getAttribute("aria-expanded")) === "true" && (await topic("Servers").isVisible()) && (await topic("Limits").isVisible()))
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

      await side.getByRole("button", { name: "New project", exact: true }).click()
      const pname = side.locator("input[aria-label='Project name']")
      await pname.fill("Work")
      await pname.press("Enter")
      await pg.waitForTimeout(300)
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
      const older = busy.locator("aside[aria-label='Conversations'] [data-topic]", { hasText: "an older one" }).locator("button").first()
      t.ok("while an answer is being written the other conversations cannot be opened", (await older.getAttribute("aria-disabled")) === "true")
      await busy.click("button[aria-label='Stop']")
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
