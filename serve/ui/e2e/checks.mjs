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
      const pages = ["#/chat", "#/dashboard", "#/live", "#/requests", id ? `#/requests/${id}` : null, "#/hardware", "#/about", "#/requests/trace"].filter(Boolean)
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
      let most = 0, digits = 0, reels = 0, moving = 0
      for (let i = 0; i < 60; i++) { await pg.waitForTimeout(100); most = Math.max(most, await popping()); digits = Math.max(digits, await pg.locator(".t-digit").count()); reels = Math.max(reels, await pg.locator(".t-reel-col").count()); moving = Math.max(moving, await pg.locator(".t-reel-strip[data-moving]").count()) }
      t.ok("a speed that goes up and down is made of reels", reels > 0, `${reels} reels`)
      t.ok("and a reel turns when its digit changes", moving > 0, `${moving} moving at once`)
      t.ok("a request makes figures out of digits", digits > 0, `${digits} digit elements`)
      t.ok("and the digits that change pop in", most > 0, `at most ${most} at once`)
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
      t.ok("the three actions are there to use", (await pg.locator(".t-morph-menu [role=option]").count()) === 3 && (await pg.locator(".t-morph-menu [role=option]").first().isVisible()))
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
    // the status marks can be orbs, a plain loading ring, or avatars (bots): the bots are loaded only when chosen; the choice is remembered
    name: "avatar: the status marks can be orbs, a plain loading ring or avatars, and it is remembered",
    async run({ browser, fast, t, errors }) {
      const pg = await open(browser, errors)
      const scripts = []
      pg.on("request", (r) => { if (/\.js(\?|$)/.test(r.url())) scripts.push(r.url()) })
      await pg.goto(fast.base + "/#/dashboard")
      await pg.waitForTimeout(2000)
      const before = scripts.length
      const btn = pg.locator("button[aria-label^='Status avatars']")
      const painted = () => pg.evaluate(() => [...document.querySelectorAll(".orb-slot canvas")].map((c) => { try { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n } catch { return -1 } }))
      t.ok("to begin with: orbs with the loading style, the orbs drawn, no avatar loaded", (await btn.getAttribute("aria-label")).includes("Orbs + Loading") && (await painted()).some((n) => n > 0))
      await btn.click()
      await pg.waitForTimeout(800)
      const marks = () => pg.evaluate(() => ({ lat: document.querySelectorAll(".orb-slot .lat").length, matrix: document.querySelectorAll(".orb-slot .t-matrix").length, canvas: document.querySelectorAll(".orb-slot canvas").length }))
      const m = await marks()
      t.ok("then the loading style alone: loaders of dots in place of the orbs, no canvas, nothing more loaded", (await btn.getAttribute("aria-label")).includes("Loading only") && m.lat + m.matrix >= 2 && m.canvas === 0 && scripts.length === before, JSON.stringify(m))
      t.ok("both loader families are used", (await pg.evaluate(() => document.querySelectorAll(".orb-slot .lat, .orb-slot .t-matrix").length)) >= 2)
      await btn.click()
      await pg.waitForTimeout(2500)
      t.ok("then avatars: they are loaded now (one more script)", (await btn.getAttribute("aria-label")).includes("Avatar") && scripts.length > before, `${before} -> ${scripts.length}`)
      const p1 = await painted()
      t.ok("and an avatar is drawn in each place", p1.length >= 2 && p1.every((n) => n > 0), p1.join(" "))
      await pg.goto(fast.base + "/#/about")
      await pg.waitForTimeout(2500)
      const tiles = pg.locator("[role=group][aria-label='Which avatar'] button")
      t.ok("About lists the choices: by status, random and eighteen avatars, each drawn", (await tiles.count()) === 20 && (await pg.locator("[role=group][aria-label='Which avatar'] canvas").count()) === 18)
      await pg.locator("button[aria-label='Cat']").click()
      await pg.waitForTimeout(500)
      t.ok("choosing Cat is kept and shown as chosen", (await pg.evaluate(() => localStorage.getItem("strata.avatar.type"))) === '"cat"' && (await pg.locator("button[aria-label='Cat']").getAttribute("aria-pressed")) === "true")
      await pg.getByRole("button", { name: "Random", exact: true }).click()
      await pg.waitForTimeout(500)
      t.ok("Random is a choice too, kept and shown as chosen", (await pg.evaluate(() => localStorage.getItem("strata.avatar.type"))) === '"random"' && (await pg.getByRole("button", { name: "Random", exact: true }).getAttribute("aria-pressed")) === "true")
      await pg.goto(fast.base + "/#/dashboard")
      await pg.waitForTimeout(2000)
      t.ok("and the avatars are drawn with a random one each", (await painted()).every((n) => n > 0))
      await pg.reload()
      await pg.waitForTimeout(2500)
      t.ok("the choice is remembered after a reload", (await btn.getAttribute("aria-label")).includes("Avatar") && (await painted()).every((n) => n > 0))
      await btn.click()
      await pg.waitForTimeout(800)
      t.ok("and it comes round to the orbs again", (await btn.getAttribute("aria-label")).includes("Orbs. Change") && (await painted()).some((n) => n > 0))
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
]
