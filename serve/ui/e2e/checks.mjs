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
