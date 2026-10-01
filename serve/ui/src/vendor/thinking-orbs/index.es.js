import { jsx } from "react/jsx-runtime";
import { useState, useEffect, useRef } from "react";
import { r as resolvePreset, s as scaleCounts, a as scaleRadii, M as MODE_FRAMES, p as paintFrame } from "./index-B8WsUNf5.js";
import { b, S, c, d, f, e, g, h, l, m, i, v } from "./index-B8WsUNf5.js";
function ancestorTheme(el) {
  let node = el;
  while (node) {
    const attr = node.getAttribute("data-theme");
    if (attr === "dark") return true;
    if (attr === "light") return false;
    if (node.classList.contains("dark")) return true;
    if (node.classList.contains("light")) return false;
    node = node.parentElement;
  }
  return null;
}
function systemDark() {
  return typeof matchMedia === "undefined" || matchMedia("(prefers-color-scheme: dark)").matches;
}
function useResolvedDark(theme, hostRef) {
  const [dark, setDark] = useState(true);
  useEffect(() => {
    if (theme === "dark") {
      setDark(true);
      return;
    }
    if (theme === "light") {
      setDark(false);
      return;
    }
    const resolve = () => {
      const fromTree = ancestorTheme(hostRef.current);
      setDark(fromTree ?? systemDark());
    };
    resolve();
    const mq2 = typeof matchMedia !== "undefined" ? matchMedia("(prefers-color-scheme: dark)") : null;
    const onMq = () => resolve();
    mq2 == null ? void 0 : mq2.addEventListener("change", onMq);
    let mo = null;
    if (typeof MutationObserver !== "undefined" && hostRef.current) {
      mo = new MutationObserver(resolve);
      mo.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class", "data-theme"],
        subtree: true
      });
    }
    return () => {
      mq2 == null ? void 0 : mq2.removeEventListener("change", onMq);
      mo == null ? void 0 : mo.disconnect();
    };
  }, [theme, hostRef]);
  return dark;
}
function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof matchMedia === "undefined") return;
    const mq2 = matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq2.matches);
    const on = (e2) => setReduced(e2.matches);
    mq2.addEventListener("change", on);
    return () => mq2.removeEventListener("change", on);
  }, []);
  return reduced;
}
const GRAVITY_DEFAULTS = Object.freeze({
  reach: 160,
  strength: 11,
  deform: 19,
  taper: 1.95,
  curve: 2.8,
  falloff: 24,
  smoothing: 0.3,
  handover: 0.6,
  squash: 1.2,
  blur: 1,
  fadeMs: 180
});
let tuning = null;
function setGravityConfig(patch) {
  tuning = patch ? { ...tuning ?? {}, ...patch } : null;
  kick();
}
function getGravityConfig() {
  return { ...GRAVITY_DEFAULTS, ...tuning ?? {} };
}
let sprite = null;
let spriteImg = null;
let src = null;
let srcDpr = 0;
let spriteDpr = 0;
let disabled = false;
let disabledReason = "";
let slowFrames = 0;
function resetGravity() {
  disabled = false;
  disabledReason = "";
  slowFrames = 0;
  kick();
}
function setGravitySprite(next) {
  if (next === sprite) return;
  sprite = next;
  spriteImg = null;
  src = null;
  disabled = false;
  slowFrames = 0;
  spriteDpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  hideCursor();
  if (!next || typeof Image === "undefined") return;
  const img = new Image();
  img.decoding = "async";
  img.onload = () => {
    if (sprite !== next) return;
    spriteImg = img;
    kick();
  };
  img.src = next.src;
}
const instances = /* @__PURE__ */ new Set();
let tracking = false;
let raf = 0;
let last = 0;
let px = Number.NaN, py = Number.NaN;
let lpx = 0, lpy = 0;
let uS = 0;
let near = null;
let pointerIsMouse = true;
let typing = false;
function attachGravity(el, options = true) {
  const o = options === true ? {} : options;
  if (o.sprite) setGravitySprite(o.sprite);
  const inst = {
    el,
    opts: {
      reach: Math.max(1, o.reach ?? GRAVITY_DEFAULTS.reach),
      strength: Math.max(0, Math.min(64, o.strength ?? GRAVITY_DEFAULTS.strength)),
      deform: Math.max(0, Math.min(32, o.deform ?? GRAVITY_DEFAULTS.deform)),
      taper: Math.max(1, Math.min(4, o.taper ?? GRAVITY_DEFAULTS.taper)),
      curve: Math.max(1, Math.min(4, o.curve ?? GRAVITY_DEFAULTS.curve)),
      falloff: Math.max(2, Math.min(200, o.falloff ?? GRAVITY_DEFAULTS.falloff)),
      smoothing: clamp01(o.smoothing ?? GRAVITY_DEFAULTS.smoothing),
      handover: clamp01(o.handover ?? GRAVITY_DEFAULTS.handover),
      squash: Math.max(0, Math.min(3, o.squash ?? GRAVITY_DEFAULTS.squash)),
      blur: Math.max(0, Math.min(24, o.blur ?? GRAVITY_DEFAULTS.blur)),
      fadeMs: Math.max(1, o.fadeMs ?? GRAVITY_DEFAULTS.fadeMs)
    }
  };
  instances.add(inst);
  ensureTracking();
  kick();
  return () => {
    instances.delete(inst);
    if (near === inst) near = null;
    if (instances.size === 0) queueMicrotask(() => {
      if (instances.size === 0) stopTracking();
    });
  };
}
const clamp01 = (v2) => Math.max(0, Math.min(1, v2));
const effective = (inst) => tuning ? { ...inst.opts, ...tuning } : inst.opts;
const mq = (q) => typeof window.matchMedia === "function" && window.matchMedia(q).matches;
function swapBlockedBy() {
  if (disabled) return disabledReason || "disabled by a fail-safe";
  if (!sprite) return "no pointer sprite set";
  if (!spriteImg) return "pointer sprite still loading";
  if (mq("(prefers-reduced-motion: reduce)")) return "prefers-reduced-motion is on";
  if (mq("(forced-colors: active)")) return "forced colours are active";
  if (!mq("(pointer: fine)") || !mq("(hover: hover)")) return "no fine pointer";
  if ((window.devicePixelRatio || 1) !== spriteDpr) return "display scale changed since the sprite was set — reload";
  const vv = window.visualViewport;
  if (vv && Math.abs(vv.scale - 1) > 1e-3) return "page is zoomed";
  return null;
}
function swapAllowed() {
  return swapBlockedBy() === null;
}
function getGravityStatus() {
  const blocked = typeof window === "undefined" ? "no window" : swapBlockedBy();
  return { orbs: instances.size, tracking, active: curShown, blockedBy: instances.size === 0 ? "no orb has gravity on" : blocked };
}
function ensureTracking() {
  if (tracking || instances.size === 0 || typeof document === "undefined") return;
  if (!mq("(pointer: fine)")) return;
  tracking = true;
  document.addEventListener("pointermove", onMove, { passive: true });
  document.addEventListener("pointerleave", onLeave);
  document.addEventListener("pointercancel", onLeave);
  document.addEventListener("keydown", onKey, { passive: true });
  document.addEventListener("visibilitychange", onLeave);
  window.addEventListener("blur", onLeave);
}
function stopTracking() {
  if (!tracking) return;
  tracking = false;
  document.removeEventListener("pointermove", onMove);
  document.removeEventListener("pointerleave", onLeave);
  document.removeEventListener("pointercancel", onLeave);
  document.removeEventListener("keydown", onKey);
  document.removeEventListener("visibilitychange", onLeave);
  window.removeEventListener("blur", onLeave);
  if (raf !== 0) {
    cancelAnimationFrame(raf);
    raf = 0;
  }
  near = null;
  uS = 0;
  hideCursor();
  if (curEl) {
    curEl.remove();
    curEl = null;
    curCanvas = null;
    curCtx = null;
  }
  if (hideStyle) {
    hideStyle.remove();
    hideStyle = null;
  }
}
function onMove(e2) {
  pointerIsMouse = e2.pointerType === "mouse" || e2.pointerType === "";
  typing = false;
  moveSeq++;
  px = lpx = e2.clientX;
  py = lpy = e2.clientY;
  if (releasePending) {
    releasePending = false;
    hideSprite();
    amp = 0;
    bend = 0;
    wcx = wcy = Number.NaN;
  }
  kick();
}
function onKey() {
  typing = true;
  hideCursor();
}
function onLeave() {
  px = py = Number.NaN;
  kick();
}
function kick() {
  if (!tracking || raf !== 0) return;
  last = performance.now();
  raf = requestAnimationFrame(step);
}
let curEl = null;
let curCanvas = null;
let curCtx = null;
let curShown = false;
const HIDE_CLASS = "thinking-orb-gravity-hide";
let hideStyle = null;
let hiding = false;
const naturalCursor = /* @__PURE__ */ new WeakMap();
let moveSeq = 0;
let claimMove = -1;
let stepSeq = 0;
let claimStep = -1;
function ensureCursor() {
  if (curEl) return true;
  const el = document.createElement("div");
  el.className = "thinking-orb-gravity-cursor";
  el.setAttribute("aria-hidden", "true");
  el.style.cssText = "position:fixed;left:0;top:0;pointer-events:none;z-index:2147483001;will-change:transform;display:none";
  const c2 = document.createElement("canvas");
  c2.style.display = "block";
  el.appendChild(c2);
  document.body.appendChild(el);
  const ctx = c2.getContext("2d");
  if (!ctx) {
    el.remove();
    return false;
  }
  curEl = el;
  curCanvas = c2;
  curCtx = ctx;
  curDpr = 0;
  return true;
}
function ensureHideStyle() {
  if (hideStyle) return;
  hideStyle = document.createElement("style");
  hideStyle.textContent = `html.${HIDE_CLASS}, html.${HIDE_CLASS} * { cursor: none !important; }`;
  document.head.appendChild(hideStyle);
}
function cursorOf(el) {
  const cached = naturalCursor.get(el);
  if (cached) return cached;
  const root = document.documentElement;
  const had = root.classList.contains(HIDE_CLASS);
  if (had) root.classList.remove(HIDE_CLASS);
  const cur = getComputedStyle(el).cursor;
  if (had) root.classList.add(HIDE_CLASS);
  naturalCursor.set(el, cur);
  return cur;
}
function claimCursor(x, y) {
  const target = document.elementFromPoint(x, y);
  if (!target) return false;
  const cur = cursorOf(target);
  if (cur !== "auto" && cur !== "default") {
    releaseCursor();
    return false;
  }
  if (!hiding) {
    ensureHideStyle();
    document.documentElement.classList.add(HIDE_CLASS);
    hiding = true;
    claimMove = moveSeq;
    claimStep = stepSeq;
  }
  return true;
}
function releaseCursor() {
  if (!hiding) return;
  document.documentElement.classList.remove(HIDE_CLASS);
  hiding = false;
  claimMove = -1;
}
function hideSprite() {
  if (curEl && curShown) {
    curEl.style.display = "none";
    curShown = false;
  }
}
const LINGER_MS = 500;
let lastReach = Number.NEGATIVE_INFINITY;
let releasePending = false;
function hideCursor() {
  releasePending = false;
  releaseCursor();
  hideSprite();
  amp = 0;
  bend = 0;
  wcx = wcy = Number.NaN;
}
const _c = { cx: 0, cy: 0, r: 0 };
let amp = 0, bend = 0;
let wcx = Number.NaN, wcy = Number.NaN;
const PAD = 48;
const PASSES = 10;
let refCanvas = null;
let refCtx = null;
let bentCanvas = null;
let bentCtx = null;
let bentData = null;
let curDpr = 0, curW = 0, curH = 0;
function readSprite(dpr) {
  if (!spriteImg || !sprite) return null;
  const c2 = document.createElement("canvas");
  c2.width = Math.ceil(sprite.width * dpr);
  c2.height = Math.ceil(sprite.height * dpr);
  const g2 = c2.getContext("2d", { willReadFrequently: true });
  if (!g2) return null;
  g2.scale(dpr, dpr);
  g2.drawImage(spriteImg, 0, 0, sprite.width, sprite.height);
  return g2.getImageData(0, 0, c2.width, c2.height);
}
function bendSprite(B, taper, P, hx, hy, cxo, cyo) {
  if (!src || !bentData) return;
  const OW = bentData.width, OH = bentData.height;
  const S2 = src, SW = S2.width, SH = S2.height, sd = S2.data;
  const od = bentData.data;
  od.fill(0);
  const L = Math.max(1, Math.hypot(SW, SH));
  const tdx = cxo - hx, tdy = cyo - hy;
  const tdl = Math.hypot(tdx, tdy) || 1;
  const tux = tdx / tdl, tuy = tdy / tdl;
  const x1 = Math.max(0, Math.floor(P + Math.min(0, B * tux) - 3)), x2 = Math.min(OW, Math.ceil(P + SW + Math.max(0, B * tux) + 3));
  const y1 = Math.max(0, Math.floor(P + Math.min(0, B * tuy) - 3)), y2 = Math.min(OH, Math.ceil(P + SH + Math.max(0, B * tuy) + 3));
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      const i2 = (y * OW + x) * 4;
      let sx = x - P, sy = y - P;
      if (B > 0.01) {
        let qx = x, qy = y;
        let m2 = 0, dx = 0, dy = 0, dl = 1;
        for (let it = 0; it < 7; it++) {
          const s = Math.hypot(qx - hx, qy - hy) / L;
          m2 = B * Math.pow(Math.min(1, s), taper);
          dx = cxo - qx;
          dy = cyo - qy;
          dl = Math.hypot(dx, dy) || 1;
          qx += (x - m2 * dx / dl - qx) * 0.5;
          qy += (y - m2 * dy / dl - qy) * 0.5;
        }
        const ex = qx + m2 * dx / dl - x, ey = qy + m2 * dy / dl - y;
        if (ex * ex + ey * ey > 2.25) {
          od[i2] = od[i2 + 1] = od[i2 + 2] = od[i2 + 3] = 0;
          continue;
        }
        sx = qx - P;
        sy = qy - P;
      }
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      if (x0 < -1 || y0 < -1 || x0 >= SW || y0 >= SH) {
        od[i2] = od[i2 + 1] = od[i2 + 2] = od[i2 + 3] = 0;
        continue;
      }
      const fx = sx - x0, fy = sy - y0;
      let r = 0, g2 = 0, b2 = 0, a = 0;
      for (let k = 0; k < 4; k++) {
        const xx = x0 + (k & 1), yy = y0 + (k >> 1);
        if (xx < 0 || yy < 0 || xx >= SW || yy >= SH) continue;
        const wgt = (k & 1 ? fx : 1 - fx) * (k >> 1 ? fy : 1 - fy);
        const j = (yy * SW + xx) * 4;
        const wa = wgt * sd[j + 3];
        r += sd[j] * wa;
        g2 += sd[j + 1] * wa;
        b2 += sd[j + 2] * wa;
        a += wa;
      }
      if (a > 0) {
        od[i2] = r / a;
        od[i2 + 1] = g2 / a;
        od[i2 + 2] = b2 / a;
        od[i2 + 3] = a;
      } else {
        od[i2] = od[i2 + 1] = od[i2 + 2] = od[i2 + 3] = 0;
      }
    }
  }
}
function drawCursor(inst, w, dt) {
  if (!curCtx || !curCanvas || !curEl || !sprite || !spriteImg) return;
  const sp = sprite;
  const o = effective(inst);
  const dpr = Math.min(3, window.devicePixelRatio || 1);
  if (!refCanvas) {
    refCanvas = document.createElement("canvas");
    refCtx = refCanvas.getContext("2d");
  }
  if (!bentCanvas) {
    bentCanvas = document.createElement("canvas");
    bentCtx = bentCanvas.getContext("2d");
  }
  if (!refCtx || !bentCtx) return;
  if (!src || srcDpr !== dpr) {
    src = readSprite(dpr);
    srcDpr = dpr;
  }
  if (!src) return;
  if (dpr !== curDpr || sp.width !== curW || sp.height !== curH) {
    curDpr = dpr;
    curW = sp.width;
    curH = sp.height;
    const OW2 = Math.ceil((sp.width + 2 * PAD) * dpr), OH2 = Math.ceil((sp.height + 2 * PAD) * dpr);
    curCanvas.width = refCanvas.width = bentCanvas.width = OW2;
    curCanvas.height = refCanvas.height = bentCanvas.height = OH2;
    curCanvas.style.width = `${sp.width + 2 * PAD}px`;
    curCanvas.style.height = `${sp.height + 2 * PAD}px`;
    bentData = bentCtx.createImageData(OW2, OH2);
  }
  const OW = curCanvas.width, OH = curCanvas.height;
  const k = Math.pow(w, o.curve);
  const ease = 1 - Math.exp(-dt / (0.012 + o.smoothing * 0.14));
  amp += (o.strength * k - amp) * ease;
  bend += (o.deform * k - bend) * ease;
  let A = amp * dpr, B = bend * dpr;
  if (Number.isNaN(wcx)) {
    wcx = _c.cx;
    wcy = _c.cy;
  } else {
    const swing = 1 - Math.exp(-dt / (0.05 + o.handover * 0.6));
    wcx += (_c.cx - wcx) * swing;
    wcy += (_c.cy - wcy) * swing;
  }
  const P = PAD * dpr;
  const hx = P + sp.hotX * dpr, hy = P + sp.hotY * dpr;
  const cxo = hx + (wcx - lpx) * dpr, cyo = hy + (wcy - lpy) * dpr;
  const dTip = Math.hypot(cxo - hx, cyo - hy) || 1;
  const ux = (cxo - hx) / dTip, uy = (cyo - hy) / dTip;
  const F = o.falloff * dpr;
  if (o.squash > 0) {
    const axL = Math.hypot(sp.width * 0.5 - sp.hotX, sp.height - sp.hotY) || 1;
    const axx = (sp.width * 0.5 - sp.hotX) / axL, axy = (sp.height - sp.hotY) / axL;
    const against = Math.max(0, -(ux * axx + uy * axy));
    const gain = 1 + o.squash * against;
    A *= gain;
    B *= gain;
  }
  const grow = Math.ceil(Math.max(B, 1)) + 4;
  const bx1 = Math.floor(Math.min(P, P + ux * A) - grow), by1 = Math.floor(Math.min(P, P + uy * A) - grow);
  const bx2 = Math.ceil(Math.max(P, P + ux * A) + sp.width * dpr + grow), by2 = Math.ceil(Math.max(P, P + uy * A) + sp.height * dpr + grow);
  const rx = Math.max(0, bx1), ry = Math.max(0, by1);
  const rw = Math.min(OW, bx2) - rx, rh = Math.min(OH, by2) - ry;
  if (B > 0.01) {
    bendSprite(B, o.taper, P, hx, hy, cxo, cyo);
    bentCtx.putImageData(bentData, 0, 0, rx, ry, rw, rh);
  } else {
    bentCtx.setTransform(1, 0, 0, 1, 0, 0);
    bentCtx.clearRect(0, 0, OW, OH);
    bentCtx.drawImage(spriteImg, P, P, sp.width * dpr, sp.height * dpr);
  }
  const ctx = curCtx, rctx = refCtx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, OW, OH);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(bentCanvas, 0, 0);
  if (A > 0.5) {
    const bcx = P + sp.width * dpr * 0.42, bcy = P + sp.height * dpr * 0.5;
    const half = F / 2;
    const mask = rctx.createLinearGradient(bcx - ux * half, bcy - uy * half, bcx + ux * half, bcy + uy * half);
    mask.addColorStop(0, "rgba(0,0,0,0)");
    mask.addColorStop(1, "rgba(0,0,0,1)");
    for (let i2 = PASSES; i2 >= 1; i2--) {
      const t = i2 / PASSES;
      rctx.setTransform(1, 0, 0, 1, 0, 0);
      rctx.globalCompositeOperation = "source-over";
      rctx.globalAlpha = 1;
      rctx.clearRect(rx, ry, rw, rh);
      rctx.drawImage(bentCanvas, rx, ry, rw, rh, rx + ux * A * t, ry + uy * A * t, rw, rh);
      rctx.globalCompositeOperation = "destination-in";
      rctx.fillStyle = mask;
      rctx.fillRect(rx, ry, rw, rh);
      ctx.globalCompositeOperation = "destination-over";
      ctx.globalAlpha = Math.pow(1 - t, 1.6) * 0.9;
      if (o.blur > 0) ctx.filter = `blur(${(o.blur * Math.sqrt(t) * dpr).toFixed(2)}px)`;
      ctx.drawImage(refCanvas, rx, ry, rw, rh, rx, ry, rw, rh);
    }
    ctx.filter = "none";
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }
  curEl.style.transform = `translate3d(${(lpx - sp.hotX - PAD).toFixed(2)}px,${(lpy - sp.hotY - PAD).toFixed(2)}px,0)`;
  if (!curShown) {
    curEl.style.display = "";
    curShown = true;
  }
}
function step(now) {
  raf = 0;
  if (!tracking) return;
  const t0 = performance.now();
  try {
    stepInner(now);
  } catch (err) {
    disabled = true;
    disabledReason = `disabled after an error (${err instanceof Error ? err.message : String(err)})`;
    hideCursor();
    near = null;
    if (typeof console !== "undefined") console.warn("thinking-orbs: gravity disabled after error", err);
    return;
  }
  const took = performance.now() - t0;
  if (took > 12) {
    if (++slowFrames >= 30 && !disabled) {
      disabled = true;
      disabledReason = `disabled after slow frames (~${Math.round(took)}ms each)`;
      hideCursor();
    }
  } else slowFrames = 0;
}
function stepInner(now) {
  const dt = Math.min(0.05, Math.max(1e-3, (now - last) / 1e3));
  last = now;
  stepSeq++;
  let best = null;
  let u = 0;
  if (!Number.isNaN(px) && swapAllowed()) {
    let bestEdge = Number.POSITIVE_INFINITY;
    for (const inst of instances) {
      if (!inst.el.isConnected) continue;
      const r = inst.el.getBoundingClientRect();
      if (r.width <= 0) continue;
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      const orbR = Math.min(r.width, r.height) / 2;
      const reach = effective(inst).reach;
      if (px < cx - orbR - reach || px > cx + orbR + reach || py < cy - orbR - reach || py > cy + orbR + reach) continue;
      const edge = Math.hypot(px - cx, py - cy) - orbR;
      if (edge <= reach && edge < bestEdge) {
        bestEdge = edge;
        best = inst;
        _c.cx = cx;
        _c.cy = cy;
        _c.r = orbR;
      }
    }
    if (best) {
      const t = 1 - Math.max(0, bestEdge) / effective(best).reach;
      u = t * t * (3 - 2 * t);
      lastReach = now;
    }
  }
  const nearest = near ?? best;
  const fade = nearest ? effective(nearest).fadeMs : GRAVITY_DEFAULTS.fadeMs;
  const a = 1 - Math.exp(-(dt * 1e3) / (fade / 3));
  uS += (u - uS) * a;
  if (best && best !== near) near = best;
  if (!best && uS < 2e-3) {
    uS = 0;
    if (near && hiding && !releasePending && pointerIsMouse && !typing && !Number.isNaN(px)) {
      if (now - lastReach < LINGER_MS) {
        if (claimCursor(px, py)) drawCursor(near, 0, dt);
        else hideSprite();
        raf = requestAnimationFrame(step);
        return;
      }
      if (curShown) {
        releaseCursor();
        releasePending = true;
        near = null;
        return;
      }
    }
    near = null;
    hideCursor();
    return;
  }
  if (!near) return;
  if (uS > 2e-3 && pointerIsMouse && !typing && !Number.isNaN(px) && ensureCursor() && claimCursor(px, py)) {
    const holding = claimMove === moveSeq && stepSeq - claimStep < 2 && !curShown;
    if (holding) hideSprite();
    else drawCursor(near, uS, dt);
  } else if (uS > 2e-3 && !Number.isNaN(px) && pointerIsMouse && !typing) {
    hideSprite();
  } else {
    hideCursor();
  }
  raf = requestAnimationFrame(step);
}
function parseTint(color) {
  if (!color) return void 0;
  const hex = color.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    let h2 = hex[1];
    if (h2.length === 3) h2 = h2.replace(/./g, (c2) => c2 + c2);
    const n = parseInt(h2, 16);
    return { r: n >> 16 & 255, g: n >> 8 & 255, b: n & 255 };
  }
  const fn = color.trim().match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i);
  if (fn) return { r: Number(fn[1]), g: Number(fn[2]), b: Number(fn[3]) };
  return void 0;
}
const LABELS = {
  working: "Working…",
  searching: "Searching…",
  solving: "Solving…",
  listening: "Listening…",
  connecting: "Connecting…",
  weaving: "Weaving…",
  composing: "Composing…",
  breathing: "Thinking…",
  shaping: "Shaping…"
};
function ThinkingOrb({
  state = "working",
  size = 64,
  theme = "auto",
  speed = 1,
  paused = false,
  color,
  dots = 1,
  dotSize = 1,
  opts: optsOverride,
  frame: customFrame,
  gravity,
  style,
  "aria-label": ariaLabel,
  ...rest
}) {
  const ref = useRef(null);
  const optsKey = optsOverride ? JSON.stringify(optsOverride) : "";
  const dark = useResolvedDark(theme, ref);
  const gravityKey = gravity ? JSON.stringify(gravity) : "";
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !gravity) return;
    return attachGravity(canvas, gravity === true ? true : gravity);
  }, [gravityKey]);
  const reduced = useReducedMotion();
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = Math.min(2, typeof devicePixelRatio !== "undefined" && devicePixelRatio || 1);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { mode, speed: baseSpeed, opts: presetOpts } = resolvePreset(state, size);
    let opts = dots !== 1 ? scaleCounts(presetOpts, Math.max(0.1, dots)) : presetOpts;
    if (dotSize !== 1) opts = scaleRadii(opts, Math.max(0.1, dotSize));
    if (optsOverride) opts = { ...opts, ...optsOverride };
    const frameFn = customFrame ?? MODE_FRAMES[mode];
    const tint = parseTint(color);
    const effSpeed = baseSpeed * speed;
    const frame = (tSec) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      paintFrame(ctx, frameFn(size, tSec, opts), dark, tint);
    };
    if (reduced) {
      frame(0.6);
      return;
    }
    let raf2 = 0;
    let running = false;
    const loop = () => {
      frame(performance.now() / 1e3 * effSpeed);
      if (running) raf2 = requestAnimationFrame(loop);
    };
    const start = () => {
      if (running || paused) return;
      running = true;
      raf2 = requestAnimationFrame(loop);
    };
    const stop = () => {
      running = false;
      cancelAnimationFrame(raf2);
    };
    frame(performance.now() / 1e3 * effSpeed);
    let visible = true;
    const io = typeof IntersectionObserver !== "undefined" ? new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible && document.visibilityState !== "hidden") start();
      else stop();
    }) : null;
    io == null ? void 0 : io.observe(canvas);
    const onVis = () => {
      if (document.visibilityState === "hidden") stop();
      else if (visible) start();
    };
    document.addEventListener("visibilitychange", onVis);
    if (!io) start();
    return () => {
      stop();
      io == null ? void 0 : io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [state, size, dark, speed, paused, reduced, color, dots, dotSize, optsKey, customFrame]);
  return /* @__PURE__ */ jsx(
    "canvas",
    {
      ref,
      role: "img",
      "aria-label": ariaLabel ?? LABELS[state],
      style: { width: size, height: size, display: "block", ...style },
      ...rest
    }
  );
}
export {
  GRAVITY_DEFAULTS,
  b as MODE_DRAWS,
  MODE_FRAMES,
  S as STATE_TO_MODE,
  ThinkingOrb,
  c as angleDelta,
  attachGravity,
  d as countDots,
  f as fibDir,
  e as finalizeFrame,
  g as frac,
  getGravityConfig,
  getGravityStatus,
  h as hashD,
  l as lerp,
  m as makeProj,
  i as radiusScale,
  resetGravity,
  resolvePreset,
  scaleCounts,
  scaleRadii,
  setGravityConfig,
  setGravitySprite,
  v as vnoise
};
