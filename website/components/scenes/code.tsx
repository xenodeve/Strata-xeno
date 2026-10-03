"use client";

import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import cpp from "highlight.js/lib/languages/cpp";
import cmake from "highlight.js/lib/languages/cmake";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import ini from "highlight.js/lib/languages/ini";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import powershell from "highlight.js/lib/languages/powershell";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

/**
 * Code on this site is coloured the way the app colours the code in an answer (serve/ui/src/lib/highlight.ts): highlight.js
 * (BSD-3-Clause) makes the spans, and the colours are Monokai Classic, drawn on its own dark ground wherever there is code, in
 * a light page as well as a dark one. The classes are highlight.js's; the colours are in app/scenes.css (`.mk`).
 */
const LANGUAGES = { bash, cpp, cmake, css, diff, ini, javascript, json, markdown, powershell, python, typescript, xml, yaml };
for (const [name, def] of Object.entries(LANGUAGES)) hljs.registerLanguage(name, def);

// the names a model writes after the fence (the app's own table, cut down to the languages registered here)
const ALIAS: Record<string, string> = {
  py: "python", python3: "python", js: "javascript", jsx: "javascript", mjs: "javascript", ts: "typescript", tsx: "typescript",
  sh: "bash", zsh: "bash", shell: "bash", console: "bash", ps1: "powershell", pwsh: "powershell", "c++": "cpp", cc: "cpp", h: "cpp", cu: "cpp",
  yml: "yaml", html: "xml", md: "markdown", patch: "diff", toml: "ini", conf: "ini", jsonc: "json",
};

const cache = new Map<string, string>();
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** The coloured HTML of `code` for the fence word `lang`; plain, escaped text when the language is not one that is known. */
export function highlight(lang: string, code: string): string {
  const l = lang.trim().toLowerCase();
  const name = ALIAS[l] ?? l;
  if (!name || !hljs.getLanguage(name)) return esc(code);
  const key = name + "\u0000" + code;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  let html: string;
  try {
    html = hljs.highlight(code, { language: name, ignoreIllegals: true }).value;
  } catch {
    html = esc(code);
  }
  if (cache.size >= 400) cache.delete(cache.keys().next().value!);
  cache.set(key, html);
  return html;
}

/**
 * A piece of code written out over time. What has been written is coloured as it is (the app colours a block that is still being
 * written again at every word, so a keyword that is not finished is plain until it is); the rest of the block is there,
 * invisible, so the box is its full size from the first letter and nothing below it ever moves.
 */
export function Code({ code, lang, shown, className = "", caret = false }: { code: string; lang: string; shown?: number; className?: string; caret?: boolean }) {
  const n = shown === undefined ? code.length : Math.max(0, Math.min(code.length, Math.floor(shown)));
  const done = n >= code.length;
  return (
    <span className={"mk__code " + className}>
      <span className="mk__ghost" aria-hidden="true" dangerouslySetInnerHTML={{ __html: highlight(lang, code) }} />
      <span className="mk__live">
        <span dangerouslySetInnerHTML={{ __html: highlight(lang, code.slice(0, n)) }} />
        {caret && !done && n > 0 && <span className="mk__caret" aria-hidden="true" />}
      </span>
    </span>
  );
}
