import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "@fontsource-variable/inter";
import "@fontsource-variable/anuphan";
import "@fontsource-variable/jetbrains-mono";
import "@fontsource/instrument-serif";
import "./globals.css";
import "./effects.css";
import "./scenes.css";
import "./monokai.css";
import "./lattice-loader.css";
import "./branched-menu.css";
import "./scene-parts.css";
import "./scenes-chat.css";
import "./scenes-monitor-a.css";
import "./scenes-monitor-b.css";
import "./scenes-panels.css";
import "./scenes-legacy.css";
import { LangProvider } from "@/lib/i18n";

const title = "Strata-xeno: a workbench around a local model";
const description =
  "A fork of Strata: a new web app with coding tools, a wider Anthropic-compatible API, and expert memory that gives RAM back as VRAM grows.";

export const metadata: Metadata = {
  title,
  description,
  applicationName: "Strata-xeno",
  authors: [{ name: "xenodeve", url: "https://github.com/xenodeve" }],
  keywords: ["Strata", "Strata-xeno", "local LLM", "mixture of experts", "Qwen3.8-Flash-Next", "coding agent", "Anthropic API"],
  openGraph: { title, description, type: "website", siteName: "Strata-xeno" },
  twitter: { card: "summary_large_image", title, description },
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f5f5f7" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0b0d" },
  ],
};

// before first paint: the saved theme, so a dark page does not flash white
const themeScript = `document.documentElement.classList.add("js");try{var t=localStorage.getItem("strata-xeno.theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <a className="skip" href="#main">
          Skip to content
        </a>
        <LangProvider>{children}</LangProvider>
      </body>
    </html>
  );
}
