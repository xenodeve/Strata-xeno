import type { NextConfig } from "next";

// Where the live demo's server is, when it is not on this PC: the repo's own mock server run as a Vercel function (api/index.py in the repo root).
// Unset (a PC with `npm run demo`), /demo/* goes through the guarded proxy in app/demo instead.
const DEMO_BACKEND = (process.env.DEMO_BACKEND || "").replace(/\/+$/, "");

const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // a separate folder for production builds, so a build never disturbs a running `next dev`
  distDir: process.env.NEXT_DIST_DIR || ".next",
  // the demo's trailing slash matters: the app's asset URLs are relative to /demo/next/
  skipTrailingSlashRedirect: true,
  async rewrites() {
    return DEMO_BACKEND ? { beforeFiles: [{ source: "/demo/:path*", destination: `${DEMO_BACKEND}/:path*` }], afterFiles: [], fallback: [] } : [];
  },
  images: { formats: ["image/avif", "image/webp"], deviceSizes: [640, 828, 1200, 1600, 1920, 2400], qualities: [75, 95] },
};

export default config;
