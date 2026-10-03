import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // a separate folder for production builds, so a build never disturbs a running `next dev`
  distDir: process.env.NEXT_DIST_DIR || ".next",
  // the demo's trailing slash matters: the app's asset URLs are relative to /demo/next/
  skipTrailingSlashRedirect: true,
  images: { formats: ["image/avif", "image/webp"], deviceSizes: [640, 828, 1200, 1600, 1920, 2400], qualities: [75, 95] },
};

export default config;
