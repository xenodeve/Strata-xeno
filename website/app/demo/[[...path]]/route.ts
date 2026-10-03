import type { NextRequest } from "next/server";

/**
 * A small reverse proxy to the demo backend (`npm run demo`: the real Strata-xeno server in its no-GPU mock mode).
 *
 * Why not a plain rewrite: the real server forbids being framed (X-Frame-Options: DENY, a clickjacking guard) and this
 * site frames it. The proxy lets this site, and only this site, frame it (frame-ancestors 'self'). The server's own
 * code is not touched.
 *
 * It keeps the server's other guards: it only serves requests for this machine (Host localhost or 127.0.0.1), and a
 * request that changes something must come from this site's own pages (same Origin).
 */
export const dynamic = "force-dynamic";

const ORIGIN = process.env.DEMO_ORIGIN || "http://127.0.0.1:8187";
const MAX_BODY = 4 * 1024 * 1024;
const LOCAL_HOSTS =new Set(["localhost", "127.0.0.1", "[::1]"]);
const DROP_REQUEST = new Set(["host", "connection", "content-length", "origin", "referer", "accept-encoding", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "cookie"]);
const DROP_RESPONSE = new Set(["x-frame-options", "content-security-policy", "content-encoding", "content-length", "transfer-encoding", "connection", "keep-alive"]);

function refuse(status: number, why: string) {
  return new Response(why, { status, headers: { "content-type": "text/plain; charset=utf-8" } });
}

async function proxy(req: NextRequest): Promise<Response> {
  const url = new URL(req.url);
  const host = (req.headers.get("host") || "").replace(/:\d+$/, "");
  if (!LOCAL_HOSTS.has(host)) return refuse(403, "The demo is served to this machine only.");

  const method = req.method.toUpperCase();
  if (method !== "GET" && method !== "HEAD") {
    const origin = req.headers.get("origin");
    if (origin && origin !== `${url.protocol}//${url.host}`) return refuse(403, "Cross-site request refused.");
    const site = req.headers.get("sec-fetch-site");
    if (site && site !== "same-origin" && site !== "none") return refuse(403, "Cross-site request refused.");
  }

  const target = ORIGIN + url.pathname.replace(/^\/demo/, "") + url.search;
  const headers = new Headers();
  req.headers.forEach((v, k) => {
    if (!DROP_REQUEST.has(k.toLowerCase())) headers.set(k, v);
  });
  // the server's own cross-site guard compares Origin with Host: the proxy is the one same-origin caller it should see
  if (method !== "GET" && method !== "HEAD") headers.set("origin", ORIGIN);

  // The backend reads a body by its Content-Length, so the body is buffered (these are small JSON requests) rather than
  // streamed, which would be sent chunked and arrive empty.
  let body: ArrayBuffer | undefined;
  if (method !== "GET" && method !== "HEAD") {
    body = await req.arrayBuffer();
    if (body.byteLength > MAX_BODY) return refuse(413, "Request too large for the demo.");
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method,
      headers,
      body,
      redirect: "manual",
      cache: "no-store",
    });
  } catch {
    return refuse(502, "The demo backend is not running. Start it with: npm run demo");
  }

  const out = new Headers();
  upstream.headers.forEach((v, k) => {
    if (!DROP_RESPONSE.has(k.toLowerCase())) out.set(k, v);
  });
  out.set("content-security-policy", "frame-ancestors 'self'");
  out.set("x-content-type-options", "nosniff");
  const loc = out.get("location");
  if (loc && loc.startsWith("/")) out.set("location", "/demo" + loc);
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export { proxy as GET, proxy as POST, proxy as DELETE, proxy as HEAD };
