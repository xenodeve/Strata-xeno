"""The plain-http warning of the servers window (#310 part 3).

An `http://` address sends the token (and every file the runner sends) unencrypted.  That is fine when the traffic
stays on this PC or inside a VPN that encrypts it: localhost, 127.0.0.0/8, ::1, Radmin VPN (26.0.0.0/8), Tailscale
(100.64.0.0/10, and its MagicDNS names under `.ts.net`).  Every other http address warns, a private LAN address
(192.168.x, 10.x) too: the token travels in clear there as well.  The rule reads the URL's text only - no DNS lookup -
and never raises on text as it is being typed.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from desktop import words

QUIET_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),        # this PC
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("26.0.0.0/8"),         # Radmin VPN
    ipaddress.ip_network("100.64.0.0/10"),      # Tailscale (CGNAT range)
)


def _quiet_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith(".ts.net"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False                             # a name: no lookup, so it warns
    if ip.version == 6 and ip.ipv4_mapped:       # ::ffff:127.0.0.1 is 127.0.0.1
        ip = ip.ipv4_mapped
    return any(ip.version == n.version and ip in n for n in QUIET_NETWORKS)


def plain_http_warning(url: str) -> str | None:
    """None for https, for http to this PC or a VPN address (above), and for text that is not an http(s) address yet;
    else `words.t("plain_http")`."""
    if not isinstance(url, str):
        return None
    try:
        u = urlsplit(url.strip())
        host = u.hostname                        # lower-cased, brackets and user info removed
    except ValueError:                           # "http://[::1" while it is typed
        return None
    if u.scheme.lower() != "http" or not host:
        return None
    return None if _quiet_host(host) else words.t("plain_http")
