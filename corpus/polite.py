"""Polite HTTP fetching shared by the corpus crawlers.

- One contact-free User-Agent for every request.
- At most one request per second per process (global, monotonic clock).
- robots.txt is checked per host before any non-robots request.
- Retries with backoff on transient errors; never follows redirects off-host silently
  (redirect targets are re-checked against robots.txt).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser

USER_AGENT = "CalcLM-corpus-research/0.1 (non-commercial research crawler; 1 req/s)"
MIN_INTERVAL_S = 1.05
MAX_BYTES_DEFAULT = 64 * 1024 * 1024

_last_request = 0.0
_robots: dict[str, urllib.robotparser.RobotFileParser] = {}


class FetchError(Exception):
    """A fetch failed after retries or was refused by robots.txt."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_no_redirect_opener = urllib.request.build_opener(_NoRedirect)


def _throttle() -> None:
    global _last_request
    wait = _last_request + MIN_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _robots_for(url: str) -> urllib.robotparser.RobotFileParser:
    parts = urllib.parse.urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = origin + "/robots.txt"
        _throttle()
        req = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rp.parse(resp.read().decode("utf-8", "replace").splitlines())
        except urllib.error.HTTPError as e:
            # 4xx means no robots.txt: everything allowed. 5xx: be conservative.
            rp.parse([] if 400 <= e.code < 500 else ["User-agent: *", "Disallow: /"])
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise FetchError(f"cannot read {robots_url}: {e}") from e
        _robots[origin] = rp
    return _robots[origin]


def allowed(url: str) -> bool:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        return False
    return _robots_for(url).can_fetch(USER_AGENT, url)


def fetch(url: str, *, headers: dict[str, str] | None = None, retries: int = 4,
          timeout: float = 120, max_bytes: int = MAX_BYTES_DEFAULT,
          follow_redirects: bool = True) -> tuple[bytes, dict]:
    """GET url politely. Returns (body, info) where info has status, final_url, headers.

    With follow_redirects=False a 3xx answer is returned as (b"", info) with info["location"].
    """
    if not allowed(url):
        raise FetchError(f"robots.txt disallows {url}")
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    delay = 5.0
    last_err: Exception | None = None
    for _attempt in range(retries):
        _throttle()
        req = urllib.request.Request(url, headers=hdrs)
        try:
            opener = urllib.request.urlopen if follow_redirects else _no_redirect_opener.open
            with opener(req, timeout=timeout) as resp:
                final = resp.geturl()
                if final != url and not allowed(final):
                    raise FetchError(f"redirect target disallowed by robots.txt: {final}")
                body = resp.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise FetchError(f"response larger than {max_bytes} bytes: {url}")
                return body, {"status": resp.status, "final_url": final,
                              "headers": dict(resp.headers.items())}
        except urllib.error.HTTPError as e:
            if not follow_redirects and 300 <= e.code < 400:
                return b"", {"status": e.code, "final_url": url,
                             "location": e.headers.get("Location", ""),
                             "headers": dict(e.headers.items())}
            if e.code in (404, 403, 410, 451):
                raise FetchError(f"HTTP {e.code} for {url}") from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
        except ValueError as e:  # e.g. http.client.InvalidURL: not retryable
            raise FetchError(f"invalid URL {url[:200]!r}: {e}") from e
        time.sleep(delay)
        delay *= 2
    raise FetchError(f"giving up on {url}: {last_err}")
