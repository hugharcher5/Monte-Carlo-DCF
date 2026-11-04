"""
Keep the Streamlit Community Cloud frontend awake.

Cloud puts an app to sleep after a period with no visitors, and a plain HTTP
GET does not count as a visit: the app only wakes when a real browser opens
its websocket session. So this opens the app in headless Chromium, clicks the
"Yes, get this app back up!" button if the sleep page is showing, and waits
for the app itself to render.

Run by .github/workflows/keep_awake.yml. Requires the app to be public;
a login wall is reported as a failure so it shows up red in Actions.

If the page never reaches a state this script recognises, it fails instead of
assuming the app is awake, and saves keep_awake.png / keep_awake.txt so the
workflow can upload what Streamlit actually showed.
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://hugh-dcf-calculator.streamlit.app/")
WAKE_BUTTON = "Yes, get this app back up"
NO_ACCESS = "You do not have access to this app"

DETECT_SECONDS = 120  # sleep page / app / login wall should appear within this
BOOT_SECONDS = 360    # a woken app can take several minutes to start
MAX_WAKE_CLICKS = 3   # the sleep page sometimes comes back while booting
SETTLE_SECONDS = 20   # keep the session open so the visit registers


def app_rendered(page) -> bool:
    # The app runs inside an iframe on *.streamlit.app, and the page also
    # carries other iframes (status embed), so check every frame
    for frame in page.frames:
        try:
            if frame.locator("[data-testid='stApp']").count():
                return True
        except Exception:
            pass
    return False


def state(page) -> str:
    if app_rendered(page):
        return "app"
    if page.get_by_role("button", name=WAKE_BUTTON).is_visible():
        return "asleep"
    if page.get_by_text(NO_ACCESS).is_visible():
        return "no_access"
    return "unknown"


def wait_for(page, wanted: set[str], seconds: int) -> str:
    deadline = time.monotonic() + seconds
    current = state(page)
    while current not in wanted and time.monotonic() < deadline:
        page.wait_for_timeout(2_000)
        current = state(page)
    return current


def save_diagnostics(page, reason: str) -> None:
    print(f"[keep_awake] {reason} Saving keep_awake.png and keep_awake.txt.")
    try:
        page.screenshot(path="keep_awake.png", full_page=True)
    except Exception as exc:
        print(f"[keep_awake] Screenshot failed: {exc}")
    lines = [f"url: {page.url}", f"frames: {[f.url for f in page.frames]}", ""]
    for frame in page.frames:
        try:
            lines.append(f"--- {frame.url}\n{frame.locator('body').inner_text(timeout=5_000)[:3000]}")
        except Exception:
            pass
    with open("keep_awake.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        # Every visitor, public app or not, is bounced through share.streamlit.io's
        # auth handshake first, so judge by what finally renders, not the URL
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)

        current = wait_for(page, {"app", "asleep", "no_access"}, DETECT_SECONDS)
        print(f"[keep_awake] Initial state: {current}")

        if current == "no_access":
            print("[keep_awake] Login wall: make the app public in Streamlit Cloud settings.")
            save_diagnostics(page, "Login wall.")
            return 1
        if current == "unknown":
            save_diagnostics(page, f"Nothing recognisable after {DETECT_SECONDS}s.")
            return 1

        clicks = 0
        deadline = time.monotonic() + BOOT_SECONDS
        while current == "asleep" and clicks < MAX_WAKE_CLICKS:
            clicks += 1
            print(f"[keep_awake] App was asleep; clicking wake button ({clicks}/{MAX_WAKE_CLICKS}).")
            page.get_by_role("button", name=WAKE_BUTTON).click()
            remaining = max(int(deadline - time.monotonic()), 30)
            current = wait_for(page, {"app", "asleep"}, remaining)

        if current != "app":
            save_diagnostics(page, f"App did not come up within {BOOT_SECONDS}s (state: {current}).")
            return 1

        page.wait_for_timeout(SETTLE_SECONDS * 1_000)
        if not app_rendered(page):
            save_diagnostics(page, "App rendered, then disappeared.")
            return 1
        print("[keep_awake] App is up.")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
