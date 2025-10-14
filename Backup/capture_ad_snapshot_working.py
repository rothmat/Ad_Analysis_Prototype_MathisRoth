# capture_ad_snapshot.py
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import re
from datetime import datetime

from playwright.async_api import (
    async_playwright,
    Page,
    Frame,
    TimeoutError as PWTimeoutError,
    Browser,
    BrowserContext,
)

# === db_client aus ../ad-db/ingest laden ===
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ad-db" / "ingest"))
from db_client import (  # type: ignore
    connect,
    get_or_create_campaign,
    upsert_ad,
    add_media_local,
    upsert_media_base64_from_path,  # NEU
)

# Schalter: Base64 direkt mit speichern?
ENCODE_BASE64_IMAGES = True  # bei Bedarf auf False setzen

# ---------------------------
# JSON laden und Ad-Liste ziehen
# ---------------------------
def load_ads(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        ads = data
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        ads = data["data"]
    else:
        ads = []
        try:
            with open(json_path, "r", encoding="utf-8") as f2:
                for line in f2:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if "data" in obj and isinstance(obj["data"], list):
                            ads.extend(obj["data"])
                        else:
                            ads.append(obj)
        except Exception:
            pass

    cleaned = []
    for ad in ads:
        aid = str(ad.get("id") or "").strip()
        url = ad.get("ad_snapshot_url")
        if aid and isinstance(url, str) and url.startswith("http"):
            cleaned.append({"id": aid, "ad_snapshot_url": url})
    return cleaned


# ---------------------------
# Cookie-Dialog schließen (schnell & robust)
# ---------------------------
ESSENTIAL_RE = re.compile(
    r"(nur|only|uniquement|solo).*?(erforderlich|notwendig|essential|essentiel|essenziali|necessari)",
    re.I | re.S,
)
ACCEPT_ALL_RE = re.compile(
    r"(alle.*(akzept|zulass)|accept.*all|allow.*all|tout.*accepter|accett.*tutti)",
    re.I | re.S,
)

async def _click_matching_button_in_dialog(frame: Frame) -> bool:
    try:
        dlg = await frame.wait_for_selector('div[role="dialog"], [data-cookiebanner]', timeout=950)
    except Exception:
        dlg = await frame.query_selector('div[role="dialog"], [data-cookiebanner]')
    if not dlg:
        return False

    try:
        await dlg.evaluate("(n) => { n.scrollTop = n.scrollHeight; }")
    except Exception:
        pass

    btns = await dlg.query_selector_all("button, [role=button]")

    async def pick_by_regex(rx):
        res = []
        for b in btns:
            try:
                t = (await b.inner_text()).strip()
            except Exception:
                continue
            if rx.search(t):
                res.append(b)
        return res

    chosen = await pick_by_regex(ESSENTIAL_RE)
    if not chosen:
        chosen = await pick_by_regex(ACCEPT_ALL_RE)

    for b in chosen:
        try:
            await b.scroll_into_view_if_needed()
        except Exception:
            pass
        try:
            await b.click(timeout=250)
            return True
        except Exception:
            try:
                await b.evaluate("(el)=>el.click()")
                return True
            except Exception:
                continue
    return False

async def try_close_cookies(page: Page) -> bool:
    clicked = False
    try:
        clicked = await _click_matching_button_in_dialog(page.main_frame)
    except Exception:
        clicked = False

    if not clicked:
        for fr in page.frames:
            url = (fr.url or "").lower()
            if any(k in url for k in ("consent", "cookie", "privacy")):
                try:
                    if await _click_matching_button_in_dialog(fr):
                        clicked = True
                        break
                except Exception:
                    continue

    if not clicked:
        for fr in page.frames:
            try:
                if await _click_matching_button_in_dialog(fr):
                    clicked = True
                    break
            except Exception:
                continue

    if clicked:
        try:
            await page.wait_for_selector('div[role="dialog"]', state="hidden", timeout=800)
        except PWTimeoutError:
            try:
                await page.wait_for_selector('div[role="dialog"]', state="detached", timeout=600)
            except Exception:
                pass

    return clicked


# ---------------------------
# „Mehr anzeigen …“ öffnen + hart unclampen
# ---------------------------
AD_CONTAINER_SELECTORS = [
    '#facebook .x1n2onr6',
    'div[data-pagelet="root"]',
    'div[role="article"]',
]

async def _force_expand_by_css(page: Page) -> None:
    try:
        await page.add_style_tag(content="""
          [data-ad-preview*="message"],
          [data-ad-rendered*="message"],
          div[role="article"],
          div[style*="-webkit-line-clamp"],
          span[style*="-webkit-line-clamp"],
          p[style*="-webkit-line-clamp"] {
            -webkit-line-clamp: unset !important;
            line-clamp: unset !important;
            display: block !important;
            overflow: visible !important;
            text-overflow: initial !important;
            max-height: none !important;
            height: auto !important;
            white-space: normal !important;
          }
          [data-ad-preview*="message"] * {
            -webkit-line-clamp: unset !important;
            line-clamp: unset !important;
            overflow: visible !important;
            text-overflow: initial !important;
            max-height: none !important;
            height: auto !important;
          }
        """)
    except Exception:
        pass

SEE_MORE_RE = re.compile(
    r"(…|\.{3}|NoBreak|\u2060)?\s*(mehr(?:\s*(an(s|z)ehen| anzeigen)?)?|see\s*more|read\s*more|"
    r"afficher\s*(plus|la\s*suite)|voir\s*plus|mostra\s*(altro|di\s*più))",
    re.I
)

async def try_expand_see_more(page: Page) -> None:
    found = False
    btns = page.locator("button, [role='button'], span[role='button'], div[role='button'], [tabindex='0']")
    try:
        count = await btns.count()
    except Exception:
        count = 0

    for i in range(count):
        el = btns.nth(i)
        try:
            txt = (await el.inner_text()).strip()
            if not txt:
                txt = (await el.evaluate("el => (el.textContent||'').trim()")) or ""
        except Exception:
            continue
        if SEE_MORE_RE.search(txt):
            try:
                await el.click(timeout=200)
                found = True
            except Exception:
                try:
                    await el.evaluate("el => el.click()")
                    found = True
                except Exception:
                    pass

    if not found:
        try:
            for t in ("Mehr anzeigen","Mehr ansehen","Mehr","See more","Read more",
                      "Afficher plus","Afficher la suite","Voir plus","Mostra altro","Mostra di più"):
                loc = page.get_by_text(t, exact=False)
                if await loc.count() > 0:
                    try:
                        await loc.first.click(timeout=200)
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    await _force_expand_by_css(page)

    try:
        await page.evaluate("""
          () => {
            document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
              try { el.click(); } catch(e) {}
            });
          }
        """)
    except Exception:
        pass


# ---------------------------
# Dropdowns schließen (Kebab/3-Punkte)
# ---------------------------
async def close_any_open_menus(page: Page) -> None:
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(15)
        await page.keyboard.press("Escape")
    except Exception:
        pass
    try:
        await page.evaluate("""
          () => {
            document.querySelectorAll('[aria-haspopup="menu"][aria-expanded="true"]').forEach(el => {
              try { el.click(); } catch(e) {}
            });
          }
        """)
    except Exception:
        pass
    try:
        await page.mouse.click(2, 2)
    except Exception:
        pass
    try:
        await page.add_style_tag(content="""[role="menu"]{display:none!important;visibility:hidden!important;}""")
    except Exception:
        pass


# ---------------------------
# Video stumm/autoplay (nur einmal pro Seite)
# ---------------------------
async def try_enable_video(page: Page) -> None:
    try:
        await page.evaluate("""
        () => {
          const vids = Array.from(document.querySelectorAll('video'));
          for (const v of vids) {
            try { v.muted = true; v.playsInline = true; v.autoplay = true; v.play().catch(()=>{}); } catch(e) {}
          }
        }""")
    except Exception:
        pass


# ---------------------------
# Laden & Warten (kurze Micro-Polls)
# ---------------------------
async def wait_for_media_loaded(page: Page, timeout_ms: int = 3200) -> None:
    step = 70
    for _ in range(max(1, timeout_ms // step)):
        try:
            done = await page.evaluate("""
                () => {
                  const imgs = Array.from(document.images);
                  const imgsOk = imgs.every(img => img.complete);
                  const vids = Array.from(document.querySelectorAll('video'));
                  const vidsOk = vids.every(v => (v.readyState||0) >= 2 || v.paused);
                  return imgsOk && vidsOk;
                }""")
            if done:
                return
        except Exception:
            pass
        await page.wait_for_timeout(step)


# ---------------------------
# Routing (blockt Fonts/Media)
# ---------------------------
async def enable_fast_routing(context: BrowserContext):
    async def handler(route, request):
        rt = request.resource_type
        url = request.url.lower()

        if rt in ("font", "media"):
            return await route.abort()

        if url.endswith((".woff", ".woff2", ".ttf", ".otf", ".eot", ".map")):
            return await route.abort()

        return await route.continue_()
    await context.route("**/*", handler)

# ---------------------------
# Screenshot-Logik
# ---------------------------
async def capture_snapshot(page: Page, url: str, out_path: Path, speed: str = "fast") -> None:
    fast = (speed == "fast")
    goto_timeout = 6000 if fast else 10000
    default_timeout = 3500 if fast else 7000

    await page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout)
    page.set_default_timeout(default_timeout)

    await try_close_cookies(page)
    await try_expand_see_more(page)
    await close_any_open_menus(page)
    await try_enable_video(page)

    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
        await try_expand_see_more(page)

        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await try_expand_see_more(page)

        await page.evaluate("window.scrollTo(0, 0)")
        await try_expand_see_more(page)
    except Exception:
        pass

    await wait_for_media_loaded(page, timeout_ms=(3200 if fast else 5200))
    await close_any_open_menus(page)

    ad = None
    for sel in AD_CONTAINER_SELECTORS:
        try:
            ad = await page.wait_for_selector(sel, timeout=(2200 if fast else 3800))
            if ad:
                break
        except Exception:
            continue

    if ad:
        await ad.screenshot(path=str(out_path))
    else:
        await page.screenshot(path=str(out_path), full_page=True)


# ---------------------------
# Hauptlauf
# ---------------------------
def _storage_path() -> Path:
    return Path(__file__).resolve().parent / "fb_storage_state.json"

async def run(json_path: Path, out_root: Path, force_setup: bool = False) -> None:
    ads = load_ads(json_path)
    if not ads:
        print("Keine gültigen Ads mit id + ad_snapshot_url gefunden.")
        return

    # ===== Neu: Datum aus JSON-Dateiname (YYYY-MM-DD) extrahieren =====
    json_stem = json_path.stem
    today = datetime.now().strftime("%Y-%m-%d")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", json_stem):
        json_date = json_stem
    else:
        json_date = today

    # Ablageort auf Platte (weiterhin tag des Laufs)
    shots_dir = out_root / "Screenshots" / today
    shots_dir.mkdir(parents=True, exist_ok=True)

    # Kampagnenname/Slug aus Ordner ableiten (out_root ist Kampagnen-Root)
    campaign_name = out_root.name
    campaign_slug = campaign_name.lower().replace(" ", "_")

    # DB-Setup (eine Connection für den ganzen Lauf)
    conn = None
    try:
        conn = connect()
        campaign_id = get_or_create_campaign(conn, campaign_name, campaign_slug)
    except Exception as e:
        print(f"⚠️  Konnte DB nicht initialisieren: {e}")
        campaign_id = None

    storage_file = _storage_path()
    needs_setup = force_setup or (not storage_file.exists())

    async with async_playwright() as p:
        # Setup-Modus (einmalig)
        if needs_setup:
            browser = await p.chromium.launch(
                headless=False,
                args=["--start-maximized", "--autoplay-policy=no-user-gesture-required"]
            )
            context = await browser.new_context(
                locale="de-CH",
                viewport=None,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
            )
            try:
                await context.grant_permissions(["autoplay"], origin="https://www.facebook.com")
            except Exception:
                pass

            page = await context.new_page()
            await page.bring_to_front()

            first_url = ads[0]["ad_snapshot_url"]
            try:
                await page.goto(first_url, wait_until="domcontentloaded", timeout=45_000)
            except Exception:
                fallback = (
                    "https://www.facebook.com/ads/library/?active_status=all"
                    "&ad_type=political_and_issue_ads&country=CH"
                )
                await page.goto(fallback, wait_until="domcontentloaded", timeout=45_000)

            try:
                auto_closed = await try_close_cookies(page)
                if not auto_closed:
                    await page.mouse.wheel(0, 1200)
                    await page.wait_for_timeout(200)
                    await try_close_cookies(page)
            except Exception:
                pass

            print("\n⚠️  Setup-Modus: Falls noch sichtbar, Cookie-Banner manuell schließen.")
            print("   Danach im Terminal ENTER drücken – der akzeptierte Zustand wird gespeichert.\n")
            input()

            await context.storage_state(path=str(storage_file))
            await context.close()
            await browser.close()
            print(f"✅ Consent gespeichert in: {storage_file}\n")

        # Produktivlauf
        browser: Browser = await p.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox", "--autoplay-policy=no-user-gesture-required"]
        )
        context: BrowserContext = await browser.new_context(
            storage_state=str(storage_file),
            locale="de-CH",
            viewport={"width": 1280, "height": 1800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        )
        try:
            await context.grant_permissions(["autoplay"], origin="https://www.facebook.com")
        except Exception:
            pass
        await enable_fast_routing(context)

        page: Page = await context.new_page()
        page.set_default_timeout(3500)

        total = len(ads)
        inserted_media = 0

        for idx, ad in enumerate(ads, 1):
            ad_id_str = str(ad["id"])
            url = ad["ad_snapshot_url"]
            out_path = shots_dir / f"{ad_id_str}.png"

            ok = False
            if out_path.exists():
                print(f"[{idx}/{total}] bereits vorhanden – reuse {ad_id_str}")
                ok = True
            else:
                print(f"[{idx}/{total}] Shot für Ad {ad_id_str} …")

                # Versuch 1: schnell
                try:
                    await capture_snapshot(page, url, out_path, speed="fast")
                    ok = True
                except Exception:
                    ok = False

                # Versuch 2: schnell
                if not ok:
                    try:
                        await capture_snapshot(page, url, out_path, speed="fast")
                        ok = True
                    except Exception:
                        ok = False

                # Versuch 3: langsam
                if not ok:
                    try:
                        await capture_snapshot(page, url, out_path, speed="slow")
                        ok = True
                    except Exception:
                        ok = False

                # Letzter Ausweg: Fullpage
                if not ok:
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=12_000)
                        await page.screenshot(path=str(out_path), full_page=True)
                        print(f"   ⚠️  Notfall-Fullpage gespeichert für {ad_id_str}")
                        ok = True
                    except Exception as e2:
                        print(f"   ❌ Konnte {ad_id_str} nicht speichern: {e2}")

            # === DB schreiben, wenn Datei existiert ===
            if ok and out_path.exists() and campaign_id is not None and conn is not None:
                try:
                    ad_pk = upsert_ad(conn, campaign_id, ad_id_str)

                    # media (Screenshot) eintragen – WICHTIG: date_folder = json_date
                    media_pk = add_media_local(
                        conn,
                        ad_pk,
                        str(out_path),
                        kind="screenshot",
                        date_folder=json_date,
                    )

                    # Base64 in separater Tabelle speichern (optional)
                    if ENCODE_BASE64_IMAGES and media_pk:
                        upsert_media_base64_from_path(conn, media_pk, str(out_path))

                    if media_pk:
                        inserted_media += 1
                        # Batch-Commit alle 20 Dateien
                        if inserted_media % 20 == 0:
                            conn.commit()

                except Exception as db_e:
                    print(f"   ⚠️  DB-Insert fehlgeschlagen (ad={ad_id_str}): {db_e}")

        # Cleanup
        try:
            await page.close()
        except Exception:
            pass
        try:
            await context.close()
        except Exception:
            pass
        await browser.close()

    # Commit/Close DB
    if conn is not None:
        try:
            conn.commit()  # restliche writes
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    print(f"Fertig. Screenshots liegen in: {shots_dir.resolve()}")
    if campaign_id is not None:
        print(f"🗄️  DB: {inserted_media} media-Einträge (Screenshots) gespeichert für Kampagne '{campaign_name}' – date_folder={json_date}.")


def _parse_args(argv: List[str]) -> tuple[Path, Path, bool]:
    """
    Usage:
      python capture_ad_snapshot.py <ads.json> [<output_root>] [--setup]
      (kompatibel) python capture_ad_snapshot.py <ads.json> <campaign_name> <output_root> [--setup]
    """
    if len(argv) < 2:
        print("Usage:")
        print("  python capture_ad_snapshot.py <ads.json> [<output_root>] [--setup]")
        print("  (kompatibel) python capture_ad_snapshot.py <ads.json> <campaign_name> <output_root> [--setup]")
        sys.exit(1)

    args = list(argv[1:])
    setup = False
    if "--setup" in args:
        setup = True
        args.remove("--setup")

    json_path = Path(args[0])

    if len(args) == 1:
        out_root = json_path.parent
    elif len(args) == 2:
        out_root = Path(args[1])
    else:
        out_root = Path(args[2])

    return json_path, out_root, setup


if __name__ == "__main__":
    json_path, out_root, setup = _parse_args(sys.argv)
    asyncio.run(run(json_path, out_root, force_setup=setup))
