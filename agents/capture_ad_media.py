# capture_ad_media.py
import asyncio
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from urllib.parse import urlparse

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Frame,
    TimeoutError as PWTimeoutError,
)

# === DB: db_client aus ../ad-db/ingest laden ===
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ad-db" / "ingest"))
from db_client import (  # type: ignore
    connect,
    get_or_create_campaign,
    upsert_ad,
    add_media_local,
    upsert_media_base64_from_path,  # optional für Base64
)

# --- Optionen ---
ENCODE_BASE64_IMAGES = True   # Base64 für Bilder zusätzlich in media_base64 speichern?
BATCH_SIZE = 20               # nach wie vielen Dateien (gesamt) committen?

# ---------------------------
# JSON laden (Array, {"data":[...]}, NDJSON)
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
# Cookie-Dialog schließen (kompakt)
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
        dlg = await frame.wait_for_selector('div[role="dialog"], [data-cookiebanner]', timeout=1200)
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

    chosen = await pick_by_regex(ESSENTIAL_RE) or await pick_by_regex(ACCEPT_ALL_RE)
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
        pass

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
            await page.wait_for_selector('div[role="dialog"]', state="hidden", timeout=1000)
        except PWTimeoutError:
            try:
                await page.wait_for_selector('div[role="dialog"]', state="detached", timeout=700)
            except Exception:
                pass
    return clicked


# ---------------------------
# Ad-Container Heuristik
# ---------------------------
AD_CONTAINER_SELECTORS = [
    '#facebook .x1n2onr6',
    'div[data-pagelet="root"]',
    'div[role="article"]',
]

# ---------------------------
# Hilfen: Endungen bestimmen
# ---------------------------
def _ext_from_url_or_ct(url: str, content_type: Optional[str]) -> str:
    path = urlparse(url).path
    lower = path.lower()
    for ext in (".mp4", ".webm", ".mov", ".m4v", ".m3u8"):
        if lower.endswith(ext):
            return ext
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        if lower.endswith(ext):
            return ext
    if content_type:
        ct = content_type.lower().split(";")[0].strip()
        mapping = {
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "video/quicktime": ".mov",
            "application/vnd.apple.mpegurl": ".m3u8",
            "application/x-mpegurl": ".m3u8",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        return mapping.get(ct, "")
    return ""


# ---------------------------
# DOM-Helfer für Medien
# ---------------------------
async def _best_img_url(el) -> Optional[str]:
    try:
        srcset = await el.get_attribute("srcset")
        if srcset:
            parts = [p.strip() for p in srcset.split(",") if p.strip()]
            if parts:
                last = parts[-1].split()[0]
                return last
        src = await el.get_attribute("src")
        if src:
            return src
    except Exception:
        pass
    return None

async def _video_urls_from_element(el) -> List[str]:
    urls = []
    try:
        current = await el.evaluate("el => el.currentSrc || el.src || ''")
        if current and current.startswith("http"):
            urls.append(current)
    except Exception:
        pass
    try:
        src = await el.get_attribute("src")
        if src and src.startswith("http"):
            urls.append(src)
    except Exception:
        pass
    try:
        sources = await el.query_selector_all("source[src]")
        for s in sources:
            u = await s.get_attribute("src")
            if u and u.startswith("http"):
                urls.append(u)
    except Exception:
        pass
    # Duplikate entfernen
    dedup = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".m4v")
HLS_EXTS = (".m3u8",)

# ---------------------------
# Responses mitschneiden + DOM auslesen
# ---------------------------
async def collect_media(page: Page) -> Tuple[List[str], List[str], List[str]]:
    """returns (image_urls, video_file_urls, hls_playlists)"""
    img_urls, vid_urls, hls_urls = set(), set(), set()

    def on_response(response):
        try:
            url = response.url
            ct = (response.headers.get("content-type") or "").lower()
            low = url.lower()
            if ct.startswith("image/") or any(low.endswith(x) for x in (".png",".jpg",".jpeg",".gif",".webp")):
                if url.startswith("http"):
                    img_urls.add(url)
            elif ct.startswith("video/") or any(low.endswith(x) for x in VIDEO_EXTS):
                if url.startswith("http"):
                    vid_urls.add(url)
            elif ("mpegurl" in ct) or any(low.endswith(x) for x in HLS_EXTS):
                if url.startswith("http"):
                    hls_urls.add(url)
        except Exception:
            pass

    page.on("response", on_response)

    # 1) Scope: Ad-Container oder ganze Seite
    container = None
    for sel in AD_CONTAINER_SELECTORS:
        try:
            container = await page.query_selector(sel)
            if container:
                break
        except Exception:
            continue
    scope = container or page

    # 2) Bilder im DOM
    try:
        for img in await scope.query_selector_all("img"):
            u = await _best_img_url(img)
            if u and u.startswith("http"):
                img_urls.add(u)
    except Exception:
        pass

    # 3) Videos im DOM (src / source)
    dom_video_urls = []
    try:
        vids = await scope.query_selector_all("video")
        for v in vids:
            dom_video_urls += await _video_urls_from_element(v)
    except Exception:
        pass

    for u in dom_video_urls:
        low = u.lower()
        if any(low.endswith(x) for x in VIDEO_EXTS):
            vid_urls.add(u)
        elif any(low.endswith(x) for x in HLS_EXTS):
            hls_urls.add(u)

    # 4) Playback kurz anstoßen – triggert zusätzliche Requests
    try:
        await page.evaluate("""
          () => {
            const vids = Array.from(document.querySelectorAll('video'));
            for (const v of vids) {
              try { v.muted = true; v.playsInline = true; v.autoplay = true; v.play().catch(()=>{}); } catch(e) {}
            }
          }""")
        await page.wait_for_timeout(900)
    except Exception:
        pass

    try:
        page.off("response", on_response)
    except Exception:
        pass

    # Filter
    img_urls = {u for u in img_urls if u.startswith("http")}
    vid_urls = {u for u in vid_urls if u.startswith("http")}
    hls_urls = {u for u in hls_urls if u.startswith("http")}
    return sorted(img_urls), sorted(vid_urls), sorted(hls_urls)

# --- Video-URL-Helfer über alle Frames + Fallback Regex ---

async def strict_video_srcs_all_frames(page: Page) -> list[str]:
    """Sucht in ALLEN Frames nach <video>-Quellen (currentSrc/src/<source>)."""
    found: list[str] = []
    seen = set()
    for fr in page.frames:
        try:
            urls = await fr.evaluate("""
            () => {
              const out = new Set();
              const vids = Array.from(document.querySelectorAll('video'));
              for (const v of vids) {
                try { if (v.currentSrc && v.currentSrc.startsWith('http')) out.add(v.currentSrc); } catch(e){}
                try { if (v.src && v.src.startsWith('http')) out.add(v.src); } catch(e){}
                const sources = v.querySelectorAll('source[src]');
                sources.forEach(s => { try { if (s.src && s.src.startsWith('http')) out.add(s.src); } catch(e){} });
              }
              return Array.from(out);
            }
            """)
        except Exception:
            urls = []
        for u in urls or []:
            if isinstance(u, str) and u.startswith("http") and u not in seen:
                seen.add(u)
                found.append(u)
    return found

async def regex_video_urls_from_html(page: Page) -> list[str]:
    """Greift als Fallback auf rohen HTML-Text zurück und extrahiert .mp4/.webm/.mov/.m4v-URLs."""
    try:
        html = await page.content()
    except Exception:
        return []
    rx = re.compile(r"https?://[^\s\"'<>]+?\.(?:mp4|webm|mov|m4v)(?:\?[^\"'<> ]*)?", re.I)
    urls = rx.findall(html)
    dedup, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup

# ---------------------------
# Download via Context.request (nimmt Cookies/Headers mit)
# ---------------------------
async def download_to(context: BrowserContext, url: str, target_path: Path, referer: Optional[str]=None) -> Optional[Path]:
    try:
        headers = {}
        if referer:
            headers["Referer"] = referer
        resp = await context.request.get(url, headers=headers)
        if not resp.ok:
            return None
        ct = resp.headers.get("content-type", "")
        ext = _ext_from_url_or_ct(url, ct) or target_path.suffix
        out = target_path.with_suffix(ext) if ext else target_path
        out.parent.mkdir(parents=True, exist_ok=True)
        data = await resp.body()
        out.write_bytes(data)
        return out
    except Exception:
        return None


# ---------------------------
# *** Robuster Video-Sniffer ***
# ---------------------------
async def play_in_all_frames(page: Page) -> None:
    for fr in page.frames:
        try:
            await fr.evaluate("""
              () => { for (const v of document.querySelectorAll('video')) {
                try { v.muted = true; v.playsInline = true; v.autoplay = true; v.play().catch(()=>{}); } catch(e) {}
              }}
            """)
        except Exception:
            pass
    await page.wait_for_timeout(800)

async def sniff_and_save_videos(page: Page, vid_dir: Path, base_name: str, sniff_ms: int = 3000) -> list[str]:
    """
    Response-Sniffer für ~sniff_ms ms.
    Wartet Tasks ab und macht Dir-Diff, damit wirklich alle gespeicherten Dateien zurückgegeben werden.
    """
    saved: list[str] = []
    tasks: list[asyncio.Task] = []

    before = {p.name for p in vid_dir.glob(f"{base_name}_*.*")}

    async def save_response_body(resp):
        try:
            url = resp.url
            ct = (resp.headers.get("content-type") or "").lower()
            if not (ct.startswith("video/") or url.lower().endswith((".mp4", ".webm", ".mov", ".m4v"))):
                return
            body = await resp.body()

            lo = url.lower()
            ext = ".mp4"
            if "webm" in ct or lo.endswith(".webm"): ext = ".webm"
            elif "quicktime" in ct or lo.endswith(".mov"): ext = ".mov"
            elif lo.endswith(".m4v"): ext = ".m4v"

            existing = sorted(vid_dir.glob(f"{base_name}_*{ext}"))
            next_idx = len(existing) + 1

            out = vid_dir / f"{base_name}_{next_idx}{ext}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
            saved.append(out.name)
        except Exception:
            pass

    def on_response(r): tasks.append(asyncio.create_task(save_response_body(r)))

    page.on("response", on_response)
    await play_in_all_frames(page)
    await page.wait_for_timeout(sniff_ms)

    try:
        page.off("response", on_response)
    except Exception:
        pass
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    after = {p.name for p in vid_dir.glob(f"{base_name}_*.*")}
    for name in sorted(after - before):
        if name not in saved:
            saved.append(name)

    return saved


# ---------------------------
# Routing: nur Fonts blocken (Media/CSS durchlassen!)
# ---------------------------
async def enable_fast_routing(context: BrowserContext):
    async def handler(route, request):
        rt = request.resource_type
        url = request.url.lower()
        if rt in ("font",):
            return await route.abort()
        if url.endswith((".woff", ".woff2", ".ttf", ".otf", ".eot", ".map")):
            return await route.abort()
        return await route.continue_()
    await context.route("**/*", handler)


# ---------------------------
# Verarbeitung einer Ad (nur Download + Konsole)
# ---------------------------
async def process_ad(page: Page, context: BrowserContext, ad_id: str, url: str, img_dir: Path, vid_dir: Path):
    await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
    await try_close_cookies(page)

    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
        await page.wait_for_timeout(120)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(120)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(100)
    except Exception:
        pass

    img_urls, vid_file_urls, hls_urls = await collect_media(page)

    saved_imgs, saved_vids, saved_hls = [], [], []
    referer = page.url

    # Bilder
    for idx, u in enumerate(img_urls, 1):
        target = img_dir / f"{ad_id}_{idx}"
        out = await download_to(context, u, target, referer=referer)
        if out:
            # Konvertierung zu PNG machst du extern in der Bild-Pipeline; hier lassen wir original
            # (alternativ: Pillow einsetzen – in diesem Skript belassen wir die Download-Endung)
            saved_imgs.append(out.name)

    # Videos: DOM-Quellen -> Datei-URLs -> Sniffer (Fallback)
    def _save_vid(u_list: List[str], start_idx: int = 1):
        idx_local = start_idx
        async def _inner():
            nonlocal idx_local
            for u in u_list:
                out = await download_video_with_headers(context, u, vid_dir / f"{ad_id}_{idx_local}", referer=referer)
                if out:
                    saved_vids.append(out.name)
                    idx_local += 1
        return _inner()

    async def download_video_with_headers(context: BrowserContext, url: str, target_base: Path, referer: str) -> Optional[Path]:
        headers = {
            "User-Agent": context._options.get("user_agent") or "Mozilla/5.0",
            "Accept": "*/*",
            "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
            "Referer": referer,
            "Origin": "https://www.facebook.com",
            "Range": "bytes=0-",
            "Connection": "keep-alive",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Dest": "video",
        }
        try:
            resp = await context.request.get(url, headers=headers, timeout=30_000)
            if not resp.ok:
                if resp.status in (403, 416):
                    headers.pop("Range", None)
                    resp = await context.request.get(url, headers=headers, timeout=30_000)
                    if not resp.ok:
                        return None
                else:
                    return None
            ct = (resp.headers.get("content-type") or "").lower()
            lo = url.lower()
            ext = ".mp4"
            if "webm" in ct or lo.endswith(".webm"): ext = ".webm"
            elif "quicktime" in ct or lo.endswith(".mov"): ext = ".mov"
            elif lo.endswith(".m4v"): ext = ".m4v"
            out = target_base.with_suffix(ext)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(await resp.body())
            return out
        except Exception:
            return None

    dom_vid_urls = await strict_video_srcs_all_frames(page)
    if not dom_vid_urls:
        await play_in_all_frames(page)
        dom_vid_urls = await strict_video_srcs_all_frames(page)
    if not dom_vid_urls:
        dom_vid_urls = await regex_video_urls_from_html(page)

    await _save_vid(dom_vid_urls, 1)
    await _save_vid(vid_file_urls, len(saved_vids) + 1)

    if not saved_vids:
        sniffed = await sniff_and_save_videos(page, vid_dir, ad_id)
        saved_vids.extend(sniffed)

    # HLS-Playlists (.m3u8)
    for idx, u in enumerate(hls_urls, 1):
        target = vid_dir / f"{ad_id}_hls_{idx}.m3u8"
        out = await download_to(context, u, target, referer=referer)
        if out:
            saved_hls.append(out.name)

    # Konsolen-Preview
    def _preview(names: list[str], n=2) -> str:
        if not names:
            return ""
        if len(names) <= n:
            return ", ".join(names)
        return f"{', '.join(names[:n])} …"

    print(f"   🖼  Bilder gespeichert: {len(saved_imgs)} ({_preview(saved_imgs)})")
    print(f"   🎬 Videos gespeichert: {len(saved_vids)} ({_preview(saved_vids)})")
    print(f"   🎞  HLS gespeichert:   {len(saved_hls)} ({_preview(saved_hls)})")

    return saved_imgs, saved_vids, saved_hls


# ---------------------------
# Storage-State Datei
# ---------------------------
def _storage_path() -> Path:
    return Path(__file__).resolve().parent / "fb_storage_state.json"


# ---------------------------
# Backfill-Helfer: noch nicht registrierte Videos in der DB anlegen
# ---------------------------
VIDEO_NAME_RE = re.compile(r'^(?P<ad>\d+)(?:_hls)?_(?P<idx>\d+)\.(mp4|webm|mov|m4v|m3u8)$', re.I)

def backfill_register_new_videos(
    conn,
    campaign_id: int,
    vid_dir: Path,
    known: Set[str],
    date_folder: str,
    batch_size: int = 100
) -> int:
    inserted = 0
    # Nur neue Dateien prüfen (gegen known)
    for p in sorted(vid_dir.glob("*")):
        if not p.is_file() or p.name in known:
            continue
        m = VIDEO_NAME_RE.match(p.name)
        if not m:
            continue
        ad_external_id = m.group("ad")
        try:
            ad_pk = upsert_ad(conn, campaign_id, ad_external_id)
            media_pk = add_media_local(conn, ad_pk, str(p), kind="video", date_folder=date_folder)
            if media_pk:
                known.add(p.name)
                inserted += 1
                if inserted % batch_size == 0:
                    conn.commit()
        except Exception:
            # nicht abbrechen – nächstes File versuchen
            continue
    return inserted


# ---------------------------
# Run (sequentiell) + DB Writes
# ---------------------------
async def run(json_path: Path, out_root: Path, force_setup: bool = False) -> None:
    ads = load_ads(json_path)
    if not ads:
        print("Keine gültigen Ads mit id + ad_snapshot_url gefunden.")
        return

    # Datum aus JSON-Dateiname (YYYY-MM-DD) oder heute
    json_stem = json_path.stem
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", json_stem):
        json_date = json_stem
    else:
        json_date = datetime.now().strftime("%Y-%m-%d")

    # Ordnerstruktur
    run_date = datetime.now().strftime("%Y-%m-%d")
    media_root = out_root / "Media" / run_date
    img_dir = media_root / "images" / json_date
    vid_dir = media_root / "videos" / json_date
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    # Kampagne aus Ordner ableiten + DB-Init
    campaign_name = out_root.name
    campaign_slug = campaign_name.lower().replace(" ", "_")
    conn = None
    campaign_id = None
    try:
        conn = connect()
        campaign_id = get_or_create_campaign(conn, campaign_name, campaign_slug)
    except Exception as e:
        print(f"⚠️  Konnte DB nicht initialisieren: {e}")

    storage_file = _storage_path()
    needs_setup = force_setup or (not storage_file.exists())

    async with async_playwright() as p:
        # Setup (einmalig)
        if needs_setup:
            import tempfile
            profdir_mgr = tempfile.TemporaryDirectory(prefix="fb_setup_profile_")
            profdir = profdir_mgr.__enter__()

            context = await p.chromium.launch_persistent_context(
                user_data_dir=profdir,
                headless=False,
                args=["--no-first-run", "--no-default-browser-check"],
                locale="de-CH",
                viewport=None,
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, wie Gecko) "
                            "Chrome/123.0.0.0 Safari/537.36"),
            )
            try:
                try:
                    await context.grant_permissions(["autoplay"], origin="https://www.facebook.com")
                except Exception:
                    pass

                page = await context.new_page()
                first_url = ads[0]["ad_snapshot_url"]
                try:
                    await page.goto(first_url, wait_until="domcontentloaded", timeout=45_000)
                except Exception:
                    fallback = ("https://www.facebook.com/ads/library/?active_status=all"
                                "&ad_type=political_and_issue_ads&country=CH")
                    await page.goto(fallback, wait_until="domcontentloaded", timeout=45_000)

                try:
                    await try_close_cookies(page)
                except Exception:
                    pass

                print("\n⚠️  Setup: Falls noch sichtbar, Cookie/Consent manuell schließen.")
                print("   Danach ENTER im Terminal drücken – der Zustand wird gespeichert.\n")
                input()

                await context.storage_state(path=str(storage_file))
            finally:
                await context.close()
                profdir_mgr.cleanup()

            print(f"✅ Consent gespeichert in: {storage_file}\n")

        # Produktivlauf (headless)
        browser: Browser = await p.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox", "--autoplay-policy=no-user-gesture-required"]
        )
        context: BrowserContext = await browser.new_context(
            storage_state=str(storage_file),
            locale="de-CH",
            viewport={"width": 1280, "height": 1800},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, wie Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"),
        )
        try:
            await context.grant_permissions(["autoplay"], origin="https://www.facebook.com")
        except Exception:
            pass
        await enable_fast_routing(context)

        page: Page = await context.new_page()
        page.set_default_timeout(12_000)

        total = len(ads)
        inserted_media = 0
        known_video_files: Set[str] = set()  # um doppelte Backfills im Lauf zu vermeiden

        for i, ad in enumerate(ads, 1):
            ad_id = str(ad["id"])
            url = ad["ad_snapshot_url"]
            print(f"[{i}/{total}] Medien für Ad {ad_id} …")

            try:
                saved_imgs, saved_vids, saved_hls = await process_ad(page, context, ad_id, url, img_dir, vid_dir)
            except Exception as e:
                print(f"   ❌ Fehler bei {ad_id}: {e}")
                continue

            # --- DB: alles zu dieser Ad registrieren ---
            if conn is not None and campaign_id is not None:
                try:
                    ad_pk = upsert_ad(conn, campaign_id, ad_id)

                    # Bilder
                    for name in saved_imgs:
                        p = img_dir / name
                        media_pk = add_media_local(conn, ad_pk, str(p), kind="image", date_folder=json_date)
                        if ENCODE_BASE64_IMAGES and media_pk:
                            upsert_media_base64_from_path(conn, media_pk, str(p))
                        if media_pk:
                            inserted_media += 1
                            if inserted_media % BATCH_SIZE == 0:
                                conn.commit()
                                # ⤵ Backfill-Scan (Videos), weil Sniffer evtl. leicht nachläuft
                                added = backfill_register_new_videos(conn, campaign_id, vid_dir, known_video_files, json_date, batch_size=BATCH_SIZE)
                                if added:
                                    print(f"   ⤴ Backfill: {added} neue Video(s) in DB registriert")

                    # Videos (inkl. .m3u8 als 'video')
                    for name in saved_vids + saved_hls:
                        p = vid_dir / name
                        media_pk = add_media_local(conn, ad_pk, str(p), kind="video", date_folder=json_date)
                        if media_pk:
                            known_video_files.add(name)
                            inserted_media += 1
                            if inserted_media % BATCH_SIZE == 0:
                                conn.commit()
                                added = backfill_register_new_videos(conn, campaign_id, vid_dir, known_video_files, json_date, batch_size=BATCH_SIZE)
                                if added:
                                    print(f"   ⤴ Backfill: {added} neue Video(s) in DB registriert")

                except Exception as db_e:
                    print(f"   ⚠️  DB-Insert fehlgeschlagen (ad={ad_id}): {db_e}")

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

    # Finaler Backfill + Commit/Close DB
    if conn is not None and campaign_id is not None:
        try:
            added = backfill_register_new_videos(conn, campaign_id, vid_dir, known_video_files, json_date, batch_size=BATCH_SIZE)
            if added:
                print(f"\n⤴ Finaler Backfill: {added} weitere Video(s) in DB registriert")
            conn.commit()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    print(f"\nFertig. Medien liegen unter: {media_root.resolve()}")
    print(f"   📁 Bilder: {img_dir.resolve()}")
    print(f"   📁 Videos: {vid_dir.resolve()}")


# ---------------------------
# CLI
# ---------------------------
def _parse_args(argv: List[str]) -> tuple[Path, Path, bool]:
    """
    Usage:
      python capture_ad_media.py <ads.json> [<output_root>] [--setup]
      (kompatibel) python capture_ad_media.py <ads.json> <campaign_name> <output_root> [--setup]
    """
    if len(argv) < 2:
        print("Usage:")
        print("  python capture_ad_media.py <ads.json> [<output_root>] [--setup]")
        print("  (kompatibel) python capture_ad_media.py <ads.json> <campaign_name> <output_root> [--setup]")
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
