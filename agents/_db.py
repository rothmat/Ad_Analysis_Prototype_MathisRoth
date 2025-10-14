# agents/_db.py
from __future__ import annotations
import os, sys, importlib.util
from pathlib import Path
from typing import Callable, List

def _try_import_direct() -> Callable | None:
    try:
        import db_client  # type: ignore
        return getattr(db_client, "connect", None)
    except Exception:
        return None

def _candidate_paths() -> List[Path]:
    here = Path(__file__).resolve()
    ps = []

    # 1) ENV überschreibt alles: AD_DB_ROOT = <pfad-zu-ad-db>
    env_root = os.getenv("AD_DB_ROOT")
    if env_root:
        ps.append(Path(env_root))

    # 2) verschiede mögliche Repo-Wurzeln relativ zur Datei
    ps += [
        here.parents[2],  # .../Code
        here.parents[1],  # .../agenticAi_system_v2
        here.parents[3],  # .../Masterthesis
        Path.cwd(),       # aktuelles Arbeitsverzeichnis
    ]

    # unter jedem Root sowohl ad-db als auch ad_db testen
    out = []
    for root in ps:
        out.append(root / "ad-db" / "ingest")
        out.append(root / "ad_db" / "ingest")
        # häufig auch im Projekt selbst eingebettet:
        out.append(root / "agenticAi_system_v2" / "ad-db" / "ingest")
    # Du kannst hier leicht weitere Varianten ergänzen
    return out

def _load_connect():
    # a) falls db_client schon importierbar ist
    fn = _try_import_direct()
    if callable(fn):
        return fn

    # b) dynamisch aus möglichen Orten laden
    tried = []
    for base in _candidate_paths():
        candidate = base / "db_client.py"
        tried.append(str(candidate))
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("db_client", candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["db_client"] = mod
                spec.loader.exec_module(mod)
                fn = getattr(mod, "connect", None)
                if callable(fn):
                    return fn

    raise FileNotFoundError(
        "db_client.py nicht gefunden.\n"
        "Versuchte Pfade:\n  - " + "\n  - ".join(tried) + "\n\n"
        "Lösung: Setze ENV AD_DB_ROOT auf den Pfad zur ad-db Repo-Wurzel\n"
        "oder verschiebe die Repos so, dass <Wurzel>/ad-db/ingest/db_client.py existiert."
    )

# öffentlich machen
connect = _load_connect()
