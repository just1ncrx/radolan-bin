import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/radarsumme")
OUT_DIR = Path("output/radarsumme24h")
OUT_NAME = "radarsumme_latest"

# raa01-rw_10000-YYMMDDHHMM-dwd---bin.hdf5  (Zeit = Ende des 1h-Fensters, UTC)
FILENAME_RE = re.compile(r"raa01-rw_10000-(\d{10})-dwd---bin\.hdf5$")
HOURS = 24
TOLERANCE_MIN = 30   # Datei gilt fuer eine Stunde, wenn ihr Zeitstempel max. so weit abweicht

RW_DEFAULT_GAIN = 0.1
RW_DEFAULT_NODATA = 65535

MAX_VALID_MM = 500.0        # Stundensumme in mm: alles darueber = ungueltig
UNIT = 0.2                  # mm je Einheit im uint8 -> max. 254 * 0.2 = 50,8 mm pro Stunde
NODATA = 255                # uint8-Wert fuer keine Daten
MAX_CODE = NODATA - 1       # groesster gueltiger Wert (254)

MASK_OUTSIDE_RADAR = True
RADAR_RANGE_KM = 150.0
RADAR_MARGIN_KM = 0.0

RADARS = {
    "asb": dict(name="ASR Borkum",     lat=53.564011, lon=6.748292),
    "boo": dict(name="Boostedt",       lat=54.00438,  lon=10.04687),
    "drs": dict(name="Dresden",        lat=51.12465,  lon=13.76865),
    "eis": dict(name="Eisberg",        lat=49.54066,  lon=12.40278),
    "emd": dict(name="Emden",          lat=53.33872,  lon=7.02377),
    "ess": dict(name="Essen",          lat=51.40563,  lon=6.96712),
    "fbg": dict(name="Feldberg",       lat=47.87361,  lon=8.00361),
    "fld": dict(name="Flechtdorf",     lat=51.31120,  lon=8.802),
    "hnr": dict(name="Hannover",       lat=52.46008,  lon=9.69452),
    "neu": dict(name="Neuhaus",        lat=50.50012,  lon=11.13504),
    "nhb": dict(name="Neuheilenbach",  lat=50.10965,  lon=6.54853),
    "oft": dict(name="Offenthal",      lat=49.9847,   lon=8.71293),
    "pro": dict(name="Prötzel",        lat=52.64867,  lon=13.85821),
    "mem": dict(name="Memmingen",      lat=48.04214,  lon=10.21924),
    "ros": dict(name="Rostock",        lat=54.17566,  lon=12.05808),
    "isn": dict(name="Isen",           lat=48.17470,  lon=12.10177),
    "tur": dict(name="Türkheim",       lat=48.58528,  lon=9.78278),
    "umd": dict(name="Ummendorf",      lat=52.16009,  lon=11.17609),
}


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_filename(filename: str) -> datetime:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Dateiname passt nicht zu 'raa01-rw_10000-YYMMDDHHMM-dwd---bin.hdf5': {filename}")
    return datetime.strptime(m.group(1), "%y%m%d%H%M").replace(tzinfo=timezone.utc)


def index_files(src_dir: Path) -> dict[datetime, Path]:
    out = {}
    for p in src_dir.glob("raa01-rw_10000-*-dwd---bin.hdf5"):
        if FILENAME_RE.match(p.name):
            out[parse_filename(p.name)] = p
    return out


def find_hour_file(index: dict[datetime, Path], t: datetime) -> Path | None:
    best, best_diff = None, timedelta(minutes=TOLERANCE_MIN, seconds=1)
    for ts, p in index.items():
        diff = abs(ts - t)
        if diff < best_diff:
            best, best_diff = p, diff
    return best


def _attr_str(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def find_data_dataset(h5file: h5py.File) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError("Kein 2D-Datensatz in der RW-Datei gefunden.")
    return candidates[0]


def read_mm(ds: h5py.Dataset) -> np.ndarray:
    """Dataset -> Stundensumme in mm (NaN = kein Wert)."""
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    attrs = what.attrs if what is not None else {}

    gain = float(_scalar(attrs.get("gain", RW_DEFAULT_GAIN)))
    offset = float(_scalar(attrs.get("offset", 0.0)))
    nodata = _scalar(attrs.get("nodata", RW_DEFAULT_NODATA))
    undetect = _scalar(attrs["undetect"]) if "undetect" in attrs else None

    raw = ds[()]
    values = raw.astype(np.float64) * gain + offset

    invalid = raw == nodata
    if np.issubdtype(raw.dtype, np.integer):
        invalid |= raw == np.iinfo(raw.dtype).max
    invalid |= ~np.isfinite(values)
    invalid |= values < 0
    invalid |= values > MAX_VALID_MM

    if undetect is not None and undetect != nodata:
        values[raw == undetect] = 0.0
        invalid &= raw != undetect
    values[invalid] = np.nan
    return values


def find_where_group(h5file: h5py.File):
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_float(key: str) -> float:
        return float(_scalar(where.attrs[key]))

    return {
        "projdef": _attr_str(where.attrs["projdef"]),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


def radar_coverage_mask(grid: dict, to_proj: Transformer) -> np.ndarray:
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    xs = ll_x + (np.arange(grid["xsize"]) + 0.5) * grid["xscale"]
    ys = ll_y + (grid["ysize"] - 1 - np.arange(grid["ysize"]) + 0.5) * grid["yscale"]
    xx, yy = np.meshgrid(xs, ys)

    limit = (RADAR_RANGE_KM + RADAR_MARGIN_KM) * 1000.0
    covered = np.zeros(xx.shape, dtype=bool)
    for info in RADARS.values():
        rx, ry = to_proj.transform(info["lon"], info["lat"])
        covered |= np.hypot(xx - rx, yy - ry) <= limit
    return covered


# --------------------------------------------------------------------------- #
# Abfrage (lat/lon -> 24 Zeitstempel + Stundensumme in mm)
# --------------------------------------------------------------------------- #
def query(data_path, lat: float, lon: float, meta: dict | None = None):
    """Liest nur die 24 Werte des Pixels per seek. Gibt None zurueck, falls ausserhalb des Rasters."""
    data_path = Path(data_path)
    if meta is None:
        meta = json.loads(data_path.with_name(data_path.name + ".json").read_text())

    w, h, t = meta["width"], meta["height"], meta["hours"]
    to_proj = Transformer.from_crs("EPSG:4326", meta["projdef"], always_xy=True)
    x, y = to_proj.transform(lon, lat)

    col = int(np.floor((x - meta["ll_x"]) / meta["pixel_size_m"]))
    row = h - 1 - int(np.floor((y - meta["ll_y"]) / meta["pixel_size_m"]))
    if not (0 <= col < w and 0 <= row < h):
        return None

    with open(data_path, "rb") as f:
        f.seek((row * w + col) * t)          # 1 Byte pro Wert
        vals = np.frombuffer(f.read(t), dtype="u1")

    return [
        {"time": ts, "mm": None if v == meta["nodata"] else round(float(v) * meta["unit_mm"], 1)}
        for ts, v in zip(meta["timestamps"], vals)
    ]


def sum_mm(series: list[dict]) -> tuple[float, int]:
    """24-h-Summe in mm (float, kein Ueberlauf) und Anzahl Stunden ohne Daten."""
    vals = [r["mm"] for r in series if r["mm"] is not None]
    return round(float(np.sum(vals)), 1), len(series) - len(vals)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        args = [a for a in sys.argv[2:] if a != "--sum"]
        want_sum = "--sum" in sys.argv[2:]
        lat, lon, path = float(args[0]), float(args[1]), args[2]
        res = query(path, lat, lon)
        if res is None:
            sys.exit("Koordinate ausserhalb des Rasters.")
        for r in res:
            print(r["time"], r["mm"])
        if want_sum:
            total, n_missing = sum_mm(res)
            print(f"Summe {len(res)} h: {total} mm" +
                  (f" ({n_missing} Stunden ohne Daten)" if n_missing else ""))
        return

    if len(sys.argv) > 1:
        end_path = Path(sys.argv[1])
        src_dir = end_path.parent
    else:
        src_dir = SRC_DIR
        candidates = sorted(p for p in src_dir.glob("raa01-rw_10000-*-dwd---bin.hdf5")
                            if FILENAME_RE.match(p.name))
        if not candidates:
            sys.exit(f"Keine RW-Datei in {src_dir} gefunden.")
        end_path = candidates[-1]

    index = index_files(src_dir)
    end_ts = parse_filename(end_path.name)
    times = [end_ts - timedelta(hours=h) for h in range(HOURS - 1, -1, -1)]   # aelteste zuerst
    print(f"Fenster: {times[0]:%Y-%m-%d %H:%M} .. {times[-1]:%Y-%m-%d %H:%M} UTC ({HOURS} Stunden)")

    grid = None
    to_proj = None
    covered = None
    stack = None      # (HOURS, ysize, xsize) uint8
    missing = []
    clipped_total = 0

    for i, t in enumerate(times):
        p = find_hour_file(index, t)
        if p is None:
            missing.append(t)
            continue

        with h5py.File(p, "r") as f:
            if grid is None:
                where = find_where_group(f)
                if where is None:
                    sys.exit("Keine 'where'-Projektionsinfo in der HDF5-Datei gefunden.")
                grid = extract_grid_info(where)
                to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
                stack = np.full((HOURS, grid["ysize"], grid["xsize"]), NODATA, dtype="u1")
                if MASK_OUTSIDE_RADAR:
                    covered = radar_coverage_mask(grid, to_proj)
            mm = read_mm(find_data_dataset(f))

        if mm.shape != stack.shape[1:]:
            sys.exit(f"{p.name}: Rastergroesse {mm.shape} passt nicht zu {stack.shape[1:]}.")
        if covered is not None:
            mm = np.where(covered, mm, np.nan)

        ok = np.isfinite(mm)
        codes = np.round(mm[ok] / UNIT)
        clipped = int(np.count_nonzero(codes > MAX_CODE))
        clipped_total += clipped

        layer = np.full(mm.shape, NODATA, dtype="u1")
        # clip ist Pflicht: astype("u1") wuerde bei Ueberlauf sonst umbrechen
        layer[ok] = np.clip(codes, 0, MAX_CODE).astype("u1")
        stack[i] = layer
        print(f"  {t:%Y-%m-%d %H:%M} UTC  <-  {p.name}"
              + (f"  ({clipped} Pixel auf {MAX_CODE * UNIT:.1f} mm gekappt)" if clipped else ""))

    if grid is None:
        sys.exit("Keine der 24 RW-Dateien vorhanden.")

    if missing:
        print("WARNUNG: fehlende Stunden (Wert = keine Daten): "
              + ", ".join(f"{m:%d.%m. %H:%M}" for m in missing), file=sys.stderr)
    if clipped_total:
        print(f"HINWEIS: {clipped_total} Stundenwerte ueber {MAX_CODE * UNIT:.1f} mm wurden gekappt.")

    # (T, H, W) -> (H, W, T): alle Stunden eines Pixels hintereinander
    stack = np.ascontiguousarray(stack.transpose(1, 2, 0))
    h, w, t = stack.shape
    print(f"Natives Raster: {w} x {h} px x {t} Stunden, {grid['xscale']} m / px")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    stack.tofile(tmp_path)

    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    meta = {
        "format": "raw-uint8",
        "layout": "row, col, hour",
        "width": w,
        "height": h,
        "hours": t,
        "unit": "mm (Stundensumme)",
        "unit_mm": UNIT,
        "max_mm": MAX_CODE * UNIT,
        "nodata": NODATA,
        "row_order": "north-to-south",
        "projdef": grid["projdef"],
        "pixel_size_m": grid["xscale"],
        "ll_x": ll_x,
        "ll_y": ll_y,
        "extent_native": [ll_x, ll_y, ll_x + w * grid["xscale"], ll_y + h * grid["yscale"]],
        "timestamps": [tt.isoformat() for tt in times],      # UTC, Ende der jeweiligen Stunde
        "hours_missing": [m.isoformat() for m in missing],
    }
    meta_json = json.dumps(meta, indent=2)

    tmp_path.replace(out_path)
    print(f"Gespeichert: {out_path} ({stack.nbytes / 1e6:.1f} MB)")
    print("Metadaten:")
    print(meta_json)


if __name__ == "__main__":
    main()