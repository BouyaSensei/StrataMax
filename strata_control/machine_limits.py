"""StrataMax: the caps of the machine, applied at every engine start.

The rule Axel gave: the engine must not exceed a limit, and the model must never stop. This module does not
invent a mechanism; it sets the ones the engine already has, at the point every start converges on
(`serve/server.py engine_args`):

  --vram-cap-act        the reserve acts by itself, at the start of each request (the engine gives back
                        its last cache segments; the experts there become CPU misses, no output change)
  --vram-reserve-mib    the target: that much VRAM must stay free for the rest of the machine
  --vram-elastic        required for a shrink to be possible (segmented arena, NVIDIA, --serve)
  --memory-limit-mib    the RAM the engine's own allocations may use; above it a parking is skipped, the
                        request is served (engine: `skip parking` then `return true`)
  STRATA_RESIDENT_HEADROOM_GIB  the RAM left free by the expert copy (it wins over the safety headroom)
  STRATA_VRAM_FREE=device       read the free VRAM of the whole card through NVML (can only make the
                        figure smaller; Windows CUDA builds only)

Opt-in: the run config carries `stratamax_limits`, e.g. {"vram_pct": 95, "ram_pct": 90}. Nothing is added
when the key is absent, and a flag the person already set is never overwritten.
"""
from __future__ import annotations

import ctypes
import os

# Axel's machine, measured on it 2026-11-16: 32 607 MiB of VRAM, 61.6 GiB of RAM reported.
DEFAULT_VRAM_PCT = 95      # 5% free = 1630 MiB, above the measured 900 MiB where decode falls to 80-115 tok/s
DEFAULT_RAM_PCT = 90       # 6.2 GiB free, above the 4 GiB a session file needs; under the 8 GiB safety headroom
MIN_FREE_VRAM_MIB = 1024   # the app's own floor: "never under 1024 MiB, below that the driver pages" (#2065)
SESSION_FILE_MIN_MIB = 4096


def total_ram_mib() -> int | None:
    """What the machine reports as its total RAM (the sticker value is not what the engine sees)."""
    if os.name == "posix":
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except OSError:
            return None
        return None
    try:
        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("dwTotalVirtual", ctypes.c_ulonglong), ("dwAvailVirtual", ctypes.c_ulonglong),
                        ("dwAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = M()
        m.dwLength = ctypes.sizeof(M)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return int(m.ullTotalPhys // (1 << 20))
    except Exception:
        return None


def limits(cfg: dict, vram_mib: int | None = None) -> dict | None:
    """The caps for this start, in MiB. None when the fork's key is not in the config."""
    raw = cfg.get("stratamax_limits")
    if not raw:
        return None
    if raw is True:
        raw = {}
    vram_pct = float(raw.get("vram_pct", DEFAULT_VRAM_PCT))
    ram_pct = float(raw.get("ram_pct", DEFAULT_RAM_PCT))
    out = {}
    card = vram_mib or raw.get("vram_total_mib")
    if card:
        free = max(MIN_FREE_VRAM_MIB, int(round(card * (100.0 - vram_pct) / 100.0)))
        out["vram_reserve_mib"] = min(free, int(card) - MIN_FREE_VRAM_MIB)
    ram = raw.get("ram_total_mib") or total_ram_mib()
    if ram:
        cap = int(round(ram * ram_pct / 100.0))
        # never leave less than a session file's room, whatever the percentage says
        free = max(ram - cap, SESSION_FILE_MIN_MIB)
        out["ram_limit_mib"] = ram - free
        out["ram_headroom_gib"] = max(1, int(round(free / 1024.0)))
    return out or None


def add_args(cfg: dict, args: list[str], vram_mib: int | None = None) -> list[str]:
    """Append the cap flags to an engine's arguments, leaving anything the person set alone."""
    lim = limits(cfg, vram_mib)
    if not lim:
        return args
    out = list(args)
    if "--vram-elastic" not in out:
        out.append("--vram-elastic")
    if "--vram-cap-act" not in out:
        out.append("--vram-cap-act")
    if "--vram-reserve-mib" not in out and "vram_reserve_mib" in lim:
        out += ["--vram-reserve-mib", str(lim["vram_reserve_mib"])]
    if "--memory-limit-mib" not in out and "ram_limit_mib" in lim:
        out += ["--memory-limit-mib", str(lim["ram_limit_mib"])]
    return out


def add_env(cfg: dict, env: dict) -> dict:
    """The engine's environment for the caps. A value the person set in the config's env is not touched."""
    lim = limits(cfg)
    if not lim:
        return env
    env.setdefault("STRATA_VRAM_FREE", "device")
    if "ram_headroom_gib" in lim:
        env.setdefault("STRATA_RESIDENT_HEADROOM_GIB", str(lim["ram_headroom_gib"]))
    return env


def note(lim: dict | None) -> str:
    if not lim:
        return ""
    bits = []
    if "vram_reserve_mib" in lim:
        bits.append(f"VRAM: keep {lim['vram_reserve_mib']} MiB free (the cache gives back its last segments below that)")
    if "ram_limit_mib" in lim:
        bits.append(f"RAM: {lim['ram_limit_mib']} MiB for the engine, {lim['ram_headroom_gib']} GiB left free "
                    "(a parking above the cap is skipped, the request is served)")
    return "[strata-max] " + "; ".join(bits)
