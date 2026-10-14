"""
emma2013_reader.py
------------------
Legge file binari in formato EMMA2013 e restituisce header e dati
come dizionari Python. Porta fedele della routine MATLAB readEMMA2013.

Uso:
    from emma2013_reader import read_emma2013
    header, data, error = read_emma2013("AQU20130101.bin")
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class EMMA2013Error:
    status: bool = False
    message: str = ""


@dataclass
class EMMA2013Header:
    # Record 1
    data_format: str = ""       # 'EMMA2013'
    source: str = ""            # es. 'UNIVAQ'
    # Record 2
    name: str = ""              # es. "L'Aquila"
    iaga_code: str = ""         # es. 'AQU'
    # Record 3
    latitude: float = 0.0       # gradi geografici (-90..90)
    longitude: float = 0.0      # gradi geografici (-180..180)
    elevation: int = 0          # metri (9999 = non noto)
    declination: float = 0.0    # gradi (default 999.99)
    reference_frame: str = ""   # es. 'HDZ'
    orientation: str = ""       # es. 'XYZ' o 'HDZ'
    # Record 4
    sps: int = 0                # campioni/secondo (sampling rate)
    data_rate: int = 0          # data rate in Hz
    data_version: str = ""      # 'R'=raw, 'C'=checked, 'D'=definitive
    data_type: str = ""         # es. 'var'
    flags: int = 0
    overwritten: int = 0
    missing: int = 0
    # Record 5
    filt_def_fi: str = ""       # filter definition file identifier
    # Record 6
    inst_def_fi: str = ""       # instrumentation definition file identifier
    # Record 7
    time_factor: int = 1000     # fattore s → int32 (default 1000 = ms)
    field_factor: int = 1000    # fattore nT → int32 (default 1000)
    year: int = 0
    month: int = 0
    day: int = 0
    extra_line_flag: str = ""
    extra_line: int = 0
    extra: list = field(default_factory=list)


@dataclass
class EMMA2013Data:
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    comp1: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    comp2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    comp3: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))


# ---------------------------------------------------------------------------
# Helpers di lettura binaria
# ---------------------------------------------------------------------------

def _read_chars(f, n: int) -> str:
    """Legge n byte e li decodifica come stringa ASCII (strip dei null)."""
    return f.read(n).decode("latin-1").rstrip("\x00").strip()


def _read_int16(f) -> int:
    return struct.unpack("<h", f.read(2))[0]


def _read_uint16(f) -> int:
    return struct.unpack("<H", f.read(2))[0]


def _read_uint8(f) -> int:
    return struct.unpack("<B", f.read(1))[0]


def _skip(f, n: int):
    f.read(n)


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def read_emma2013(filename: str | Path):
    """
    Legge un file in formato EMMA2013.

    Parametri
    ----------
    filename : str o Path
        Percorso al file binario EMMA2013.

    Restituisce
    -----------
    header : EMMA2013Header  (None in caso di errore)
    data   : EMMA2013Data    (None in caso di errore)
    error  : EMMA2013Error
    """
    error = EMMA2013Error()
    header = EMMA2013Header()
    data = EMMA2013Data()

    try:
        f = open(filename, "rb")
    except OSError as e:
        error.status = True
        error.message = str(e)
        return None, None, error

    try:
        # ── Record 1 (byte 1-16) ────────────────────────────────────────────
        header.data_format = _read_chars(f, 8)   # byte  1- 8
        header.source      = _read_chars(f, 8)   # byte  9-16

        # ── Record 2 (byte 17-32) ───────────────────────────────────────────
        header.name      = _read_chars(f, 12)    # byte 17-28
        _skip(f, 1)                              # byte 29  (non usato)
        header.iaga_code = _read_chars(f, 3)     # byte 30-32

        # ── Record 3 (byte 33-48) ───────────────────────────────────────────
        header.latitude        = _read_int16(f) / 100.0   # byte 33-34
        header.longitude       = _read_int16(f) / 100.0   # byte 35-36
        header.elevation       = _read_int16(f)            # byte 37-38
        header.declination     = _read_int16(f) / 100.0   # byte 39-40
        _skip(f, 1)                                        # byte 41
        header.reference_frame = _read_chars(f, 3)        # byte 42-44
        _skip(f, 1)                                        # byte 45
        header.orientation     = _read_chars(f, 3)        # byte 46-48

        # ── Record 4 (byte 49-64) ───────────────────────────────────────────
        header.sps          = _read_uint16(f)            # byte 49-50
        header.data_rate    = _read_uint16(f)            # byte 51-52
        header.data_version = _read_chars(f, 1)          # byte 53
        header.data_type    = _read_chars(f, 3)          # byte 54-56
        header.flags        = _read_uint16(f)            # byte 57-58
        header.overwritten  = _read_uint16(f)            # byte 59-60
        header.missing      = _read_uint16(f)            # byte 61-62
        _skip(f, 2)                                      # byte 63-64

        # ── Record 5 (byte 65-80) ───────────────────────────────────────────
        header.filt_def_fi = _read_chars(f, 16)          # byte 65-80

        # ── Record 6 (byte 81-96) ───────────────────────────────────────────
        header.inst_def_fi = _read_chars(f, 16)          # byte 81-96

        # ── Record 7 (byte 97-112) ──────────────────────────────────────────
        header.time_factor      = _read_uint16(f)        # byte  97-98
        header.field_factor     = _read_uint16(f)        # byte  99-100
        header.year             = _read_uint16(f)        # byte 101-102
        header.month            = _read_uint8(f)         # byte 103
        header.day              = _read_uint8(f)         # byte 104
        _skip(f, 6)                                      # byte 105-110
        header.extra_line_flag  = _read_chars(f, 1)      # byte 111
        header.extra_line       = _read_uint8(f)         # byte 112

        # ── Record extra (opzionale) ─────────────────────────────────────────
        if header.extra_line_flag == "+":
            for _ in range(header.extra_line):
                header.extra.append(list(f.read(16)))

        # ── Dati binari (4 × int32 per campione) ────────────────────────────
        raw = np.frombuffer(f.read(), dtype="<i4").reshape(-1, 4)

        if raw.size == 0:
            pass  # data rimane con array vuoti
        else:
            data.time  = raw[:, 0]
            data.comp1 = raw[:, 1]
            data.comp2 = raw[:, 2]
            data.comp3 = raw[:, 3]

    except Exception as e:
        error.status = True
        error.message = str(e)
        f.close()
        return None, None, error

    f.close()
    return header, data, error


# ---------------------------------------------------------------------------
# Stampa riassuntiva dell'header (utile per debug)
# ---------------------------------------------------------------------------

def print_header(header: EMMA2013Header):
    print(f"{'─'*40}")
    print(f"  Formato  : {header.data_format}")
    print(f"  Sorgente : {header.source}")
    print(f"  Stazione : {header.name}  [{header.iaga_code}]")
    print(f"  Lat/Lon  : {header.latitude:.2f}° / {header.longitude:.2f}°")
    print(f"  Elev.    : {header.elevation} m")
    print(f"  Data     : {header.year:04d}-{header.month:02d}-{header.day:02d}")
    print(f"  SPS      : {header.sps} Hz  |  DataRate: {header.data_rate} Hz")
    print(f"  Frame    : {header.reference_frame}  |  Orient.: {header.orientation}")
    print(f"  Versione : {header.data_version}  |  Tipo: {header.data_type}")
    print(f"  Mancanti : {header.missing}  |  Sovrascritti: {header.overwritten}")
    print(f"  TimeFact.: {header.time_factor}  |  FieldFact.: {header.field_factor}")
    print(f"{'─'*40}")
