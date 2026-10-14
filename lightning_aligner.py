"""
lightning_aligner.py
--------------------
Allinea i dati del magnetometro (EMMA2013, 1 Hz) con il catalogo fulmini
(CSV con precisione al millisecondo) e costruisce il dataset di finestre
pronte per la rete neurale.

Output principale: LightningDataset
    - windows       : np.ndarray  shape (N, window_len, 3)   [nT, float32]
    - labels        : np.ndarray  shape (N,)                 [float32, offset in sec]
    - has_lightning : np.ndarray  shape (N,) bool            True = finestra con fulmine
    - valid_mask    : np.ndarray  shape (N,) bool            False = scartata (NaN / no segnale / ...)
    - meta          : lista di dict con info per ogni finestra

Motivi di scarto (meta["discard_reason"]):
    "nan"               finestra con troppi NaN
    "no_signal"         positiva: fulmine in catalogo ma magnetometro piatto
    "overlap"           positiva: si sovrappone a un'altra finestra positiva
    "unlabeled_signal"  negativa: segnale impulsivo nel magnetometro ma nessun
                        fulmine in catalogo (probabile fulmine non registrato)

Uso rapido:
    from lightning_aligner import build_dataset, load_lightning_csv
    from emma2013_reader  import read_emma2013

    header, raw_data, err = read_emma2013("AQU20130101.bin")
    lightning_df = load_lightning_csv("fulmini.csv")
    dataset = build_dataset(header, raw_data, lightning_df, half_window=5)
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from emma2013_reader import EMMA2013Header, EMMA2013Data


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Valore sentinella usato in EMMA2013 per i dati mancanti (da specifica)
MISSING_VALUE_INT = 999999999   # valore int32 convenzionale per "dato mancante"

# Codici dei motivi di scarto (array LightningDataset.reason)
REASON_CODES = {None: 0, "nan": 1, "no_signal": 2, "overlap": 3, "unlabeled_signal": 4}
REASON_NAMES = {v: (k or "ok") for k, v in REASON_CODES.items()}


# ---------------------------------------------------------------------------
# Struttura dati del dataset
# ---------------------------------------------------------------------------

@dataclass
class LightningDataset:
    """
    Contenitore del dataset allineato.

    Attributi
    ---------
    windows : ndarray (N, window_len, 3)
        Finestre di segnale magnetometrico in nT (float32, NaN dove mancante).
        Dim 2: [comp1, comp2, comp3] → tipicamente [H, D, Z] o [X, Y, Z].

    labels : ndarray (N,) float32
        Offset in secondi del fulmine rispetto al centro della finestra.
        0.0 per le finestre senza fulmine (has_lightning=False).

    has_lightning : ndarray (N,) bool
        True se la finestra contiene un fulmine.

    valid_mask : ndarray (N,) bool
        True se la finestra è utilizzabile (niente NaN, niente falsi negativi
        ambigui, niente sovrapposizioni). Usare questo per filtrare prima
        del training.

    meta : list of dict
        Per ogni finestra: t_center, window_start, window_end, n_nan,
        lightning_times, type, discard_reason, max_deriv, ...
        (può essere vuota se build_dataset_from_files(keep_meta=False))

    [NUOVI] array per finestra, sempre presenti e salvati da save_dataset():
    t_center      : datetime64[s]  centro della finestra
    max_deriv     : float32  max |Δcomp| nella finestra (nT/s)
    noise_local   : float32  mediana di max|Δcomp| nei noise_window secondi
                             PRIMA della finestra (livello di rumore locale)
    dt_lightning  : float32  secondi (con segno) dal fulmine di catalogo più
                             vicino al centro finestra; +inf se nessuno
    day_has_lightning : bool  il giorno UTC della finestra ha fulmini in catalogo
    reason        : int8     0 = ok, poi vedi REASON_CODES
    """
    windows: np.ndarray = field(default_factory=lambda: np.empty((0, 11, 3), dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    has_lightning: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    valid_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    meta: list = field(default_factory=list)
    t_center: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="datetime64[s]"))
    max_deriv: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    noise_local: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    dt_lightning: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    day_has_lightning: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    reason: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))

    @property
    def snr(self) -> np.ndarray:
        """Salto massimo diviso per il rumore locale (adimensionale)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.max_deriv / self.noise_local

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def n_positive(self) -> int:
        return int((self.has_lightning & self.valid_mask).sum())

    @property
    def n_negative(self) -> int:
        return int((~self.has_lightning & self.valid_mask).sum())

    def discard_counts(self) -> dict:
        """Conteggio delle finestre scartate per motivo, es. {'no_signal': 12, 'overlap': 3}."""
        if len(self.reason) == self.n_samples:
            r = self.reason[~self.valid_mask]
            return {REASON_NAMES[int(c)]: int((r == c).sum()) for c in np.unique(r)}
        return dict(Counter(
            m.get("discard_reason") or "other"
            for m, ok in zip(self.meta, self.valid_mask) if not ok
        ))

    @property
    def n_no_signal(self) -> int:
        """Finestre positive scartate perché prive di segnale nel magnetometro."""
        return int((self.reason == REASON_CODES["no_signal"]).sum())

    def summary(self):
        n_discarded = self.n_samples - self.n_valid
        counts = self.discard_counts()
        labels_it = {
            "nan":              "con troppi NaN",
            "no_signal":        "positive senza segnale (mag piatto)",
            "overlap":          "positive sovrapposte",
            "unlabeled_signal": "negative con segnale (fulmine non in catalogo?)",
            "other":            "altri motivi",
        }
        print(f"{'─'*45}")
        print(f"  Finestre totali    : {self.n_samples}")
        print(f"  Finestre valide    : {self.n_valid}")
        print(f"    di cui positive  : {self.n_positive}  (con fulmine)")
        print(f"    di cui negative  : {self.n_negative}  (senza fulmine)")
        print(f"  Finestre scartate  : {n_discarded}")
        for key, txt in labels_it.items():
            if counts.get(key):
                print(f"    {txt:<48s}: {counts[key]}")
        print(f"{'─'*45}")


# ---------------------------------------------------------------------------
# 1. Lettura CSV fulmini
# ---------------------------------------------------------------------------

def load_lightning_csv(
    filepath: str | Path,
    time_col: str = "Time",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    time_format: Optional[str] = None,
    drop_duplicate_times: bool = True,
) -> pd.DataFrame:
    """
    Carica il CSV dei fulmini e restituisce un DataFrame con colonna
    'timestamp' (pandas Timestamp, UTC-naive) ordinato nel tempo.

    Il CSV atteso ha almeno:
        Time,Latitude,Longitude
        2013-01-01T00:00:00.123,42.35,13.39
        ...

    Parametri
    ----------
    filepath    : percorso al CSV
    time_col    : nome della colonna con il timestamp
    lat_col     : nome della colonna latitudine
    lon_col     : nome della colonna longitudine
    time_format : formato strftime (None = inferito automaticamente da pandas)
    """
    df = pd.read_csv(filepath)

    # Rinomina le colonne per uniformità interna
    rename = {}
    if time_col != "timestamp":
        rename[time_col] = "timestamp"
    if lat_col != "latitude":
        rename[lat_col] = "latitude"
    if lon_col != "longitude":
        rename[lon_col] = "longitude"
    df.rename(columns=rename, inplace=True)

    # Converti in datetime
    if time_format:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=time_format)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Rimuovi timezone per lavorare in UTC-naive (coerente con EMMA2013)
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # [FIX] Il catalogo contiene molte righe con lo STESSO timestamp al ms
    # (stesso evento localizzato in più punti). Ognuna generava una finestra
    # positiva identica, poi scartata come "overlap" (16804 scarti!).
    # Teniamo una riga per timestamp.
    if drop_duplicate_times:
        n_before = len(df)
        df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
        if n_before != len(df):
            print(f"[CSV fulmini] Rimossi {n_before - len(df)} eventi con timestamp duplicato")

    print(f"[CSV fulmini] Caricati {len(df)} eventi  "
          f"({df['timestamp'].min()} → {df['timestamp'].max()})")
    return df


# ---------------------------------------------------------------------------
# 2. Conversione dati EMMA2013 → Serie temporale in nT con index datetime
# ---------------------------------------------------------------------------

def emma_to_series(
    header: EMMA2013Header,
    data: EMMA2013Data,
    despike_nt: Optional[float] = 50.0,
    despike_window: int = 61,
) -> pd.DataFrame:
    """
    Converte i dati grezzi EMMA2013 in un DataFrame pandas con:
        - index: DatetimeIndex al secondo (UTC-naive)
        - comp1, comp2, comp3: float64 in nT (NaN dove mancante)

    La conversione usa:
        - time_factor  : divide il valore int32 per ottenere secondi
          (default 1000 → il time int32 è in ms dall'inizio del giorno)
        - field_factor : divide i valori int32 per ottenere nT
          (default 1000 → i valori int32 sono in pT)
        - Il timestamp 0 corrisponde alla mezzanotte del giorno in header.
    """
    if data is None or len(data.time) == 0:
        raise ValueError("Nessun dato nel file EMMA2013.")

    # Mezzanotte del giorno registrato
    day_origin = pd.Timestamp(
        year=header.year, month=header.month, day=header.day
    )

    # Converti time int32 → secondi → Timestamp
    time_s = data.time / header.time_factor          # secondi dall'inizio del giorno
    timestamps = day_origin + pd.to_timedelta(time_s, unit="s")

    # Converti componenti int32 → nT, sostituendo il valore sentinella con NaN
    tf = float(header.field_factor)

    def to_nt(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float64) / tf
        out[arr == MISSING_VALUE_INT] = np.nan
        # Sostituisci anche valori anomali molto grandi (overflow int32 / 1000)
        out[np.abs(out) > 1e4] = np.nan
        return out

    df = pd.DataFrame(
        {
            "comp1": to_nt(data.comp1),
            "comp2": to_nt(data.comp2),
            "comp3": to_nt(data.comp3),
        },
        index=timestamps,
    )
    df.index.name = "timestamp"

    # Rimuovi duplicati di timestamp (mantieni il primo)
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        warnings.warn(f"Rimossi {n_dup} timestamp duplicati.")
        df = df[~df.index.duplicated(keep="first")]

    # Reindex su griglia regolare al secondo (inserisce NaN dove mancano campioni)
    grid = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1s")
    df = df.reindex(grid)
    df.index.name = "timestamp"

    # [FIX] Despiking. Il filtro |x| > 1e4 nT non toglie glitch come il
    # -2483 nT trovato in X_train (il resto dei dati sta tra ~136 e ~1150 nT).
    # Un campione che si discosta dalla mediana mobile di più di despike_nt
    # è fisicamente impossibile a 1 Hz per un fulmine → NaN.
    # (I veri segnali di fulmine nel catalogo hanno |Δ| mediano ≈ 0.04 nT.)
    if despike_nt is not None:
        n_spk = 0
        for col in ("comp1", "comp2", "comp3"):
            med = df[col].rolling(despike_window, center=True, min_periods=1).median()
            spk = (df[col] - med).abs() > despike_nt
            n_spk += int(spk.sum())
            df.loc[spk, col] = np.nan
        if n_spk:
            print(f"[Magnetometro] Despiking: {n_spk} campioni anomali → NaN")

    n_nan = df.isna().any(axis=1).sum()
    print(f"[Magnetometro] {len(df)} campioni  |  NaN: {n_nan} "
          f"({100*n_nan/len(df):.1f}%)")
    return df


# ---------------------------------------------------------------------------
# 3. Costruzione delle finestre
# ---------------------------------------------------------------------------

def build_dataset(
    header: EMMA2013Header,
    raw_data: EMMA2013Data,
    lightning_df: pd.DataFrame,
    half_window: int = 5,
    nan_threshold: float = 0.3,
    negative_stride: int = 11,
    guard_seconds: int = 10,
    deriv_threshold: Optional[float] = None,
    deriv_components: tuple[int, ...] = (0, 1, 2),
    neg_deriv_threshold: Optional[float] = None,
    non_overlap_positives: bool = True,
    overlap_mode: str = "keep_first",
    min_separation: Optional[float] = None,
    center_offset: int = 0,
    noise_window: int = 300,
    dist_lightning_times: Optional[np.ndarray] = None,
    day_has_lightning: Optional[bool] = None,
) -> LightningDataset:
    """
    Costruisce il dataset di finestre allineate.

    Parametri
    ----------
    header, raw_data   : output di read_emma2013()
    lightning_df       : output di load_lightning_csv()
    half_window        : semiampiezza della finestra in secondi (default 5 → 11 camp.)
    nan_threshold      : frazione massima di NaN tollerata in una finestra (default 30%)
    negative_stride    : passo in secondi per campionare le finestre negative
    guard_seconds      : secondi di margine tra i BORDI di una finestra negativa
                         e il fulmine più vicino (prima era misurato dal centro)
    deriv_threshold    : soglia in nT/s sulla derivata massima assoluta.
                         Se None il controllo è disabilitato.
                         Le finestre positive con max|Δcomp| < soglia vengono
                         marcate valid=False con discard_reason='no_signal'.
    deriv_components   : indici delle componenti su cui applicare le soglie
                         (default: tutte e tre, 0=comp1, 1=comp2, 2=comp3).

    neg_deriv_threshold : soglia in nT/s per individuare i FALSI NEGATIVI di
                         catalogo. Le finestre negative con max|Δcomp| >= soglia
                         (segnale impulsivo ma nessun fulmine registrato) vengono
                         marcate valid=False con discard_reason='unlabeled_signal'.
                         Se None il controllo è disabilitato.
                         NB: va scelta più alta di deriv_threshold (che è un
                         percentile basso dei positivi e sarebbe superata anche
                         dal semplice rumore). Il max_deriv di ogni negativa è
                         sempre salvato in meta per aiutare la scelta.
    non_overlap_positives : se True, le finestre positive non possono
                         sovrapporsi (fulmini ravvicinati). Le scartate hanno
                         discard_reason='overlap'.
    overlap_mode       : come risolvere una sovrapposizione
                         'keep_first' → si tiene il primo fulmine in ordine
                                        temporale, si scartano quelli che si
                                        sovrappongono a una finestra già tenuta
                         'drop_all'   → si scartano TUTTE le positive che si
                                        sovrappongono ad almeno un'altra
                                        (più severo, elimina ogni ambiguità di
                                        etichetta)
    center_offset      : [NUOVO] secondi di cui spostare in AVANTI il centro delle
                         finestre positive rispetto al fulmine. La superposed
                         epoch analysis mostra che la risposta del magnetometro
                         arriva tra +0 e +3 s (picco a +2 s): con offset 1 e
                         half_window 4 la finestra copre [-3, +5] s.
    min_separation     : distanza minima in secondi tra i centri di due finestre
                         positive. Default None → window_len = 2*half_window+1,
                         cioè finestre senza alcun campione in comune.
    noise_window       : [NUOVO] secondi PRIMA di ogni finestra su cui si stima
                         il rumore locale (mediana di max|Δcomp|).
    dist_lightning_times : [NUOVO] array ordinato datetime64[s] dei fulmini da
                         usare per calcolare dt_lightning. Default: quelli di
                         lightning_df. Passare l'intero catalogo (o uno più
                         ampio, es. raggio > 50 km) per non perdere i fulmini
                         dei giorni adiacenti.
    day_has_lightning  : [NUOVO] valore da salvare per tutte le finestre;
                         default: len(lightning_df) > 0.

    Restituisce
    -----------
    LightningDataset
    """
    if overlap_mode not in ("keep_first", "drop_all"):
        raise ValueError("overlap_mode deve essere 'keep_first' o 'drop_all'.")

    mag = emma_to_series(header, raw_data)
    window_len = 2 * half_window + 1
    if min_separation is None:
        min_separation = float(window_len)

    # Converti i timestamp dei fulmini in array numpy per lookup veloce.
    # [FIX] prima si usava .astype("datetime64[s]") che TRONCA (12:00:03.9 → :03)
    # mentre le positive usano .round("s") (→ :04): incoerenza di 1 s proprio
    # sul bordo della zona di guardia. Ora si arrotonda come per le positive.
    if len(lightning_df):
        lt_array = np.sort(
            lightning_df["timestamp"].dt.round("s").values.astype("datetime64[s]")
        )
    else:
        lt_array = np.array([], dtype="datetime64[s]")

    windows_list = []
    labels_list = []
    has_lt_list = []
    valid_list = []
    meta_list = []

    # ── A. Finestre positive (centrate su ogni fulmine) ──────────────────────
    for _, row in lightning_df.iterrows():
        # arrotonda al secondo e sposta in avanti di center_offset (ritardo strumentale)
        t_center = row["timestamp"].round("s") + pd.Timedelta(seconds=center_offset)
        t_start  = t_center - pd.Timedelta(seconds=half_window)
        t_end    = t_center + pd.Timedelta(seconds=half_window)

        win = _extract_window(mag, t_start, t_end, window_len)
        if win is None:
            continue  # fuori range del magnetometro

        n_nan = int(np.isnan(win).any(axis=1).sum())
        nan_frac = n_nan / window_len

        # Offset del fulmine rispetto al centro: 0.0 perché è già centrato
        label = float(
            (row["timestamp"] - t_center).total_seconds()
        )  # ≈ -center_offset ± 0.5 s

        max_deriv = _max_abs_deriv(win, deriv_components)

        # Controlla che il magnetometro mostri effettivamente un segnale
        discard_reason = None
        if nan_frac > nan_threshold:
            valid = False
            discard_reason = "nan"
        elif deriv_threshold is not None:
            has_signal, _ = check_lightning_signal(
                win, deriv_threshold, deriv_components
            )
            if not has_signal:
                valid = False
                discard_reason = "no_signal"
            else:
                valid = True
        else:
            valid = True

        windows_list.append(win)
        labels_list.append(label)
        has_lt_list.append(True)
        valid_list.append(valid)
        meta_list.append({
            "t_center":      t_center,
            "window_start":  t_start,
            "window_end":    t_end,
            "n_nan":         n_nan,
            "lightning_times": [row["timestamp"]],
            "type":          "positive",
            "discard_reason": discard_reason,
            "reference_frame": header.reference_frame,
            "max_deriv":     max_deriv,
        })

    # ── A2. Controllo sovrapposizione tra positive ───────────────────────────
    # A questo punto in liste ci sono solo le positive (in ordine temporale).
    # Si considerano solo quelle ancora valide: una positiva già scartata
    # (NaN / no_signal) non deve "bloccare" le vicine.
    if non_overlap_positives and windows_list:
        centers_s = np.array([
            (m["t_center"] - mag.index[0]).total_seconds() for m in meta_list
        ])
        valid_arr = np.array(valid_list, dtype=bool)
        to_drop = _flag_overlapping(centers_s, valid_arr, min_separation, overlap_mode)
        for i in np.where(to_drop)[0]:
            valid_list[i] = False
            meta_list[i]["discard_reason"] = "overlap"

    # ── B. Finestre negative (sliding window, evitando i fulmini) ───────────
    t_mag_start = mag.index[0]
    t_mag_end   = mag.index[-1]

    t_cursor = t_mag_start + pd.Timedelta(seconds=half_window)
    guard = pd.Timedelta(seconds=guard_seconds)

    while t_cursor + pd.Timedelta(seconds=half_window) <= t_mag_end:
        t_start = t_cursor - pd.Timedelta(seconds=half_window)
        t_end   = t_cursor + pd.Timedelta(seconds=half_window)

        # Controlla se ci sono fulmini nella zona di guardia.
        # [FIX] prima la guardia era misurata dal CENTRO: con guard=3 e
        # half_window=3 un fulmine poteva cadere a 1 s dal bordo della
        # finestra negativa. Ora il margine è misurato dai BORDI della finestra:
        # nessun fulmine in [t_start - guard, t_end + guard].
        in_guard = _any_lightning_in_range(
            lt_array, t_start - guard, t_end + guard
        )

        if not in_guard:
            win = _extract_window(mag, t_start, t_end, window_len)
            if win is not None:
                n_nan    = int(np.isnan(win).any(axis=1).sum())
                nan_frac = n_nan / window_len
                max_deriv = _max_abs_deriv(win, deriv_components)

                discard_reason = None
                if nan_frac > nan_threshold:
                    valid = False
                    discard_reason = "nan"
                elif neg_deriv_threshold is not None and max_deriv >= neg_deriv_threshold:
                    # Segnale impulsivo ma nessun fulmine in catalogo:
                    # probabile fulmine non registrato → finestra ambigua
                    valid = False
                    discard_reason = "unlabeled_signal"
                else:
                    valid = True

                windows_list.append(win)
                labels_list.append(0.0)
                has_lt_list.append(False)
                valid_list.append(valid)
                meta_list.append({
                    "t_center":      t_cursor,
                    "window_start":  t_start,
                    "window_end":    t_end,
                    "n_nan":         n_nan,
                    "lightning_times": [],
                    "type": "negative",
                    "discard_reason": discard_reason,
                    "reference_frame": header.reference_frame,
                    "max_deriv":     max_deriv,
                })

        t_cursor += pd.Timedelta(seconds=negative_stride)

    # ── Assembla il dataset ──────────────────────────────────────────────────
    if not windows_list:
        warnings.warn("Nessuna finestra costruita — controlla la sovrapposizione temporale.")
        return LightningDataset()

    # ── [NUOVO] array per finestra: tempo, salto, rumore locale, distanza ───
    t_center_arr = np.array([m["t_center"] for m in meta_list], dtype="datetime64[s]")
    max_deriv_arr = np.array([m["max_deriv"] for m in meta_list], dtype=np.float32)
    reason_arr = np.array([REASON_CODES[m["discard_reason"]] for m in meta_list], dtype=np.int8)

    # Rumore locale: d[t] = max sulle componenti di |x[t] - x[t-1]|,
    # poi mediana mobile sui noise_window secondi che PRECEDONO la finestra.
    comps = ["comp1", "comp2", "comp3"]
    d = mag[[comps[c] for c in deriv_components]].diff().abs().max(axis=1, skipna=True)
    noise = d.rolling(noise_window, min_periods=max(10, noise_window // 5)).median().shift(1)
    t_start_arr = t_center_arr - np.timedelta64(half_window, "s")
    noise_arr = noise.reindex(pd.DatetimeIndex(t_start_arr)).to_numpy(dtype=np.float32)

    # Distanza (con segno) dal fulmine di catalogo più vicino
    ref = lt_array if dist_lightning_times is None else np.asarray(
        dist_lightning_times, dtype="datetime64[s]")
    dt_arr = _signed_distance_s(t_center_arr, ref)

    dhl = (len(lightning_df) > 0) if day_has_lightning is None else bool(day_has_lightning)

    dataset = LightningDataset(
        windows      = np.stack(windows_list).astype(np.float32),
        labels       = np.array(labels_list, dtype=np.float32),
        has_lightning= np.array(has_lt_list, dtype=bool),
        valid_mask   = np.array(valid_list, dtype=bool),
        meta         = meta_list,
        t_center     = t_center_arr,
        max_deriv    = max_deriv_arr,
        noise_local  = noise_arr,
        dt_lightning = dt_arr,
        day_has_lightning = np.full(len(meta_list), dhl, dtype=bool),
        reason       = reason_arr,
    )
    dataset.summary()
    return dataset


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def _signed_distance_s(t: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Per ogni tempo in t (datetime64[s]) restituisce t - fulmine più vicino in
    ref (ordinato), in secondi con segno (negativo = prima del fulmine).
    +inf se ref è vuoto.
    """
    out = np.full(len(t), np.inf, dtype=np.float32)
    if len(ref) == 0 or len(t) == 0:
        return out
    ti = t.astype("int64")
    ri = np.sort(ref.astype("datetime64[s]").astype("int64"))
    j = np.searchsorted(ri, ti)
    left  = ri[np.clip(j - 1, 0, len(ri) - 1)]
    right = ri[np.clip(j,     0, len(ri) - 1)]
    dl, dr = ti - left, ti - right
    return np.where(np.abs(dl) <= np.abs(dr), dl, dr).astype(np.float32)


def _max_abs_deriv(win: np.ndarray, components: tuple[int, ...] = (0, 1, 2)) -> float:
    """
    Massimo di |Δcomp| campione-per-campione sulle componenti scelte.
    Restituisce 0.0 se tutti i valori sono NaN.
    """
    sub = win[:, list(components)]
    abs_diff = np.abs(np.diff(sub, axis=0))
    if abs_diff.size == 0 or np.all(np.isnan(abs_diff)):
        return 0.0
    return float(np.nanmax(abs_diff))


def _flag_overlapping(
    centers_s: np.ndarray,
    valid: np.ndarray,
    min_sep: float,
    mode: str,
) -> np.ndarray:
    """
    Restituisce una maschera bool (stessa lunghezza di centers_s) con True per
    le finestre VALIDE da scartare perché troppo vicine ad altre finestre valide.

    Due finestre si sovrappongono se la distanza tra i centri è < min_sep.

    mode='keep_first' : scansione in ordine temporale, si tiene una finestra
                        solo se dista >= min_sep dall'ultima tenuta.
    mode='drop_all'   : si scartano tutte le finestre che hanno almeno un
                        vicino (valido) più vicino di min_sep.
    """
    drop = np.zeros(len(centers_s), dtype=bool)
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return drop

    order = idx[np.argsort(centers_s[idx], kind="stable")]
    c = centers_s[order]

    if mode == "keep_first":
        last_kept = -np.inf
        for k, i in enumerate(order):
            if c[k] - last_kept < min_sep:
                drop[i] = True
            else:
                last_kept = c[k]
    else:  # drop_all
        close = np.diff(c) < min_sep          # close[k]: k e k+1 troppo vicini
        flag = np.zeros(len(order), dtype=bool)
        flag[1:]  |= close
        flag[:-1] |= close
        drop[order[flag]] = True

    return drop


def check_lightning_signal(
    win: np.ndarray,
    deriv_threshold: float,
    components: tuple[int, ...] = (0, 1, 2),
) -> tuple[bool, float]:
    """
    Verifica se una finestra positiva mostra un segnale coerente con un fulmine,
    usando la derivata prima (variazione campione-per-campione) come indicatore.

    Un fulmine genera tipicamente un transiente impulsivo nel magnetometro:
    almeno un salto |Δcomp| > deriv_threshold in almeno una delle componenti
    selezionate.

    Parametri
    ----------
    win              : ndarray (window_len, 3) in nT
    deriv_threshold  : soglia in nT/s (a 1 Hz corrisponde a nT/campione)
    components       : indici delle componenti da controllare (default: tutte)

    Restituisce
    -----------
    has_signal : bool   — True se il segnale supera la soglia
    max_deriv  : float  — valore massimo della derivata assoluta trovato
                          (utile per calibrare la soglia sui propri dati)
    """
    max_deriv = _max_abs_deriv(win, components)
    has_signal = max_deriv >= deriv_threshold
    return has_signal, max_deriv


def _extract_window(
    mag: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    window_len: int,
) -> Optional[np.ndarray]:
    """
    Estrae una finestra [t_start, t_end] dal DataFrame del magnetometro.
    Restituisce ndarray (window_len, 3) o None se fuori range.
    """
    try:
        win_df = mag.loc[t_start:t_end]
    except KeyError:
        return None

    if len(win_df) != window_len:
        return None  # finestra incompleta (bordi del file)

    return win_df[["comp1", "comp2", "comp3"]].values.astype(np.float64)


def _any_lightning_in_range(
    lt_array: np.ndarray,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> bool:
    """
    Controlla se esiste almeno un fulmine nell'intervallo [t_start, t_end].
    lt_array deve essere ORDINATO e già in datetime64[s] (vedi build_dataset).
    Ricerca binaria: O(log n) invece di O(n) per ogni finestra.
    """
    if lt_array.size == 0:
        return False
    t0 = np.datetime64(t_start, "s")
    t1 = np.datetime64(t_end, "s")
    i = np.searchsorted(lt_array, t0, side="left")
    return bool(i < lt_array.size and lt_array[i] <= t1)


# ---------------------------------------------------------------------------
# Calibrazione deriv_threshold
# ---------------------------------------------------------------------------

def compute_deriv_threshold(
    mag_series: pd.DataFrame,
    lightning_df: pd.DataFrame,
    half_window: int = 5,
    percentile: float = 10.0,
    deriv_components: tuple[int, ...] = (0, 1, 2),
    center_offset: int = 0,
) -> tuple[float, np.ndarray]:
    """
    Calcola deriv_threshold su una singola serie temporale del magnetometro.

    Itera su tutte le finestre positive (centrate sui fulmini di lightning_df
    che ricadono nel range di mag_series), calcola la derivata massima assoluta
    per ognuna e restituisce il percentile scelto come soglia.

    Parametri
    ----------
    mag_series       : DataFrame pandas prodotto da emma_to_series()
    lightning_df     : catalogo fulmini (anche l'intero, viene filtrato internamente)
    half_window      : semiampiezza finestra in secondi
    percentile       : percentile da usare come soglia (default 10)
    deriv_components : componenti su cui calcolare la derivata

    Restituisce
    -----------
    threshold  : float — soglia calcolata (nT/s)
    all_derivs : ndarray — array di tutte le derivate massime raccolte
                           (utile per fare statistiche o grafici)
    """
    window_len = 2 * half_window + 1

    # Filtra i fulmini nel range della serie
    t_min = mag_series.index.min()
    t_max = mag_series.index.max()
    mask  = (lightning_df["timestamp"] >= t_min) & \
            (lightning_df["timestamp"] <= t_max)
    local_lt = lightning_df.loc[mask]

    derivs = []
    for _, row in local_lt.iterrows():
        t_center = row["timestamp"].round("s") + pd.Timedelta(seconds=center_offset)
        t_start  = t_center - pd.Timedelta(seconds=half_window)
        t_end    = t_center + pd.Timedelta(seconds=half_window)
        try:
            win = mag_series.loc[t_start:t_end][["comp1", "comp2", "comp3"]].values
            # [FIX] escludi finestre con NaN: davano max_deriv = 0.0 e
            # abbassavano artificialmente il percentile (min = 0.000 nT/s)
            if len(win) == window_len and not np.isnan(win).any():
                _, md = check_lightning_signal(win, deriv_threshold=0.0,
                                               components=deriv_components)
                derivs.append(md)
        except Exception:
            pass

    if not derivs:
        return 0.0, np.array([])

    arr = np.array(derivs)
    threshold = float(np.percentile(arr, percentile))
    return threshold, arr


def compute_deriv_threshold_from_files(
    mag_files: list[str | Path],
    lightning_df: pd.DataFrame,
    half_window: int = 5,
    percentile: float = 10.0,
    deriv_components: tuple[int, ...] = (0, 1, 2),
    skip_errors: bool = True,
    center_offset: int = 0,
) -> tuple[float, np.ndarray]:
    """
    Calcola deriv_threshold iterando su una lista di file EMMA2013.

    Raccoglie le derivate massime da tutti i file e calcola il percentile
    sull'intera distribuzione aggregata — così la soglia è rappresentativa
    di tutti i giorni, non di uno solo.

    Parametri
    ----------
    mag_files        : lista di percorsi a file EMMA2013
    lightning_df     : catalogo fulmini completo (tutti i giorni)
    half_window      : semiampiezza finestra in secondi
    percentile       : percentile da usare come soglia (default 10)
    deriv_components : componenti su cui calcolare la derivata
    skip_errors      : se True salta i file con errori di lettura

    Restituisce
    -----------
    threshold  : float — soglia da passare a build_dataset_from_files()
    all_derivs : ndarray — tutte le derivate raccolte (per grafici/statistiche)
    """
    from emma2013_reader import read_emma2013

    all_derivs = []
    n_files = len(mag_files)

    for i, mag_path in enumerate(mag_files):
        mag_path = Path(mag_path)
        print(f"  [{i+1}/{n_files}] calibrazione su {mag_path.name} ...", end=" ")

        header, raw_data, error = read_emma2013(mag_path)
        if error.status:
            msg = f"Errore lettura {mag_path.name}: {error.message}"
            if skip_errors:
                warnings.warn(msg)
                print("SKIP")
                continue
            else:
                raise RuntimeError(msg)

        try:
            mag_series = emma_to_series(header, raw_data)
            _, day_derivs = compute_deriv_threshold(
                mag_series, lightning_df,
                half_window=half_window,
                percentile=percentile,          # non usato nel calcolo parziale
                deriv_components=deriv_components,
                center_offset=center_offset,
            )
            all_derivs.append(day_derivs)
            print(f"{len(day_derivs)} finestre")
        except Exception as e:
            msg = f"Errore calibrazione {mag_path.name}: {e}"
            if skip_errors:
                warnings.warn(msg)
                print("SKIP")
            else:
                raise

    if not all_derivs:
        warnings.warn("Nessuna derivata raccolta — calibrazione impossibile.")
        return 0.0, np.array([])

    arr = np.concatenate(all_derivs)
    threshold = float(np.percentile(arr, percentile))

    print(f"\n  Finestre totali analizzate : {len(arr)}")
    print(f"  Percentile {percentile:.0f}             : {threshold:.3f} nT/s")
    return threshold, arr


# ---------------------------------------------------------------------------
# Costruzione dataset da lista di file
# ---------------------------------------------------------------------------

def merge_datasets(*datasets: LightningDataset) -> LightningDataset:
    """
    Concatena più LightningDataset in uno solo.

    I meta di ogni dataset vengono uniti mantenendo l'ordine di arrivo.
    Utile per combinare dataset costruiti su file diversi prima di fare
    lo split train/test.

    Parametri
    ----------
    *datasets : LightningDataset
        Due o più dataset da unire.

    Restituisce
    -----------
    LightningDataset unificato.
    """
    valid = [d for d in datasets if d.n_samples > 0]
    if not valid:
        return LightningDataset()

    merged = LightningDataset(
        windows       = np.concatenate([d.windows       for d in valid], axis=0),
        labels        = np.concatenate([d.labels        for d in valid], axis=0),
        has_lightning = np.concatenate([d.has_lightning for d in valid], axis=0),
        valid_mask    = np.concatenate([d.valid_mask    for d in valid], axis=0),
        meta          = [m for d in valid for m in d.meta],
        **{k: np.concatenate([getattr(d, k) for d in valid], axis=0) for k in _ARRAY_FIELDS},
    )
    return merged


# campi array aggiunti (salvati/caricati e concatenati automaticamente)
_ARRAY_FIELDS = ("t_center", "max_deriv", "noise_local", "dt_lightning",
                 "day_has_lightning", "reason")


def build_dataset_from_files(
    mag_files: list[str | Path],
    lightning_df: pd.DataFrame,
    half_window: int = 5,
    nan_threshold: float = 0.3,
    negative_stride: int = 11,
    guard_seconds: int = 10,
    deriv_threshold: Optional[float] = None,
    deriv_components: tuple[int, ...] = (0, 1, 2),
    skip_errors: bool = True,
    neg_deriv_threshold: Optional[float] = None,
    non_overlap_positives: bool = True,
    overlap_mode: str = "keep_first",
    min_separation: Optional[float] = None,
    center_offset: int = 0,
    noise_window: int = 300,
    dist_lightning_df: Optional[pd.DataFrame] = None,
    keep_meta: bool = False,
) -> LightningDataset:
    """
    Costruisce un dataset unico iterando su una lista di file EMMA2013.

    Per ogni file viene estratto il sottoinsieme di fulmini del giorno
    corrispondente (in base a year/month/day nell'header), costruito il
    dataset parziale con build_dataset(), e infine tutti i dataset parziali
    vengono uniti con merge_datasets().

    Parametri
    ----------
    mag_files        : lista di percorsi a file EMMA2013 (uno per giorno)
    lightning_df     : catalogo fulmini completo (tutti i giorni)
    half_window      : semiampiezza finestra in secondi
    nan_threshold    : soglia NaN per validità finestra
    negative_stride  : passo sliding window negativa (secondi)
    guard_seconds    : buffer anti-falso-negativo (secondi)
    deriv_threshold  : soglia derivata sulle positive (None = disabilitata)
    deriv_components : componenti su cui applicare le soglie
    skip_errors      : se True salta i file con errori di lettura
                       (stampa un warning); se False rilancia l'eccezione
    neg_deriv_threshold, non_overlap_positives, overlap_mode, min_separation,
    center_offset, noise_window : vedi build_dataset()
    dist_lightning_df : [NUOVO] catalogo da usare per dt_lightning (default:
                       lightning_df intero). Può essere un catalogo più ampio.
    keep_meta        : [NUOVO] se False non conserva la lista di dict meta
                       (milioni di dict → diversi GB di RAM). Tutte le
                       informazioni utili sono già negli array del dataset.

    Restituisce
    -----------
    LightningDataset unificato su tutti i file.
    """
    from emma2013_reader import read_emma2013   # import locale per evitare circolarità

    partial_datasets: list[LightningDataset] = []
    n_files = len(mag_files)

    ref_df = lightning_df if dist_lightning_df is None else dist_lightning_df
    all_times = np.sort(ref_df["timestamp"].dt.round("s").values.astype("datetime64[s]"))

    for i, mag_path in enumerate(mag_files):
        mag_path = Path(mag_path)
        print(f"\n[{i+1}/{n_files}] {mag_path.name}")

        # ── Leggi il file magnetometro ────────────────────────────────────
        header, raw_data, error = read_emma2013(mag_path)
        if error.status:
            msg = f"Errore lettura {mag_path.name}: {error.message}"
            if skip_errors:
                warnings.warn(msg)
                continue
            else:
                raise RuntimeError(msg)

        # ── Filtra i fulmini per il giorno dell'header ────────────────────
        day_start = pd.Timestamp(year=header.year, month=header.month, day=header.day)
        day_end   = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask      = (lightning_df["timestamp"] >= day_start) & \
                    (lightning_df["timestamp"] <= day_end)
        day_lightning = lightning_df.loc[mask].reset_index(drop=True)

        if day_lightning.empty:
            print(f"  → Nessun fulmine in catalogo per questo giorno, solo finestre negative.")

        # ── Costruisci il dataset parziale ────────────────────────────────
        try:
            ds = build_dataset(
                header           = header,
                raw_data         = raw_data,
                lightning_df     = day_lightning,
                half_window      = half_window,
                nan_threshold    = nan_threshold,
                negative_stride  = negative_stride,
                guard_seconds    = guard_seconds,
                deriv_threshold  = deriv_threshold,
                deriv_components = deriv_components,
                neg_deriv_threshold   = neg_deriv_threshold,
                non_overlap_positives = non_overlap_positives,
                overlap_mode          = overlap_mode,
                min_separation        = min_separation,
                center_offset         = center_offset,
                noise_window          = noise_window,
                dist_lightning_times  = all_times,
            )
            if not keep_meta:
                ds.meta = []
            partial_datasets.append(ds)
        except Exception as e:
            msg = f"Errore build_dataset per {mag_path.name}: {e}"
            if skip_errors:
                warnings.warn(msg)
            else:
                raise

    if not partial_datasets:
        warnings.warn("Nessun dataset parziale costruito.")
        return LightningDataset()

    # ── Unisci tutti i dataset parziali ───────────────────────────────────
    full_dataset = merge_datasets(*partial_datasets)

    print(f"\n{'═'*45}")
    print(f"  File elaborati : {len(partial_datasets)} / {n_files}")
    print(f"  Dataset finale :")
    full_dataset.summary()

    return full_dataset


# ---------------------------------------------------------------------------
# Salvataggio / caricamento del dataset
# ---------------------------------------------------------------------------

def save_dataset(dataset: LightningDataset, path: str | Path):
    """Salva il dataset in formato .npz compresso."""
    path = Path(path)
    np.savez_compressed(
        path,
        windows       = dataset.windows,
        labels        = dataset.labels,
        has_lightning = dataset.has_lightning,
        valid_mask    = dataset.valid_mask,
        **{k: getattr(dataset, k) for k in _ARRAY_FIELDS},
    )
    print(f"Dataset salvato in {path}")


def load_dataset(path: str | Path) -> LightningDataset:
    """Carica un dataset salvato con save_dataset()."""
    d = np.load(path)
    return LightningDataset(
        windows       = d["windows"],
        labels        = d["labels"],
        has_lightning = d["has_lightning"],
        valid_mask    = d["valid_mask"],
        **{k: d[k] for k in _ARRAY_FIELDS if k in d.files},
    )

def save_training_dataset(
    filename,
    X_train,
    y_train,
    c_train,
    t_train,
    X_test,
    y_test,
    c_test,
    t_test,
    *,
    deriv_threshold=None,
    half_window=None,
    train_fraction=None,
    negative_ratio=None,
    X_val=None,
    y_val=None,
    c_val=None,
    t_val=None,
    **extra_arrays,
):
    """
    Salva il dataset finale già suddiviso in train/test.

    Parameters
    ----------
    X_train, X_test : ndarray
        Finestre magnetometro.
    y_train, y_test : ndarray
        Offset del fulmine rispetto al centro finestra.
    c_train, c_test : ndarray
        Label di classificazione (1=fulmine, 0=no fulmine).
    """

    # [FIX] set di validazione separato (per checkpoint/scheduler):
    # il test non va MAI usato durante il training.
    extra = {}
    if X_val is not None:
        extra = dict(X_val=X_val, y_val=y_val, c_val=c_val, t_val=t_val)

    # [NUOVO] array aggiuntivi liberi (es. g_test, dt_test, md_test, ...)
    extra.update(extra_arrays)

    np.savez_compressed(
        filename,
        **extra,
        X_train=X_train,
        y_train=y_train,
        c_train=c_train,
        t_train=t_train,
        X_test=X_test,
        y_test=y_test,
        c_test=c_test,
        t_test=t_test,

        deriv_threshold=deriv_threshold,
        half_window=half_window,
        train_fraction=train_fraction,
        negative_ratio=negative_ratio,
    )

    print(f"Training dataset salvato in: {filename}")

def load_training_dataset(filename):
    data = np.load(filename, allow_pickle=True)

    base = {"X_train", "y_train", "c_train", "t_train", "X_test", "y_test",
            "c_test", "t_test", "deriv_threshold", "half_window",
            "train_fraction", "negative_ratio"}
    out = {k: data[k] for k in data.files if k not in base}   # val + array extra

    return out | {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "c_train": data["c_train"],
        "t_train": data["t_train"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "c_test": data["c_test"],
        "t_test": data["t_test"],

        "deriv_threshold": data.get("deriv_threshold", None),
        "half_window": data.get("half_window", None),
        "train_fraction": data.get("train_fraction", None),
        "negative_ratio": data.get("negative_ratio", None),
    }


# ---------------------------------------------------------------------------
# Diagnostica: Superposed Epoch Analysis
# ---------------------------------------------------------------------------

def superposed_epoch_analysis(
    mag_files: list[str | Path],
    lightning_df: pd.DataFrame,
    max_lag_s: int = 7200,
    components: tuple[int, ...] = (0, 1, 2),
    skip_errors: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Media di |Δcomp| (somma sulle componenti) in funzione del ritardo rispetto
    al tempo del fulmine, sovrapponendo TUTTI gli eventi del catalogo.

    Serve a rispondere a due domande prima ancora di addestrare una rete:
      1. Il magnetometro a 1 Hz "vede" davvero i fulmini?
         → ci deve essere un picco a lag ≈ 0 ben sopra il livello di fondo.
      2. I due orologi sono allineati?
         → se il picco cade a ±3600 / ±7200 s il CSV è in ora locale
           (CET/CEST) e non in UTC; se cade a ±1-2 s c'è un ritardo
           strumentale da correggere nelle finestre positive.

    Restituisce
    -----------
    lags    : ndarray (2*max_lag_s+1,)  secondi
    mean_d  : ndarray  media di |Δ| a ogni lag (nT/s)
    count   : ndarray  numero di eventi che contribuiscono a ogni lag
    """
    from emma2013_reader import read_emma2013

    L = int(max_lag_s)
    acc = np.zeros(2 * L + 1)
    cnt = np.zeros(2 * L + 1)
    t_all = lightning_df["timestamp"].dt.round("s")

    for mag_path in mag_files:
        header, raw_data, error = read_emma2013(Path(mag_path))
        if error.status:
            if skip_errors:
                continue
            raise RuntimeError(error.message)
        mag = emma_to_series(header, raw_data)

        # |Δ| sommata sulle componenti, allineata alla griglia (d[k] = |x[k]-x[k-1]|)
        x = mag[["comp1", "comp2", "comp3"]].values[:, list(components)]
        d = np.full(len(mag), np.nan)
        d[1:] = np.abs(np.diff(x, axis=0)).sum(axis=1)

        t0 = mag.index[0]
        sel = (t_all >= t0 - pd.Timedelta(seconds=L)) & \
              (t_all <= mag.index[-1] + pd.Timedelta(seconds=L))
        idx = ((t_all[sel] - t0).dt.total_seconds()).astype(int).to_numpy()

        for i in idx:
            a, b = i - L, i + L + 1
            lo, hi = max(a, 0), min(b, len(d))
            if lo >= hi:
                continue
            seg = d[lo:hi]
            ok = ~np.isnan(seg)
            acc[lo - a: hi - a][ok] += seg[ok]
            cnt[lo - a: hi - a][ok] += 1

    lags = np.arange(-L, L + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_d = acc / cnt
    return lags, mean_d, cnt
