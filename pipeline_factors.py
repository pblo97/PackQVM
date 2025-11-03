# pipeline_factors.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable


# ===========================================================
# 1. Traer fundamentales crudos
# ===========================================================
def fetch_raw_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Devuelve métricas contables / balance / flujo / tecnicas básicas POR TICKER.
    Esta función es donde tú conectas tu data real:
    - tus cachés FMP
    - tus funciones tipo download_guardrails_batch / download_vfq_batch
    - cualquier df local que armes con FCF yield, EV/EBITDA, etc.

    POR AHORA: mock aleatorio para que el pipeline no se caiga.
    Reemplaza esto apenas lo tengas.
    """
    rows = []
    for sym in tickers:
        rows.append({
            "symbol": sym,
            "sector": "Technology",  # TODO: cámbialo con el sector real
            "market_cap": np.random.uniform(5e9, 5e11),

            # --- métricas fundamentales para guardrails ---
            "netdebt_ebitda":  np.random.uniform(-1, 3),
            "accruals_ta":     np.random.uniform(-0.1, 0.1),
            "asset_growth":    np.random.uniform(-0.05, 0.3),
            "profit_hits":     np.random.randint(0, 5),       # cuántos trimestres ganando plata
            "share_issuance":  np.random.uniform(-0.05, 0.05),# dilución ~ emisión de acciones

            # --- métricas de flujo / momentum que VFQ quiere mirar ---
            "BreakoutScore": np.random.uniform(0, 100),
            "RVOL20":        np.random.uniform(0.5, 3),
            "hits":          np.random.randint(0, 5),
        })
    return pd.DataFrame(rows)


# ===========================================================
# 2. Guardrails (pasa / no pasa "empresa sana")
# ===========================================================
def apply_guardrails_logic(
    df: pd.DataFrame,
    *,
    PROFIT_MIN_HITS: int    = 2,     # al menos 2 de {EBIT, CFO, FCF} > 0
    MAX_ISSUANCE: float     = 0.03,  # dilución neta máx (3%)
    MAX_ASSET_GROWTH: float = 0.20,  # |asset growth| máx
    MAX_ACCRUALS_ABS: float = 0.10,  # |accruals/TA| máx
    MAX_NETDEBT_EBITDA: float = 3.0, # net debt / EBITDA máx
    MIN_COVERAGE: int       = 2      # # de bloques disponibles mín
) -> pd.DataFrame:
    """
    Marca pass_* y pass_all en base a umbrales centrales.
    - coverage_count cuenta BLOQUES disponibles: 
      1) bloque rentabilidad si hay al menos UNO de {ebit_margin, cfo_margin, fcf_margin} no-NaN
      2) netdebt_ebitda
      3) accruals_ta
      4) asset_growth
      5) share_issuance
    - pass_profit usa 'profit_hits' (>= PROFIT_MIN_HITS) pero OJO: coverage NO se basa en profit_hits,
      se basa en la DISPONIBILIDAD de márgenes (para evitar el bug de coverage=5 para todos).
    """
    out = df.copy()

    # -------- columnas numéricas seguras --------
    # (no rellenamos NaN aquí salvo donde es necesario para la lógica)
    ebit_m = pd.to_numeric(out.get("ebit_margin"), errors="coerce")
    cfo_m  = pd.to_numeric(out.get("cfo_margin"),  errors="coerce")
    fcf_m  = pd.to_numeric(out.get("fcf_margin"),  errors="coerce")

    ndebt  = pd.to_numeric(out.get("netdebt_ebitda"), errors="coerce")
    accr   = pd.to_numeric(out.get("accruals_ta"),    errors="coerce")
    ag     = pd.to_numeric(out.get("asset_growth"),   errors="coerce")
    iss    = pd.to_numeric(out.get("share_issuance"), errors="coerce")

    profit_hits = pd.to_numeric(out.get("profit_hits"), errors="coerce")  # puede ser Int64 con <NA>

    # -------- coverage_count: disponibilidad de BLOQUES --------
    # 1) bloque rentabilidad: hay al menos un margen disponible?
    if (ebit_m is not None) or (cfo_m is not None) or (fcf_m is not None):
        has_profit_any = pd.concat(
            [ebit_m.notna(), cfo_m.notna(), fcf_m.notna()],
            axis=1
        ).any(axis=1)
    else:
        # fallback si por alguna razón no tenemos márgenes pero sí profit_hits:
        # cuenta el bloque si profit_hits no es NaN
        has_profit_any = profit_hits.notna()

    blocks = pd.concat([
        has_profit_any,
        ndebt.notna(),
        accr.notna(),
        ag.notna(),
        iss.notna(),
    ], axis=1)

    out["coverage_count"] = blocks.sum(axis=1).astype(int)

    # -------- reglas pass_* --------
    # Para pass_profit, tratamos NaN en profit_hits como 0 (no pasa)
    out["pass_profit"]   = profit_hits.fillna(0).astype(int) >= int(PROFIT_MIN_HITS)
    out["pass_issuance"] = iss.abs()   <= float(MAX_ISSUANCE)
    out["pass_assets"]   = ag.abs()    <= float(MAX_ASSET_GROWTH)
    out["pass_accruals"] = accr.abs()  <= float(MAX_ACCRUALS_ABS)
    out["pass_ndebt"]    = ndebt       <= float(MAX_NETDEBT_EBITDA)
    out["pass_coverage"] = out["coverage_count"] >= int(MIN_COVERAGE)

    # NaN → False
    for c in ["pass_profit","pass_issuance","pass_assets","pass_accruals","pass_ndebt","pass_coverage"]:
        out[c] = out[c].fillna(False)

    out["pass_all"] = (
        out["pass_profit"]
        & out["pass_issuance"]
        & out["pass_assets"]
        & out["pass_accruals"]
        & out["pass_ndebt"]
        & out["pass_coverage"]
    )

    return out



# ===========================================================
# 3. VFQ scores (Quality / Value / Flow)
# ===========================================================
def compute_quality_value_flow(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las columnas que Tab3 necesita para rankear:
      - quality_adj_neut (0..1)
      - value_adj_neut   (0..1)
      - acc_pct          (0..100, más alto = accruals más 'limpios')
      - prob_up          (0..1)
    Este es tu sitio para meter ROIC, FCF yield, etc.
    Por ahora quedan fórmulas dummy, pero NUMÉRICAMENTE CONSISTENTES.
    """
    out = df_raw.copy()

    # QUALITY = "gana plata y no está ahogado en deuda".
    q_raw = (
        (out["profit_hits"].fillna(0) / 4.0)                     # más hits => mejor
        + (3 - out["netdebt_ebitda"].clip(-1, 3).fillna(0)) / 3  # menos deuda => mejor
    )
    q_min, q_max = q_raw.min(), q_raw.max()
    if q_min == q_max:
        out["quality_adj_neut"] = 0.0
    else:
        out["quality_adj_neut"] = (q_raw - q_min) / (q_max - q_min)

    # VALUE = "no me diluyes y no estás maquillando ganancias".
    v_raw = (
        (-out["share_issuance"].fillna(0)) * 1.0
        + (-out["accruals_ta"].fillna(0))   * 0.5
    )
    v_min, v_max = v_raw.min(), v_raw.max()
    if v_min == v_max:
        out["value_adj_neut"] = 0.0
    else:
        out["value_adj_neut"] = (v_raw - v_min) / (v_max - v_min)

    # acc_pct = percentil "bueno" de accruals (cero accruals = sano)
    acc_abs = out["accruals_ta"].abs()
    ranks = acc_abs.rank(pct=True)  # 0..1 peor=1
    out["acc_pct"] = (1 - ranks) * 100  # 100 = accruals súper limpios

    # prob_up = "flujo + volumen + setup de ruptura"
    pu_raw = (
        0.6 * (out["BreakoutScore"].fillna(0) / 100.0)
        + 0.4 * (out["RVOL20"].fillna(0) / 3.0)
    )
    pu_min, pu_max = pu_raw.min(), pu_raw.max()
    if pu_min == pu_max:
        out["prob_up"] = 0.0
    else:
        out["prob_up"] = (pu_raw - pu_min) / (pu_max - pu_min)

    return out[[
        "symbol",
        "quality_adj_neut",
        "value_adj_neut",
        "acc_pct",
        "prob_up",
    ]]


# ===========================================================
# 4. FUNCIÓN ÚNICA que usa TODO el resto de la app
# ===========================================================
def build_factor_frame(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Entrada: lista de tickers.
    Salida: dataframe maestro con TODAS las columnas que van a usar Tab2 y Tab3.

    Columnas clave que devolvemos SIEMPRE:
      symbol, sector, market_cap,
      profit_hits, coverage_count, asset_growth, accruals_ta,
      netdebt_ebitda, pass_profit, pass_issuance, pass_assets,
      pass_accruals, pass_ndebt, pass_coverage, pass_all,
      quality_adj_neut, value_adj_neut, acc_pct, hits,
      BreakoutScore, RVOL20, prob_up
    """

    tickers = list(dict.fromkeys(tickers))  # unique
    if not tickers:
        return pd.DataFrame(columns=[
            "symbol","sector","market_cap",
            "profit_hits","coverage_count","asset_growth","accruals_ta",
            "netdebt_ebitda","pass_profit","pass_issuance","pass_assets",
            "pass_accruals","pass_ndebt","pass_coverage","pass_all",
            "quality_adj_neut","value_adj_neut","acc_pct",
            "hits","BreakoutScore","RVOL20","prob_up",
        ])

    # 1. Data cruda
    raw = fetch_raw_fundamentals(tickers)

    # 2. Guardrails
    guarded = apply_guardrails_logic(raw)

    # 3. VFQ scores
    qvf = compute_quality_value_flow(raw)

    # 4. Merge final
    out = guarded.merge(qvf, on="symbol", how="left", suffixes=("",""))

    # 5. Fill columnas que podrían faltar
    needed_cols_defaults = {
        "sector": "Unknown",
        "market_cap": np.nan,

        "profit_hits": np.nan,
        "coverage_count": 0,
        "asset_growth": np.nan,
        "accruals_ta": np.nan,
        "netdebt_ebitda": np.nan,

        "pass_profit": False,
        "pass_issuance": False,
        "pass_assets": False,
        "pass_accruals": False,
        "pass_ndebt": False,
        "pass_coverage": False,
        "pass_all": False,

        "quality_adj_neut": 0.0,
        "value_adj_neut": 0.0,
        "acc_pct": 0.0,
        "hits": 0.0,
        "BreakoutScore": 0.0,
        "RVOL20": 0.0,
        "prob_up": 0.0,
    }
    for col, default_val in needed_cols_defaults.items():
        if col not in out.columns:
            out[col] = default_val

    out["symbol"] = out["symbol"].astype(str)
    return out
