"""
Fundamentals - Refactorizado
============================

PROBLEMA ACTUAL:
- apply_quality_guardrails tiene 200+ líneas
- Lógica duplicada entre funciones
- Nombres inconsistentes (_safe_series vs _num_or_nan)

SOLUCIÓN:
- Funciones pequeñas con responsabilidad única
- Naming consistente
- Separación clara: fetch / compute / apply
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


# ============================================================================
# LAYER 1: HELPERS BÁSICOS (TIPO CONVERSIÓN)
# ============================================================================

def to_numeric(s: pd.Series | pd.DataFrame | np.ndarray | None) -> pd.Series:
    """
    Convierte CUALQUIER input a Serie numérica (float64).
    
    Rules:
    - None → Serie vacía
    - DataFrame → primera columna o promedio
    - ndarray → aplanar si es 2D
    - NaN preservados (NO se rellenan con 0)
    """
    if s is None:
        return pd.Series(dtype=float)
    
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            s = s.iloc[:, 0]
        else:
            s = s.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    
    elif isinstance(s, np.ndarray):
        if s.ndim > 1:
            s = s[:, 0]
        s = pd.Series(s)
    
    elif not isinstance(s, pd.Series):
        s = pd.Series(s)
    
    return pd.to_numeric(s, errors="coerce").astype(float)


def safe_column(
    df: pd.DataFrame,
    candidates: list[str],
    default: float = np.nan,
) -> pd.Series:
    """
    Devuelve la PRIMERA columna encontrada de 'candidates'.
    Si ninguna existe → Serie con 'default'.
    
    IMPORTANTE: NaN en la columna existente se preserva (no se reemplaza con default).
    """
    for col in candidates:
        if col in df.columns:
            return to_numeric(df[col])
    
    return pd.Series(default, index=df.index, dtype=float)


# ============================================================================
# LAYER 2: TRANSFORMACIONES ESTADÍSTICAS
# ============================================================================

def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """
    Winsoriza serie al percentil p (ambas colas).
    
    Args:
        s: Serie numérica
        p: Percentil (0.01 = 1%)
    
    Returns:
        Serie con outliers truncados
    """
    s = to_numeric(s)
    
    if s.notna().sum() < 3 or p <= 0:
        return s
    
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    """
    Z-score robusto con manejo de casos edge.
    
    Rules:
    - sd=0 o sd=NaN → usar sd=1 (evita división por 0)
    - Inf → NaN
    - Missing → 0 (para agregaciones downstream)
    """
    s = to_numeric(s)
    
    mu = s.mean()
    sd = s.std(ddof=0)
    
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    
    z = (s - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan)
    
    return z.fillna(0.0)


def safe_divide(a, b) -> pd.Series:
    """División segura: a/b con Inf→NaN"""
    a = to_numeric(a)
    b = to_numeric(b)
    
    result = a / b
    return result.replace([np.inf, -np.inf], np.nan)


def rank_pct(s: pd.Series) -> pd.Series:
    """Percentil ranking (0-1)"""
    s = to_numeric(s)
    
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    
    return s.rank(pct=True, method="average").fillna(0.0)


# ============================================================================
# LAYER 3: LÓGICA DE NEGOCIO (GUARDRAILS)
# ============================================================================

class GuardrailConfig:
    """Configuración centralizada de umbrales"""
    def __init__(
        self,
        profit_min_hits: int = 2,
        max_issuance: float = 0.03,
        max_asset_growth: float = 0.20,
        max_accruals: float = 0.10,
        max_netdebt_ebitda: float = 3.0,
        min_coverage: int = 2,
    ):
        self.profit_min_hits = profit_min_hits
        self.max_issuance = max_issuance
        self.max_asset_growth = max_asset_growth
        self.max_accruals = max_accruals
        self.max_netdebt_ebitda = max_netdebt_ebitda
        self.min_coverage = min_coverage


def check_profit_floor(df: pd.DataFrame, min_hits: int) -> pd.Series:
    """
    Check: ¿Tiene suficiente rentabilidad?
    
    Logic:
    - Si profit_hits ≥ min_hits → True
    - Si profit_hits es NaN (sin info) → True (benefit of doubt)
    
    Returns: Serie booleana
    """
    hits = pd.to_numeric(df.get("profit_hits", np.nan), errors="coerce")
    
    # NaN = sin info → pasa (no penalizamos falta de datos)
    return (hits >= min_hits) | hits.isna()


def check_dilution(df: pd.DataFrame, max_issuance: float) -> pd.Series:
    """Check: ¿No está diluyendo agresivamente?"""
    issuance = to_numeric(df.get("share_issuance", np.nan))
    
    # Abs porque emisión puede ser negativa (recompra) y es bueno
    return (issuance.abs() <= max_issuance) | issuance.isna()


def check_asset_growth(df: pd.DataFrame, max_growth: float) -> pd.Series:
    """Check: ¿Crecimiento de activos no inflado?"""
    growth = to_numeric(df.get("asset_growth", np.nan))
    
    return (growth.abs() <= max_growth) | growth.isna()


def check_accruals(df: pd.DataFrame, max_accruals: float) -> pd.Series:
    """Check: ¿Accruals no excesivos?"""
    accr = to_numeric(df.get("accruals_ta", np.nan))
    
    return (accr.abs() <= max_accruals) | accr.isna()


def check_leverage(df: pd.DataFrame, max_ndebt: float) -> pd.Series:
    """Check: ¿Deuda neta manejable?"""
    ndebt = to_numeric(df.get("netdebt_ebitda", np.nan))
    
    return (ndebt <= max_ndebt) | ndebt.isna()


def check_coverage(df: pd.DataFrame, min_cov: int) -> pd.Series:
    """Check: ¿Suficiente cobertura de datos?"""
    cov = pd.to_numeric(df.get("coverage_count", 0), errors="coerce").fillna(0)
    
    return cov >= min_cov


# ============================================================================
# FUNCIÓN PÚBLICA SIMPLIFICADA
# ============================================================================

"""
Fundamentals - Módulo de Guardrails COMPLETO
============================================

Reemplaza apply_quality_guardrails() de fundamentals.py.
Código limpio, modular y testeable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# HELPERS BÁSICOS
# ============================================================================

def to_numeric_safe(s: pd.Series | None) -> pd.Series:
    """Convierte a numérico preservando NaN"""
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")


def get_column_safe(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Obtiene columna o devuelve Serie con default"""
    if col in df.columns:
        return to_numeric_safe(df[col])
    return pd.Series(default, index=df.index, dtype=float)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

@dataclass
class GuardrailConfig:
    """Configuración centralizada de umbrales"""
    profit_min_hits: int = 2
    max_issuance: float = 0.03
    max_asset_growth: float = 0.20
    max_accruals: float = 0.10
    max_netdebt_ebitda: float = 3.0
    min_coverage: int = 2


# ============================================================================
# CHECKS INDIVIDUALES (funciones puras)
# ============================================================================

def check_profit_floor(df: pd.DataFrame, min_hits: int) -> pd.Series:
    """
    Rentabilidad mínima.
    
    Logic:
    - profit_hits ≥ min_hits → True
    - profit_hits = NaN (sin info) → True (benefit of doubt)
    - profit_hits < min_hits → False
    """
    hits = to_numeric_safe(df.get("profit_hits"))
    
    # NaN = sin info → pasa
    return (hits >= min_hits) | hits.isna()


def check_dilution(df: pd.DataFrame, max_issuance: float) -> pd.Series:
    """
    Dilución controlada.
    
    Acepta:
    - |share_issuance| ≤ max_issuance
    - NaN (sin info)
    """
    issuance = to_numeric_safe(df.get("share_issuance"))
    
    return (issuance.abs() <= max_issuance) | issuance.isna()


def check_asset_growth(df: pd.DataFrame, max_growth: float) -> pd.Series:
    """
    Crecimiento de activos no inflado.
    
    Rechaza crecimiento excesivo que puede indicar adquisiciones
    mal estructuradas o expansión insostenible.
    """
    growth = to_numeric_safe(df.get("asset_growth"))
    
    return (growth.abs() <= max_growth) | growth.isna()


def check_accruals(df: pd.DataFrame, max_accruals: float) -> pd.Series:
    """
    Accruals limpios (earnings quality).
    
    Accruals altos = posible manipulación contable.
    """
    accr = to_numeric_safe(df.get("accruals_ta"))
    
    return (accr.abs() <= max_accruals) | accr.isna()


def check_leverage(df: pd.DataFrame, max_ndebt: float) -> pd.Series:
    """
    Apalancamiento manejable.
    
    NetDebt/EBITDA alto = riesgo financiero.
    """
    ndebt = to_numeric_safe(df.get("netdebt_ebitda"))
    
    return (ndebt <= max_ndebt) | ndebt.isna()


def check_coverage(df: pd.DataFrame, min_cov: int) -> pd.Series:
    """
    Cobertura de datos mínima.
    
    Rechaza si no tenemos suficiente información para evaluar.
    """
    cov = to_numeric_safe(df.get("coverage_count")).fillna(0)
    
    return cov >= min_cov


# ============================================================================
# APLICADOR PRINCIPAL
# ============================================================================

def apply_quality_guardrails(
    df: pd.DataFrame,
    config: Optional[GuardrailConfig] = None,
    require_profit_floor: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica guardrails de calidad y devuelve diagnóstico detallado.
    
    Args:
        df: DataFrame con columnas:
            - profit_hits (int)
            - coverage_count (int)
            - share_issuance (float)
            - asset_growth (float)
            - accruals_ta (float)
            - netdebt_ebitda (float)
        config: Configuración de umbrales (usa default si None)
        require_profit_floor: Si exigir profit floor (True recomendado)
    
    Returns:
        (kept, diagnostics)
        
        kept: DataFrame con símbolos que pasaron TODO
            Columnas: ['symbol']
        
        diagnostics: DataFrame detallado con:
            - symbol
            - pass_profit, pass_issuance, pass_assets, 
              pass_accruals, pass_ndebt, pass_coverage
            - pass_all (True si pasó todo)
            - reason (motivos de rechazo si aplica)
    """
    if config is None:
        config = GuardrailConfig()
    
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Ejecutar checks
    # -------------------------------------------------------------------------
    
    if require_profit_floor:
        df["pass_profit"] = check_profit_floor(df, config.profit_min_hits)
    else:
        df["pass_profit"] = True
    
    df["pass_issuance"] = check_dilution(df, config.max_issuance)
    df["pass_assets"] = check_asset_growth(df, config.max_asset_growth)
    df["pass_accruals"] = check_accruals(df, config.max_accruals)
    df["pass_ndebt"] = check_leverage(df, config.max_netdebt_ebitda)
    df["pass_coverage"] = check_coverage(df, config.min_coverage)
    
    # Asegurar booleanos (por si alguno quedó con NaN)
    check_cols = [
        "pass_profit", "pass_issuance", "pass_assets",
        "pass_accruals", "pass_ndebt", "pass_coverage"
    ]
    for col in check_cols:
        df[col] = df[col].fillna(False).astype(bool)
    
    # -------------------------------------------------------------------------
    # Agregado: pass_all
    # -------------------------------------------------------------------------
    
    df["pass_all"] = df[check_cols].all(axis=1)
    
    # -------------------------------------------------------------------------
    # Razón de rechazo
    # -------------------------------------------------------------------------
    
    def _build_reason(row):
        failed = [
            col.replace("pass_", "")
            for col in check_cols
            if not row[col]
        ]
        return ",".join(failed) if failed else ""
    
    df["reason"] = df.apply(_build_reason, axis=1)
    
    # -------------------------------------------------------------------------
    # Split: kept vs diagnostics
    # -------------------------------------------------------------------------
    
    # Kept: solo símbolos que pasaron
    kept = df[df["pass_all"] == True][["symbol"]].copy()
    
    # Diagnostics: todo con checks + reason
    diag_cols = ["symbol"] + check_cols + ["pass_all", "reason"]
    
    # Añadir métricas raw para debugging (opcional)
    optional_cols = [
        "profit_hits", "coverage_count",
        "share_issuance", "asset_growth",
        "accruals_ta", "netdebt_ebitda"
    ]
    diag_cols += [c for c in optional_cols if c in df.columns]
    
    diagnostics = df[diag_cols].copy()
    
    return kept, diagnostics



# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def add_guardrail_flags(
    df: pd.DataFrame,
    config: Optional[GuardrailConfig] = None,
) -> pd.DataFrame:
    """
    Añade columnas pass_* al DataFrame sin hacer split.
    
    Útil cuando quieres los flags pero mantener todas las filas.
    
    Returns:
        DataFrame original + columnas pass_*
    """
    _, diag = apply_quality_guardrails(df, config)
    
    # Merge back
    result = df.merge(
        diag[["symbol", "pass_all", "reason"]],
        on="symbol",
        how="left"
    )
    
    return result

def test_guardrails():
    """Test básico de funcionamiento"""
    print("🧪 Testing guardrails...")
    
    # DataFrame de prueba
    df = pd.DataFrame({
        "symbol": ["PASS", "FAIL_PROFIT", "FAIL_ISSUANCE", "NO_DATA"],
        "profit_hits": [3, 0, 3, np.nan],
        "coverage_count": [4, 4, 4, 1],
        "share_issuance": [0.01, 0.01, 0.10, np.nan],
        "asset_growth": [0.15, 0.15, 0.15, np.nan],
        "accruals_ta": [0.05, 0.05, 0.05, np.nan],
        "netdebt_ebitda": [2.0, 2.0, 2.0, np.nan],
    })
    
    # Aplicar con config estricta
    config = GuardrailConfig(
        profit_min_hits=2,
        max_issuance=0.03,
    )
    
    kept, diag = apply_quality_guardrails(df, config)
    
    # Validaciones
    assert len(kept) == 1, f"Expected 1 pass, got {len(kept)}"
    assert kept["symbol"].iloc[0] == "PASS", "Wrong symbol passed"
    
    assert diag[diag["symbol"] == "FAIL_PROFIT"]["pass_profit"].iloc[0] == False
    assert diag[diag["symbol"] == "FAIL_ISSUANCE"]["pass_issuance"].iloc[0] == False
    
    # NO_DATA pasa profit (NaN = benefit of doubt) pero falla coverage
    assert diag[diag["symbol"] == "NO_DATA"]["pass_coverage"].iloc[0] == False
    
    print("✅ Tests passed")
    print(f"\n📊 Diagnostics:\n{diag}")


if __name__ == "__main__":
    test_guardrails()
    
    print("\n💡 Ejemplo de uso:")
    print("""
    from fundamentals_guardrails import apply_quality_guardrails, GuardrailConfig
    
    # Custom config
    config = GuardrailConfig(
        profit_min_hits=3,
        max_issuance=0.02,
    )
    
    # Aplicar
    kept, diag = apply_quality_guardrails(df, config)
    
    print(f"Pasaron: {len(kept)}")
    print(f"Rechazados por profit: {(diag['pass_profit'] == False).sum()}")
    """)

def _example():
    """Ejemplo de cómo usar la nueva API"""
    
    # 1. Crear config personalizada
    config = GuardrailConfig(
        profit_min_hits=3,  # Más estricto
        max_issuance=0.02,  # Menos dilución
    )
    
    # 2. DataFrame de prueba
    df = pd.DataFrame({
        "symbol": ["AAPL", "TSLA", "GME"],
        "profit_hits": [3, 2, 0],
        "share_issuance": [0.01, 0.05, -0.02],
        "coverage_count": [4, 3, 1],
    })
    
    # 3. Aplicar guardrails
    kept, diag = apply_quality_guardrails(df, config)
    
    print("✅ Pasaron:", kept["symbol"].tolist())
    print("\n📊 Diagnóstico:")
    print(diag)


if __name__ == "__main__":
    _example()


