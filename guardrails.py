# qvm_pack/guardrails.py
from dataclasses import dataclass
import pandas as pd

COVERAGE_COLS = ["profit_hits","netdebt_ebitda","accruals_ta","asset_growth","share_issuance"]

@dataclass
class GuardrailsCfg:
    min_coverage: int
    profit_min_hits: int
    max_net_issuance: float
    max_asset_growth: float
    max_accruals_ta: float
    max_netdebt_ebitda: float

def build_guardrails_base(rows_df: pd.DataFrame) -> pd.DataFrame:
    base = rows_df.copy()
    base["coverage_count"] = (
        base[COVERAGE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .sum(axis=1)
        .astype(int)
    )
    return base

def apply_guardrails(base: pd.DataFrame, cfg: GuardrailsCfg) -> pd.DataFrame:
    pass_profit   = base["profit_hits"] >= cfg.profit_min_hits
    pass_issuance = base["share_issuance"].abs() <= cfg.max_net_issuance
    pass_assets   = base["asset_growth"].abs()   <= cfg.max_asset_growth
    pass_accruals = base["accruals_ta"].abs()    <= cfg.max_accruals_ta
    pass_leverage = base["netdebt_ebitda"]       <= cfg.max_netdebt_ebitda
    pass_cover    = base["coverage_count"]       >= cfg.min_coverage

    strict = pass_cover & pass_profit & pass_issuance & pass_assets & pass_accruals & pass_leverage
    return base.assign(
        pass_profit=pass_profit, pass_issuance=pass_issuance,
        pass_assets=pass_assets, pass_accruals=pass_accruals,
        pass_leverage=pass_leverage, pass_all=strict
    )
