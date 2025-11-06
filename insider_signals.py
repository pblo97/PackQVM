"""
Insider Trading Signals - Detectar Compras/Ventas de Insiders
===============================================================

Basado en:
- Seyhun (1986): "Insiders' Profits, Costs of Trading, and Market Efficiency"
- Lakonishok & Lee (2001): "Are Insider Trades Informative?"
- Jeng, Metrick & Zeckhauser (2003): "Estimating the Returns to Insider Trading"

Hallazgos clave:
- Insider BUYING → +6-8% next year (señal fuerte)
- Insider SELLING → débil predictor (venden por múltiples razones)
- Cluster de compras → señal más fuerte
- Open market purchases > options exercise (más significativo)

API: Financial Modeling Prep
Endpoint: /v4/insider-trading?symbol=AAPL
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


# ============================================================================
# CLASIFICACIÓN DE TRANSACCIONES
# ============================================================================

def classify_insider_transaction(transaction_type: str) -> str:
    """
    Clasifica transacción de insider en: BUY, SELL, NEUTRAL

    FMP codes:
    - P-Purchase: Compra en open market (MÁS SIGNIFICATIVA)
    - S-Sale: Venta
    - A-Award: Grant de acciones (neutral)
    - M-ExerciseOption: Ejercicio de opciones (menos significativo)
    - D-Return: Devolución de shares
    """
    t = str(transaction_type).upper()

    # COMPRAS (señal positiva)
    if 'P-PURCHASE' in t or 'P' == t:
        return 'BUY'

    # VENTAS (señal negativa débil)
    if 'S-SALE' in t or 'S-SALE (PARTIAL)' in t or 'S' == t:
        return 'SELL'

    # EJERCICIO DE OPCIONES (neutral a ligeramente positivo)
    if 'M-' in t or 'EXERCISE' in t:
        return 'NEUTRAL'

    # AWARDS/GRANTS (neutral)
    if 'A-' in t or 'AWARD' in t or 'GRANT' in t:
        return 'NEUTRAL'

    # Default
    return 'NEUTRAL'


def calculate_insider_score(transactions: pd.DataFrame, lookback_days: int = 90) -> Dict:
    """
    Calcula insider trading score basado en transacciones recientes.

    Args:
        transactions: DataFrame con columnas:
                     - transactionDate
                     - transactionType (P-Purchase, S-Sale, etc)
                     - shares
                     - price (opcional)
        lookback_days: Días hacia atrás para considerar (default 90)

    Returns:
        Dict con métricas:
        - net_shares: Net compras - ventas
        - buy_count: # de transacciones de compra
        - sell_count: # de transacciones de venta
        - insider_score: Score 0-100 (100 = heavy buying)
    """
    if transactions is None or transactions.empty:
        return {
            'net_shares': 0,
            'buy_count': 0,
            'sell_count': 0,
            'insider_score': 50.0,  # Neutral
            'signal': 'NEUTRAL'
        }

    # Convertir fecha
    transactions = transactions.copy()
    transactions['transactionDate'] = pd.to_datetime(transactions['transactionDate'], errors='coerce')

    # Filtrar últimos N días
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    recent = transactions[transactions['transactionDate'] >= cutoff_date]

    if recent.empty:
        return {
            'net_shares': 0,
            'buy_count': 0,
            'sell_count': 0,
            'insider_score': 50.0,
            'signal': 'NEUTRAL'
        }

    # Clasificar transacciones
    recent['classification'] = recent['transactionType'].apply(classify_insider_transaction)

    # Contar compras y ventas
    buys = recent[recent['classification'] == 'BUY']
    sells = recent[recent['classification'] == 'SELL']

    buy_count = len(buys)
    sell_count = len(sells)

    # Net shares
    buy_shares = buys['shares'].sum() if not buys.empty else 0
    sell_shares = sells['shares'].sum() if not sells.empty else 0
    net_shares = buy_shares - sell_shares

    # Calcular score (0-100)
    # Lógica:
    # - Solo compras → 100
    # - Solo ventas → 0
    # - Net neutral → 50
    # - Peso mayor a compras (más significativas según paper)

    if buy_count > 0 and sell_count == 0:
        # Solo compras (muy bullish)
        score = 80 + min(20, buy_count * 5)  # Max 100

    elif sell_count > 0 and buy_count == 0:
        # Solo ventas (bearish débil)
        score = 30 - min(20, sell_count * 2)  # Min 10

    elif buy_count > 0 and sell_count > 0:
        # Mixto - depende de ratio
        ratio = buy_shares / (sell_shares + 1)
        if ratio > 2:
            score = 70  # Net buyers
        elif ratio > 1:
            score = 60  # Ligeramente más compras
        elif ratio > 0.5:
            score = 45  # Ligeramente más ventas
        else:
            score = 35  # Net sellers

    else:
        # Sin transacciones de compra/venta
        score = 50.0

    # Signal
    if score >= 65:
        signal = 'BUY'
    elif score <= 35:
        signal = 'SELL'
    else:
        signal = 'NEUTRAL'

    return {
        'net_shares': int(net_shares),
        'buy_count': int(buy_count),
        'sell_count': int(sell_count),
        'insider_score': float(score),
        'signal': signal
    }


# ============================================================================
# FILTROS
# ============================================================================

def filter_positive_insider_activity(
    transactions: pd.DataFrame,
    min_score: float = 60.0,
    lookback_days: int = 90,
) -> bool:
    """
    Filtrar acciones con insider buying positivo.

    Args:
        transactions: DataFrame de insider trades
        min_score: Score mínimo para pasar (default 60 = net buyers)
        lookback_days: Ventana de tiempo

    Returns:
        True si pasa filtro (positive insider activity)
    """
    metrics = calculate_insider_score(transactions, lookback_days)

    return metrics['insider_score'] >= min_score


def filter_no_heavy_insider_selling(
    transactions: pd.DataFrame,
    max_sell_count: int = 5,
    lookback_days: int = 90,
) -> bool:
    """
    Filtrar acciones sin heavy insider selling.

    Rationale: Heavy selling es red flag (insiders saben algo malo)

    Args:
        transactions: DataFrame de insider trades
        max_sell_count: Máximo # de ventas permitidas
        lookback_days: Ventana de tiempo

    Returns:
        True si pasa filtro (no heavy selling)
    """
    metrics = calculate_insider_score(transactions, lookback_days)

    return metrics['sell_count'] <= max_sell_count


# ============================================================================
# CLUSTER DETECTION (Señal más fuerte)
# ============================================================================

def detect_insider_buying_cluster(
    transactions: pd.DataFrame,
    min_insiders: int = 3,
    window_days: int = 30,
) -> bool:
    """
    Detecta "cluster" de compras de insiders (señal MÁS fuerte).

    Paper: Lakonishok & Lee (2001) - múltiples insiders comprando
    simultáneamente es señal más confiable que 1 insider.

    Args:
        transactions: DataFrame de insider trades
        min_insiders: Mínimo # de insiders distintos comprando
        window_days: Ventana de tiempo para cluster

    Returns:
        True si hay cluster de compras
    """
    if transactions is None or transactions.empty:
        return False

    transactions = transactions.copy()
    transactions['transactionDate'] = pd.to_datetime(transactions['transactionDate'], errors='coerce')
    transactions['classification'] = transactions['transactionType'].apply(classify_insider_transaction)

    # Filtrar solo compras
    buys = transactions[transactions['classification'] == 'BUY']

    if buys.empty:
        return False

    # Filtrar ventana reciente
    cutoff_date = datetime.now() - timedelta(days=window_days)
    recent_buys = buys[buys['transactionDate'] >= cutoff_date]

    if recent_buys.empty:
        return False

    # Contar insiders únicos (por nombre o ID)
    # Asumiendo que hay columna 'reportingName' o 'insiderName'
    unique_insiders = 0

    if 'reportingName' in recent_buys.columns:
        unique_insiders = recent_buys['reportingName'].nunique()
    elif 'insiderName' in recent_buys.columns:
        unique_insiders = recent_buys['insiderName'].nunique()
    else:
        # Fallback: contar # de transacciones
        unique_insiders = len(recent_buys)

    return unique_insiders >= min_insiders


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def calculate_insider_scores_batch(
    insider_data_dict: Dict[str, pd.DataFrame],
    lookback_days: int = 90,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calcula insider scores para múltiples símbolos.

    Args:
        insider_data_dict: {symbol: transactions_df}
        lookback_days: Ventana de tiempo
        verbose: Print progress

    Returns:
        DataFrame con insider metrics por símbolo
    """
    results = []

    for symbol, transactions in insider_data_dict.items():
        try:
            metrics = calculate_insider_score(transactions, lookback_days)

            # Detectar cluster
            has_cluster = detect_insider_buying_cluster(transactions)

            results.append({
                'symbol': symbol,
                'insider_score': metrics['insider_score'],
                'signal': metrics['signal'],
                'buy_count': metrics['buy_count'],
                'sell_count': metrics['sell_count'],
                'net_shares': metrics['net_shares'],
                'has_buying_cluster': has_cluster,
            })

        except Exception as e:
            if verbose:
                print(f"⚠️  {symbol}: error calculando insider score - {e}")
            continue

    df = pd.DataFrame(results)

    if verbose and not df.empty:
        print(f"\n✅ Insider scores calculados para {len(df)} símbolos")
        buy_signals = len(df[df['signal'] == 'BUY'])
        clusters = df['has_buying_cluster'].sum()
        print(f"   BUY signals: {buy_signals} ({100*buy_signals/len(df):.1f}%)")
        print(f"   Buying clusters: {clusters}")

    return df


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def add_insider_signals_to_df(
    df: pd.DataFrame,
    insider_data_dict: Dict[str, pd.DataFrame],
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Agrega insider signals al DataFrame principal.

    Args:
        df: DataFrame con símbolos
        insider_data_dict: {symbol: transactions_df}
        lookback_days: Ventana de tiempo

    Returns:
        DataFrame con columnas de insider signals
    """
    df = df.copy()

    # Calcular scores batch
    insider_df = calculate_insider_scores_batch(insider_data_dict, lookback_days)

    if insider_df.empty:
        df['insider_score'] = 50.0  # Neutral
        df['insider_signal'] = 'NEUTRAL'
        df['has_buying_cluster'] = False
        return df

    # Merge
    df = df.merge(
        insider_df[['symbol', 'insider_score', 'signal', 'has_buying_cluster']],
        on='symbol',
        how='left',
        suffixes=('', '_insider')
    )

    # Rename si hay conflicto
    if 'signal_insider' in df.columns:
        df['insider_signal'] = df['signal_insider']
        df = df.drop('signal_insider', axis=1)
    elif 'signal' in df.columns:
        df['insider_signal'] = df['signal']

    # Fill NaN
    df['insider_score'] = df['insider_score'].fillna(50.0)
    df['insider_signal'] = df['insider_signal'].fillna('NEUTRAL')
    df['has_buying_cluster'] = df['has_buying_cluster'].fillna(False)

    return df


# ============================================================================
# NOTA: FETCH DE DATOS
# ============================================================================
"""
Para usar este módulo, necesitas fetch de insider trading data desde FMP:

from data_fetcher import fetch_insider_trading

# Fetch para un símbolo
insider_trades = fetch_insider_trading('AAPL')

# Si no existe la función, agregar a data_fetcher.py:
def fetch_insider_trading(symbol: str, api_key: Optional[str] = None) -> pd.DataFrame:
    if api_key is None:
        api_key = get_api_key()

    url = f"https://financialmodelingprep.com/api/v4/insider-trading"
    params = {
        'symbol': symbol,
        'apikey': api_key
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data:
            return pd.DataFrame(data)

    return pd.DataFrame()
"""


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing insider_signals...")

    # Mock data: Heavy insider buying
    heavy_buying = pd.DataFrame({
        'transactionDate': ['2024-10-01', '2024-10-05', '2024-10-10'],
        'transactionType': ['P-Purchase', 'P-Purchase', 'P-Purchase'],
        'shares': [10000, 5000, 8000],
        'reportingName': ['John Doe', 'Jane Smith', 'Bob Wilson']
    })

    # Mock data: Heavy insider selling
    heavy_selling = pd.DataFrame({
        'transactionDate': ['2024-10-01', '2024-10-03', '2024-10-05', '2024-10-08'],
        'transactionType': ['S-Sale', 'S-Sale', 'S-Sale', 'S-Sale'],
        'shares': [20000, 15000, 10000, 25000],
        'reportingName': ['CEO', 'CFO', 'Director', 'VP Sales']
    })

    # Test 1: Heavy buying
    print("\n📈 Heavy Insider Buying:")
    metrics_buy = calculate_insider_score(heavy_buying)
    print(f"   Score: {metrics_buy['insider_score']:.1f}/100")
    print(f"   Signal: {metrics_buy['signal']}")
    print(f"   Buy count: {metrics_buy['buy_count']}")
    print(f"   Net shares: {metrics_buy['net_shares']:,}")

    has_cluster = detect_insider_buying_cluster(heavy_buying)
    print(f"   Has buying cluster: {has_cluster}")

    # Test 2: Heavy selling
    print("\n📉 Heavy Insider Selling:")
    metrics_sell = calculate_insider_score(heavy_selling)
    print(f"   Score: {metrics_sell['insider_score']:.1f}/100")
    print(f"   Signal: {metrics_sell['signal']}")
    print(f"   Sell count: {metrics_sell['sell_count']}")

    print("\n✅ Tests complete!")
