"""
Rate Limiter Thread-Safe
=========================

Reemplaza el rate limiting frágil de data_io.py con token bucket thread-safe.
"""

import time
import threading
from typing import Optional


class TokenBucket:
    """
    Rate limiter basado en Token Bucket (thread-safe).
    
    Permite burst controlado sin violar el rate limit promedio.
    """
    
    def __init__(
        self,
        rate: float,           # Tokens por segundo (ej: 4.0 = 4 req/s)
        capacity: Optional[float] = None,  # Max tokens acumulados
        initial: Optional[float] = None,   # Tokens iniciales
    ):
        """
        Args:
            rate: Tasa de generación de tokens (req/s)
            capacity: Capacidad máxima (default: rate * 2)
            initial: Tokens iniciales (default: capacity)
        """
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate * 2
        self.tokens = initial if initial is not None else self.capacity
        
        self._lock = threading.Lock()
        self._last_update = time.monotonic()
    
    def _refill(self):
        """Rellena tokens basado en tiempo transcurrido"""
        now = time.monotonic()
        elapsed = now - self._last_update
        
        # Generar tokens por tiempo transcurrido
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self._last_update = now
    
    def acquire(self, tokens: float = 1.0, block: bool = True) -> bool:
        """
        Intenta consumir 'tokens' tokens.
        
        Args:
            tokens: Número de tokens a consumir
            block: Si esperar cuando no hay tokens suficientes
        
        Returns:
            True si se consumieron, False si no había y block=False
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if not block:
                return False
            
            # Calcular cuánto esperar
            deficit = tokens - self.tokens
            wait_time = deficit / self.rate
        
        # Esperar FUERA del lock para no bloquear otros threads
        time.sleep(wait_time)
        
        with self._lock:
            self._refill()
            self.tokens -= tokens
            return True
    
    def __enter__(self):
        """Context manager: acquire 1 token"""
        self.acquire(1.0)
        return self
    
    def __exit__(self, *args):
        pass


# ============================================================================
# DECORADOR PARA FUNCIONES
# ============================================================================

def rate_limit(rate: float, capacity: Optional[float] = None):
    """
    Decorador que aplica rate limiting a una función.
    
    Usage:
        @rate_limit(rate=4.0)  # Max 4 llamadas/s
        def api_call():
            ...
    """
    bucket = TokenBucket(rate, capacity)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            bucket.acquire()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# INTEGRACION CON FMP
# ============================================================================

class FMPRateLimiter:
    """
    Rate limiter específico para Financial Modeling Prep API.
    
    - Free tier: 250 req/day, ~10 req/min
    - Paid tier: Hasta 300 req/min
    """
    
    def __init__(self, tier: str = "free"):
        """
        Args:
            tier: 'free' o 'paid'
        """
        if tier == "free":
            # Conservador: 4 req/s = 240/min (bajo límite de 300)
            self.bucket = TokenBucket(rate=4.0, capacity=10.0)
        elif tier == "paid":
            # Agresivo: 10 req/s = 600/min (tolerante a burst)
            self.bucket = TokenBucket(rate=10.0, capacity=30.0)
        else:
            raise ValueError(f"Tier '{tier}' inválido (usa 'free' o 'paid')")
    
    def acquire(self):
        """Espera hasta que esté disponible el siguiente slot"""
        return self.bucket.acquire()
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, *args):
        pass


# Global instance (importar esto en data_io.py)
fmp_limiter = FMPRateLimiter(tier="free")


# ============================================================================
# TESTS
# ============================================================================

def test_token_bucket():
    """Test básico de funcionamiento"""
    print("🧪 Testing TokenBucket...")
    
    # 2 req/s, capacity 4
    limiter = TokenBucket(rate=2.0, capacity=4.0)
    
    # Primera ráfaga (hasta 4 tokens disponibles)
    start = time.time()
    for i in range(4):
        assert limiter.acquire(block=False), f"Failed at {i}"
    print(f"✅ Burst de 4 req OK ({time.time() - start:.3f}s)")
    
    # Siguiente debe esperar
    assert not limiter.acquire(block=False), "Debería rechazar (sin tokens)"
    print("✅ Correctamente rechazó sin block")
    
    # Con block=True debe esperar ~0.5s (1 token a 2/s)
    start = time.time()
    limiter.acquire(block=True)
    elapsed = time.time() - start
    assert 0.4 < elapsed < 0.7, f"Esperó {elapsed:.3f}s (esperado ~0.5s)"
    print(f"✅ Block funcionó correctamente ({elapsed:.3f}s)")
    
    print("✅ Todos los tests pasaron\n")


def test_concurrent():
    """Test de thread-safety"""
    import concurrent.futures
    
    print("🧪 Testing concurrencia...")
    
    limiter = TokenBucket(rate=10.0)  # 10 req/s
    results = []
    
    def worker(i):
        limiter.acquire()
        results.append(time.time())
        return i
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(worker, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    
    elapsed = time.time() - start
    
    # 20 req a 10 req/s = ~2s (con burst inicial puede ser ~1.5s)
    assert 1.5 < elapsed < 3.0, f"Tardó {elapsed:.3f}s (esperado ~2s)"
    print(f"✅ Concurrencia OK: {len(results)} req en {elapsed:.3f}s")
    print(f"   Rate real: {len(results)/elapsed:.1f} req/s")


if __name__ == "__main__":
    test_token_bucket()
    test_concurrent()
    
    print("\n💡 Ejemplo de uso:")
    print("""
    # En data_io.py:
    from rate_limiter import fmp_limiter
    
    def _http_get(url, params):
        fmp_limiter.acquire()  # Espera su turno
        r = session.get(url, params=params)
        return r.json()
    """)