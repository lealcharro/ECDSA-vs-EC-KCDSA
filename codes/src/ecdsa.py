"""
ECDSA — Algoritmo de Firma Digital por Curvas Elípticas.

Estándares: FIPS 186-5, ANSI X9.62, IEEE 1363-2000, ISO/IEC 15946-2.

────────────────────────────────────────────────────────────────
Generación de claves (parámetros de dominio p, E, G, n):
    d  ←R [1, n−1]          clave privada
    Q  = d · G              clave pública

Firma (mensaje m, clave privada d):
    1.  k  ←R [1, n−1]
    2.  (x₁, y₁) = k · G
    3.  r  = x₁ mod n        (reintentar si r = 0)
    4.  e  = H(m)
    5.  s  = k⁻¹(e + d·r) mod n   (reintentar si s = 0)
    Devolver (r, s)

Verificación (mensaje m, firma (r, s), clave pública Q):
    1.  Verificar r, s ∈ [1, n−1]
    2.  e  = H(m)
    3.  w  = s⁻¹ mod n
    4.  u₁ = e·w mod n,   u₂ = r·w mod n
    5.  X  = u₁·G + u₂·Q
    6.  Aceptar si x(X) mod n = r
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import secrets
from .ec import Curve, Point, SECP256K1

# Funciones auxiliares internas

def _hash_a_entero(mensaje: bytes) -> int:
    """Devuelve SHA-256(mensaje) como entero big-endian."""
    return int.from_bytes(hashlib.sha256(mensaje).digest(), "big")

# API pública

def keygen(curva: Curve = SECP256K1) -> tuple[int, Point]:
    """
    Genera un par de claves ECDSA sobre curva.
    """
    d = secrets.randbelow(curva.n - 1) + 1   # d ∈ [1, n−1]
    Q = curva.mul(d, curva.G)
    return d, Q


def sign(
    mensaje: bytes,
    d: int,
    curva: Curve = SECP256K1,
) -> tuple[int, int]:
    """
    Genera una firma ECDSA para mensaje.
    """
    n = curva.n
    e = _hash_a_entero(mensaje)

    while True:
        # Paso 1 - secreto aleatorio
        k = secrets.randbelow(n - 1) + 1           # k ∈ [1, n−1]

        # Paso 2 - clave publica efímera
        R = curva.mul(k, curva.G)
        assert R is not None                       # k.G nunca es infinito para k ∈ [1, n−1]
        x1, _ = R

        # Paso 3 - primera componente de la firma
        r = x1 % n
        if r == 0:
            continue

        # Paso 5 - segunda componente de la firma
        s = pow(k, -1, n) * (e + d * r) % n
        if s == 0:
            continue

        return r, s


def verify(
    mensaje: bytes,
    firma: tuple[int, int],
    Q: Point,
    curva: Curve = SECP256K1,
) -> bool:
    """
    Verifica una firma ECDSA.
    """
    r, s = firma
    n = curva.n

    # Paso 1 - verificacion de rango
    if not (1 <= r <= n - 1 and 1 <= s <= n - 1):
        return False

    # Paso 2 - hash del mensaje
    e = _hash_a_entero(mensaje)

    # Pasos 3-4 - valores intermedios
    w  = pow(s, -1, n)
    u1 = e * w % n
    u2 = r * w % n

    # Paso 5 - reconstruir el punto
    X = curva.add(curva.mul(u1, curva.G), curva.mul(u2, Q))
    if X is None:
        return False

    # Paso 6 - comparar
    x1, _ = X
    return x1 % n == r
