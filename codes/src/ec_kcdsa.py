from __future__ import annotations

import hashlib
import secrets
from .ec import Curve, Point, SECP256K1

# Longitud del digest SHA-256 en bytes
_L_H: int = 32

# Funciones auxiliares internas

def _sha256(datos: bytes) -> bytes:
    return hashlib.sha256(datos).digest()


def _x_a_bytes(x: int) -> bytes:
    """Codifica un entero x como cadena de 32 bytes big-endian."""
    return x.to_bytes(32, "big")


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR byte a byte de dos cadenas de igual longitud."""
    return bytes(x ^ y for x, y in zip(a, b))


# API publica

def keygen(curva: Curve = SECP256K1) -> tuple[int, Point, bytes]:
    """
    Genera un par de claves EC-KCDSA sobre curva
    """
    n = curva.n
    d     = secrets.randbelow(n - 1) + 1   # d ∈ [1, n−1]
    d_inv = pow(d, -1, n)                  # d⁻¹ mod n
    Q     = curva.mul(d_inv, curva.G)
    assert Q is not None

    Qx, Qy = Q
    h_cert = _sha256(_x_a_bytes(Qx) + _x_a_bytes(Qy))
    return d, Q, h_cert


def sign(
    mensaje: bytes,
    d: int,
    h_cert: bytes,
    curva: Curve = SECP256K1,
) -> tuple[bytes, int]:
    """
    Genera una firma EC-KCDSA para mensaje.
    """
    n = curva.n

    while True:
        # Paso 1 - secreto aleatorio
        k = secrets.randbelow(n - 1) + 1           # k ∈ [1, n−1]

        # Paso 2 - clave publica efimera
        kG = curva.mul(k, curva.G)
        assert kG is not None
        x1, _ = kG

        # Paso 3 - r = H(x₁)
        r = _sha256(_x_a_bytes(x1))

        # Paso 4 - e = H(h_cert ‖ m)
        e = _sha256(h_cert + mensaje)

        # Paso 5 - w̄ = int(r ⊕ e); reducir mod n si es necesario
        w_bar = int.from_bytes(_xor_bytes(r, e), "big")
        if w_bar >= n:
            w_bar -= n

        # Paso 6 - s = d(k − w̄) mod n
        s = d * (k - w_bar) % n
        if s == 0:
            continue

        return r, s


def verify(
    mensaje: bytes,
    firma: tuple[bytes, int],
    Q: Point,
    h_cert: bytes,
    curva: Curve = SECP256K1,
) -> bool:
    """
    Verifica una firma EC-KCDSA.
    """
    r, s = firma
    n = curva.n

    # Paso 1 - validacion de entradas
    if len(r) > _L_H:
        return False
    if not (1 <= s <= n - 1):
        return False

    # Paso 2 - e = H(h_cert ‖ m)
    e = _sha256(h_cert + mensaje)

    # Paso 3 - w̄ = int(r ⊕ e)
    w_bar = int.from_bytes(_xor_bytes(r, e), "big")
    if w_bar >= n:
        w_bar -= n

    # Paso 4 - X = s·Q + w̄·G
    X = curva.add(curva.mul(s, Q), curva.mul(w_bar, curva.G))
    if X is None:
        return False

    # Pasos 5-6 - v = H(x(X));  aceptar si v = r
    x1, _ = X
    v = _sha256(_x_a_bytes(x1))
    return v == r
