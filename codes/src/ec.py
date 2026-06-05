from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Un punto de la curva: (x, y) ∈ F_p × F_p  o  None (= punto en el infinito).
Point = Optional[tuple[int, int]]


@dataclass(frozen=True)
class Curve:
    """Contenedor inmutable de parametros de dominio de curva eliptica."""
    name: str
    p: int   # Primo del campo (característica de F_p)
    a: int   # Coeficiente de Weierstrass  a  en y² = x³ + ax + b
    b: int   # Coeficiente de Weierstrass  b
    Gx: int  # Coordenada x del punto base G
    Gy: int  # Coordenada y del punto base G
    n: int   # Orden de G  (primo)
    h: int   # Cofactor  h = #E(F_p) / n

    # Propiedades de conveniencia

    @property
    def G(self) -> Point:
        """Punto base (generador)."""
        return (self.Gx, self.Gy)

    @property
    def bits(self) -> int:
        """Longitud en bits del primo del campo p."""
        return self.p.bit_length()

    # Validación

    def esta_en_curva(self, P: Point) -> bool:
        """Devuelve True si P satisface la ecuacion de la curva (None -> True)."""
        if P is None:
            return True
        x, y = P
        lado_izq = pow(y, 2, self.p)
        lado_der = (pow(x, 3, self.p) + self.a * x + self.b) % self.p
        return lado_izq == lado_der

    # Alias en ingles para compatibilidad con los tests
    def is_on_curve(self, P: Point) -> bool:
        return self.esta_en_curva(P)

    # Operaciones del grupo

    def neg(self, P: Point) -> Point:
        """Devuelve -P (el inverso del grupo para P)."""
        if P is None:
            return None
        return (P[0], (-P[1]) % self.p)

    def add(self, P: Point, Q: Point) -> Point:
        """
        Calcula P + Q usando la regla de cuerda y tangente.

        Casos tratados:
          P = infinito  →  Q
          Q = infinito  →  P
          P = Q  →  doblado (tangente)
          P = -Q →  infinito
          otro   →  formula de la cuerda
        """
        if P is None:
            return Q
        if Q is None:
            return P

        x1, y1 = P
        x2, y2 = Q
        p = self.p

        if x1 == x2:
            if y1 != y2:
                # P = -Q  →  P + Q = infinito
                return None
            # P = Q  →  doblado de punto
            if y1 == 0:
                # Tangente vertical, resultado es 
                return None
            # λ = (3x₁² + a) / (2y₁)  mod p
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, p) % p
        else:
            # λ = (y₂ - y₁) / (x₂ - x₁)  mod p
            lam = (y2 - y1) * pow(x2 - x1, -1, p) % p

        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def mul(self, k: int, P: Point) -> Point:
        """
        Calcula el multiplo escalar k.P (doble y suma, izquierda a derecha).

        Maneja k = 0, k < 0 y P = infinito como casos especiales.
        """
        if k == 0 or P is None:
            return None
        if k < 0:
            return self.mul(-k, self.neg(P))

        resultado: Point = None
        sumando: Point = P
        while k:
            if k & 1:
                resultado = self.add(resultado, sumando)
            sumando = self.add(sumando, sumando)
            k >>= 1
        return resultado


# Parametros de curvas estándar

# secp256k1 - curva de Koblitz usada por Bitcoin y Ethereum.
# a = 0, b = 7  →  y² = x³ + 7
SECP256K1 = Curve(
    name="secp256k1",
    p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
    a=0,
    b=7,
    Gx=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    Gy=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
    h=1,
)

# NIST P-256 (secp256r1) - usada en TLS 1.3 y FIPS 186-5.
# a = p - 3  (≡ -3 mod p)
P256 = Curve(
    name="P-256",
    p=0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF,
    a=0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC,
    b=0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B,
    Gx=0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
    Gy=0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5,
    n=0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551,
    h=1,
)
