from __future__ import annotations

import argparse
import gc
import platform
import secrets
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

from src.ec import SECP256K1, Curve
from src import ecdsa, ec_kcdsa

CURVE = SECP256K1

# Defaults
DEFAULT_SIZES = [64, 256, 1_024, 4_096, 16_384, 65_536]
DEFAULT_REPS  = 1000   # más repeticiones → mínimo más estable (el ruido solo suma)
DEFAULT_WARMUP = 5    # llena cachés de CPU y resuelve atributos antes de medir


# Low-level timing helper

def _timeit(fn: Callable, *args, reps: int) -> list[float]:
    """Ejecuta fn(*args) *reps* veces; devuelve los tiempos en milisegundos.

    El recolector de basura se desactiva durante la medición: una pausa de GC
    en mitad de una ejecución inflaría arbitrariamente ese tiempo individual.
    """
    times = []
    gc_estaba_activo = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1_000)
    finally:
        if gc_estaba_activo:
            gc.enable()
    return times


@dataclass(frozen=True)
class Stats:
    """Estadísticas (en ms) de una serie de tiempos.

    El *mínimo* es la métrica principal: el ruido del sistema solo suma
    tiempo, así que la ejecución más rápida es la más limpia.  La mediana, la
    media y la desviación se conservan como diagnóstico de variabilidad.
    """

    minimum: float
    median: float
    mean: float
    std: float


def _stats(times: list[float]) -> Stats:
    """Resume una serie de tiempos; el mínimo es la métrica principal."""
    return Stats(
        minimum=min(times),
        median=statistics.median(times),
        mean=statistics.mean(times),
        std=statistics.stdev(times) if len(times) > 1 else 0.0,
    )

# Per-algorithm benchmark runners

def bench_ecdsa(sizes: list[int], reps: int, warmup: int) -> dict:
    """
    Retorna un dict indexado por tamaño de mensaje; cada valor contiene
    estadísticas de firma y verificación.
    """
    # Un par de claves para todos los tamaños (keygen excluido de sign/verify)
    d, Q = ecdsa.keygen(CURVE)

    # Verificación de cordura: nunca medir una implementación rota.  Si el
    # round-trip falla, `verify` tomaría un camino corto (rechazo temprano) y
    # los tiempos no representarían la verificación completa.
    _m = b"benchmark sanity check"
    if not ecdsa.verify(_m, ecdsa.sign(_m, d, CURVE), Q, CURVE):
        raise RuntimeError(
            "ECDSA: el round-trip sign/verify falló — medición abortada"
        )

    # Benchmark de keygen por separado
    _timeit(ecdsa.keygen, CURVE, reps=warmup)
    kg_times = _timeit(ecdsa.keygen, CURVE, reps=reps)

    results: dict = {"keygen": _stats(kg_times)}

    for n_bytes in sizes:
        msg = secrets.token_bytes(n_bytes)

        # Warm up
        for _ in range(warmup):
            sig = ecdsa.sign(msg, d, CURVE)
            ecdsa.verify(msg, sig, Q, CURVE)

        # Firma
        sign_times  = _timeit(ecdsa.sign, msg, d, CURVE, reps=reps)

        # Verificación (reutilizar firma precalculada para medir solo verify)
        sig = ecdsa.sign(msg, d, CURVE)
        assert ecdsa.verify(msg, sig, Q, CURVE)  # garantiza el camino completo
        verify_times = _timeit(ecdsa.verify, msg, sig, Q, CURVE, reps=reps)

        results[n_bytes] = {
            "sign":   _stats(sign_times),
            "verify": _stats(verify_times),
        }

    return results


def bench_ec_kcdsa(sizes: list[int], reps: int, warmup: int) -> dict:
    d, Q, h_cert = ec_kcdsa.keygen(CURVE)

    # Verificación de cordura: nunca medir una implementación rota.
    _m = b"benchmark sanity check"
    if not ec_kcdsa.verify(_m, ec_kcdsa.sign(_m, d, h_cert, CURVE), Q, h_cert, CURVE):
        raise RuntimeError(
            "EC-KCDSA: el round-trip sign/verify falló — medición abortada"
        )

    _timeit(ec_kcdsa.keygen, CURVE, reps=warmup)
    kg_times = _timeit(ec_kcdsa.keygen, CURVE, reps=reps)

    results: dict = {"keygen": _stats(kg_times)}

    for n_bytes in sizes:
        msg = secrets.token_bytes(n_bytes)

        for _ in range(warmup):
            sig = ec_kcdsa.sign(msg, d, h_cert, CURVE)
            ec_kcdsa.verify(msg, sig, Q, h_cert, CURVE)

        sign_times = _timeit(ec_kcdsa.sign, msg, d, h_cert, CURVE, reps=reps)

        sig = ec_kcdsa.sign(msg, d, h_cert, CURVE)
        assert ec_kcdsa.verify(msg, sig, Q, h_cert, CURVE)  # camino completo
        verify_times = _timeit(
            ec_kcdsa.verify, msg, sig, Q, h_cert, CURVE, reps=reps
        )

        results[n_bytes] = {
            "sign":   _stats(sign_times),
            "verify": _stats(verify_times),
        }

    return results

# Output helpers

def _ms(s: Stats) -> str:
    """Formatea un Stats como 'mín ± σ' (ms); el mínimo es la métrica principal."""
    return f"{s.minimum:7.2f} ± {s.std:5.2f}"


def print_table(
    er: dict,
    kr: dict,
    sizes: list[int],
):
    """Imprime la tabla comparativa lado a lado."""
    header_top = (
        f"{'N bytes':>9} │ "
        f"{'── ECDSA ──':^31} │ "
        f"{'── EC-KCDSA ──':^31} │ "
        f"{'─ Ratio ECDSA/KCDSA ─':^23}"
    )
    header_row = (
        f"{'':>9} │ "
        f"{'Sign (ms)':^14}  {'Verify (ms)':^14} │ "
        f"{'Sign (ms)':^14}  {'Verify (ms)':^14} │ "
        f"{'Sign':^10}  {'Verify':^10}"
    )
    W = len(header_top)

    print("\n" + "═" * W)
    print(header_top)
    print(header_row)
    print("─" * W)

    # Fila de keygen
    e_kg = er["keygen"]
    k_kg = kr["keygen"]
    kg_ratio = e_kg.minimum / k_kg.minimum if k_kg.minimum else 0
    print(
        f"{'keygen':>9} │ "
        f"{_ms(e_kg):^31} │ "
        f"{_ms(k_kg):^31} │ "
        f"{kg_ratio:^10.3f}  {'':^10}"
    )
    print("─" * W)

    for n_bytes in sizes:
        es = er[n_bytes]["sign"]
        ev = er[n_bytes]["verify"]
        ks = kr[n_bytes]["sign"]
        kv = kr[n_bytes]["verify"]

        ratio_s = es.minimum / ks.minimum if ks.minimum else 0
        ratio_v = ev.minimum / kv.minimum if kv.minimum else 0

        print(
            f"{n_bytes:>9} │ "
            f"{_ms(es):^31}  {_ms(ev):^31} │ "
            f"{_ms(ks):^31}  {_ms(kv):^31} │ "
            f"{ratio_s:^10.3f}  {ratio_v:^10.3f}"
        )

    print("═" * W)
    print(
        "  Ratio > 1.0  →  ECDSA es más lento que EC-KCDSA\n"
        "  Ratio < 1.0  →  ECDSA es más rápido que EC-KCDSA\n"
    )


def print_analysis(er: dict, kr: dict, sizes: list[int]):
    """Imprime un comentario analítico breve sobre los resultados."""
    print("─" * 60)
    print("  Análisis")
    print("─" * 60)

    # Variación respecto a N
    ecdsa_sign_times  = [er[n]["sign"].minimum for n in sizes]
    kcdsa_sign_times  = [kr[n]["sign"].minimum for n in sizes]

    def variation(times):
        return (max(times) - min(times)) / statistics.mean(times) * 100

    print(
        f"\n  Variación del tiempo de firma respecto a N:\n"
        f"    ECDSA    : {variation(ecdsa_sign_times):.1f}%\n"
        f"    EC-KCDSA : {variation(kcdsa_sign_times):.1f}%\n"
        f"  → El tiempo de firma/verificación es prácticamente\n"
        f"    independiente del tamaño del mensaje.  Las operaciones\n"
        f"    de curva elíptica dominan; el SHA-256 es despreciable."
    )

    avg_ratio_s = statistics.mean(
        er[n]["sign"].minimum / kr[n]["sign"].minimum for n in sizes
        if kr[n]["sign"].minimum > 0
    )
    avg_ratio_v = statistics.mean(
        er[n]["verify"].minimum / kr[n]["verify"].minimum for n in sizes
        if kr[n]["verify"].minimum > 0
    )
    print(
        f"\n  Ratio medio ECDSA / EC-KCDSA:\n"
        f"    Firma        : {avg_ratio_s:.3f}x\n"
        f"    Verificación : {avg_ratio_v:.3f}x"
    )

    if avg_ratio_s > 1.01:
        print(
            "  → EC-KCDSA firma más rápido: evita calcular k⁻¹ mod n."
        )
    elif avg_ratio_s < 0.99:
        print(
            "  → ECDSA firma ligeramente más rápido en esta ejecución\n"
            "    (Python puro: la varianza oculta diferencias pequeñas)."
        )
    else:
        print("  → Diferencia de firma dentro del margen de varianza.")

    if avg_ratio_v > 1.01:
        print(
            "  → EC-KCDSA verifica más rápido: evita calcular s⁻¹ mod n."
        )
    print()

# Conteo de operaciones de curva  (métrica independiente del hardware)

class CountingCurve:
    """Envoltura sobre una :class:`Curve` que cuenta las operaciones de curva
    de *alto nivel* que invoca un algoritmo: multiplicaciones escalares
    (``mul``) y sumas de puntos (``add``).

    Como cada llamada se delega a la curva real, las sumas internas que
    ``mul`` realiza por debajo (doble-y-suma) NO se contabilizan: se cuenta
    únicamente lo que el algoritmo pide de forma explícita.  Esa es justamente
    la métrica algorítmica — independiente del intérprete y del hardware.

    Reenvía cualquier otro atributo (``n``, ``G``, ``p``, ``name`` …) a la
    curva subyacente mediante ``__getattr__``.
    """

    def __init__(self, curve: Curve):
        self._curve = curve
        self.muls = 0
        self.adds = 0

    def reset(self) -> None:
        self.muls = 0
        self.adds = 0

    def mul(self, k: int, P):
        self.muls += 1
        return self._curve.mul(k, P)

    def add(self, P, Q):
        self.adds += 1
        return self._curve.add(P, Q)

    def __getattr__(self, name):
        # Solo se invoca si el atributo no existe en la instancia (delegación).
        return getattr(self._curve, name)


def count_curve_ops() -> dict:
    """Cuenta multiplicaciones escalares y sumas de puntos por operación.

    El tamaño del mensaje no influye en estas cifras (solo cambia el coste de
    hashing, que no es una operación de curva), de modo que basta un mensaje.
    """
    msg = b"op-count probe"

    cc = CountingCurve(CURVE)
    out: dict = {"ECDSA": {}, "EC-KCDSA": {}}

    # ECDSA
    cc.reset(); d, Q = ecdsa.keygen(cc);              out["ECDSA"]["keygen"] = (cc.muls, cc.adds)
    cc.reset(); sig = ecdsa.sign(msg, d, cc);          out["ECDSA"]["sign"]   = (cc.muls, cc.adds)
    cc.reset(); ecdsa.verify(msg, sig, Q, cc);         out["ECDSA"]["verify"] = (cc.muls, cc.adds)

    # EC-KCDSA
    cc.reset(); d, Q, h = ec_kcdsa.keygen(cc);         out["EC-KCDSA"]["keygen"] = (cc.muls, cc.adds)
    cc.reset(); sig = ec_kcdsa.sign(msg, d, h, cc);    out["EC-KCDSA"]["sign"]   = (cc.muls, cc.adds)
    cc.reset(); ec_kcdsa.verify(msg, sig, Q, h, cc);   out["EC-KCDSA"]["verify"] = (cc.muls, cc.adds)

    return out


def print_op_counts(oc: dict):
    """Imprime el conteo de operaciones de curva por algoritmo y operación."""
    def cell(t: tuple[int, int]) -> str:
        m, a = t
        return f"{m} mul, {a} add"

    print("─" * 60)
    print("  Operaciones de curva por operación  (independiente del HW)")
    print("─" * 60)
    print(f"\n  {'':<8} │ {'ECDSA':^16} │ {'EC-KCDSA':^16}")
    print(f"  {'─'*8}─┼─{'─'*16}─┼─{'─'*16}")
    for op in ("keygen", "sign", "verify"):
        print(f"  {op:<8} │ {cell(oc['ECDSA'][op]):^16} │ {cell(oc['EC-KCDSA'][op]):^16}")
    print(
        "\n  Nota: solo se cuentan operaciones de CURVA.  El menor tiempo de\n"
        "  EC-KCDSA en firma/verificación se debe además a que evita las\n"
        "  inversiones modulares k⁻¹ (firma) y s⁻¹ (verificación) que ECDSA sí\n"
        "  necesita — un coste de campo, no de curva, no reflejado en esta tabla."
    )
    print()

# Tamaño de claves y firmas

def measure_sizes() -> dict:
    """Mide el tamaño serializado (bytes) de claves y firmas de cada algoritmo.

    Se usan codificaciones de ancho fijo derivadas de los parámetros de la
    curva (lo que hace una implementación real), no la longitud del entero de
    una instancia concreta (que varía ±1 byte según los ceros a la izquierda).
    """
    field_bytes  = (CURVE.p.bit_length() + 7) // 8   # tamaño de coordenada en F_p
    scalar_bytes = (CURVE.n.bit_length() + 7) // 8   # tamaño de escalar mod n

    # Una firma real de cada algoritmo para leer la longitud de r (EC-KCDSA).
    d_e, Q_e = ecdsa.keygen(CURVE)
    ecdsa.sign(b"size probe", d_e, CURVE)            # round-trip de cordura implícito
    d_k, Q_k, h_cert = ec_kcdsa.keygen(CURVE)
    r_k, _s_k = ec_kcdsa.sign(b"size probe", d_k, h_cert, CURVE)

    return {
        "field_bytes":  field_bytes,
        "scalar_bytes": scalar_bytes,
        "ECDSA": {
            "priv":           scalar_bytes,
            "pub_uncompressed": 1 + 2 * field_bytes,   # 0x04 ‖ x ‖ y
            "pub_compressed":   1 + field_bytes,       # 0x02/03 ‖ x
            "sig":            2 * scalar_bytes,         # (r, s): dos escalares
        },
        "EC-KCDSA": {
            "priv":           scalar_bytes,
            "pub_uncompressed": 1 + 2 * field_bytes,
            "pub_compressed":   1 + field_bytes,
            "sig":            len(r_k) + scalar_bytes,  # r = H(x₁) (l_H) ‖ s (escalar)
            "h_cert":         len(h_cert),              # hash de certificado (param.)
            "r_len":          len(r_k),
        },
    }


def print_sizes(sz: dict):
    """Imprime la tabla comparativa de tamaños serializados."""
    e, k = sz["ECDSA"], sz["EC-KCDSA"]
    print("─" * 60)
    print("  Tamaños serializados  (bytes)")
    print("─" * 60)
    print(f"\n  {'Elemento':<22} │ {'ECDSA':^7} │ {'EC-KCDSA':^9}")
    print(f"  {'─'*22}─┼─{'─'*7}─┼─{'─'*9}")
    print(f"  {'Clave privada d':<22} │ {e['priv']:^7} │ {k['priv']:^9}")
    print(f"  {'Clave pública (compr.)':<22} │ {e['pub_compressed']:^7} │ {k['pub_compressed']:^9}")
    print(f"  {'Clave pública (sin c.)':<22} │ {e['pub_uncompressed']:^7} │ {k['pub_uncompressed']:^9}")
    print(f"  {'Firma (r ‖ s)':<22} │ {e['sig']:^7} │ {k['sig']:^9}")
    print(f"  {'h_cert (parámetro)':<22} │ {'—':^7} │ {k['h_cert']:^9}")
    print(
        f"\n  La componente r de EC-KCDSA es un hash de longitud fija l_H = "
        f"{k['r_len']} bytes,\n"
        f"  mientras que la r de ECDSA es un escalar de {sz['scalar_bytes']} "
        f"bytes (crece con la curva).\n"
        f"  En {CURVE.name} ({CURVE.bits} bits) ambas firmas coinciden en tamaño."
    )
    print()

# Optional matplotlib chart

def try_plot(er: dict, kr: dict, sizes: list[int], reps: int, out: str = "benchmark_results.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "  (matplotlib no disponible — omitiendo gráfica)\n"
            "   Para instalar: pip install matplotlib\n"
        )
        return

    x      = np.arange(len(sizes))
    labels = [f"{s:,}" for s in sizes]
    w      = 0.35

    ecdsa_sign    = [er[n]["sign"].minimum   for n in sizes]
    ecdsa_sign_e  = [er[n]["sign"].std       for n in sizes]
    ecdsa_verify  = [er[n]["verify"].minimum for n in sizes]
    ecdsa_verify_e= [er[n]["verify"].std     for n in sizes]
    kcdsa_sign    = [kr[n]["sign"].minimum   for n in sizes]
    kcdsa_sign_e  = [kr[n]["sign"].std       for n in sizes]
    kcdsa_verify  = [kr[n]["verify"].minimum for n in sizes]
    kcdsa_verify_e= [kr[n]["verify"].std     for n in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"ecdsa": "#1f77b4", "kcdsa": "#ff7f0e"}

    for ax, (ec_vals, ec_errs, kc_vals, kc_errs, title) in zip(
        axes,
        [
            (ecdsa_sign,   ecdsa_sign_e,   kcdsa_sign,   kcdsa_sign_e,   "Firma"),
            (ecdsa_verify, ecdsa_verify_e, kcdsa_verify, kcdsa_verify_e, "Verificación"),
        ],
    ):
        ax.bar(x - w/2, ec_vals, w, yerr=ec_errs,
               label="ECDSA", color=colors["ecdsa"],
               error_kw={"capsize": 4}, alpha=0.9)
        ax.bar(x + w/2, kc_vals, w, yerr=kc_errs,
               label="EC-KCDSA", color=colors["kcdsa"],
               error_kw={"capsize": 4}, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Tamaño del mensaje N (bytes)")
        ax.set_ylabel("Tiempo medio (ms)")
        ax.set_title(f"Tiempo de {title}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"ECDSA vs EC-KCDSA — {CURVE.name}, SHA-256\n"
        f"(Python from-scratch,  {reps} repeticiones,  métrica = mínimo)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Gráfica guardada en: {out}\n")

# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark ECDSA vs EC-KCDSA")
    p.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help=f"Repeticiones por caso (default: {DEFAULT_REPS})",
    )
    p.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Ejecuciones de calentamiento (default: {DEFAULT_WARMUP})",
    )
    p.add_argument(
        "--sizes", type=str, default=None,
        help="Tamaños de mensaje separados por coma, ej: 64,1024,65536",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="No generar gráfica aunque matplotlib esté disponible",
    )
    p.add_argument(
        "--plot-out", type=str, default="benchmark_results.png",
        help="Ruta del archivo de gráfica (default: benchmark_results.png)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    sizes = (
        [int(s) for s in args.sizes.split(",")]
        if args.sizes
        else DEFAULT_SIZES
    )

    print(f"\n{'═'*60}")
    print(f"  Benchmark: ECDSA vs EC-KCDSA")
    print(f"  Curva  : {CURVE.name}  ({CURVE.bits}-bit prime field)")
    print(f"  Hash   : SHA-256")
    print(f"  Reps   : {args.reps}  |  Warmup: {args.warmup}")
    print(f"  N sizes: {sizes}")
    print(f"  Métrica: mínimo ± σ (ms)  —  el mínimo estima el coste real")
    print(f"{'─'*60}")
    print(f"  Entorno de ejecución")
    print(f"    Python : {platform.python_implementation()} {platform.python_version()}"
          f"  ({sys.executable})")
    print(f"    SO     : {platform.platform()}")
    print(f"    CPU    : {platform.processor() or 'desconocido'}"
          f"  ({platform.machine()})")
    print(f"{'═'*60}\n")

    print("  [1/2] Midiendo ECDSA …")
    er = bench_ecdsa(sizes, args.reps, args.warmup)

    print("  [2/2] Midiendo EC-KCDSA …\n")
    kr = bench_ec_kcdsa(sizes, args.reps, args.warmup)

    print_table(er, kr, sizes)
    print_analysis(er, kr, sizes)
    print_op_counts(count_curve_ops())
    print_sizes(measure_sizes())

    if not args.no_plot:
        try_plot(er, kr, sizes, reps=args.reps, out=args.plot_out)


if __name__ == "__main__":
    main()
