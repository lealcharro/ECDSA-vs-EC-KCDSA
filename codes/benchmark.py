#!/usr/bin/env python3
"""
Benchmark: ECDSA vs EC-KCDSA — rendimiento con mensajes de longitud N.

Métricas medidas
────────────────
  • keygen   — generación de par de claves
  • sign     — firma de un mensaje de N bytes
  • verify   — verificación de una firma

Para cada N se realizan REPS repeticiones tras WARMUP ejecuciones de
calentamiento (no contabilizadas).  Se reporta el MÍNIMO ± desviación
estándar: en benchmarking el ruido del sistema (planificador del SO, GC,
frecuencia de CPU) solo AÑADE tiempo, por lo que el mínimo es el estimador
más estable del coste real del algoritmo.

Hipótesis a observar
─────────────────────
  1. El tiempo de firma/verificación es esencialmente CONSTANTE respecto
     a N: las operaciones de curva elíptica dominan; el SHA-256 es
     despreciable incluso para N = 64 KB.
  2. EC-KCDSA sign prescinde de k⁻¹ mod n → debería ser ligeramente
     más rápido que ECDSA sign.
  3. EC-KCDSA verify prescinde de s⁻¹ mod n → idem para verify.

Uso:
    cd codes/
    python benchmark.py
    python benchmark.py --reps 30 --sizes 64,1024,65536
"""

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

from src.ec import SECP256K1
from src import ecdsa, ec_kcdsa

CURVE = SECP256K1

# Defaults
DEFAULT_SIZES = [64, 256, 1_024, 4_096, 16_384, 65_536]
DEFAULT_REPS  = 20
DEFAULT_WARMUP = 3


# ──────────────────────────────────────────────────────────────────────
# Low-level timing helper
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Per-algorithm benchmark runners
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Optional matplotlib chart
# ──────────────────────────────────────────────────────────────────────

def try_plot(er: dict, kr: dict, sizes: list[int], out: str = "benchmark_results.png"):
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
        f"(Python from-scratch,  {DEFAULT_REPS} repeticiones)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Gráfica guardada en: {out}\n")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

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

    if not args.no_plot:
        try_plot(er, kr, sizes, out=args.plot_out)


if __name__ == "__main__":
    main()
