"""
Benchmark suite registry for CEC-style experiments.

This module exposes a unified interface for selecting benchmark suites
without changing the optimisation pipeline.
"""

import numpy as np

from benchmark_functions import BOUNDS as CEC2014_BOUNDS
from benchmark_functions import FUNCTIONS as CEC2014_FUNCTIONS
from benchmark_functions import FUNC_NAMES as CEC2014_FUNC_NAMES

AVAILABLE_SUITES = ("cec2014", "cec2017", "cec2020", "cec2022")


def _make_shifted_wrapper(base_func, seed, bias):
    """Create a deterministic transformed function from a base benchmark."""

    def wrapped(x):
        x = np.asarray(x, dtype=float)
        dim = len(x)
        rng = np.random.default_rng(seed + dim * 1000)
        shift = rng.uniform(-20.0, 20.0, dim)
        return float(base_func(x - shift) + bias)

    return wrapped


def _build_transformed_suite(tag, num_funcs, seed_base, bias_step):
    functions = []
    names = []

    for idx in range(num_funcs):
        base_func = CEC2014_FUNCTIONS[idx % len(CEC2014_FUNCTIONS)]
        wrapped = _make_shifted_wrapper(
            base_func=base_func,
            seed=seed_base + idx + 1,
            bias=bias_step * (idx + 1),
        )
        functions.append(wrapped)
        names.append(f"F{idx+1}")

    return {
        "tag": tag,
        "functions": functions,
        "func_names": names,
        "bounds": CEC2014_BOUNDS,
    }


def get_suite(name):
    """
    Return a suite dict with keys: tag, functions, func_names, bounds.

    Notes:
    - cec2014 uses the project's original 30 CEC-style functions.
    - cec2017/2020/2022 provide additional CEC-style suites using
      deterministic transformed variants for reproducible comparisons.
    """
    suite = name.lower().strip()
    if suite not in AVAILABLE_SUITES:
        raise ValueError(f"Unknown suite '{name}'. Available: {AVAILABLE_SUITES}")

    if suite == "cec2014":
        return {
            "tag": "cec2014",
            "functions": CEC2014_FUNCTIONS,
            "func_names": CEC2014_FUNC_NAMES,
            "bounds": CEC2014_BOUNDS,
        }

    if suite == "cec2017":
        return _build_transformed_suite("cec2017", num_funcs=30, seed_base=1700, bias_step=100)

    if suite == "cec2020":
        return _build_transformed_suite("cec2020", num_funcs=10, seed_base=2000, bias_step=200)

    return _build_transformed_suite("cec2022", num_funcs=12, seed_base=2200, bias_step=220)
