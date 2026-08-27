"""Rewrite the two equity columns of example_returns.csv as calibrated synthetics.

Dates, the risk-free column, and the four Fama/French style factors stay exactly
as downloaded. portfolio_return and benchmark_return are teaching series: they
are simulated so every headline figure the dashboard reports for them is hit
exactly, not approximately:

    benchmark volatility        15.00% annualized
    benchmark CAGR              10.00%
    portfolio CAPM beta         1.200
    portfolio Jensen's alpha    +2.00% annualized
    portfolio volatility        20.00% annualized
    portfolio style loadings    the four constants in STYLE_TILTS

Three properties of least squares make that possible. Adding a constant to a
series moves its mean and leaves its standard deviation alone, so the drift and
volatility calibrations do not interfere. Noise made exactly sample-orthogonal
to a design matrix contributes nothing to any coefficient fitted on it. And a
regressor whose sample covariance with the market is zero cannot pull the
market's slope, which is why the tilt vector is nudged onto that constraint.

Run with the dashboard's own interpreter:

    QuantDashboard/.venv/Scripts/python.exe ExampleCSV/synthesize_returns.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CSV_PATH = Path(__file__).with_name("example_returns.csv")

# Fixed so the synthetic equity columns can be rebuilt byte for byte.
SEED = 20260826

# The dashboard's own annualization factor. Anything derived here has to use the
# same one or the targets will not read back as stated.
TRADING_DAYS_PER_YEAR = 252

BENCHMARK_VOLATILITY = 0.15
# Compounded, i.e. the CAGR the dashboard prints, not the arithmetic drift.
BENCHMARK_CAGR = 0.10

PORTFOLIO_BETA = 1.2
PORTFOLIO_VOLATILITY = 0.20
PORTFOLIO_ALPHA = 0.02

# GARCH(1,1) on the benchmark. Persistence of 0.95 puts the half-life of a
# shock near 13 trading days: enough clustering to give the drawdown and tail
# panels something to show, short of the near-unit-root persistence fitted to
# real equity indices.
GARCH_ARCH = 0.05
GARCH_LAG = 0.90
GARCH_BURN_IN = 1000

# A growth and quality tilt: small positive size, negative value, positive
# profitability.
STYLE_TILTS = {"smb": 0.25, "hml": -0.15, "rmw": 0.10, "cma": -0.05}

# Matches the precision the columns are already written at.
DECIMALS = 8

PORTFOLIO_FIELD = 1
BENCHMARK_FIELD = 2
RF_FIELD = 3
# smb, hml, rmw, cma, in the order STYLE_TILTS names them.
FACTOR_FIELDS = (4, 5, 6, 7)


def read_csv_lines() -> tuple[list[str], list[str]]:
    """The file split into lines that keep their own endings.

    Rewriting two fields per line as text, rather than round-tripping the whole
    table through a parser, is what guarantees the columns this script must not
    touch come out identical: they are never converted to a float and back.
    """
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)

    header, rows = lines[0], [line for line in lines[1:] if line.strip()]
    return [header], rows


def column(rows: list[str], index: int) -> np.ndarray:
    return np.array([float(row.split(",")[index]) for row in rows])


def simulate_garch(count: int, daily_variance: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Shocks from a GARCH(1,1) whose unconditional variance is as given.

    Returns the shocks and the conditional volatility path that produced them.
    The path is handed back so the portfolio's idiosyncratic noise can ride the
    same regimes: a turbulent month should be turbulent for both series.
    """
    omega = daily_variance * (1.0 - GARCH_ARCH - GARCH_LAG)
    total = GARCH_BURN_IN + count

    innovations = rng.standard_normal(total)
    variance = np.empty(total)
    shocks = np.empty(total)

    variance[0] = daily_variance
    shocks[0] = np.sqrt(variance[0]) * innovations[0]
    for t in range(1, total):
        variance[t] = omega + GARCH_ARCH * shocks[t - 1] ** 2 + GARCH_LAG * variance[t - 1]
        shocks[t] = np.sqrt(variance[t]) * innovations[t]

    # Discard the burn-in so the sample opens in the stationary distribution
    # rather than at the unconditional variance it was seeded with.
    return shocks[GARCH_BURN_IN:], np.sqrt(variance[GARCH_BURN_IN:])


def volatility_multiplier(values: np.ndarray,
                          annualized_volatility: float) -> float:
    """The single factor that puts a series at an exact annualized volatility.

    One multiplier for the whole path, so the relative size of calm and
    turbulent stretches survives. ddof=1 matches ``series_performance``.
    """
    target = annualized_volatility / np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(target / values.std(ddof=1))


def solve_drift(shocks: np.ndarray, cagr: float) -> float:
    """The constant that makes simple returns compound to an exact CAGR.

    Total growth rises monotonically with the constant, so bisection converges
    on the one root. Solved in logs because the product of 2,472 gross returns
    is better behaved as a sum.
    """
    target = (len(shocks) / TRADING_DAYS_PER_YEAR) * np.log1p(cagr)

    def excess_growth(drift: float) -> float:
        return float(np.sum(np.log1p(drift + shocks)) - target)

    low, high = -0.01, 0.01
    if excess_growth(low) > 0.0 or excess_growth(high) < 0.0:
        raise RuntimeError("drift root is outside the bracket")

    for _ in range(200):
        middle = 0.5 * (low + high)
        if excess_growth(middle) < 0.0:
            low = middle
        else:
            high = middle

    return 0.5 * (low + high)


def orthogonalize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """The part of a series no column of the design matrix can explain.

    With an intercept in the design the result also has a sample mean of
    exactly zero, so it moves neither a slope nor an intercept fitted on those
    columns. That is what leaves the planted coefficients recoverable.
    """
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def project_out_market(tilts: np.ndarray, factors: np.ndarray,
                       market: np.ndarray) -> np.ndarray:
    """Nudge the tilt vector until the tilted return cannot bias market beta.

    The five-factor fit holds the factors fixed and so returns the planted
    market slope whatever the tilts are, but the single-regressor CAPM of the
    scatter panel does not: it absorbs cov(tilt, market)/var(market) into beta.
    The style columns are real data and the market is simulated, so their sample
    covariance is small noise rather than zero. Removing the component of the
    tilt vector that lies along that covariance sets the bias to zero exactly.

    What it costs is the round numbers: the constants above are a direction to
    tilt in, not the loadings that come back out. The removed component is a
    projection of the whole tilt vector, so however small the covariances are,
    the correction is a fixed fraction of the tilt's own size -- second decimal,
    in practice. The signs and rough magnitudes are what survive.
    """
    centered = market - market.mean()
    covariances = (factors - factors.mean(axis=0)).T @ centered / (len(market) - 1)
    return tilts - covariances * (tilts @ covariances) / (covariances @ covariances)


def build_benchmark(rows: list[str], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    daily_variance = (BENCHMARK_VOLATILITY / np.sqrt(TRADING_DAYS_PER_YEAR)) ** 2
    shocks, conditional_volatility = simulate_garch(len(rows), daily_variance, rng)

    # The simulated path lands near its target volatility but not on it, so both
    # it and the conditional volatilities behind it take the same correction.
    multiplier = volatility_multiplier(shocks, BENCHMARK_VOLATILITY)
    shocks = shocks * multiplier

    return (solve_drift(shocks, BENCHMARK_CAGR) + shocks,
            conditional_volatility * multiplier)


def build_portfolio(benchmark: np.ndarray, rf: np.ndarray, factors: np.ndarray,
                    conditional_volatility: np.ndarray,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The portfolio column, and the style loadings it will regress out at."""
    market = benchmark - rf
    tilts = project_out_market(
        np.array([STYLE_TILTS[name] for name in ("smb", "hml", "rmw", "cma")]),
        factors, market)
    tilt = factors @ tilts

    design = np.column_stack([np.ones(len(market)), market, factors, rf])
    residual = orthogonalize(
        rng.standard_normal(len(market)) * conditional_volatility, design)

    # Jensen's alpha is the intercept of the CAPM, which the style premia feed
    # into: a tilted portfolio earns them on top of its market exposure. Netting
    # off the tilt's own mean is what leaves the reported figure at the target.
    alpha = PORTFOLIO_ALPHA / TRADING_DAYS_PER_YEAR - tilt.mean()
    explained = rf + alpha + PORTFOLIO_BETA * market + tilt

    # The residual is orthogonal to every column of the design, and so to any
    # combination of them, which makes the variances add and the scale a matter
    # of one square root rather than a search.
    target_variance = (PORTFOLIO_VOLATILITY / np.sqrt(TRADING_DAYS_PER_YEAR)) ** 2
    shortfall = target_variance - explained.var(ddof=1)
    if shortfall <= 0.0:
        raise RuntimeError(
            f"beta {PORTFOLIO_BETA} on a {BENCHMARK_VOLATILITY:.0%} benchmark already "
            f"exceeds the {PORTFOLIO_VOLATILITY:.0%} volatility budget")

    return explained + residual * np.sqrt(shortfall / residual.var(ddof=1)), tilts


def rewrite(rows: list[str], portfolio: np.ndarray,
            benchmark: np.ndarray) -> list[str]:
    """Swap two fields per line and pass every other character through."""
    rewritten = []
    for row, portfolio_value, benchmark_value in zip(rows, portfolio, benchmark):
        body = row.rstrip("\r\n")
        ending = row[len(body):]

        fields = body.split(",")
        fields[PORTFOLIO_FIELD] = f"{portfolio_value:.{DECIMALS}f}"
        fields[BENCHMARK_FIELD] = f"{benchmark_value:.{DECIMALS}f}"
        rewritten.append(",".join(fields) + ending)

    return rewritten


def main() -> None:
    header, rows = read_csv_lines()
    rng = np.random.default_rng(SEED)

    rf = column(rows, RF_FIELD)
    factors = np.column_stack([column(rows, field) for field in FACTOR_FIELDS])

    benchmark, conditional_volatility = build_benchmark(rows, rng)
    portfolio, tilts = build_portfolio(
        benchmark, rf, factors, conditional_volatility, rng)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(header + rewrite(rows, portfolio, benchmark))

    print(f"rewrote {len(rows)} rows of {CSV_PATH.name}")
    print("realized style loadings: " + ", ".join(
        f"{name} {value:+.6f}" for name, value in zip(STYLE_TILTS, tilts)))


if __name__ == "__main__":
    main()
