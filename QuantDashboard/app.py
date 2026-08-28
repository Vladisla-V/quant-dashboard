"""
The risk-free rate and the Fama-French style factors are hand-supplied in the
upload rather than downloaded, so a portfolio from any market can be analyzed
against its own baseline. The market factor is not uploaded: it is derived as
the benchmark return net of the risk-free column.

Parsing, math and chart building all live in this module.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from flask import Flask, render_template, request

app = Flask(__name__)
# A daily return series is tiny; the cap exists to block abusive uploads.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

DATE_COLUMN = "date"
PORTFOLIO_COLUMN = "portfolio_return"
BENCHMARK_COLUMN = "benchmark_return"
RF_COLUMN = "rf"

# Fama-French style factors, as uploaded. The fifth regressor of the 5-factor
# model, the market factor, is derived as benchmark_return - rf instead.
FACTOR_COLUMNS = ("smb", "hml", "rmw", "cma")

# Display names of the five regressors, in the order they are charted.
FF5_FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")

# The 3-factor model drops the profitability and investment styles.
FF3_FACTORS = FF5_FACTORS[:3]

# What each style factor stands for, spelled out in its own chart title.
STYLE_FACTOR_THEMES = {
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
}

# Hover text for the factor-table rows, so a loading reads without the README.
FACTOR_DESCRIPTIONS = {
    "Mkt-RF": "Market excess return: the benchmark net of the risk-free column.",
    "SMB": "Small Minus Big: small-cap minus large-cap returns (size).",
    "HML": "High Minus Low: value minus growth returns (value).",
    "RMW": "Robust Minus Weak: high minus low profitability firms (profitability).",
    "CMA": "Conservative Minus Aggressive: low minus high investment firms (investment).",
    "Alpha": "Intercept: the excess return the regressors leave unexplained.",
    "Beta (Market)": "Market exposure: how far the portfolio moves with the benchmark.",
}

# The conventional bar a loading has to clear to be called significant.
SIGNIFICANCE_LEVEL = 0.05

REQUIRED_COLUMNS = (DATE_COLUMN, PORTFOLIO_COLUMN, BENCHMARK_COLUMN, RF_COLUMN,
                    *FACTOR_COLUMNS)
# The two series that get compounded into equity curves and charted side by side.
RETURN_COLUMNS = (PORTFOLIO_COLUMN, BENCHMARK_COLUMN)
NUMERIC_COLUMNS = (*RETURN_COLUMNS, RF_COLUMN, *FACTOR_COLUMNS)
# Without these three a row cannot be compounded or measured against a baseline,
# so it leaves the sample entirely. A blank style factor only costs it the
# factor fits, which is why the two groups are kept apart.
CORE_COLUMNS = (*RETURN_COLUMNS, RF_COLUMN)

# A quarter of trading days: enough to fit the five-factor model and still leave
# the shortest rolling window something to say.
MIN_ROWS = 63

# Roughly a century of daily returns. The byte cap above admits far more than
# that, and every chart and rolling fit on the page grows with the row count.
MAX_ROWS = 25200

# Dates are required in one unambiguous format, so 04/01 cannot quietly mean
# April 1st in one upload and January 4th in the next.
DATE_FORMAT = "%Y-%m-%d"
DATE_FORMAT_LABEL = "YYYY-MM-DD"

# A parsed date outside this window is a misread number rather than an
# observation; the upper bound is resolved per upload so it tracks today.
EARLIEST_DATE = pd.Timestamp("1900-01-01")

# Text that is entirely numeric, which is how spreadsheets store a date once the
# cell loses its formatting.
_NUMERIC_TEXT = r"-?\d+(?:\.\d+)?"

# A number written with a comma for its decimal point, as European Excel exports
# it. Only reachable in a comma-delimited file when the field is quoted.
_DECIMAL_COMMA_TEXT = r"-?\d+,\d+"

# Separators mistaken for commas often enough to be worth naming in the error.
_ALTERNATE_DELIMITERS = {";": "semicolons", "\t": "tabs"}

# Standard trading-day count used to annualize daily statistics.
TRADING_DAYS_PER_YEAR = 252

# Left-tail probability behind the reported VaR and CVaR figures.
TAIL_PROBABILITY = 0.05

# Bin count for the overlaid return histograms.
DISTRIBUTION_BINS = 48

# Trailing windows for the rolling ratio charts: roughly a month, a quarter and
# a year of trading days.
ROLLING_WINDOWS = (21, 63, 252)
ROLLING_WINDOW_LABELS = {21: "month", 63: "quarter", 252: "year"}

# A daily move beyond +/-150% is far more likely percent-scaled input.
PERCENT_SCALE_THRESHOLD = 1.5

_PORTFOLIO_COLOR = "#e30b5d"
# Mkt-RF below shares this color: it is the benchmark's excess return.
_BENCHMARK_COLOR = "#009193"
# One color per factor so the five beta lines stay distinguishable when they cross.
_FACTOR_COLORS = {
    "Mkt-RF": "#009193",
    "SMB": "#1108b1",
    "HML": "#488a2e",
    "RMW": "#ffbf00",
    "CMA": "#5f08af",
}
# Every scatter already colors its points by the series being regressed on, so
# the fitted lines share one neutral color instead of restating that hue.
_FIT_LINE_COLOR = "#ffd894"
_TEXT = "#e8a54b"
_TITLE = "#ffd894"
_MUTED = "#ad7d40"
_GRID = "rgba(232,165,75,0.12)"
_AXIS_ZERO = "rgba(232,165,75,0.30)"
_TRANSPARENT = "rgba(0,0,0,0)"
_FONT = "Cascadia Code, Cascadia Mono, Consolas, ui-monospace, monospace"
_HOVER_LABEL = dict(bgcolor="#15110d", bordercolor="rgba(232,165,75,0.35)",
                    font=dict(family=_FONT, color=_TEXT, size=11))
_CHART_CONFIG = {"displayModeBar": False, "responsive": True}


class CsvFormatError(ValueError):
    """Raised when an upload cannot be read as the expected returns CSV."""


def _read_csv(raw: bytes, **options) -> pd.DataFrame:
    """Read the upload with nothing inferred from the bytes themselves.

    ``compression=None`` is the load-bearing argument: pandas will unpack an
    archive whenever it can infer one, which would turn a small upload into an
    unbounded one. ``utf-8-sig`` costs nothing and strips the byte-order mark
    that Excel writes ahead of the first column name.
    """
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig",
                           compression=None, **options)
    except UnicodeDecodeError as exc:
        raise CsvFormatError(
            "The file is not UTF-8 text, so it cannot be read as a CSV. In Excel, "
            "save it as 'CSV UTF-8' and try again."
        ) from exc
    except Exception as exc:
        raise CsvFormatError(f"The file could not be read as CSV ({exc}).") from exc


def _delimiter_name(header: str) -> str | None:
    """Name the separator of a header row that arrived as a single column.

    Every required name landing in one field means the file is delimited by
    something other than a comma, and saying which one is far more useful than
    reporting all eight columns as missing.
    """
    for delimiter, name in _ALTERNATE_DELIMITERS.items():
        if header.count(delimiter) >= len(REQUIRED_COLUMNS) - 1:
            return name
    return None


def _column_names(raw: bytes) -> list[str]:
    """The upload's column names, with case and padding normalized away."""
    header = _read_csv(raw, nrows=0)
    return [str(column).strip().lower() for column in header.columns]


def _check_schema(names: list[str]) -> None:
    """Reject a header that repeats a name or does not carry all eight columns."""
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise CsvFormatError(
            f"These column names appear more than once: {', '.join(duplicated)}. "
            "Capitalisation and surrounding spaces are ignored, so 'Date' and "
            "'date' count as the same column. Each one must appear exactly once."
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in names]
    if not missing:
        return

    if len(names) == 1:
        delimiter = _delimiter_name(names[0])
        if delimiter is not None:
            raise CsvFormatError(
                f"The columns in this file are separated by {delimiter} rather "
                "than commas, so it reads as a single column. Re-save it as a "
                "comma-separated CSV and try again."
            )

    raise CsvFormatError(
        f"Missing required column(s): {', '.join(missing)}. "
        f"Expected exactly: {', '.join(REQUIRED_COLUMNS)}."
    )


def _check_raw_text(frame: pd.DataFrame) -> None:
    """Catch spreadsheet export habits before the values are coerced to numbers.

    Both mistakes below survive coercion as silence rather than an error: a
    serial date parses as a timestamp microseconds into 1970, and a comma
    decimal becomes NaN and takes its whole row out of the sample.
    """
    dates = frame[DATE_COLUMN].str.strip().dropna()
    if len(dates) and dates.str.fullmatch(_NUMERIC_TEXT).all():
        raise CsvFormatError(
            "The date column holds plain numbers, which is how a spreadsheet "
            "stores dates once the cell loses its date formatting. Please export "
            f"the column as text in {DATE_FORMAT_LABEL} form, such as 2023-01-03."
        )

    for column in NUMERIC_COLUMNS:
        values = frame[column].str.strip().dropna()
        if len(values) and values.str.fullmatch(_DECIMAL_COMMA_TEXT).any():
            raise CsvFormatError(
                f"Column '{column}' writes its decimals with a comma, such as "
                "0,01. Please re-save the file using a period instead (0.01)."
            )


def load_returns(raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Parse raw CSV bytes into a date-indexed frame of returns plus advisories.

    The frame keeps every row that carries a usable date and a complete set of
    core returns, so a day with a blank style factor still compounds into the
    equity curves; the factor fits narrow the sample themselves through
    :func:`factor_ready`. The second value is the list of notices the dashboard
    shows above the charts, which is where anything dropped is accounted for.
    """
    if not raw.strip():
        raise CsvFormatError("The uploaded file is empty.")

    _check_schema(_column_names(raw))

    # Reading one row past the cap is what distinguishes a file at the limit
    # from one over it, and the columns are narrowed here so a file padded with
    # thousands of unused ones is never materialized.
    frame = _read_csv(
        raw,
        usecols=lambda column: str(column).strip().lower() in REQUIRED_COLUMNS,
        nrows=MAX_ROWS + 1,
        dtype=str,
    )
    if len(frame) > MAX_ROWS:
        raise CsvFormatError(
            f"This file holds more than {MAX_ROWS:,} rows, about a century of "
            "daily returns, which is more than the dashboard will chart. Please "
            "shorten the sample and upload it again."
        )

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.loc[:, list(REQUIRED_COLUMNS)]
    _check_raw_text(frame)

    dates = frame[DATE_COLUMN].str.strip()
    blank_dates = int(dates.isna().sum())
    frame[DATE_COLUMN] = pd.to_datetime(dates, format=DATE_FORMAT, errors="coerce")
    misformatted = int(frame[DATE_COLUMN].isna().sum()) - blank_dates

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # Infinity survives every later check -- it is neither missing nor at or
        # below -1 -- and turns the compounded curves into a page of N/A.
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise CsvFormatError(
                f"Column '{column}' contains an infinite value. Every return has "
                "to be a finite decimal."
            )

    supplied_rows = len(frame)
    frame = frame.dropna(subset=[DATE_COLUMN, *CORE_COLUMNS])
    dropped_rows = supplied_rows - len(frame)

    if len(frame):
        newest_allowed = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        oldest, newest = frame[DATE_COLUMN].min(), frame[DATE_COLUMN].max()
        if oldest < EARLIEST_DATE or newest > newest_allowed:
            raise CsvFormatError(
                f"The dates run from {oldest.date()} to {newest.date()}, outside "
                f"the {EARLIEST_DATE.date()} to {newest_allowed.date()} range the "
                "dashboard accepts. Please check the date column."
            )

    frame = frame.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)

    repeated = frame.index.duplicated()
    if repeated.any():
        examples = [str(date.date()) for date in frame.index[repeated].unique()[:3]]
        raise CsvFormatError(
            f"{int(repeated.sum())} row(s) repeat a date that already appears in "
            f"the file, such as {', '.join(examples)}. Each date must appear once, "
            "so please combine or remove the duplicates."
        )

    if len(frame) < MIN_ROWS:
        detail = ""
        if misformatted:
            detail = (f" {misformatted} row(s) carried a date that is not in "
                      f"{DATE_FORMAT_LABEL} form.")
        raise CsvFormatError(
            f"Need at least {MIN_ROWS} rows -- about a quarter of trading days -- "
            "with a valid date and a portfolio, benchmark and risk-free return on "
            f"each, but only {len(frame)} usable row(s) were found.{detail}"
        )

    # Only the two compounded series are bounded below by -1. The risk-free
    # column and the style factors are regressors and spreads, never wealth
    # paths, so a large negative value there is data rather than an error.
    for column in RETURN_COLUMNS:
        if (frame[column] <= -1).any():
            raise CsvFormatError(
                f"Column '{column}' contains values of -1 or lower. Returns must be "
                "simple decimals above -1 (0.01 = 1%); a value such as -2.5 suggests "
                "the file holds percentages rather than decimals."
            )

    notices = []
    if dropped_rows:
        notices.append(
            f"Dropped {dropped_rows} row(s) that were missing a portfolio, "
            f"benchmark or risk-free return, or a date in {DATE_FORMAT_LABEL} form."
        )
    incomplete = int(frame[list(FACTOR_COLUMNS)].isna().any(axis=1).sum())
    if incomplete:
        notices.append(
            f"{incomplete} row(s) are missing at least one style factor. They still "
            "count towards the equity curves, the risk metrics and the CAPM, but "
            "the Fama-French regressions and style scatters leave them out."
        )

    return frame, notices


def factor_ready(returns: pd.DataFrame) -> pd.DataFrame:
    """The rows carrying all four style factors, the only ones a factor fit can use."""
    return returns.dropna(subset=list(FACTOR_COLUMNS))


def percent_scale_warning(returns: pd.DataFrame) -> str | None:
    """Flag the common mistake of uploading percents (1.5) instead of decimals (0.015)."""
    largest = float(np.nanmax(np.abs(returns.to_numpy(dtype=float))))
    if largest <= PERCENT_SCALE_THRESHOLD:
        return None
    return (
        f"The largest absolute daily return is {largest:.2f}. Returns are expected as "
        "decimals (0.01 = 1%), so these values look percent-scaled and the curves "
        "below may be unrealistic."
    )


def equity_curves(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound simple returns into a growth-of-$1 wealth index for each series."""
    return pd.DataFrame(
        {
            "Portfolio": (1.0 + returns[PORTFOLIO_COLUMN]).cumprod(),
            "Benchmark": (1.0 + returns[BENCHMARK_COLUMN]).cumprod(),
        }
    )


def excess_returns(returns: pd.DataFrame, column: str) -> pd.Series:
    """One return series net of the risk-free column supplied in the CSV.

    Whatever baseline the file carries -- local T-bills, cash, an inflation
    series -- is what Sharpe, Sortino and the CAPM fit measure returns against.
    """
    return returns[column] - returns[RF_COLUMN]


def _reportable(value: float | None) -> bool:
    """Whether a computed figure is a real number worth printing.

    A degenerate sample leaves NaN or infinity behind in almost any statistic,
    and every formatter below routes through this so none of them can put the
    word ``nan`` on the page.
    """
    return value is not None and bool(np.isfinite(value))


def _pct(value: float, digits: int = 2) -> str:
    """Format a decimal as a percent string; anything unmeasurable becomes N/A."""
    if not _reportable(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _signed_pct(value: float, digits: int = 2) -> str:
    """Format a decimal as a percent string that always carries its sign."""
    if not _reportable(value):
        return "N/A"
    return f"{value * 100:+.{digits}f}%"


def _num(value: float, digits: int = 2) -> str:
    """Format a plain number; anything unmeasurable becomes N/A."""
    if not _reportable(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _signed(value: float, digits: int = 4) -> str:
    """Format a plain number that always carries its sign, such as a loading."""
    if not _reportable(value):
        return "N/A"
    return f"{value:+.{digits}f}"


def _money(value: float, digits: int = 2) -> str:
    """Format a dollar figure; a curve that overflowed becomes N/A."""
    if not _reportable(value):
        return "N/A"
    return f"${value:,.{digits}f}"


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Percentage drawdown from the running peak (the underwater curve)."""
    return equity / equity.cummax() - 1.0


def longest_underwater_days(equity: pd.Series) -> int:
    """Longest contiguous stretch of observations spent below the prior peak.

    An open drawdown at the end of the sample counts through the last date.
    """
    underwater = drawdown_series(equity) < 0
    longest = 0
    current = 0
    for is_under in underwater.to_numpy():
        if is_under:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def time_underwater_series(equity: pd.Series) -> pd.Series:
    """Observations elapsed since the last equity peak, zero on a peak day.

    Counted in observations rather than calendar days, the same convention as
    the longest-underwater statistic in the metrics table.
    """
    underwater = (drawdown_series(equity) < 0).to_numpy()
    days = np.zeros(len(underwater), dtype=int)
    current = 0
    for i, is_under in enumerate(underwater):
        current = current + 1 if is_under else 0
        days[i] = current
    return pd.Series(days, index=equity.index)


def rolling_sharpe(excess: pd.Series, window: int) -> pd.Series:
    """Annualized Sharpe of an excess-return series, NaN until the window fills."""
    mean = excess.rolling(window, min_periods=window).mean()
    sd = excess.rolling(window, min_periods=window).std(ddof=1)
    return (mean / sd.replace(0.0, np.nan)) * np.sqrt(TRADING_DAYS_PER_YEAR)


def _downside_deviation(window_returns: np.ndarray) -> float:
    """Standard deviation of the losing days inside one rolling window."""
    downside = window_returns[window_returns < 0]
    # One losing day gives no sample spread, so the ratio is undefined.
    if len(downside) < 2:
        return float("nan")
    sd = float(downside.std(ddof=1))
    return sd if sd > 0 else float("nan")


def rolling_sortino(excess: pd.Series, window: int) -> pd.Series:
    """Annualized Sortino of an excess-return series, NaN until the window fills.

    Matches the table statistic: the denominator uses only the days that fell
    short of the risk-free rate, so a stretch without enough of them leaves a gap.
    """
    mean = excess.rolling(window, min_periods=window).mean()
    downside = excess.rolling(window, min_periods=window).apply(
        _downside_deviation, raw=True)
    return (mean / downside) * np.sqrt(TRADING_DAYS_PER_YEAR)


def series_performance(returns: pd.Series, rf: pd.Series) -> dict[str, float]:
    """Headline risk/return stats for one daily simple-return series.

    Growth, volatility and drawdowns describe the total returns an investor
    actually earned; Sharpe and Sortino measure the returns left over once the
    risk-free series from the CSV is subtracted.
    """
    n = len(returns)
    total_growth = float((1.0 + returns).prod())
    if n == 0:
        cagr = float("nan")
    elif total_growth <= 0:
        cagr = -1.0
    else:
        cagr = total_growth ** (TRADING_DAYS_PER_YEAR / n) - 1.0

    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    excess = returns - rf
    sd = float(excess.std(ddof=1))
    if sd == 0.0 or np.isnan(sd):
        sharpe = float("nan")
    else:
        sharpe = float(excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))

    downside = excess[excess < 0]
    dd = float(downside.std(ddof=1)) if len(downside) else float("nan")
    if dd == 0.0 or np.isnan(dd):
        sortino = float("nan")
    else:
        sortino = float(excess.mean() / dd * np.sqrt(TRADING_DAYS_PER_YEAR))

    equity = (1.0 + returns).cumprod()
    return {
        "cagr": float(cagr),
        "volatility": vol,
        "max_drawdown": float(drawdown_series(equity).min()),
        "longest_underwater_days": float(longest_underwater_days(equity)),
        "sharpe": sharpe,
        "sortino": sortino,
    }


def format_performance(stats: dict[str, float], correlation: float) -> dict[str, str]:
    """Turn raw performance floats into template-ready display strings."""
    return {
        "cagr": _pct(stats["cagr"]),
        "volatility": _pct(stats["volatility"]),
        "max_drawdown": _pct(stats["max_drawdown"]),
        "longest_underwater_days": str(int(stats["longest_underwater_days"])),
        "sharpe": _num(stats["sharpe"]),
        "sortino": _num(stats["sortino"]),
        "correlation": _num(correlation, 3),
    }


def distribution_stats(returns: pd.Series) -> dict[str, float]:
    """Left-tail and shape statistics for one daily simple-return series.

    VaR is the empirical 5th percentile and CVaR the mean of the days at or
    below it, so both are returns on the same scale as the input (a loss is
    negative). Kurtosis is Fisher's excess version, zero for a normal sample.
    """
    var = float(returns.quantile(TAIL_PROBABILITY))
    tail = returns[returns <= var]
    return {
        "var": var,
        "cvar": float(tail.mean()) if len(tail) else float("nan"),
        "skewness": float(returns.skew()),
        "excess_kurtosis": float(returns.kurtosis()),
    }


def benchmark_correlation(returns: pd.DataFrame) -> float:
    """Pearson correlation of the two total daily return series."""
    return float(np.corrcoef(returns[BENCHMARK_COLUMN].to_numpy(),
                             returns[PORTFOLIO_COLUMN].to_numpy())[0, 1])


def capm_regression(returns: pd.DataFrame) -> dict[str, float]:
    """Fit the CAPM by ordinary least squares on excess returns.

        (r_p - rf) = alpha + beta * (r_b - rf) + epsilon

    Both sides are net of the risk-free column from the CSV, so alpha is
    Jensen's alpha: the daily return earned beyond what the portfolio's market
    exposure alone would have paid.
    """
    benchmark = excess_returns(returns, BENCHMARK_COLUMN).to_numpy()
    portfolio = excess_returns(returns, PORTFOLIO_COLUMN).to_numpy()

    if benchmark.var() == 0.0:
        raise CsvFormatError(
            "Every benchmark excess return in the file is the same value, so there "
            "is no market move to regress against. Beta, alpha and R² need a "
            "benchmark that varies relative to the risk-free rate."
        )

    beta, alpha = np.polyfit(benchmark, portfolio, 1)
    correlation = float(np.corrcoef(benchmark, portfolio)[0, 1])
    return {
        "alpha": float(alpha),
        "alpha_annualized": float(alpha) * TRADING_DAYS_PER_YEAR,
        "beta": float(beta),
        "r_squared": correlation ** 2,
    }


def factor_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """The five Fama-French regressors, one column per factor.

    Only the four style factors are uploaded. The market factor is the
    benchmark's own excess return -- the same regressor the CAPM fit uses -- so
    the 5-factor model is the CAPM with four style exposures added to it.
    """
    columns = {"Mkt-RF": excess_returns(returns, BENCHMARK_COLUMN)}
    columns.update(
        {name: returns[column] for name, column in zip(FF5_FACTORS[1:], FACTOR_COLUMNS)}
    )
    return pd.DataFrame(columns, index=returns.index)


def single_factor_fit(portfolio: pd.Series,
                      factor: pd.Series) -> tuple[float, float] | None:
    """Fit portfolio excess returns on one factor by itself.

        (r_p - rf) = alpha + beta * factor + epsilon

    The slope is a simple beta: it measures how the portfolio moved with that
    factor alone, without holding the other four fixed the way the five-factor
    loadings do. Returns None for a factor that never moves, since a flat
    regressor pins down no slope; the CAPM's regressor is rejected outright in
    that case, but one dead style column should not sink the whole upload.
    """
    values = factor.to_numpy()
    if values.var() == 0.0:
        return None

    beta, alpha = np.polyfit(values, portfolio.to_numpy(), 1)
    return float(alpha), float(beta)


def rolling_factor_betas(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Re-fit the five-factor model on every trailing window of the given length.

        (r_p - rf) = alpha + sum_i beta_i * factor_i + epsilon

    Each row holds the loadings estimated from the window ending on that date,
    so reading across the rows shows style drift: whether the portfolio's
    exposures held steady or wandered. Only the betas are kept; the intercept is
    the window's alpha and is not charted.

    Rows before the window fills stay NaN, as do windows whose regressors are
    collinear or constant, since those pin down no unique set of loadings.
    """
    design = np.column_stack(
        [np.ones(len(returns)), factor_matrix(returns).to_numpy()])
    target = excess_returns(returns, PORTFOLIO_COLUMN).to_numpy()

    betas = np.full((len(returns), len(FF5_FACTORS)), np.nan)
    for end in range(window, len(returns) + 1):
        rows = slice(end - window, end)
        window_design = design[rows]
        coefficients, _, rank, _ = np.linalg.lstsq(
            window_design, target[rows], rcond=None)
        if rank == window_design.shape[1]:
            # Drop the intercept: the remaining coefficients are the loadings.
            betas[end - 1] = coefficients[1:]

    return pd.DataFrame(betas, index=returns.index, columns=list(FF5_FACTORS))


def _has_unique_fit(design: pd.DataFrame) -> bool:
    """Whether a design matrix pins down exactly one coefficient per column.

    Constant regressors, collinear ones, or fewer observations than coefficients
    all leave the least-squares problem without a unique answer. Statsmodels
    solves it through a pseudo-inverse anyway and reports numbers that look like
    loadings, so the callers below ask this first and report nothing if it fails.
    """
    matrix = np.asarray(design, dtype=float)
    return (matrix.shape[0] > matrix.shape[1]
            and int(np.linalg.matrix_rank(matrix)) == matrix.shape[1])


def _fit_summary(model) -> dict:
    """Reduce a fitted OLS model to the numbers a factor table reports.

    The intercept is renamed Alpha and is the only coefficient annualized: it is
    a return per day, so 252 of them compound the model's unexplained edge into
    a yearly figure, whereas a loading is a ratio that no scaling applies to.
    """
    loadings = []
    for name in model.params.index:
        coefficient = float(model.params[name])
        loadings.append({
            "name": "Alpha" if name == "const" else name,
            "coef": coefficient,
            "annualized": (coefficient * TRADING_DAYS_PER_YEAR
                           if name == "const" else None),
            "tstat": float(model.tvalues[name]),
            "pvalue": float(model.pvalues[name]),
        })

    return {"loadings": loadings, "r_squared": float(model.rsquared)}


def factor_regression(returns: pd.DataFrame,
                      factors: tuple[str, ...]) -> dict | None:
    """Fit the named factors on portfolio excess returns over the whole sample.

        (r_p - rf) = alpha + sum_i beta_i * factor_i + epsilon

    Every loading is a partial beta -- what that factor explains once the others
    are held fixed -- so it will not match the simple beta of the scatter fitted
    on the same factor alone. Alongside each coefficient the fit reports the
    t-statistic and p-value that say whether it is distinguishable from zero,
    and R-squared reports how much of the portfolio's excess-return variance the
    whole set accounts for.

    Returns None when these regressors cannot be told apart on this sample, so
    the dashboard can drop the table and say why instead of printing loadings
    the data does not support.
    """
    design = sm.add_constant(factor_matrix(returns)[list(factors)],
                             has_constant="add")
    if not _has_unique_fit(design):
        return None

    model = sm.OLS(excess_returns(returns, PORTFOLIO_COLUMN), design).fit()
    return _fit_summary(model)


def capm_table_regression(returns: pd.DataFrame) -> dict | None:
    """The CAPM of :func:`capm_regression`, restated for the factor table.

    Same single-regressor model on the same days -- the market needs no style
    factor, so this is given the whole sample the stat cards and the scatter
    use, not the narrower one the Fama-French tables are fitted on -- but run
    through the shared OLS so the market loading arrives with the inference the
    scatter's fit does not carry. The factor is renamed Beta (Market) because on
    its own it is the CAPM beta rather than one loading among five. None has the
    same meaning as above.
    """
    market = excess_returns(returns, BENCHMARK_COLUMN).rename("Mkt-RF")
    design = sm.add_constant(market.to_frame(), has_constant="add")
    if not _has_unique_fit(design):
        return None

    model = sm.OLS(excess_returns(returns, PORTFOLIO_COLUMN), design).fit()

    summary = _fit_summary(model)
    for loading in summary["loadings"]:
        if loading["name"] == "Mkt-RF":
            loading["name"] = "Beta (Market)"
    return summary


def format_factor_table(regression: dict | None) -> list[dict]:
    """Turn one regression's loadings into display-ready table rows.

    A regression the sample could not support has no rows, which is what leaves
    its table out of the page.
    """
    if regression is None:
        return []

    rows = []
    for loading in regression["loadings"]:
        annualized = loading["annualized"]
        pvalue = loading["pvalue"]
        rows.append({
            "name": loading["name"],
            "coef": (_signed_pct(loading["coef"], 3)
                     if loading["name"] == "Alpha"
                     else _signed(loading["coef"])),
            "annualized": _pct(annualized) if annualized is not None else "-",
            "tstat": _num(loading["tstat"]),
            "pvalue": _num(pvalue, 3),
            "significant": _reportable(pvalue) and pvalue < SIGNIFICANCE_LEVEL,
            "description": FACTOR_DESCRIPTIONS.get(loading["name"], ""),
        })
    return rows


def build_equity_chart(curves: pd.DataFrame, title: str, div_id: str,
                       log_scale: bool = False) -> str:
    """Render the two equity curves as a self-contained Plotly HTML fragment.

    Plotly's JS is loaded once from a CDN in the template, so each fragment ships
    only its own data and layout.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curves.index, y=curves["Portfolio"], name="Portfolio", mode="lines",
        line=dict(color=_PORTFOLIO_COLOR, width=2.2),
        hovertemplate="Portfolio: $%{y:,.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=curves.index, y=curves["Benchmark"], name="Benchmark", mode="lines",
        line=dict(color=_BENCHMARK_COLOR, width=1.8),
        hovertemplate="Benchmark: $%{y:,.4f}<extra></extra>"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_TITLE), x=0.01, xanchor="left"),
        template="plotly_dark",
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font=dict(family=_FONT, color=_MUTED, size=11),
        hoverlabel=_HOVER_LABEL,
        margin=dict(l=58, r=24, t=48, b=44),
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor=_TRANSPARENT, font=dict(size=11)),
        xaxis=dict(gridcolor=_GRID, zeroline=False),
        yaxis=dict(gridcolor=_GRID, zeroline=False, tickprefix="$",
                   type="log" if log_scale else "linear"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config=_CHART_CONFIG, div_id=div_id)


def _time_series_layout(title: str, yaxis: dict) -> dict:
    """Shared dark-theme layout for the date-indexed portfolio/benchmark charts."""
    return dict(
        title=dict(text=title, font=dict(size=14, color=_TITLE), x=0.01, xanchor="left"),
        template="plotly_dark",
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font=dict(family=_FONT, color=_MUTED, size=11),
        hoverlabel=_HOVER_LABEL,
        margin=dict(l=64, r=24, t=48, b=44),
        height=340,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor=_TRANSPARENT, font=dict(size=11)),
        xaxis=dict(gridcolor=_GRID, zeroline=False),
        yaxis=yaxis,
    )


def build_drawdown_chart(curves: pd.DataFrame) -> str:
    """Overlay both underwater curves: how far each series sits below its peak."""
    fig = go.Figure()
    for name, color, width in (
        ("Portfolio", _PORTFOLIO_COLOR, 2.2),
        ("Benchmark", _BENCHMARK_COLOR, 1.8),
    ):
        depth = drawdown_series(curves[name])
        fig.add_trace(go.Scatter(
            x=depth.index, y=depth, name=name, mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=f"{name}: %{{y:.2%}}<extra></extra>"))

    fig.update_layout(**_time_series_layout(
        "Drawdown Depth - Decline from the Running Peak",
        dict(title="Drawdown", gridcolor=_GRID, tickformat=".0%",
             zeroline=True, zerolinecolor=_AXIS_ZERO),
    ))
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config=_CHART_CONFIG, div_id="chart-drawdown-depth")


def build_time_underwater_chart(curves: pd.DataFrame) -> str:
    """Overlay how long each series has gone without setting a new peak."""
    fig = go.Figure()
    for name, color, width in (
        ("Portfolio", _PORTFOLIO_COLOR, 2.2),
        ("Benchmark", _BENCHMARK_COLOR, 1.8),
    ):
        days = time_underwater_series(curves[name])
        fig.add_trace(go.Scatter(
            x=days.index, y=days, name=name, mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=f"{name}: %{{y:,d}} days<extra></extra>"))

    fig.update_layout(**_time_series_layout(
        "Time Underwater - Days Since the Last Peak",
        dict(title="Days", gridcolor=_GRID, rangemode="tozero",
             zeroline=True, zerolinecolor=_AXIS_ZERO),
    ))
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config=_CHART_CONFIG, div_id="chart-time-underwater")


def build_rolling_ratio_chart(returns: pd.DataFrame, window: int, kind: str) -> str:
    """Overlay the rolling Sharpe or Sortino ratio of both series for one window.

    Gaps are left unconnected: the leading days before the window fills, and for
    Sortino any window without enough losing days to measure downside spread.
    """
    compute = rolling_sharpe if kind == "Sharpe" else rolling_sortino

    fig = go.Figure()
    drawn = False
    for name, column, color, width in (
        ("Portfolio", PORTFOLIO_COLUMN, _PORTFOLIO_COLOR, 2.2),
        ("Benchmark", BENCHMARK_COLUMN, _BENCHMARK_COLOR, 1.8),
    ):
        ratio = compute(excess_returns(returns, column), window)
        drawn = drawn or bool(ratio.notna().any())
        fig.add_trace(go.Scatter(
            x=ratio.index, y=ratio, name=name, mode="lines", connectgaps=False,
            line=dict(color=color, width=width),
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>"))

    fig.add_hline(y=0.0, line=dict(color=_AXIS_ZERO, width=1))
    if not drawn:
        fig.add_annotation(
            text=f"The sample is shorter than the {window}-day window.",
            xref="paper", yref="paper", x=0.5, y=0.5, xanchor="center",
            showarrow=False, font=dict(size=11, color=_MUTED))

    label = ROLLING_WINDOW_LABELS[window]
    fig.update_layout(**_time_series_layout(
        f"Rolling {kind} Ratio - {window}-day window (~1 {label})",
        dict(title=f"{kind} ratio", gridcolor=_GRID,
             zeroline=True, zerolinecolor=_AXIS_ZERO),
    ))
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_CHART_CONFIG,
                       div_id=f"chart-rolling-{kind.lower()}-{window}")


def build_rolling_factor_betas_chart(returns: pd.DataFrame, window: int) -> str:
    """Overlay the portfolio's five rolling factor loadings for one window."""
    betas = rolling_factor_betas(returns, window)

    fig = go.Figure()
    for factor in FF5_FACTORS:
        fig.add_trace(go.Scatter(
            x=betas.index, y=betas[factor], name=factor, mode="lines",
            connectgaps=False,
            line=dict(color=_FACTOR_COLORS[factor], width=1.8),
            hovertemplate=f"{factor}: %{{y:.2f}}<extra></extra>"))

    fig.add_hline(y=0.0, line=dict(color=_AXIS_ZERO, width=1))
    if not betas.notna().to_numpy().any():
        message = (
            f"The sample is shorter than the {window}-day window."
            if len(returns) < window else
            "No window in this sample pins down a unique set of loadings."
        )
        fig.add_annotation(
            text=message, xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor="center", showarrow=False, font=dict(size=11, color=_MUTED))

    label = ROLLING_WINDOW_LABELS[window]
    fig.update_layout(**_time_series_layout(
        f"Rolling Fama-French Factor Betas - {window}-day window (~1 {label})",
        dict(title="Beta", gridcolor=_GRID,
             zeroline=True, zerolinecolor=_AXIS_ZERO),
    ))
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_CHART_CONFIG,
                       div_id=f"chart-rolling-ff5-betas-{window}")


def _build_regression_scatter(regressor: pd.Series, portfolio: pd.Series,
                              fit: tuple[float, float] | None, title: str,
                              x_title: str, hover_label: str, marker_color: str,
                              line_color: str, div_id: str) -> str:
    """Scatter one regressor against portfolio excess returns with its OLS line.

    Every point is one trading day. ``fit`` is the ``(alpha, beta)`` of the
    univariate regression of the portfolio's excess return on this regressor; it
    is None when the regressor never moves, which leaves no slope to draw, so
    the points are shown on their own under a note saying why.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=regressor, y=portfolio, name="Daily excess returns", mode="markers",
        marker=dict(color=marker_color, size=6, opacity=0.55),
        hovertemplate=(f"{hover_label}: %{{x:.2%}}"
                       "<br>Portfolio: %{y:.2%}<extra></extra>")))

    if fit is None:
        fig.add_annotation(
            text=f"{hover_label} is the same value every day, so no line can be fitted.",
            xref="paper", yref="paper", x=0.5, y=0.94, xanchor="center",
            showarrow=False, font=dict(size=11, color=_MUTED))
    else:
        alpha, beta = fit
        # A straight line only needs its endpoints.
        fit_x = np.array([regressor.min(), regressor.max()])
        fig.add_trace(go.Scatter(
            x=fit_x, y=alpha + beta * fit_x, name="OLS fit", mode="lines",
            line=dict(color=line_color, width=2.2),
            hovertemplate="Fitted: %{y:.2%}<extra></extra>"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_TITLE), x=0.01,
                   xanchor="left"),
        template="plotly_dark",
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font=dict(family=_FONT, color=_MUTED, size=11),
        hoverlabel=_HOVER_LABEL,
        margin=dict(l=64, r=24, t=48, b=56),
        height=440,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor=_TRANSPARENT, font=dict(size=11)),
        xaxis=dict(title=x_title, gridcolor=_GRID, tickformat=".1%",
                   zeroline=True, zerolinecolor=_AXIS_ZERO),
        yaxis=dict(title="Portfolio excess return", gridcolor=_GRID, tickformat=".1%",
                   zeroline=True, zerolinecolor=_AXIS_ZERO),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config=_CHART_CONFIG, div_id=div_id)


def build_scatter_chart(returns: pd.DataFrame, alpha: float, beta: float) -> str:
    """Scatter every day's paired excess returns and overlay the fitted OLS line."""
    return _build_regression_scatter(
        excess_returns(returns, BENCHMARK_COLUMN),
        excess_returns(returns, PORTFOLIO_COLUMN),
        (alpha, beta),
        title="CAPM Model - Daily Excess Returns vs Benchmark",
        x_title="Benchmark excess return",
        hover_label="Benchmark",
        marker_color=_BENCHMARK_COLOR,
        line_color=_FIT_LINE_COLOR,
        div_id="chart-capm",
    )


def build_style_factor_scatter_charts(returns: pd.DataFrame) -> list[str]:
    """One CAPM-style scatter per uploaded style factor, each fitted on its own.

    The market factor already has the CAPM chart, so only the four style columns
    get one here.
    """
    factors = factor_matrix(returns)
    portfolio = excess_returns(returns, PORTFOLIO_COLUMN)

    charts = []
    for name in FF5_FACTORS[1:]:
        factor = factors[name]
        charts.append(_build_regression_scatter(
            factor, portfolio, single_factor_fit(portfolio, factor),
            title=(f"{name} - Daily Excess Returns vs the "
                   f"{STYLE_FACTOR_THEMES[name]} Factor"),
            x_title=f"{name} return",
            hover_label=name,
            marker_color=_FACTOR_COLORS[name],
            line_color=_FIT_LINE_COLOR,
            div_id=f"chart-factor-{name.lower()}",
        ))
    return charts


def _normal_pdf(grid: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Normal density on ``grid`` for the given mean and standard deviation."""
    return np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def _distribution_summary(name: str, color: str, stats: dict[str, float]) -> str:
    """One annotation line pairing a series with its tail and shape numbers."""
    return (
        f"<span style='color:{color}'><b>{name}</b></span>"
        f"  VaR 5% {_pct(stats['var'])}"
        f"  ·  CVaR 5% {_pct(stats['cvar'])}"
        f"  ·  Skew {_num(stats['skewness'])}"
        f"  ·  Excess kurtosis {_num(stats['excess_kurtosis'])}"
    )


def build_distribution_chart(returns: pd.DataFrame,
                             portfolio_stats: dict[str, float],
                             benchmark_stats: dict[str, float]) -> str:
    """Overlay both return histograms with fitted normal curves and VaR markers.

    Both series share one bin grid so the bars are directly comparable, and the
    densities are normalized so a taller bar means a larger share of days rather
    than a longer sample.
    """
    portfolio = returns[PORTFOLIO_COLUMN]
    benchmark = returns[BENCHMARK_COLUMN]

    low = float(min(portfolio.min(), benchmark.min()))
    high = float(max(portfolio.max(), benchmark.max()))
    # A constant series has no spread of its own, so borrow a visible window.
    span = high - low if high > low else max(abs(high), 0.01) * 2.0
    low -= span * 0.05
    high += span * 0.05
    bin_size = (high - low) / DISTRIBUTION_BINS
    grid = np.linspace(low, high, 240)

    fig = go.Figure()
    for series, name, color in (
        (portfolio, "Portfolio", _PORTFOLIO_COLOR),
        (benchmark, "Benchmark", _BENCHMARK_COLOR),
    ):
        fig.add_trace(go.Histogram(
            x=series, name=name, histnorm="probability density",
            xbins=dict(start=low, end=high, size=bin_size),
            marker=dict(color=color, line=dict(width=0)), opacity=0.45,
            hovertemplate=f"{name}<br>Return: %{{x:.2%}}<br>"
                          "Density: %{y:.2f}<extra></extra>"))

        sd = float(series.std(ddof=1))
        if sd > 0 and not np.isnan(sd):
            fig.add_trace(go.Scatter(
                x=grid, y=_normal_pdf(grid, float(series.mean()), sd),
                name=f"{name} normal fit", mode="lines",
                line=dict(color=color, width=2.2),
                hovertemplate=f"{name} normal fit<br>Return: %{{x:.2%}}<extra></extra>"))

    # Portfolio labels sit above the benchmark's so the two markers stay legible.
    for stats, name, color, position in (
        (portfolio_stats, "Portfolio", _PORTFOLIO_COLOR, "top left"),
        (benchmark_stats, "Benchmark", _BENCHMARK_COLOR, "bottom left"),
    ):
        fig.add_vline(
            x=stats["var"], line=dict(color=color, width=1.6, dash="dash"),
            annotation_text=f"{name} VaR 5% {_pct(stats['var'])}",
            annotation_position=position,
            annotation_font=dict(size=10, color=color))

    fig.add_annotation(
        text=(_distribution_summary("Portfolio", _PORTFOLIO_COLOR, portfolio_stats)
              + "<br>"
              + _distribution_summary("Benchmark", _BENCHMARK_COLOR, benchmark_stats)),
        xref="paper", yref="paper", x=0.5, y=-0.24, xanchor="center",
        yanchor="top", align="center", showarrow=False,
        font=dict(size=11, color=_MUTED))

    fig.update_layout(
        title=dict(text="Distribution of Daily Returns",
                   font=dict(size=14, color=_TITLE), x=0.01, xanchor="left"),
        template="plotly_dark",
        barmode="overlay",
        bargap=0.02,
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font=dict(family=_FONT, color=_MUTED, size=11),
        hoverlabel=_HOVER_LABEL,
        # The bottom margin holds the two-line statistics annotation.
        margin=dict(l=64, r=24, t=48, b=120),
        hovermode="closest",
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor=_TRANSPARENT, font=dict(size=11)),
        xaxis=dict(title="Daily return", gridcolor=_GRID, tickformat=".1%",
                   zeroline=True, zerolinecolor=_AXIS_ZERO),
        yaxis=dict(title="Probability density", gridcolor=_GRID, zeroline=False),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config=_CHART_CONFIG, div_id="chart-return-distribution")


@app.route("/")
def index():
    return render_template("index.html")


def _render_dashboard(filename: str, raw: bytes) -> str:
    """Parse one upload and render the whole dashboard for it.

    Everything downstream of the parse lives here so a single ``try`` in the
    view covers the regressions and the chart building too, not just the read.
    """
    returns, notices = load_returns(raw)
    regression = capm_regression(returns)

    scale_warning = percent_scale_warning(returns)
    if scale_warning:
        notices.append(scale_warning)

    # The CAPM regresses on the market alone, so it keeps every parsed day, the
    # same sample the stat cards and the scatter above it report. Only the
    # multi-factor models need the style columns, and they share the narrower
    # sample those columns leave so their R-squared figures stay comparable.
    capm_table = capm_table_regression(returns)
    if capm_table is None:
        notices.append(
            "The CAPM regression table is left out: on this sample its "
            "regressors are constant or move together too closely to "
            "separate one loading from another."
        )

    factors = factor_ready(returns)
    if len(factors) < MIN_ROWS:
        notices.append(
            f"Only {len(factors)} row(s) carry all four style factors, fewer than "
            f"the {MIN_ROWS} a factor fit needs, so the Fama-French regressions, "
            "style scatters and rolling factor betas are left out."
        )
        ff5 = ff3 = None
        style_scatters: list[str] = []
        rolling_beta_charts: list[str] = []
    else:
        ff5 = factor_regression(factors, FF5_FACTORS)
        ff3 = factor_regression(factors, FF3_FACTORS)
        for label, fit in (("five-factor", ff5), ("three-factor", ff3)):
            if fit is None:
                notices.append(
                    f"The {label} regression table is left out: on this sample its "
                    "regressors are constant or move together too closely to "
                    "separate one loading from another."
                )
        style_scatters = build_style_factor_scatter_charts(factors)
        rolling_beta_charts = [
            build_rolling_factor_betas_chart(factors, window)
            for window in ROLLING_WINDOWS
        ]

    curves = equity_curves(returns)
    rf = returns[RF_COLUMN]
    portfolio_perf = format_performance(
        series_performance(returns[PORTFOLIO_COLUMN], rf),
        benchmark_correlation(returns),
    )
    benchmark_perf = format_performance(
        series_performance(returns[BENCHMARK_COLUMN], rf),
        1.0,
    )

    return render_template(
        "dashboard.html",
        filename=filename,
        warnings=notices,
        n_obs=len(returns),
        start_date=returns.index.min().strftime(DATE_FORMAT),
        end_date=returns.index.max().strftime(DATE_FORMAT),
        final_portfolio=_money(curves["Portfolio"].iloc[-1]),
        final_benchmark=_money(curves["Benchmark"].iloc[-1]),
        portfolio_perf=portfolio_perf,
        benchmark_perf=benchmark_perf,
        linear_chart=build_equity_chart(
            curves, "Equity Curve - Growth of $1 (linear scale)", "chart-equity-linear"),
        log_chart=build_equity_chart(
            curves, "Equity Curve - Growth of $1 (logarithmic scale)",
            "chart-equity-log", log_scale=True),
        drawdown_chart=build_drawdown_chart(curves),
        time_underwater_chart=build_time_underwater_chart(curves),
        rolling_sharpe_charts=[
            build_rolling_ratio_chart(returns, window, "Sharpe")
            for window in ROLLING_WINDOWS
        ],
        rolling_sortino_charts=[
            build_rolling_ratio_chart(returns, window, "Sortino")
            for window in ROLLING_WINDOWS
        ],
        distribution_chart=build_distribution_chart(
            returns,
            distribution_stats(returns[PORTFOLIO_COLUMN]),
            distribution_stats(returns[BENCHMARK_COLUMN]),
        ),
        beta=_num(regression["beta"], 3),
        alpha=_signed_pct(regression["alpha"], 3),
        alpha_annualized=_signed_pct(regression["alpha_annualized"]),
        r_squared=_pct(regression["r_squared"], 1),
        scatter_chart=build_scatter_chart(
            returns, regression["alpha"], regression["beta"]),
        style_factor_scatter_charts=style_scatters,
        ff5_rows=format_factor_table(ff5),
        ff3_rows=format_factor_table(ff3),
        capm_rows=format_factor_table(capm_table),
        ff5_r2=_num(ff5["r_squared"], 3) if ff5 else None,
        ff3_r2=_num(ff3["r_squared"], 3) if ff3 else None,
        capm_r2=_num(capm_table["r_squared"], 3) if capm_table else None,
        rolling_ff5_beta_charts=rolling_beta_charts,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return render_template(
            "index.html", error="Please choose a CSV file to analyze."), 400
    if not upload.filename.lower().endswith(".csv"):
        return render_template(
            "index.html", error="Please upload a file with a .csv extension."), 400

    try:
        return _render_dashboard(upload.filename, upload.read())
    except CsvFormatError as exc:
        return render_template("index.html", error=str(exc)), 400
    except Exception:
        # Anything reaching here is a bug rather than a bad upload, so the trace
        # belongs in the log and the visitor gets a message instead of it.
        app.logger.exception("Analysis failed for upload %r", upload.filename)
        return render_template(
            "index.html",
            error="Something went wrong while analyzing that file. Please check "
                  "it against the expected format and try again."), 500


@app.errorhandler(413)
def too_large(_error):
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return render_template(
        "index.html", error=f"That file is larger than the {limit_mb} MB limit."), 413


if __name__ == "__main__":
    # PORT lets this run alongside another local Flask app already on 5000.
    # Debug is opt-in through FLASK_DEBUG because its reloader also exposes an
    # interactive console on every traceback, which must never be reachable
    # anywhere the app is served to someone else.
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug, port=int(os.environ.get("PORT", 5000)))
