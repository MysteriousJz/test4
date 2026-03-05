"""
binary_mask_generator.py
========================
Generates a binary mask DataFrame of 54 trading-signal boolean columns
from a 21-column indicator DataFrame produced by the oh2.py data pipeline.

Each rule uses only current and historical data (no lookahead). NaN inputs
always produce False outputs. All operations are vectorised; no Python-level
row loops are used.

Author:  production module – see README for full rule descriptions
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
REQUIRED_INPUT_COLUMNS: list[str] = [
    "TIME",
    "CURRENT_RATE",
    "PHI_5_7",
    "PHI_9_2",
    "PHI_14_8",
    "PHI_24",
    "CRAP_5_7",
    "CRAP_9_2",
    "CRAP_14_8",
    "CRAP_24",
    "RSI_21",
    "RSI_34",
    "STD_D1",
    "STD_U1",
    "STD_D2",
    "STD_U2",
    "STD_D3",
    "STD_U3",
    "THETA_144",
    "THETA_233",
    "THETA_377",
    "THETA_610",
]

# Exact threshold used in CRAP level rules
CRAP_THRESHOLD: float = 0.003105620015142

# RSI extremes
RSI_OVERSOLD_LEVEL: float = 9.01
RSI_OVERBOUGHT_LEVEL: float = 90.99

# THETA boundary shared by both THETA rules
THETA_BOUNDARY: float = 9.01

# Rolling window sizes (rows) for multi-timeframe percentile rules.
# Assumes 2-second data cadence: rows = hours × 3600 / 2.
WIN_5_7H: int = 10260   # 5.7 hours
WIN_9_2H: int = 16560   # 9.2 hours
WIN_14_8H: int = 26640  # 14.8 hours
WIN_24H: int = 43200    # 24 hours

# Percentile boundaries for bottom / top rules
PCT_LOW: float = 0.0901   # 9.01th percentile
PCT_HIGH: float = 0.9099  # 90.99th percentile

# Ordered list of the 54 output column names (matches rule numbering in spec)
OUTPUT_COLUMNS: list[str] = [
    # Category 1 – PHI Gates
    "PHI_OOS_UP",
    "PHI_OOS_DOWN",
    # Category 2 – PHI Cross / Trend
    "PHI_CROSS_UP",
    "PHI_CROSS_DOWN",
    "PHI_TREND_UP",
    "PHI_TREND_DOWN",
    # Category 3 – Bollinger Walls (rule column names differ from input column names)
    "BOOL_STD_D1",
    "BOOL_STD_D2",
    "BOOL_STD_D3",
    "BOOL_STD_U1",
    "BOOL_STD_U2",
    "BOOL_STD_U3",
    # Category 4 – CRAP Breath (Peak / Trough only; U/D/MU/MD removed)
    "CRAP5_7_PEAK",
    "CRAP9_2_PEAK",
    "CRAP14_8_PEAK",
    "CRAP24_PEAK",
    "CRAP5_7_TROUGH",
    "CRAP9_2_TROUGH",
    "CRAP14_8_TROUGH",
    "CRAP24_TROUGH",
    # Category 5 – RSI Pulse
    "RSI_OVERSOLD",
    "RSI_OVERBOUGHT",
    # Category 6 – THETA Spine
    "THETA_BUY",
    "THETA_SELL",
    # --- New masks (added) ---
    # CRAP Alignment
    "CRAP_ALL_UP",
    "CRAP_ALL_DOWN",
    # PHI Crossovers (rate crossing each PHI level)
    "PHI_CROSS_UP_5_7",
    "PHI_CROSS_DOWN_5_7",
    "PHI_CROSS_UP_9_2",
    "PHI_CROSS_DOWN_9_2",
    "PHI_CROSS_UP_14_8",
    "PHI_CROSS_DOWN_14_8",
    "PHI_CROSS_UP_24",
    "PHI_CROSS_DOWN_24",
    # Multi-Timeframe Price Percentile
    "PRICE_BOTTOM_5H",
    "PRICE_TOP_5H",
    "PRICE_BOTTOM_9H",
    "PRICE_TOP_9H",
    "PRICE_BOTTOM_14H",
    "PRICE_TOP_14H",
    "PRICE_BOTTOM_24H",
    "PRICE_TOP_24H",
    # Extreme Price Alignment
    "EXTREME_OVERSOLD",
    "EXTREME_OVERBOUGHT",
    # CRAP Percentile
    "CRAP_BOTTOM_5H",
    "CRAP_TOP_5H",
    "CRAP_BOTTOM_9H",
    "CRAP_TOP_9H",
    "CRAP_BOTTOM_14H",
    "CRAP_TOP_14H",
    "CRAP_BOTTOM_24H",
    "CRAP_TOP_24H",
    # CRAP Extreme Alignment
    "CRAP_EXTREME_LOW",
    "CRAP_EXTREME_HIGH",
]


class BinaryMaskGenerator:
    """Compute 54 binary trading-signal masks from a pipeline indicator DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all 21 indicator columns listed in
        ``REQUIRED_INPUT_COLUMNS``.  The ``TIME`` column is not used in
        calculations but must be present for schema validation.
    window : int, optional
        Rolling-window size (in rows) used for Peak/Trough rules.
        Default is 60 (= 2 minutes at 2-second data cadence).

    Raises
    ------
    ValueError
        If any required input column is missing from *df*.

    Examples
    --------
    >>> gen = BinaryMaskGenerator(df)
    >>> masks = gen.generate()
    >>> print(masks.columns.tolist())
    ['PHI_OOS_UP', 'PHI_OOS_DOWN', ...]
    """

    def __init__(self, df: pd.DataFrame, window: int = 60) -> None:
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing}"
            )
        self._df = df
        self._window = window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Compute all 54 binary masks and return them as a DataFrame.

        The returned DataFrame shares the same index as the input DataFrame.
        Every column is of dtype ``bool`` (Python ``True`` / ``False``).
        Rows with insufficient history (e.g. first *window* rows for rolling
        rules) produce ``False``.

        Returns
        -------
        pd.DataFrame
            Shape ``(len(df), 54)``, columns as listed in ``OUTPUT_COLUMNS``.
        """
        df = self._df
        w = self._window

        # ----------------------------------------------------------------
        # Convenience aliases – avoid repeated attribute look-ups
        # ----------------------------------------------------------------
        rate = df["CURRENT_RATE"]
        phi5 = df["PHI_5_7"]
        phi9 = df["PHI_9_2"]
        phi14 = df["PHI_14_8"]
        phi24 = df["PHI_24"]
        crap5 = df["CRAP_5_7"]
        crap9 = df["CRAP_9_2"]
        crap14 = df["CRAP_14_8"]
        crap24 = df["CRAP_24"]
        rsi21 = df["RSI_21"]
        rsi34 = df["RSI_34"]
        std_d1 = df["STD_D1"]
        std_d2 = df["STD_D2"]
        std_d3 = df["STD_D3"]
        std_u1 = df["STD_U1"]
        std_u2 = df["STD_U2"]
        std_u3 = df["STD_U3"]
        theta144 = df["THETA_144"]
        theta233 = df["THETA_233"]
        theta377 = df["THETA_377"]
        theta610 = df["THETA_610"]

        # ----------------------------------------------------------------
        # Category 1 – PHI Gates (Rules 1–2)
        # ----------------------------------------------------------------

        # Rule 1: PHI_OOS_UP
        # Price is above ALL four PHI values → extreme upward overextension.
        phi_max = phi5.combine(phi9, max).combine(phi14, max).combine(phi24, max)
        phi_oos_up = _mask(rate > phi_max)

        # Rule 2: PHI_OOS_DOWN
        # Price is below ALL four PHI values → extreme downward underextension.
        phi_min = phi5.combine(phi9, min).combine(phi14, min).combine(phi24, min)
        phi_oos_down = _mask(rate < phi_min)

        # ----------------------------------------------------------------
        # Category 2 – PHI Cross / Trend (Rules 3–6)
        # ----------------------------------------------------------------

        phi5_prev = phi5.shift(1)
        phi24_prev = phi24.shift(1)

        # Rule 3: PHI_CROSS_UP
        # Short PHI just crossed above long PHI (bullish crossover).
        phi_cross_up = _mask((phi5 > phi24) & (phi5_prev <= phi24_prev))

        # Rule 4: PHI_CROSS_DOWN
        # Short PHI just crossed below long PHI (bearish crossover).
        phi_cross_down = _mask((phi5 < phi24) & (phi5_prev >= phi24_prev))

        # Rule 5: PHI_TREND_UP
        # All PHI values in perfect ascending order → strong bullish alignment.
        phi_trend_up = _mask((phi5 > phi9) & (phi9 > phi14) & (phi14 > phi24))

        # Rule 6: PHI_TREND_DOWN
        # All PHI values in perfect descending order → strong bearish alignment.
        phi_trend_down = _mask((phi5 < phi9) & (phi9 < phi14) & (phi14 < phi24))

        # ----------------------------------------------------------------
        # Category 3 – Bollinger Walls (Rules 7–12)
        # ----------------------------------------------------------------

        # Rules 7–9: price below lower bands (increasingly extreme)
        bool_std_d1 = _mask(rate < std_d1)
        bool_std_d2 = _mask(rate < std_d2)
        bool_std_d3 = _mask(rate < std_d3)

        # Rules 10–12: price above upper bands (increasingly extreme)
        bool_std_u1 = _mask(rate > std_u1)
        bool_std_u2 = _mask(rate > std_u2)
        bool_std_u3 = _mask(rate > std_u3)

        # ----------------------------------------------------------------
        # Category 4 – CRAP Breath (Rules 13–20: Peak / Trough only)
        # Note: CRAP U/D/MU/MD level and momentum rules removed as per spec.
        # ----------------------------------------------------------------

        crap5_prev = crap5.shift(1)
        crap9_prev = crap9.shift(1)
        crap14_prev = crap14.shift(1)
        crap24_prev = crap24.shift(1)

        # --- Peak / Trough rules: 60-bar rolling window ---
        # min_periods=window ensures False (not True) when history is short.

        roll5 = crap5.rolling(w, min_periods=w)
        roll9 = crap9.rolling(w, min_periods=w)
        roll14 = crap14.rolling(w, min_periods=w)
        roll24 = crap24.rolling(w, min_periods=w)

        # At rolling maximum AND still rising
        crap5_7_peak = _mask((crap5 >= roll5.max()) & (crap5 > crap5_prev))
        crap9_2_peak = _mask((crap9 >= roll9.max()) & (crap9 > crap9_prev))
        crap14_8_peak = _mask((crap14 >= roll14.max()) & (crap14 > crap14_prev))
        crap24_peak = _mask((crap24 >= roll24.max()) & (crap24 > crap24_prev))

        # At rolling minimum AND still falling
        crap5_7_trough = _mask((crap5 <= roll5.min()) & (crap5 < crap5_prev))
        crap9_2_trough = _mask((crap9 <= roll9.min()) & (crap9 < crap9_prev))
        crap14_8_trough = _mask((crap14 <= roll14.min()) & (crap14 < crap14_prev))
        crap24_trough = _mask((crap24 <= roll24.min()) & (crap24 < crap24_prev))

        # ----------------------------------------------------------------
        # Category 5 – RSI Pulse (Rules 37–38)
        # ----------------------------------------------------------------

        # Rule 37: RSI_OVERSOLD – both RSI lines below 9.01
        rsi_oversold = _mask((rsi34 < RSI_OVERSOLD_LEVEL) & (rsi21 < RSI_OVERSOLD_LEVEL))

        # Rule 38: RSI_OVERBOUGHT – both RSI lines above 90.99
        rsi_overbought = _mask((rsi34 > RSI_OVERBOUGHT_LEVEL) & (rsi21 > RSI_OVERBOUGHT_LEVEL))

        # ----------------------------------------------------------------
        # Category 6 – THETA Spine (Rules 39–40)
        # ----------------------------------------------------------------

        # Strict monotonic ascending: 144 < 233 < 377 < 610
        theta_asc = (
            (theta144 < theta233)
            & (theta233 < theta377)
            & (theta377 < theta610)
        )
        # Strict monotonic descending: 144 > 233 > 377 > 610
        theta_desc = (
            (theta144 > theta233)
            & (theta233 > theta377)
            & (theta377 > theta610)
        )
        theta_ordered = theta_asc | theta_desc

        # All four THETA values below boundary (oversold zone)
        theta_all_low = (
            (theta144 < THETA_BOUNDARY)
            & (theta233 < THETA_BOUNDARY)
            & (theta377 < THETA_BOUNDARY)
            & (theta610 < THETA_BOUNDARY)
        )
        # All four THETA values above boundary (overbought zone)
        theta_all_high = (
            (theta144 > THETA_BOUNDARY)
            & (theta233 > THETA_BOUNDARY)
            & (theta377 > THETA_BOUNDARY)
            & (theta610 > THETA_BOUNDARY)
        )

        # Rule 39: THETA_BUY – ordered alignment while all values are oversold
        theta_buy = _mask(theta_ordered & theta_all_low)

        # Rule 40: THETA_SELL – ordered alignment while all values are overbought
        theta_sell = _mask(theta_ordered & theta_all_high)

        # ================================================================
        # New masks
        # ================================================================

        # ----------------------------------------------------------------
        # CRAP Alignment
        # ----------------------------------------------------------------

        # CRAP_ALL_UP: CRAP values in perfect ascending order (5_7 < 9_2 < 14_8 < 24)
        crap_all_up = _mask((crap5 < crap9) & (crap9 < crap14) & (crap14 < crap24))

        # CRAP_ALL_DOWN: CRAP values in perfect descending order (5_7 > 9_2 > 14_8 > 24)
        crap_all_down = _mask((crap5 > crap9) & (crap9 > crap14) & (crap14 > crap24))

        # ----------------------------------------------------------------
        # PHI Crossovers – rate crossing each individual PHI level
        # ----------------------------------------------------------------

        rate_prev = rate.shift(1)
        phi9_prev = phi9.shift(1)
        phi14_prev = phi14.shift(1)
        # phi5_prev and phi24_prev are already defined above

        phi_cross_up_5_7 = _mask((rate >= phi5) & (rate_prev < phi5_prev))
        phi_cross_down_5_7 = _mask((rate <= phi5) & (rate_prev > phi5_prev))

        phi_cross_up_9_2 = _mask((rate >= phi9) & (rate_prev < phi9_prev))
        phi_cross_down_9_2 = _mask((rate <= phi9) & (rate_prev > phi9_prev))

        phi_cross_up_14_8 = _mask((rate >= phi14) & (rate_prev < phi14_prev))
        phi_cross_down_14_8 = _mask((rate <= phi14) & (rate_prev > phi14_prev))

        phi_cross_up_24 = _mask((rate >= phi24) & (rate_prev < phi24_prev))
        phi_cross_down_24 = _mask((rate <= phi24) & (rate_prev > phi24_prev))

        # ----------------------------------------------------------------
        # Multi-Timeframe Price Percentile
        # ----------------------------------------------------------------

        pq5 = rate.rolling(WIN_5_7H, min_periods=WIN_5_7H)
        pq9 = rate.rolling(WIN_9_2H, min_periods=WIN_9_2H)
        pq14 = rate.rolling(WIN_14_8H, min_periods=WIN_14_8H)
        pq24 = rate.rolling(WIN_24H, min_periods=WIN_24H)

        price_bottom_5h = _mask(rate <= pq5.quantile(PCT_LOW))
        price_top_5h = _mask(rate >= pq5.quantile(PCT_HIGH))
        price_bottom_9h = _mask(rate <= pq9.quantile(PCT_LOW))
        price_top_9h = _mask(rate >= pq9.quantile(PCT_HIGH))
        price_bottom_14h = _mask(rate <= pq14.quantile(PCT_LOW))
        price_top_14h = _mask(rate >= pq14.quantile(PCT_HIGH))
        price_bottom_24h = _mask(rate <= pq24.quantile(PCT_LOW))
        price_top_24h = _mask(rate >= pq24.quantile(PCT_HIGH))

        # Extreme Price Alignment
        extreme_oversold = _mask(
            price_bottom_5h & price_bottom_9h & price_bottom_14h & price_bottom_24h
        )
        extreme_overbought = _mask(
            price_top_5h & price_top_9h & price_top_14h & price_top_24h
        )

        # ----------------------------------------------------------------
        # CRAP Percentile (uses crap24 series across each time window)
        # ----------------------------------------------------------------

        cq5 = crap24.rolling(WIN_5_7H, min_periods=WIN_5_7H)
        cq9 = crap24.rolling(WIN_9_2H, min_periods=WIN_9_2H)
        cq14 = crap24.rolling(WIN_14_8H, min_periods=WIN_14_8H)
        cq24 = crap24.rolling(WIN_24H, min_periods=WIN_24H)

        crap_bottom_5h = _mask(crap24 <= cq5.quantile(PCT_LOW))
        crap_top_5h = _mask(crap24 >= cq5.quantile(PCT_HIGH))
        crap_bottom_9h = _mask(crap24 <= cq9.quantile(PCT_LOW))
        crap_top_9h = _mask(crap24 >= cq9.quantile(PCT_HIGH))
        crap_bottom_14h = _mask(crap24 <= cq14.quantile(PCT_LOW))
        crap_top_14h = _mask(crap24 >= cq14.quantile(PCT_HIGH))
        crap_bottom_24h = _mask(crap24 <= cq24.quantile(PCT_LOW))
        crap_top_24h = _mask(crap24 >= cq24.quantile(PCT_HIGH))

        # CRAP Extreme Alignment
        crap_extreme_low = _mask(
            crap_bottom_5h & crap_bottom_9h & crap_bottom_14h & crap_bottom_24h
        )
        crap_extreme_high = _mask(
            crap_top_5h & crap_top_9h & crap_top_14h & crap_top_24h
        )

        # ----------------------------------------------------------------
        # Assemble result DataFrame
        # ----------------------------------------------------------------
        result = pd.DataFrame(
            {
                # Category 1 – PHI Gates
                "PHI_OOS_UP": phi_oos_up,
                "PHI_OOS_DOWN": phi_oos_down,
                # Category 2 – PHI Cross / Trend
                "PHI_CROSS_UP": phi_cross_up,
                "PHI_CROSS_DOWN": phi_cross_down,
                "PHI_TREND_UP": phi_trend_up,
                "PHI_TREND_DOWN": phi_trend_down,
                # Category 3 – Bollinger Walls
                "BOOL_STD_D1": bool_std_d1,
                "BOOL_STD_D2": bool_std_d2,
                "BOOL_STD_D3": bool_std_d3,
                "BOOL_STD_U1": bool_std_u1,
                "BOOL_STD_U2": bool_std_u2,
                "BOOL_STD_U3": bool_std_u3,
                # Category 4 – CRAP Breath (Peak / Trough only)
                "CRAP5_7_PEAK": crap5_7_peak,
                "CRAP9_2_PEAK": crap9_2_peak,
                "CRAP14_8_PEAK": crap14_8_peak,
                "CRAP24_PEAK": crap24_peak,
                "CRAP5_7_TROUGH": crap5_7_trough,
                "CRAP9_2_TROUGH": crap9_2_trough,
                "CRAP14_8_TROUGH": crap14_8_trough,
                "CRAP24_TROUGH": crap24_trough,
                # Category 5 – RSI Pulse
                "RSI_OVERSOLD": rsi_oversold,
                "RSI_OVERBOUGHT": rsi_overbought,
                # Category 6 – THETA Spine
                "THETA_BUY": theta_buy,
                "THETA_SELL": theta_sell,
                # New: CRAP Alignment
                "CRAP_ALL_UP": crap_all_up,
                "CRAP_ALL_DOWN": crap_all_down,
                # New: PHI Crossovers (rate vs each PHI level)
                "PHI_CROSS_UP_5_7": phi_cross_up_5_7,
                "PHI_CROSS_DOWN_5_7": phi_cross_down_5_7,
                "PHI_CROSS_UP_9_2": phi_cross_up_9_2,
                "PHI_CROSS_DOWN_9_2": phi_cross_down_9_2,
                "PHI_CROSS_UP_14_8": phi_cross_up_14_8,
                "PHI_CROSS_DOWN_14_8": phi_cross_down_14_8,
                "PHI_CROSS_UP_24": phi_cross_up_24,
                "PHI_CROSS_DOWN_24": phi_cross_down_24,
                # New: Multi-Timeframe Price Percentile
                "PRICE_BOTTOM_5H": price_bottom_5h,
                "PRICE_TOP_5H": price_top_5h,
                "PRICE_BOTTOM_9H": price_bottom_9h,
                "PRICE_TOP_9H": price_top_9h,
                "PRICE_BOTTOM_14H": price_bottom_14h,
                "PRICE_TOP_14H": price_top_14h,
                "PRICE_BOTTOM_24H": price_bottom_24h,
                "PRICE_TOP_24H": price_top_24h,
                # New: Extreme Price Alignment
                "EXTREME_OVERSOLD": extreme_oversold,
                "EXTREME_OVERBOUGHT": extreme_overbought,
                # New: CRAP Percentile
                "CRAP_BOTTOM_5H": crap_bottom_5h,
                "CRAP_TOP_5H": crap_top_5h,
                "CRAP_BOTTOM_9H": crap_bottom_9h,
                "CRAP_TOP_9H": crap_top_9h,
                "CRAP_BOTTOM_14H": crap_bottom_14h,
                "CRAP_TOP_14H": crap_top_14h,
                "CRAP_BOTTOM_24H": crap_bottom_24h,
                "CRAP_TOP_24H": crap_top_24h,
                # New: CRAP Extreme Alignment
                "CRAP_EXTREME_LOW": crap_extreme_low,
                "CRAP_EXTREME_HIGH": crap_extreme_high,
            },
            index=df.index,
        )

        return result

    def save(self, path: str) -> None:
        """Persist the generated mask DataFrame to disk.

        The file format is inferred from the file extension:

        * ``.feather`` – Apache Arrow / Feather format (fast, columnar).
        * ``.db`` / ``.sqlite`` / ``.sqlite3`` – SQLite table named
          ``binary_masks``.
        * Any other extension defaults to Feather.

        Parameters
        ----------
        path : str
            Destination file path.

        Raises
        ------
        ValueError
            If :meth:`generate` has not been called yet (internal masks not
            available).  Call ``gen.save(path)`` after ``gen.generate()``, or
            use the convenience pattern::

                masks = BinaryMaskGenerator(df).generate()
                BinaryMaskGenerator.save_df(masks, path)
        """
        masks = self.generate()
        _save_df(masks, path)

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load a previously saved mask DataFrame from disk.

        The file format is inferred from the file extension (same logic as
        :meth:`save`).

        Parameters
        ----------
        path : str
            Source file path.

        Returns
        -------
        pd.DataFrame
            The mask DataFrame with boolean columns.

        Raises
        ------
         FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not recognised.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask file not found: {path}")
        return _load_df(path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _mask(series: pd.Series) -> pd.Series:
    """Convert a boolean series to strict ``bool`` dtype, mapping NaN → False.

    pandas comparisons involving NaN produce NaN rather than False, so we
    must explicitly fill before casting.

    Parameters
    ----------
    series : pd.Series
        Result of a boolean comparison that may contain NaN values.

    Returns
    -------
    pd.Series
        Same index, dtype ``bool``, NaN replaced with ``False``.
    """
    return series.fillna(False).astype(bool)


def _save_df(df: pd.DataFrame, path: str) -> None:
    """Save *df* to *path* using format inferred from extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".db", ".sqlite", ".sqlite3"):
        import sqlite3

        with sqlite3.connect(path) as conn:
            df.to_sql("binary_masks", conn, if_exists="replace", index=True)
    else:
        # Default: Feather (also handles explicit .feather extension)
        df.reset_index().to_feather(path)


def _load_df(path: str) -> pd.DataFrame:
    """Load *df* from *path* using format inferred from extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".db", ".sqlite", ".sqlite3"):
        import sqlite3

        with sqlite3.connect(path) as conn:
            df = pd.read_sql("SELECT * FROM binary_masks", conn, index_col="index")
        # Restore bool dtype (SQLite stores booleans as integers)
        for col in df.columns:
            if col in OUTPUT_COLUMNS:
                df[col] = df[col].astype(bool)
        return df
    else:
        df = pd.read_feather(path)
        # Feather resets the index; restore it if an 'index' column was saved
        if "index" in df.columns:
            df = df.set_index("index")
        return df



if __name__ == "__main__":
    import sqlite3
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "indicators2.sqlite"

    # Load indicators
    conn = sqlite3.connect(input_path)
    df = pd.read_sql_query("SELECT * FROM indicators ORDER BY TIME", conn)
    conn.close()

    # Generate masks
    gen = BinaryMaskGenerator(df)
    masks = gen.generate()

    # Create output filename based on input
    input_file = Path(input_path)
    output_file = input_file.parent / f"{input_file.stem}_masks.sqlite"

    # Save
    gen.save(str(output_file))
    print(f"Masks created successfully at {output_file}")
