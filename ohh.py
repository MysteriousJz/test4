# --- Imports (grouped, minimal changes to preserve original behavior) ---
# Based on prov2.4.py, adapted for SQLite loading instead of Feather
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# stdlib
from math import atan, degrees

# third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateFormatter, AutoDateLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Kivy GUI (required by this app)
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.properties import NumericProperty
from kivy.clock import Clock
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

import sqlite3  # Added for SQLite loading
import datetime  # For backup timestamps
import os  # For file operations
    
# CONSTANTS: DON'T ASK, DON'T TELL.
DEFAULT_PATH = "/home/anon/program/indicators2.sqlite"  # Changed from Feather to SQLite
LR_WINDOWS = [144,233,377,610] #fib amounts
POS_THETA_DEG = 9.01  # CORRECTED: Changed from 23.6 to 9.01
NEG_THETA_DEG = -9.01  # CORRECTED: Changed from -23.6 to -9.01
DEBOUNCE_TICKS = 1 #maybe unused
FEE = 0.0025 #starter fee
PROFIT_TARGET_PCT = 0.6  # auto-sell profit target in percent
# Fibonacci rolling-window lengths used for buy/sell peak detection.
# A "peak" bar is when the mask-count reaches its highest value in this many bars.
# 55 (buy) and 89 (sell) are consecutive Fibonacci numbers chosen so buy signals
# look back over a slightly tighter horizon, keeping entries responsive, while sell
# signals use a wider horizon to avoid premature exits.
BUY_PEAK_WINDOW  = 55   # bars looked back to confirm a buy-count peak
SELL_PEAK_WINDOW = 89   # bars looked back to confirm a sell-count peak
BUY_MARKER='s'; SELL_MARKER='o'
PRICE_COLOR='#000000'

# Binary mask categories for the switchboard UI
MASK_CATEGORIES = {
    "PHI Gates (2)": [
        ("PHI_OOS_UP", "Price above all PHI values"),
        ("PHI_OOS_DOWN", "Price below all PHI values"),
    ],
    "PHI Cross/Trend (4)": [
        ("PHI_CROSS_UP", "Short PHI crossed above long PHI"),
        ("PHI_CROSS_DOWN", "Short PHI crossed below long PHI"),
        ("PHI_TREND_UP", "All PHI values ascending"),
        ("PHI_TREND_DOWN", "All PHI values descending"),
    ],
    "Bollinger Walls (6)": [
        ("BOOL_STD_D1", "Price below first lower band"),
        ("BOOL_STD_D2", "Price below second lower band"),
        ("BOOL_STD_D3", "Price below third lower band"),
        ("BOOL_STD_U1", "Price above first upper band"),
        ("BOOL_STD_U2", "Price above second upper band"),
        ("BOOL_STD_U3", "Price above third upper band"),
    ],
    "CRAP Breath (24)": [
        ("CRAP5_7_U",      "CRAP_5_7 above +threshold"),
        ("CRAP9_2_U",      "CRAP_9_2 above +threshold"),
        ("CRAP14_8_U",     "CRAP_14_8 above +threshold"),
        ("CRAP24_U",       "CRAP_24 above +threshold"),
        ("CRAP5_7_D",      "CRAP_5_7 below -threshold"),
        ("CRAP9_2_D",      "CRAP_9_2 below -threshold"),
        ("CRAP14_8_D",     "CRAP_14_8 below -threshold"),
        ("CRAP24_D",       "CRAP_24 below -threshold"),
        ("CRAP5_7_MU",     "CRAP_5_7 increasing"),
        ("CRAP9_2_MU",     "CRAP_9_2 increasing"),
        ("CRAP14_8_MU",    "CRAP_14_8 increasing"),
        ("CRAP24_MU",      "CRAP_24 increasing"),
        ("CRAP5_7_MD",     "CRAP_5_7 decreasing"),
        ("CRAP9_2_MD",     "CRAP_9_2 decreasing"),
        ("CRAP14_8_MD",    "CRAP_14_8 decreasing"),
        ("CRAP24_MD",      "CRAP_24 decreasing"),
        ("CRAP5_7_PEAK",   "CRAP_5_7 at 2-min peak"),
        ("CRAP9_2_PEAK",   "CRAP_9_2 at 2-min peak"),
        ("CRAP14_8_PEAK",  "CRAP_14_8 at 2-min peak"),
        ("CRAP24_PEAK",    "CRAP_24 at 2-min peak"),
        ("CRAP5_7_TROUGH", "CRAP_5_7 at 2-min trough"),
        ("CRAP9_2_TROUGH", "CRAP_9_2 at 2-min trough"),
        ("CRAP14_8_TROUGH","CRAP_14_8 at 2-min trough"),
        ("CRAP24_TROUGH",  "CRAP_24 at 2-min trough"),
    ],
    "RSI Pulse (2)": [
        ("RSI_OVERSOLD",   "Both RSIs below 9.01"),
        ("RSI_OVERBOUGHT", "Both RSIs above 89.99"),
    ],
    "THETA Spine (2)": [
        ("THETA_BUY",  "Thetas aligned & below 9.01"),
        ("THETA_SELL", "Thetas aligned & above 9.01"),
    ],
}

# Per-mask colors: BUY signals = shades of green, SELL signals = shades of red
MASK_COLORS = {
    # --- BUY SIGNALS (shades of green) ---
    # PHI Gates (Buy)
    "PHI_OOS_DOWN":    "#006400",   # dark green
    # PHI Cross/Trend (Buy)
    "PHI_CROSS_UP":    "#228B22",   # forest green
    "PHI_TREND_UP":    "#2E8B57",   # sea green
    # Bollinger Walls (Buy)
    "BOOL_STD_D1":     "#3CB371",   # medium sea green
    "BOOL_STD_D2":     "#32CD32",   # lime green
    "BOOL_STD_D3":     "#00FF7F",   # spring green
    # CRAP Breath (Buy – Oversold)
    "CRAP5_7_D":       "#009933",   # deep green
    "CRAP9_2_D":       "#00AA44",
    "CRAP14_8_D":      "#00CC44",
    "CRAP24_D":        "#00EE55",
    # CRAP Breath (Buy – Trough)
    "CRAP5_7_TROUGH":  "#66FF66",   # light green
    "CRAP9_2_TROUGH":  "#44CC44",
    "CRAP14_8_TROUGH": "#33AA33",
    "CRAP24_TROUGH":   "#00FF66",
    # CRAP Breath (Buy – Momentum Up)
    "CRAP5_7_MU":      "#99FF99",   # pale green
    "CRAP9_2_MU":      "#88EE88",
    "CRAP14_8_MU":     "#77DD77",
    "CRAP24_MU":       "#66CC66",
    # RSI Pulse (Buy)
    "RSI_OVERSOLD":    "#00FA9A",   # medium spring green
    # THETA Spine (Buy)
    "THETA_BUY":       "#7CFC00",   # lawn green
    # --- SELL SIGNALS (shades of red) ---
    # PHI Gates (Sell)
    "PHI_OOS_UP":      "#8B0000",   # dark red
    # PHI Cross/Trend (Sell)
    "PHI_CROSS_DOWN":  "#B22222",   # firebrick
    "PHI_TREND_DOWN":  "#DC143C",   # crimson
    # Bollinger Walls (Sell)
    "BOOL_STD_U1":     "#FF4500",   # orange red
    "BOOL_STD_U2":     "#FF6347",   # tomato
    "BOOL_STD_U3":     "#FF7F7F",   # light coral
    # CRAP Breath (Sell – Overbought)
    "CRAP5_7_U":       "#CC0000",
    "CRAP9_2_U":       "#DD1111",
    "CRAP14_8_U":      "#EE2222",
    "CRAP24_U":        "#FF3333",
    # CRAP Breath (Sell – Peak)
    "CRAP5_7_PEAK":    "#FF8080",
    "CRAP9_2_PEAK":    "#FF6666",
    "CRAP14_8_PEAK":   "#FF4444",
    "CRAP24_PEAK":     "#FF2222",
    # CRAP Breath (Sell – Momentum Down)
    "CRAP5_7_MD":      "#FFB3B3",   # pale red/pink
    "CRAP9_2_MD":      "#FFA0A0",
    "CRAP14_8_MD":     "#FF8888",
    "CRAP24_MD":       "#FF7070",
    # RSI Pulse (Sell)
    "RSI_OVERBOUGHT":  "#FF0055",   # hot red
    # THETA Spine (Sell)
    "THETA_SELL":      "#A50000",   # very dark red
}

# HELPERS

#HELP ME HELP YOU HELP US


#scales the price graph to look most pretty having a complete focus on price itself, other things are just letting you know they are happening with price, around price, about price but are not as important as price.
def autoscale_price_axis(ax, x, prices, pad_frac=0.02):
    x_arr = np.asarray(x)
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return
    pmin, pmax = float(np.min(p)), float(np.max(p))
    span_y = max(pmax - pmin, 1e-8)
    pad_y = span_y * pad_frac
    ax.set_ylim(pmin - pad_y, pmax + pad_y)

    if x_arr.size == 0:
        return
    if x_arr.dtype.kind in ('M', 'm'):
        xmin = x_arr.min(); xmax = x_arr.max(); span_x = xmax - xmin
        pad_x = span_x * pad_frac
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
    else:
        xn = x_arr.astype(float); xn = xn[np.isfinite(xn)]
        if xn.size == 0: return
        xmin, xmax = float(np.min(xn)), float(np.max(xn))
        span_x = max(xmax - xmin, 1e-8); pad_x = span_x * pad_frac
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
'''
This computes the slope angle (degrees) of a linear fit of series vs its index. Steps: convert to float array, bail if too short or all-NaN, build x = 0..n-1, center x and y, compute slope by least-squares formula, then return atan(slope) in degrees. It assumes uniform time spacing and ignores NaNs by early bail.
'''
def linreg_slope_angle(series):
    y=np.asarray(series,dtype=float); n=len(y)
    if n<2 or np.all(np.isnan(y)): return np.nan
    x=np.arange(n); xm,ym=x.mean(),np.nanmean(y); xz,yz=x-xm,y-ym
    denom=np.sum(xz*xz)
    if denom==0: return np.nan
    slope = np.sum(xz*yz)/denom
    return degrees(atan(slope))

'''
compute_theta_series builds slope-angle time series for several linear-regression window lengths.

    Input: prices (sequence), windows (list of integer window sizes; default LR_WINDOWS).
    It converts prices to a float numpy array p and creates an output dict thetas.
    For each window w it allocates an array arr of length n filled with NaN.
        If w <= 1 it stores the NaN array.
        Otherwise it walks i from w-1 to n-1 and sets arr[i] = linreg_slope_angle(p[i-w+1 : i+1]). That means arr[i] is the slope-angle (in degrees, from linreg_slope_angle) of the linear fit over the w most recent points ending at i.
    The function returns a dict mapping each window size to its angle array.

Notes:

    It assumes uniform spacing (index-as-time). Angle range is roughly (-90, +90) degrees; positive = upward slope.
    Early indices (< w-1) remain NaN because there aren’t enough points for that window.
'''
def compute_theta_series(prices, windows=LR_WINDOWS):
    p=np.asarray(prices,dtype=float); n=len(p); thetas={}
    for w in windows:
        arr=np.full(n,np.nan)
        if w<=1: thetas[w]=arr; continue
        for i in range(w-1,n): arr[i]=linreg_slope_angle(p[i-w+1:i+1])
        thetas[w]=arr
    return thetas



'''
Purpose: find contiguous True runs in a boolean mask and return a list of (xstart, width) spans suitable for matplotlib.broken_barh.

How it works (stepwise):

    Coerce mask to a boolean numpy array.
    If no True values, return empty list.
    Convert mask to ints and pad with 0 at both ends; take the discrete difference to locate run starts (diff==1) and ends (diff==-1).
    For each start s and end e (Python slice [s,e)):
        Try: xstart = xvals[s], xend = xvals[e-1], width = xend - xstart.
        If width equals 0 or subtraction fails, fall back to width = e - s (an integer length).
        On other indexing/subtraction errors, fall back to xstart=float(s) and width=float(e-s).
    Append (xstart, width) pairs and return the list.

Common failure modes (why plots break):

    xvals is a pandas RangeIndex or has non-numeric types (Timestamps); subtracting can yield timedeltas or non-finite results that broken_barh rejects.
    Mixing positional widths (integers) with time/float xstart units causes inconsistent units.
    Using pandas objects directly can trigger pandas positional-index vs label-index mismatches.
'''
def spans_from_mask(mask, xvals):
    mask=np.asarray(mask,dtype=bool); spans=[]
    if not mask.any(): return spans
    arr=mask.astype(int); padded=np.concatenate([[0],arr,[0]]); diff=np.diff(padded)
    starts=np.where(diff==1)[0]; ends=np.where(diff==-1)[0]
    for s,e in zip(starts,ends):
        try:
            xstart=xvals[s]; xend=xvals[e-1]; width=xend-xstart
            if width==0: width=float(e-s)
        except Exception:
            xstart=float(s); width=float(e-s)
        spans.append((xstart,width))
    return spans

def check_ball_states(row):
    vals=[row.get(c,np.nan) for c in ('LVAL','MVAL','SVAL','TVAL')]
    price=row.get('Composit Rate', row.get('Rate', np.nan))
    try:
        arr=np.array(vals,dtype=float)
        if np.isnan(arr).any(): return False,False
        return float(price) < arr.min(), float(price) > arr.max()
    except Exception:
        return False,False


#used for the trend values computation of order
def order_pattern(row):
    try:
        L=float(row.get('LVAL',np.nan)); M=float(row.get('MVAL',np.nan)); S=float(row.get('SVAL',np.nan)); T=float(row.get('TVAL',np.nan))
        if not np.isfinite([L,M,S,T]).all(): return False,False
        if (L>M) and (M>S) and (S>T): return True,False
        if (L<M) and (M<S) and (S<T): return False,True
    except Exception:
        pass
    return False,False





    """
    Inspect theta{w} values in `row` for the given windows (in LR_WINDOWS order).
    Returns (theta_rally, theta_drop):
      - theta_rally: True when theta values strictly increase across windows AND min(theta) >= pos_threshold
      - theta_drop:  True when theta values strictly decrease across windows AND max(theta) <= neg_threshold
    Non-numeric or missing thetas cause (False, False).
    """
def detect_theta_order(row, windows=LR_WINDOWS, pos_threshold=POS_THETA_DEG, neg_threshold=NEG_THETA_DEG):
    vals = []
    for w in windows:
        key = f"theta{w}"
        if key in row:
            try:
                vals.append(float(row.get(key, np.nan)))
            except Exception:
                vals.append(np.nan)
    if len(vals) < 2:
        return False, False
    vals = np.asarray(vals, dtype=float)
    if not np.isfinite(vals).all():
        return False, False
    inc = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
    dec = all(vals[i] > vals[i+1] for i in range(len(vals)-1))
    theta_rally = bool(inc and np.min(vals) >= pos_threshold)
    theta_drop  = bool(dec and np.max(vals) <= neg_threshold)
    return theta_rally, theta_drop


#arrangements of certain data points in certain sequence over an amount of time
def compute_pattern_events(view):
    events = []
    prev_tdown = False; prev_tup = False
    prev_theta_rally = False; prev_theta_drop = False
    rows = view.reset_index(drop=True)
    for idx, row in rows.iterrows():
        t = float(row['TIME']) if 'TIME' in view.columns else float(idx)
        tdown, tup = order_pattern(row)
        try:
            theta_rally, theta_drop = detect_theta_order(row)
        except Exception:
            theta_rally, theta_drop = False, False
        if tdown != prev_tdown:
            events.append((t, 'tdown_start' if tdown else 'tdown_stop'))
        if tup != prev_tup:
            events.append((t, 'tup_start' if tup else 'tup_stop'))
        if theta_rally != prev_theta_rally:
            events.append((t, 'theta_rally_start' if theta_rally else 'theta_rally_stop'))
        if theta_drop != prev_theta_drop:
            events.append((t, 'theta_drop_start' if theta_drop else 'theta_drop_stop'))
        prev_tdown, prev_tup, prev_theta_rally, prev_theta_drop = tdown, tup, theta_rally, theta_drop
    cleaned = []; last = None
    for ev in events:
        if ev != last: cleaned.append(ev)
        last = ev
    return cleaned


# Flags compute where the flags are all binary reductions of indactors at or not certain values
def compute_flags_dataframe(view):
    flags = pd.DataFrame(index=view.index)
    price_series = pd.to_numeric(view['CURRENT_RATE'] if 'CURRENT_RATE' in view.columns else view.get('Rate'), errors='coerce')
    if 'RSI_21' in view.columns or 'RSI_34' in view.columns:
        # Use RSI_34 for simplicity, or compute if needed
        rsi_raw = pd.to_numeric(view.get('RSI_34', view.get('RSI_21')), errors='coerce')
        view['RSI_SMA34'] = rsi_raw.ewm(span=34, adjust=False).mean() 
        flags['FLAG_RSI_LOW'] = view['RSI_SMA34'] < 40
        flags['FLAG_RSI_HIGH'] = view['RSI_SMA34'] > 160
    else:
        view['RSI_SMA34']=np.nan; flags['FLAG_RSI_LOW']=False; flags['FLAG_RSI_HIGH']=False
    if any(col in view.columns for col in ['CRAP_5_7', 'CRAP_9_2', 'CRAP_14_8', 'CRAP_24']):
        # Use CRAP_24 as representative, or combine if needed
        cr = pd.to_numeric(view.get('CRAP_24', view.get('CRAP_14_8', view.get('CRAP_9_2', view.get('CRAP_5_7')))), errors='coerce')
        flags['FLAG_CRAP_LOW'] = cr <= -0.003105620015142
        flags['FLAG_CRAP_HIGH'] = cr >= 0.003105620015142
        flags['FLAG_CRAP'] = flags['FLAG_CRAP_LOW'] | flags['FLAG_CRAP_HIGH']
    else:
        flags['FLAG_CRAP_LOW']=False; flags['FLAG_CRAP_HIGH']=False; flags['FLAG_CRAP']=False
    balls=[]; aballs=[]
    for _,r in view.iterrows():
        b,a = check_ball_states(r); balls.append(b); aballs.append(a)
    flags['FLAG_BALL']=balls; flags['FLAG_ABALL']=aballs
    tdowns=[]; tups=[]
    for _,r in view.iterrows():
        td,tp = order_pattern(r); tdowns.append(td); tups.append(tp)
    flags['FLAG_VAL_TDOWN']=tdowns; flags['FLAG_VAL_TUP']=tups
    stdu1 = pd.to_numeric(view['STD_U1'], errors='coerce') if 'STD_U1' in view.columns else pd.Series([np.nan]*len(view))
    stdu2 = pd.to_numeric(view['STD_U2'], errors='coerce') if 'STD_U2' in view.columns else pd.Series([np.nan]*len(view))
    stdd1 = pd.to_numeric(view['STD_D1'], errors='coerce') if 'STD_D1' in view.columns else pd.Series([np.nan]*len(view))
    stdd2 = pd.to_numeric(view['STD_D2'], errors='coerce') if 'STD_D2' in view.columns else pd.Series([np.nan]*len(view))
    flags['FLAG_STDD2_ESC'] = (~stdd2.isna()) & (price_series < stdd2)
    flags['FLAG_STDD1_ESC'] = (~stdd1.isna()) & (price_series < stdd1)
    flags['FLAG_STDU2_ESC'] = (~stdu2.isna()) & (price_series > stdu2)
    flags['FLAG_STDU1_ESC'] = (~stdu1.isna()) & (price_series > stdu1)
    return flags, view



class RiverDetector:
    """Minimal detector: stores latest flags/events and current position."""
    def __init__(self):
        self.current_flags = None
        self.current_events = None
        self.position = 'USD'  # track current position here

    def reset(self):
        self.current_flags = None
        self.current_events = None


    def ingest(self, view):
        # compute and store canonical signals for the given slice
        flags, view = compute_flags_dataframe(view)
        events = compute_pattern_events(view)
        self.current_flags = flags
        self.current_events = events

    def evaluate(self):
        # Bare minimum: return no automatic action and a compact summary
        summary = {'position': self.position, 'last_flags_row': None, 'last_events': []}
        if self.current_events:
            summary['last_events'] = list(self.current_events)
        if self.current_flags is not None and len(self.current_flags) > 0:
            last = self.current_flags.iloc[-1].to_dict()
            summary['last_flags_row'] = {k: bool(v) for k, v in last.items()}
        return None, summary


def simulate_trades(prices, detector, start_usd=1000.0, fee=FEE, close_at_end=True):
    """
    Full-capital buys/sells using Kraken buy math.
    Forced end-sell only if the detector's final position is BTC and close_at_end is True.
    Returns (trades, final_usd).
    """
    detector.reset()                 # reset once at start
    detector.position = 'USD'
    usd = float(start_usd); btc = 0.0; trades = []
    n = len(prices)
    if n == 0:
        return trades, float(usd)

    for i in range(n):
        price = float(prices[i])
        sub = pd.DataFrame({'Composit Rate': prices[:i+1]})
        detector.ingest(sub)
        action, _ = detector.evaluate()   # must return 'BUY'/'SELL'/None

        if action == 'BUY' and detector.position == 'USD' and usd > 0.0:
            usd_before = usd
            btc = usd_before / (price * (1.0 + fee))   # Kraken full-capital buy math
            trades.append(('BUY', i, price, usd_before, btc))
            usd = 0.0
            detector.position = 'BTC'

        elif action == 'SELL' and detector.position == 'BTC' and btc > 0.0:
            btc_before = btc
            usd_after = btc_before * price * (1.0 - fee)
            trades.append(('SELL', i, price, usd_after, btc_before))
            usd = usd_after
            btc = 0.0
            detector.position = 'USD'

    last_price = float(prices[-1])
    # Force sell only if detector ended in BTC and caller asked to close
    if close_at_end and detector.position == 'BTC' and btc > 0.0:
        btc_before = btc
        usd_after = btc_before * last_price * (1.0 - fee)
        # FIXED: Added missing closing parenthesis
        trades.append(('SELL', n-1, last_price, usd_after, btc_before))
        usd = usd_after
        btc = 0.0
        detector.position = 'USD'

    # final_usd: if we sold above, usd contains final USD.
    # If we didn't sell and still hold BTC, report market value without applying a sell fee.
    final_usd = float(usd + (btc * last_price))
    return trades, final_usd


def compute_trade_metrics(trades, start_usd=1000.0):
    paired = []; buy_rec = None
    for rec in trades:
        if rec[0] == 'BUY': buy_rec = rec
        elif rec[0] == 'SELL' and buy_rec is not None:
            paired.append((buy_rec, rec)); buy_rec = None
    rois = []
    for buy, sell in paired:
        buy_usd = float(buy[3]); sell_usd = float(sell[3])
        if buy_usd > 0:
            r = (sell_usd / buy_usd) - 1.0
            rois.append(r)
    total = float(start_usd)
    for r in rois: total *= (1.0 + r)
    total_roi = (total / float(start_usd)) - 1.0 if rois else None
    return {'total_roi': total_roi, 'trades': len(paired), 'per_trade_rois': rois, 'final_value': total}


# ---------------------------------------------------------------------------
# MaskSwitchboard: collapsible toggle panel for all 40 binary masks
# ---------------------------------------------------------------------------
class MaskSwitchboard(BoxLayout):
    """
    A vertical BoxLayout containing one collapsible section per mask category.
    Each section has:
      • a header row with the category name, a ▼/▶ collapse button, and
        "All ON" / "All OFF" buttons
      • one toggle row per mask (green = ON, gray = OFF)
    Toggle states are stored in self.toggle_states keyed by mask name.
    """

    # Colours used for toggle buttons
    _COLOR_ON  = (0.18, 0.65, 0.18, 1)   # green
    _COLOR_OFF = (0.45, 0.45, 0.45, 1)   # gray

    def __init__(self, categories, **kwargs):
        kwargs.setdefault('orientation', 'vertical')
        kwargs.setdefault('size_hint_y', None)
        kwargs.setdefault('spacing', 5)
        super().__init__(**kwargs)
        self.bind(minimum_height=self.setter('height'))

        # Flat dict: mask_name -> bool (initial state = False)
        self.toggle_states = {
            name: False
            for masks in categories.values()
            for name, _ in masks
        }
        # Keep references to per-mask toggle buttons so colours stay in sync
        self._toggle_buttons = {}   # mask_name -> Button

        for cat_name, masks in categories.items():
            self._build_category(cat_name, masks)

    # ------------------------------------------------------------------
    # Build one collapsible category block
    # ------------------------------------------------------------------
    def _build_category(self, cat_name, masks):
        # Outer container for the whole category (header + body)
        cat_box = BoxLayout(orientation='vertical', size_hint_y=None, spacing=2)
        cat_box.bind(minimum_height=cat_box.setter('height'))

        # --- Header row ---
        header = BoxLayout(size_hint_y=None, height=32, spacing=4)

        # Collapse toggle (▼ = expanded, ▲ = collapsed)
        collapse_btn = Button(
            text='▼', size_hint_x=None, width=28,
            background_color=(0.25, 0.25, 0.45, 1), font_size=13,
        )

        cat_label = Label(
            text=f'[b]{cat_name}[/b]', markup=True,
            halign='left', size_hint_x=1, font_size=13,
        )
        cat_label.bind(size=cat_label.setter('text_size'))

        all_on_btn  = Button(text='All ON',  size_hint_x=None, width=60,
                             background_color=self._COLOR_ON,  font_size=11)
        all_off_btn = Button(text='All OFF', size_hint_x=None, width=60,
                             background_color=self._COLOR_OFF, font_size=11)

        header.add_widget(collapse_btn)
        header.add_widget(cat_label)
        header.add_widget(all_on_btn)
        header.add_widget(all_off_btn)
        cat_box.add_widget(header)

        # --- Body: one row per mask ---
        body = BoxLayout(orientation='vertical', size_hint_y=None, spacing=2)
        body.bind(minimum_height=body.setter('height'))

        for mask_name, tooltip in masks:
            row = self._build_mask_row(mask_name, tooltip)
            body.add_widget(row)

        cat_box.add_widget(body)
        self.add_widget(cat_box)

        # --- Wire up collapse ---
        def toggle_collapse(instance):
            if body.height > 0:
                # Collapse: hide body
                body.height = 0
                body.opacity = 0
                body.disabled = True
                instance.text = '▶'
            else:
                # Expand: restore body
                body.disabled = False
                body.opacity = 1
                body.bind(minimum_height=body.setter('height'))
                body.height = body.minimum_height
                instance.text = '▼'

        collapse_btn.bind(on_press=toggle_collapse)

        # --- Wire up All ON / All OFF ---
        mask_names = [n for n, _ in masks]

        def set_all_on(instance, names=mask_names):
            for nm in names:
                self._set_mask(nm, True)

        def set_all_off(instance, names=mask_names):
            for nm in names:
                self._set_mask(nm, False)

        all_on_btn.bind(on_press=set_all_on)
        all_off_btn.bind(on_press=set_all_off)

    # ------------------------------------------------------------------
    # Build a single mask toggle row
    # ------------------------------------------------------------------
    def _build_mask_row(self, mask_name, tooltip):
        row = BoxLayout(size_hint_y=None, height=26, spacing=4)

        btn = Button(
            text='OFF',
            size_hint_x=None, width=46,
            background_color=self._COLOR_OFF,
            font_size=11,
        )
        btn.bind(on_press=lambda instance, nm=mask_name: self._toggle_mask(nm))
        self._toggle_buttons[mask_name] = btn

        lbl = Label(
            text=mask_name,
            halign='left', size_hint_x=1,
            font_size=11,
        )
        lbl.bind(size=lbl.setter('text_size'))

        row.add_widget(btn)
        row.add_widget(lbl)
        return row

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------
    def _toggle_mask(self, mask_name):
        """Flip a single mask on/off."""
        self._set_mask(mask_name, not self.toggle_states[mask_name])

    def _set_mask(self, mask_name, state):
        """Set a single mask to True/False and update its button colour."""
        self.toggle_states[mask_name] = state
        btn = self._toggle_buttons.get(mask_name)
        if btn is not None:
            btn.text = 'ON' if state else 'OFF'
            btn.background_color = self._COLOR_ON if state else self._COLOR_OFF

    def get_active_masks(self):
        """Return the list of mask names that are currently ON."""
        return [name for name, state in self.toggle_states.items() if state]


# UI with debug box and event-values box
class PrototypeUI(BoxLayout):
    start_index = NumericProperty(1000); page_size = NumericProperty(2000); df = None
    def __init__(self, **kw):
        super().__init__(orientation='horizontal', **kw)

        left = ScrollView(size_hint_x=0.2360679775)
        lc = GridLayout(cols=1, size_hint_y=None, spacing=6, padding=6)
        lc.bind(minimum_height=lc.setter('height'))

        # File row
        lc.add_widget(Label(text='SQLite path (none):', size_hint_y=None, height=20))
        fr = BoxLayout(size_hint_y=None, height=36)
        self.file_input = TextInput(text=DEFAULT_PATH, multiline=False)
        load = Button(text='Load file', size_hint_x=None, width=120)
        load.bind(on_press=self.load_file)
        fr.add_widget(self.file_input); fr.add_widget(load); lc.add_widget(fr)

        # Simple navigation / slice controls
        self.status_label = Label(text='Status: idle', size_hint_y=None, height=24); lc.add_widget(self.status_label)
        self.rows_label = Label(text='Rows: N/A', size_hint_y=None, height=20)
        lc.add_widget(self.rows_label)
        nav = BoxLayout(size_hint_y=None, height=36)
        prev = Button(text='< Prev'); next = Button(text='Next >')
        prev.bind(on_press=self.prev_page); next.bind(on_press=self.next_page)
        nav.add_widget(prev); nav.add_widget(next); lc.add_widget(nav)

        # tiny debug box
        self.debug_box = Label(text='', size_hint_y=None, height=36)
        lc.add_widget(self.debug_box)

        row = BoxLayout(size_hint_y=None, height=30)
        row.add_widget(Label(text='Start idx:', size_hint_x=0.45))
        self.start_input = TextInput(text=str(self.start_index), multiline=False, size_hint_x=0.35)
        self.start_input.bind(on_text_validate=self.update_start_index)
        row.add_widget(self.start_input); lc.add_widget(row)
        ps_row = BoxLayout(size_hint_y=None, height=30)
        ps_row.add_widget(Label(text='Page size:', size_hint_x=0.45))
        self.page_input = TextInput(text=str(self.page_size), multiline=False, size_hint_x=0.35)
        self.page_input.bind(on_text_validate=self.update_page_size)
        ps_row.add_widget(self.page_input); lc.add_widget(ps_row)

        # --- Binary Mask Switchboard (replaces Feature 1 / Feature 2) ---
        lc.add_widget(Label(
            text='[b]Binary Mask Toggles (40)[/b]', markup=True,
            size_hint_y=None, height=24, font_size=13,
        ))
        self.mask_switchboard = MaskSwitchboard(MASK_CATEGORIES)
        lc.add_widget(self.mask_switchboard)

        # Mode toggle: Visualization Only vs Strategy Mode
        mode_row = BoxLayout(size_hint_y=None, height=30)
        mode_row.add_widget(Label(text='Mode:', size_hint_x=0.25))
        self.viz_cb = CheckBox(group='run_mode', active=True, size_hint_x=0.12)
        mode_row.add_widget(self.viz_cb)
        mode_row.add_widget(Label(text='Viz', size_hint_x=0.25))
        self.strat_cb = CheckBox(group='run_mode', active=False, size_hint_x=0.12)
        mode_row.add_widget(self.strat_cb)
        mode_row.add_widget(Label(text='Strategy', size_hint_x=0.26))
        lc.add_widget(mode_row)

        # Run Strategy Button
        sim_button = Button(text='Run Strategy', size_hint_y=None, height=40, background_color=(0.2, 0.6, 1, 1))
        sim_button.bind(on_press=self.on_run_button)
        lc.add_widget(sim_button)

        # Metrics — shows avg-user ROI and visible price stats; updated by plot_all().
        # This label is intentionally NEVER overwritten by the strategy runner so that
        # the "average user context" numbers remain visible alongside trade results.
        self.metrics_label = Label(text='Metrics:\nAvg User ROI: N/A\n\nVisible Avg Price: N/A\n0.61% of Avg: N/A', size_hint_y=None, height=120)
        lc.add_widget(self.metrics_label)

        # Strategy stats — populated only when Strategy mode runs.
        # Kept separate from metrics_label so avg-user numbers are never hidden.
        self.strategy_stats_label = Label(text='Strategy: (not run yet)', size_hint_y=None, height=60)
        lc.add_widget(self.strategy_stats_label)

        # Scrollable trade log (populated by run_strategy)
        lc.add_widget(Label(text='[b]Trade Log[/b]', markup=True, size_hint_y=None, height=20))
        self.trade_log = ScrollView(size_hint_y=None, height=200)
        self.trade_log_container = GridLayout(cols=1, size_hint_y=None, spacing=2)
        self.trade_log_container.bind(minimum_height=self.trade_log_container.setter('height'))
        self.trade_log.add_widget(self.trade_log_container)
        lc.add_widget(self.trade_log)

        left.clear_widgets(); left.add_widget(lc); self.add_widget(left)

        # Right plots
        right = BoxLayout(orientation='vertical')
        self.fig, (self.ax_price, self.ax_buy, self.ax_sell) = plt.subplots(3, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
        self.canvas_widget = FigureCanvasKivyAgg(self.fig)
        right.add_widget(self.canvas_widget)
        self.add_widget(right)

        # Initialize toggle_states as a reference to the switchboard's dict
        # so callers can still read self.toggle_states directly
        self.toggle_states = self.mask_switchboard.toggle_states

        # Holds the DataFrame loaded from the companion masks.sqlite file.
        # None until a SQLite file is successfully loaded and masks.sqlite is found.
        self.masks_df = None

        Clock.schedule_once(lambda dt: self._try_load_default(), 0.1)

    def set_toggle(self, toggle_id, state):
        # Delegate to switchboard if the key belongs to a mask; kept for
        # any external callers that still use the old API.
        if toggle_id in self.mask_switchboard.toggle_states:
            self.mask_switchboard._set_mask(toggle_id, state)
        else:
            self.toggle_states[toggle_id] = state
        self.debug_box.text = f"{toggle_id} set to {'ON' if state else 'OFF'}"

    def run_simulation(self, instance):
        """
        Visualize active mask signals as category-colored dots overlaid on the price chart.
        No trading logic is executed — this is a pure visualization tool.
        Each active mask is represented by dots in its category color wherever that mask is True.
        A legend identifies which color corresponds to which category.
        """
        if self.df is None:
            self.debug_box.text = "No data loaded"
            return

        # Guard: mask file must be loaded before visualization can run
        if self.masks_df is None:
            self.debug_box.text = "No mask file loaded. Please run aa.py first."
            return

        # Determine which masks are currently toggled ON by the user
        active_masks = set(self.mask_switchboard.get_active_masks())

        # 1. Create view slice for indicator data
        s = int(self.start_index)
        e = min(len(self.df), s + int(self.page_size))
        view = self.df.iloc[s:e].reset_index(drop=True)
        if view.empty:
            self.debug_box.text = "Empty view"
            return

        # 2. Slice the mask DataFrame to exactly the same rows as 'view'.
        #    reset_index(drop=True) ensures integer positions align with 'view'.
        view_masks = self.masks_df.iloc[s:e].reset_index(drop=True)
        # Sanity-check: both slices must cover the same number of rows
        if len(view_masks) != len(view):
            self.debug_box.text = "Mask/indicator length mismatch"
            return

        # 3. Define buy-side masks for legend bucketing
        BUY_MASKS = {
            "PHI_OOS_DOWN", "PHI_CROSS_UP", "PHI_TREND_UP",
            "BOOL_STD_D1", "BOOL_STD_D2", "BOOL_STD_D3",
            "CRAP5_7_D", "CRAP9_2_D", "CRAP14_8_D", "CRAP24_D",
            "CRAP5_7_TROUGH", "CRAP9_2_TROUGH", "CRAP14_8_TROUGH", "CRAP24_TROUGH",
            "CRAP5_7_MU", "CRAP9_2_MU", "CRAP14_8_MU", "CRAP24_MU",
            "RSI_OVERSOLD", "THETA_BUY",
        }

        # 4. Refresh base plot — this also updates metrics_label with avg-user ROI/price stats
        active_count = len(active_masks)  # used below for the debug status message
        self.plot_all()
        x_vals = view['TIME']

        # 5. Track buy/sell signal types plotted for the legend
        plotted_buy = False
        plotted_sell = False

        # 6. Colored dots for every active mask
        for mask_name in active_masks:
            if mask_name in view_masks.columns:
                color = MASK_COLORS.get(mask_name)
                if color:
                    if mask_name in BUY_MASKS:
                        plotted_buy = True
                    else:
                        plotted_sell = True
                    mask_bool = view_masks[mask_name].fillna(0).astype(bool)
                    if mask_bool.any():
                        self.ax_price.scatter(
                            x_vals[mask_bool].values,
                            view['CURRENT_RATE'][mask_bool].values,
                            color=color, s=15, marker='.', zorder=5, alpha=0.5
                        )

        # 7. Add legend for buy/sell signal types
        legend_elements = []
        if plotted_buy:
            legend_elements.append(Patch(facecolor="#228B22", label="Buy Signals"))
        if plotted_sell:
            legend_elements.append(Patch(facecolor="#CC0000", label="Sell Signals"))
        if legend_elements:
            self.ax_price.legend(handles=legend_elements, loc='upper right', fontsize='small')

        # 8. Refresh canvas to display all new annotations
        self.canvas_widget.draw()

        self.debug_box.text = f"Visualization: {active_count} masks active"

    def on_run_button(self, instance):
        """Dispatcher: run visualization or strategy depending on the mode toggle."""
        if self.strat_cb.active:
            self.run_strategy(instance)
        else:
            self.run_simulation(instance)

    def run_strategy(self, instance):
        """
        Peak-detection trading strategy using binary masks only.
        A buy peak occurs when the total count of active buy masks reaches a local
        maximum (greater than both the previous and next bar).  A sell peak is the
        mirror for sell masks.  Trades fire at those peaks; fees use the global FEE
        constant.  All existing visualization (dots, heat-map, bars) is preserved and
        trade entry/exit markers are added on top.
        """
        if self.df is None:
            self.debug_box.text = "No data loaded"
            return
        if self.masks_df is None:
            self.debug_box.text = "No mask file loaded. Please run aa.py first."
            return

        active_masks = set(self.mask_switchboard.get_active_masks())

        s = int(self.start_index)
        e = min(len(self.df), s + int(self.page_size))
        view = self.df.iloc[s:e].reset_index(drop=True)
        if view.empty:
            self.debug_box.text = "Empty view"
            return

        view_masks = self.masks_df.iloc[s:e].reset_index(drop=True)
        if len(view_masks) != len(view):
            self.debug_box.text = "Mask/indicator length mismatch"
            return

        # Same buy/sell mask classification used in run_simulation
        BUY_MASKS = {
            "PHI_OOS_DOWN", "PHI_CROSS_UP", "PHI_TREND_UP",
            "BOOL_STD_D1", "BOOL_STD_D2", "BOOL_STD_D3",
            "CRAP5_7_D", "CRAP9_2_D", "CRAP14_8_D", "CRAP24_D",
            "CRAP5_7_TROUGH", "CRAP9_2_TROUGH", "CRAP14_8_TROUGH", "CRAP24_TROUGH",
            "CRAP5_7_MU", "CRAP9_2_MU", "CRAP14_8_MU", "CRAP24_MU",
            "RSI_OVERSOLD", "THETA_BUY",
        }

        # --- Step 1: accumulate buy / sell counts per bar ---
        buy_counts = pd.Series(0, index=view_masks.index, dtype=float)
        sell_counts = pd.Series(0, index=view_masks.index, dtype=float)
        for mask_name in active_masks:
            if mask_name in view_masks.columns:
                col = view_masks[mask_name].fillna(0).astype(int)
                if mask_name in BUY_MASKS:
                    buy_counts += col
                else:
                    sell_counts += col

        # --- Step 2: peak detection — first bar where count reaches a new local high ---
        # We use two different Fibonacci look-back windows:
        #   BUY_PEAK_WINDOW  (55 bars) for buy peaks  — tighter horizon keeps entries responsive
        #   SELL_PEAK_WINDOW (89 bars) for sell peaks — wider horizon avoids premature exits
        #
        # How it works:
        #   rolling().max() computes the highest count seen in the last N bars.
        #   A peak fires when the current bar EQUALS that rolling high AND is strictly
        #   greater than the previous bar (rising-edge condition).  The rising-edge check
        #   means only the very FIRST bar of a plateau triggers a trade, not every bar
        #   that stays at the plateau level.
        rolling_max_buy  = buy_counts.rolling(BUY_PEAK_WINDOW,  min_periods=BUY_PEAK_WINDOW  // 2).max()
        rolling_max_sell = sell_counts.rolling(SELL_PEAK_WINDOW, min_periods=SELL_PEAK_WINDOW // 2).max()
        buy_peak  = (buy_counts  == rolling_max_buy)  & (buy_counts  > buy_counts.shift(1).fillna(0))
        sell_peak = (sell_counts == rolling_max_sell) & (sell_counts > sell_counts.shift(1).fillna(0))

        # --- Step 3: trade execution ---
        usd = 1000.0
        btc = 0.0
        position = 'USD'
        trades = []

        x_vals = view['TIME']

        # PHI_OOS column availability — checked once, outside the trade loop.
        # PHI_OOS (out-of-sample confirmation) is an independent signal computed on
        # data the other indicators have NOT seen.  Requiring it as a gate keeps us
        # from trading on in-sample noise.
        has_phi_oos_down = 'PHI_OOS_DOWN' in view_masks.columns
        has_phi_oos_up   = 'PHI_OOS_UP'   in view_masks.columns

        # ── Patience filter ───────────────────────────────────────────────────────
        # Goal: never buy at the EXACT bottom of a 2-minute (60-bar) capitulation dip.
        #
        # Why?  When price is at its rolling minimum it is still FALLING — the low
        # has not been confirmed yet.  We want to wait 1-2 bars until the price has
        # started to recover before committing capital.
        #
        # How:
        #   price_rolling_min  — the lowest price seen in any of the last 60 bars.
        #   at_2min_low        — True on any bar where the current price IS that low.
        #   low_occurred_recently — True when a low was present 1 or 2 bars AGO
        #                           (shift(1) moves the window back so the current bar
        #                            is never counted as "recently past").
        #   confirmed_exit_from_low — True when BOTH conditions hold:
        #       1. We are NOT currently at the rolling low  (price has moved up)
        #       2. A low WAS present 1-2 bars ago           (the dip just happened)
        #   Only bars where confirmed_exit_from_low is True are allowed to buy.
        price_rolling_min = view['CURRENT_RATE'].rolling(60, min_periods=30).min()
        at_2min_low = (view['CURRENT_RATE'] == price_rolling_min)
        low_occurred_recently = at_2min_low.rolling(2).max().shift(1).fillna(False).astype(bool)
        confirmed_exit_from_low = (~at_2min_low) & low_occurred_recently

        # avg_buy_price tracks the price we paid on entry.
        # Used exclusively by the profit-target check — reset to None when flat.
        avg_buy_price = None

        # ── Main trade loop ───────────────────────────────────────────────────────
        # Each bar (i) we:
        #   1. Read the current price and active mask counts.
        #   2. Combine the peak-detection signal with the PHI_OOS gate.
        #   3. Check the profit target FIRST (fast path — exits immediately).
        #   4. Otherwise check the buy or sell condition (signal-driven path).
        for i in range(len(view)):
            price = float(view.loc[i, 'CURRENT_RATE'])

            # Total number of BUY-side masks that are True on this bar.
            current_buy_count  = buy_counts.iloc[i]
            # Total number of SELL-side masks that are True on this bar.
            current_sell_count = sell_counts.iloc[i]

            # Read the out-of-sample confirmation flags for this bar.
            # PHI_OOS_DOWN  → OOS model says market is trending DOWN  (buy dip signal)
            # PHI_OOS_UP    → OOS model says market is trending UP     (sell peak signal)
            # If the column doesn't exist in the mask file, default to False so trades
            # are suppressed rather than accidentally firing on every bar.
            phi_oos_down = bool(view_masks['PHI_OOS_DOWN'].iloc[i]) if has_phi_oos_down else False
            phi_oos_up   = bool(view_masks['PHI_OOS_UP'].iloc[i])   if has_phi_oos_up   else False

            # Final composite signals.
            # buy_signal  = "buy-count just hit a Fibonacci-window peak"
            #               AND "OOS model confirms a downtrend dip to buy into"
            # sell_signal = "sell-count just hit a Fibonacci-window peak"
            #               AND "OOS model confirms an uptrend peak to sell into"
            buy_signal  = buy_peak.iloc[i]  and phi_oos_down
            sell_signal = sell_peak.iloc[i] and phi_oos_up

            # ── PRIORITY 1: profit target ─────────────────────────────────────────
            # Check this BEFORE the normal signal path so a profitable position is
            # never held past the target just because no sell peak is firing.
            # If we're in BTC and our unrealised gain has reached PROFIT_TARGET_PCT,
            # sell immediately and skip the rest of this bar's logic (continue).
            if position == 'BTC' and btc > 0 and avg_buy_price is not None:
                profit_pct = (price - avg_buy_price) / avg_buy_price * 100
                if profit_pct >= PROFIT_TARGET_PCT:
                    # Lock in the gain: convert BTC back to USD, deduct trading fee.
                    usd = btc * price * (1.0 - FEE)
                    trades.append(('SELL', i, price, usd, btc))
                    btc = 0.0
                    position = 'USD'
                    avg_buy_price = None
                    continue  # move to next bar — no further action needed this bar

            # ── PRIORITY 2: buy condition ─────────────────────────────────────────
            # ALL of the following must be True to open a position:
            #   buy_signal              — peak AND OOS confirmation (see above)
            #   confirmed_exit_from_low — price just exited a rolling-low dip (patience)
            #   position == 'USD'       — we must not already be in a position
            #   usd > 0                 — we need money to spend
            #   sell_count <= buy_count — buy-side masks DOMINATE (no counter-trend longs)
            if (buy_signal and confirmed_exit_from_low.iloc[i]
                    and position == 'USD' and usd > 0
                    and current_sell_count <= current_buy_count):
                # Convert all USD to BTC, paying the taker fee on the way in.
                btc = usd / (price * (1.0 + FEE))
                trades.append(('BUY', i, price, usd, btc))
                usd = 0.0
                position = 'BTC'
                avg_buy_price = price  # remember entry price for profit-target tracking

            # ── PRIORITY 3: signal-driven sell ───────────────────────────────────
            # Fires when a sell peak is confirmed AND we hold BTC AND sell masks dominate.
            # This is the "normal" exit path when the profit target hasn't been hit.
            elif (sell_signal and position == 'BTC' and btc > 0
                    and current_buy_count <= current_sell_count):
                # Convert all BTC back to USD, paying the taker fee on the way out.
                usd = btc * price * (1.0 - FEE)
                trades.append(('SELL', i, price, usd, btc))
                btc = 0.0
                position = 'USD'
                avg_buy_price = None

        # Force-close any open position at end of view
        if position == 'BTC' and btc > 0:
            last_price = float(view.iloc[-1]['CURRENT_RATE'])
            usd = btc * last_price * (1.0 - FEE)
            trades.append(('SELL', len(view) - 1, last_price, usd, btc))
            btc = 0.0
            position = 'USD'
            avg_buy_price = None

        final_usd = usd
        roi = ((final_usd - 1000.0) / 1000.0) * 100
        buy_trade_count = sum(1 for t in trades if t[0] == 'BUY')

        # --- Step 4: draw base chart + existing visualization (dots, heat-map, bars) ---
        self.plot_all()

        plotted_buy = False
        plotted_sell = False

        for mask_name in active_masks:
            if mask_name in view_masks.columns:
                color = MASK_COLORS.get(mask_name)
                if color:
                    if mask_name in BUY_MASKS:
                        plotted_buy = True
                    else:
                        plotted_sell = True
                    mask_bool = view_masks[mask_name].fillna(0).astype(bool)
                    if mask_bool.any():
                        self.ax_price.scatter(
                            x_vals[mask_bool].values,
                            view['CURRENT_RATE'][mask_bool].values,
                            color=color, s=15, marker='.', zorder=5, alpha=0.5
                        )

        # Legend for mask signal types
        legend_elements = []
        if plotted_buy:
            legend_elements.append(Patch(facecolor="#228B22", label="Buy Signals"))
        if plotted_sell:
            legend_elements.append(Patch(facecolor="#CC0000", label="Sell Signals"))

        # --- Step 5: overlay trade entry / exit markers ---
        for trade in trades:
            xi = trade[1]
            if 0 <= xi < len(x_vals):
                x_pos_t = x_vals.iloc[xi]
                if trade[0] == 'BUY':
                    self.ax_price.scatter(x_pos_t, trade[2], color='lime', marker='^', s=120, zorder=11, edgecolors='darkgreen', linewidths=0.8)
                else:
                    self.ax_price.scatter(x_pos_t, trade[2], color='red', marker='v', s=120, zorder=11, edgecolors='darkred', linewidths=0.8)

        # Add trade markers to legend
        legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=9, label='Buy (peak)'))
        legend_elements.append(Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markeredgecolor='darkred', markersize=9, label='Sell (peak)'))
        if legend_elements:
            self.ax_price.legend(handles=legend_elements, loc='upper right', fontsize='small')

        self.canvas_widget.draw()

        # Update the STRATEGY stats label (separate from metrics_label so that
        # avg-user ROI / visible-price stats set by plot_all() are never hidden).
        self.strategy_stats_label.text = (
            f"Strategy ROI: {roi:.2f}%  |  Buys: {buy_trade_count}  |  Final: ${final_usd:.2f}"
        )
        self.debug_box.text = f"Strategy: {len(trades)} signals, ROI={roi:.2f}%"

        # --- Step 6: populate trade log ---
        self.trade_log_container.clear_widgets()
        # Calculate pair profits first so we can annotate each SELL
        pair_profits = {}
        pair_num = 0
        for k in range(len(trades) - 1):
            if trades[k][0] == 'BUY' and trades[k + 1][0] == 'SELL':
                buy_price = trades[k][2]
                sell_price = trades[k + 1][2]
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                pair_num += 1
                pair_profits[k + 1] = (pair_num, profit_pct)

        wins = sum(1 for _, pct in pair_profits.values() if pct > 0)
        total_pairs = len(pair_profits)
        win_rate = (wins / total_pairs * 100) if total_pairs > 0 else 0.0

        summary = Label(
            text=f"Pairs: {total_pairs}  Wins: {wins}  Win%: {win_rate:.0f}%",
            size_hint_y=None, height=20, font_size=11,
        )
        self.trade_log_container.add_widget(summary)

        for j, trade in enumerate(trades):
            if trade[0] == 'BUY':
                detail = f"USD ${trade[3]:.2f} → BTC {trade[4]:.6f}"
            else:
                pp = pair_profits.get(j)
                pnl = f"  P&L: {pp[1]:+.2f}%" if pp else ""
                detail = f"BTC {trade[4]:.6f} → USD ${trade[3]:.2f}{pnl}"
            trade_text = f"{j + 1}. {trade[0]} bar {trade[1]} @ ${trade[2]:.2f} | {detail}"
            row_lbl = Label(text=trade_text, size_hint_y=None, height=18, font_size=10)
            self.trade_log_container.add_widget(row_lbl)

    def _try_load_default(self):
        try:
            if DEFAULT_PATH:
                self.file_input.text = DEFAULT_PATH
                self.load_file(None)
        except Exception:
            pass

    def load_file(self, instance):
        path = self.file_input.text.strip()
        if not path:
            self.status_label.text = "Status: no path"
            self.debug_box.text = "No path provided"
            return
        try:
            # Load from SQLite instead of Feather
            conn = sqlite3.connect(path)
            df = pd.read_sql_query("SELECT * FROM indicators ORDER BY TIME", conn)
            conn.close()
            self.df = df
            self.start_index = 1000 if len(df) > 1000 else 0
            self.page_size = min(2000, len(df))
            self.start_input.text = str(self.start_index)
            self.page_input.text = str(self.page_size)
            self.status_label.text = f"Loaded: {path}"
            self.rows_label.text = f"Rows: {len(df)}"

            # --- Load companion mask file (masks.sqlite) ---
            # Look for masks.sqlite in the same directory as the indicator file.
            mask_path = os.path.join(os.path.dirname(path), 'masks.sqlite')
            if os.path.exists(mask_path):
                # Read all mask rows from the masks table in masks.sqlite.
                # Use a context manager so the connection is always closed cleanly.
                try:
                    with sqlite3.connect(mask_path) as mask_conn:
                        self.masks_df = pd.read_sql_query("SELECT * FROM binary_masks", mask_conn)
                except Exception:
                    self.masks_df = None
                    self.debug_box.text = "masks.sqlite found but 'binary_masks' table missing"
                    self.on_redraw_dots(None)
                    return
                # Restore stored row index if present so iloc slicing stays aligned
                if 'index' in self.masks_df.columns:
                    self.masks_df = self.masks_df.set_index('index')
                # Warn if the mask row count doesn't match the indicator row count
                if len(self.masks_df) != len(self.df):
                    self.debug_box.text = (
                        f"Warning: Mask rows ({len(self.masks_df)}) "
                        f"!= indicators ({len(self.df)})"
                    )
                else:
                    # Report the number of mask columns that were loaded
                    self.debug_box.text = f"Loaded {len(self.masks_df.columns)} masks"
            else:
                # No masks.sqlite found next to the indicator file
                self.masks_df = None
                self.debug_box.text = "No mask file found - run aa.py first"
            # --- end mask loading ---

            # initial draw
            self.on_redraw_dots(None)
        except Exception as ex:
            msg = f"Load error: {ex}"
            self.df = None
            # Clear any previously loaded masks so simulation can't use stale data
            self.masks_df = None
            self.status_label.text = msg
            try:
                self.rows_label.text = "Rows: N/A"
                self.debug_box.text = msg
                # FIXED: Removed self.event_text.text = msg
                self.metrics_label.text = msg
            except Exception:
                pass
    def prev_page(self,instance):
        if self.df is None: return
        self.start_index=max(0,int(self.start_index)-int(self.page_size)); self.start_input.text=str(self.start_index); self.on_redraw_dots(None)
    def next_page(self,instance):
        if self.df is None: return
        self.start_index=min(max(0,len(self.df)-1),int(self.start_index)+int(self.page_size)); self.start_input.text=str(self.start_index); self.on_redraw_dots(None)
    def update_start_index(self,instance):
        try: v=int(self.start_input.text); self.start_index=max(0,v)
        except Exception: self.start_input.text=str(self.start_index)
        self.on_redraw_dots(None)
    def update_page_size(self,instance):
        try: v=int(self.page_input.text); self.page_size=v
        except Exception: self.page_input.text=str(self.page_size)
        self.on_redraw_dots(None)



    def on_redraw_dots(self, instance):
        # Core plotting routine: price (with STD clouds), theta (middle), meta (bottom), legend aggregation
        if self.df is None:
            self.canvas_widget.draw(); return
        s = int(self.start_index)
        e = min(len(self.df), s + int(self.page_size))
        view = self.df.iloc[s:e].reset_index(drop=True)

        # compute flags and events, catching any exceptions and writing to debug box
        try:
            flags, view = compute_flags_dataframe(view)
            self.current_flags = flags
            self.pattern_events = compute_pattern_events(view)
            self._last_view = view
            ev_text = (
                f"LR_WINDOWS={LR_WINDOWS}\n"
            )
            # FIXED: Removed self.event_values.text = ev_text
            self.debug_box.text = ""
        except Exception as ex:
            msg = f"compute error: {ex}"
            self.debug_box.text = msg
            # FIXED: Removed self.event_text.text = msg
            return

        # draw the plots
        try:
            self.plot_all()
        except Exception as ex:
            self.debug_box.text = f"plot error: {ex}"


    def plot_all(self):
        # Core plotting routine: 4 panels - price (top), CRAP (2nd), RSI (3rd), THETA (4th)
        if self.df is None:
            self.canvas_widget.draw(); return
        s = int(self.start_index)
        e = min(len(self.df), s + int(self.page_size))
        view = self.df.iloc[s:e].reset_index(drop=True)
        if view.empty:
            self.canvas_widget.draw(); return

 

        # recompute flags / events
        try:
            flags, view = compute_flags_dataframe(view)
            self.current_flags = flags
            self.pattern_events = compute_pattern_events(view)
            self._last_view = view
        except Exception as ex:
            self.debug_box.text = f"flags compute error: {ex}"
            return


        x = view['TIME']
        price_col = 'CURRENT_RATE'
        prices = pd.to_numeric(view[price_col], errors='coerce').to_numpy()


        # clear axes
        self.ax_price.clear(); self.ax_buy.clear(); self.ax_sell.clear()

        # FIXED: UPDATED COLORS FOR BOLLINGER BANDS - More visible, progressive blues
        CLOUD_COLORS = [
            ('#2A73B9', '#7FB3E6', 0.32),  # inner cloud: edge, fill, alpha
            ('#4A90E2', '#A5C8F0', 0.26),  # middle cloud: edge, fill, alpha - MORE VISIBLE
            ('#6AAEE6', '#C0E0FF', 0.20)   # outer cloud: edge, fill, alpha - STILL VISIBLE
        ]

        def plot_std_cloud(up_col, down_col, tier=1):
            """Robust cloud drawer with palette tiers (1 inner, 2 middle, 3 outer)."""
            x_arr = np.asarray(x)
            if tier < 1 or tier > len(CLOUD_COLORS): tier = 1
            edge_color, face_color, alpha_fill = CLOUD_COLORS[tier-1]

            up = pd.to_numeric(view[up_col], errors='coerce').to_numpy() if up_col in view.columns else None
            down = pd.to_numeric(view[down_col], errors='coerce').to_numpy() if down_col in view.columns else None

            if up is None and down is None:
                return

            # If both exist, mask to paired finite points for solid fill; otherwise outline dotted
            if up is not None and down is not None:
                valid = (~np.isnan(up)) & (~np.isnan(down))
                if valid.any():
                    up_f = np.where(valid, up, np.nan)
                    down_f = np.where(valid, down, np.nan)
                    self.ax_price.fill_between(x_arr, down_f, up_f, color=face_color, alpha=alpha_fill, zorder=2, linewidth=0)
                    # FIXED: THICKER EDGES - as dark/thick as price line (2.2)
                    lw = 2.2
                    self.ax_price.plot(x_arr, up_f, color=edge_color, linewidth=lw, zorder=4, label=up_col)
                    self.ax_price.plot(x_arr, down_f, color=edge_color, linewidth=lw, zorder=4, label=down_col)
                    return
                # fallthrough: draw dotted outlines for context
            if up is not None:
                self.ax_price.plot(x_arr, up, linestyle=':', color=edge_color, linewidth=1.0, zorder=4, label=up_col)
            if down is not None:
                self.ax_price.plot(x_arr, down, linestyle=':', color=edge_color, linewidth=1.0, zorder=4, label=down_col)

        # Call clouds (inner first so outer lies under) - UPDATED FOR 3 TIERS
        plot_std_cloud('STD_U1', 'STD_D1', tier=1)
        plot_std_cloud('STD_U2', 'STD_D2', tier=2)
        plot_std_cloud('STD_U3', 'STD_D3', tier=3)
        
        # Price line
        price_line, = self.ax_price.plot(x, prices, color=PRICE_COLOR, linewidth=2.2, label='Price', zorder=6)
        autoscale_price_axis(self.ax_price, x, prices, pad_frac=0.02)


                # SILENCED: Comment out debug XTICKS print
        # ticks = self.ax_price.get_xticks()
        # fixed-format with up to 8 fractional digits, trim trailing zeros
        # tick_texts = [("{:.8f}".format(float(t)).rstrip('0').rstrip('.')) for t in ticks]
        # print("XTICKS (raw positions):", tick_texts)


        # PHI lines - ADDED AS PER FEEDBACK
        PHI_COLORS = {'PHI_5_7': 'indigo', 'PHI_9_2': 'violet', 'PHI_14_8': 'darkorange', 'PHI_24': 'gold'}
        phi_lines = []
        for col, color in PHI_COLORS.items():
            if col in view.columns:
                vals = pd.to_numeric(view[col], errors='coerce')
                if not vals.isna().all():
                    line, = self.ax_price.plot(x, vals, '-', color=color, alpha=0.9, lw=2.2, label=col, zorder=4)
                    phi_lines.append(line)

        self.ax_price.set_ylabel('Price'); self.ax_price.grid(True)

        # Buy/Sell count panels — plot active mask counts with Fibonacci rolling averages.
        # If mask data is not yet loaded, the panels render empty with correct labels/grid.
        BUY_MASKS_SET = {
            "PHI_OOS_DOWN", "PHI_CROSS_UP", "PHI_TREND_UP",
            "BOOL_STD_D1", "BOOL_STD_D2", "BOOL_STD_D3",
            "CRAP5_7_D", "CRAP9_2_D", "CRAP14_8_D", "CRAP24_D",
            "CRAP5_7_TROUGH", "CRAP9_2_TROUGH", "CRAP14_8_TROUGH", "CRAP24_TROUGH",
            "CRAP5_7_MU", "CRAP9_2_MU", "CRAP14_8_MU", "CRAP24_MU",
            "RSI_OVERSOLD", "THETA_BUY",
        }
        if self.masks_df is not None:
            view_masks_pa = self.masks_df.iloc[s:e].reset_index(drop=True)
            if len(view_masks_pa) == len(view):
                active_masks_pa = set(self.mask_switchboard.get_active_masks())
                buy_counts_pa = pd.Series(0, index=view_masks_pa.index, dtype=float)
                sell_counts_pa = pd.Series(0, index=view_masks_pa.index, dtype=float)
                for mask_name in active_masks_pa:
                    if mask_name in view_masks_pa.columns:
                        col = view_masks_pa[mask_name].fillna(0).astype(int)
                        if mask_name in BUY_MASKS_SET:
                            buy_counts_pa += col
                        else:
                            sell_counts_pa += col

                # Four distinct-colored Fibonacci MAs — same periods/colors on both panels
                # so each line is immediately identifiable regardless of panel.
                PANEL_AVERAGES = [
                    (144,  '#800080'),  # Purple
                    (377,  '#FFD700'),  # Gold
                    (987,  '#008080'),  # Teal
                    (1597, '#000000'),  # Black
                ]

                # Panel A: Buy signal counts
                self.ax_buy.plot(x, buy_counts_pa.values, color='darkgreen', linewidth=1.5, label='Buy Count')
                for period, color in PANEL_AVERAGES:
                    if len(buy_counts_pa) >= period // 2:
                        ma = buy_counts_pa.rolling(period, min_periods=period // 2).mean()
                        self.ax_buy.plot(x, ma.values, color=color, linewidth=1, alpha=0.7, label=f'{period}-MA')

                # Panel B: Sell signal counts
                self.ax_sell.plot(x, sell_counts_pa.values, color='darkred', linewidth=1.5, label='Sell Count')
                for period, color in PANEL_AVERAGES:
                    if len(sell_counts_pa) >= period // 2:
                        ma = sell_counts_pa.rolling(period, min_periods=period // 2).mean()
                        self.ax_sell.plot(x, ma.values, color=color, linewidth=1, alpha=0.7, label=f'{period}-MA')

        self.ax_buy.set_ylabel('Buy Count')
        self.ax_buy.legend(loc='upper left', fontsize='x-small')
        self.ax_buy.grid(True, alpha=0.3)
        self.ax_sell.set_ylabel('Sell Count')
        self.ax_sell.legend(loc='upper left', fontsize='x-small')
        self.ax_sell.grid(True, alpha=0.3)

        # Metrics text formatting (clean)
        try:
            first_price = float(prices[0])
            last_price = float(prices[-1])
            avg_user_roi = (last_price / first_price - 1.0) * 100.0
            roi_text = f"{avg_user_roi:.2f}%"
            avg_price = float(np.nanmean(prices))
            pct_value = avg_price * 0.0061
            self.metrics_label.text = (
                f"Metrics:\nAvg User ROI: {roi_text}\n\n"
                f"Visible Avg Price: {avg_price:,.2f}\n0.61% of Avg: {pct_value:,.2f}"
            )
        except Exception:
            pass

        try:
            h, l = self.ax_price.get_legend_handles_labels()
            # map raw plot labels to user-facing legend names (legend-only change)
            remap = {
                'STD_U1': 'Bollinger 1', 'STD_D1': 'Bollinger 1',
                'STD_U2': 'Bollinger 2', 'STD_D2': 'Bollinger 2',
                'STD_U3': 'Bollinger 3', 'STD_D3': 'Bollinger 3'
            }
            seen = set(); h2 = []; l2 = []
            for hh, ll in zip(h, l):
                mapped = remap.get(ll, ll)           # map only legend text, leave others alone
                if mapped not in seen:
                    seen.add(mapped)
                    h2.append(hh)
                    l2.append(mapped)
            if h2:
                self.ax_price.legend(h2, l2, loc='upper left', fontsize='small')
        except Exception:
            pass

        # DISABLED: TIME FORMATTING COMPLETELY - use raw timestamps
        # All datetime conversion code removed to prevent crashes
        # Raw timestamps are acceptable for now

        self.fig.tight_layout(); self.canvas_widget.draw()

if __name__ == '__main__':
    class PrototypeApp(App):
        def build(self): return PrototypeUI()
    PrototypeApp().run()

