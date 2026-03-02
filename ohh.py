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

# Category colors for mask visualization
MASK_COLORS = {
    "PHI Gates (2)":        "purple",
    "PHI Cross/Trend (4)":  "blue",
    "Bollinger Walls (6)":  "cyan",
    "CRAP Breath (24)":     "orange",
    "RSI Pulse (2)":        "pink",
    "THETA Spine (2)":      "magenta",
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

        # Run Trading Simulation Button
        sim_button = Button(text='Run Trading Simulation', size_hint_y=None, height=40, background_color=(0.2, 0.6, 1, 1))
        sim_button.bind(on_press=self.run_simulation)
        lc.add_widget(sim_button)

        # Metrics
        self.metrics_label = Label(text='Metrics:\nAvg User ROI: N/A\n\nVisible Avg Price: N/A\n0.61% of Avg: N/A', size_hint_y=None, height=120)
        lc.add_widget(self.metrics_label)

        left.clear_widgets(); left.add_widget(lc); self.add_widget(left)

        # Right plots
        right = BoxLayout(orientation='vertical')
        self.fig, (self.ax_price, self.ax_crap, self.ax_rsi, self.ax_theta) = plt.subplots(4, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
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

        # 3. Build a reverse mapping: mask column name → category name
        mask_to_category = {}
        for cat_name, masks in MASK_CATEGORIES.items():
            for mask_name, _ in masks:
                mask_to_category[mask_name] = cat_name

        # 4. Update metrics label (visualization mode — no trading)
        active_count = len(active_masks)
        self.metrics_label.text = (
            f"MASK VISUALIZATION:\n"
            f"Active masks: {active_count}\n"
            f"View rows: {len(view)}"
        )

        # 5. Refresh base plot then overlay mask signal dots for ALL active masks
        self.plot_all()
        x_vals = view['TIME']

        # 6. Track which categories have been plotted for the legend
        plotted_categories = set()

        # 7. Colored dots for every active mask
        for mask_name in active_masks:
            if mask_name in view_masks.columns:
                category = mask_to_category.get(mask_name)
                if category and category in MASK_COLORS:
                    color = MASK_COLORS[category]
                    plotted_categories.add(category)
                    dot_indices = view_masks[view_masks[mask_name]].index
                    for dot_idx in dot_indices:
                        self.ax_price.scatter(
                            x_vals.iloc[dot_idx],
                            view['CURRENT_RATE'].iloc[dot_idx],
                            color=color, s=15, marker='.', zorder=5, alpha=0.5
                        )

        # 8. Add legend for active categories
        if plotted_categories:
            legend_elements = [Patch(facecolor=MASK_COLORS[cat], label=cat)
                               for cat in plotted_categories]
            self.ax_price.legend(handles=legend_elements, loc='upper right', fontsize='small')

        # 9. Refresh canvas to display all new annotations
        self.canvas_widget.draw()

        self.debug_box.text = f"Visualization: {active_count} masks active"


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
                        self.masks_df = pd.read_sql_query("SELECT * FROM masks", mask_conn)
                except Exception:
                    self.masks_df = None
                    self.debug_box.text = "masks.sqlite found but 'masks' table missing"
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
        self.ax_price.clear(); self.ax_crap.clear(); self.ax_rsi.clear(); self.ax_theta.clear()

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

        # CRAP panel (2nd panel) - FIXED: REGULAR LINE PLOTS, NO STEP, NO FLAG SHADING
        self.ax_crap.clear()
        CRAP_COLORS = {'CRAP_5_7': 'indigo', 'CRAP_9_2': 'violet', 'CRAP_14_8': 'darkorange', 'CRAP_24': 'gold'}
        crap_lines = []
        for col, color in CRAP_COLORS.items():
            if col in view.columns:
                vals = pd.to_numeric(view[col], errors='coerce')
                if not vals.isna().all():
                    line, = self.ax_crap.plot(x, vals, color=color, linewidth=1.2, label=col)
                    crap_lines.append(line)
        # FIXED: ADD THRESHOLD LINES - solid black at 0, green/red dotted at extremes
        self.ax_crap.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)
        self.ax_crap.axhline(-0.003105620015142, color='green', linestyle=':', linewidth=1.0, alpha=0.8)
        self.ax_crap.axhline(0.003105620015142, color='red', linestyle=':', linewidth=1.0, alpha=0.8)
        self.ax_crap.set_ylabel('CRAP'); self.ax_crap.set_yticks([])

        # RSI panel (3rd panel) - UPDATED TO PLOT ACTUAL RSI VALUES AS LINES
        self.ax_rsi.clear()
        xvals = np.array(x)
        rsi_lines = []
        if 'RSI_21' in view.columns:
            vals = pd.to_numeric(view['RSI_21'], errors='coerce')
            if not vals.isna().all():
                line, = self.ax_rsi.plot(xvals, vals, color='blue', linewidth=1.2, label='RSI_21')
                rsi_lines.append(line)
        if 'RSI_34' in view.columns:
            vals = pd.to_numeric(view['RSI_34'], errors='coerce')
            if not vals.isna().all():
                line, = self.ax_rsi.plot(xvals, vals, color='orange', linewidth=1.2, label='RSI_34')
                rsi_lines.append(line)
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.axhline(9.01, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        self.ax_rsi.axhline(89.99, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        # FIXED: RSI shading based on BOTH RSI_21 and RSI_34 being extreme
        if 'RSI_21' in view.columns and 'RSI_34' in view.columns:
            rsi21 = pd.to_numeric(view['RSI_21'], errors='coerce')
            rsi34 = pd.to_numeric(view['RSI_34'], errors='coerce')
            # Green when BOTH are below 9.01, red when BOTH are above 89.99
            low_mask = (~rsi21.isna()) & (~rsi34.isna()) & (rsi21 < 9.01) & (rsi34 < 9.01)
            high_mask = (~rsi21.isna()) & (~rsi34.isna()) & (rsi21 > 89.99) & (rsi34 > 89.99)
            if low_mask.any():
                spans_low = spans_from_mask(low_mask, xvals)
                if spans_low:
                    self.ax_rsi.broken_barh(spans_low, (0-0.4,0.8), facecolors='green', edgecolors='k', zorder=1, alpha=0.3)
            if high_mask.any():
                spans_high = spans_from_mask(high_mask, xvals)
                if spans_high:
                    self.ax_rsi.broken_barh(spans_high, (0-0.4,0.8), facecolors='red', edgecolors='k', zorder=1, alpha=0.3)
        self.ax_rsi.set_ylabel('RSI'); self.ax_rsi.set_yticks([])

        # THETA panel (4th panel)
        # prepare theta history so long windows have enough prior points
        maxw = max(LR_WINDOWS)
        # Theta / VAL color map (ordered by window size in LR_WINDOWS)
        VAL_COLORS = {
            144: 'indigo',       # shortest
            233: 'violet',
            377: 'darkorange',
            610: 'gold'          # longest (most prominent)
        }
        # theta labels mapping (SHORT, MEDIUM, LONG, X-LONG)
        THETA_LABELS = {144: 'Short', 233: 'Medium', 377: 'Long', 610: 'X-Long'}
        start_for_theta = max(0, s - maxw + 1)
        prices_for_theta = pd.to_numeric(self.df.iloc[start_for_theta:e][price_col], errors='coerce').to_numpy()
        thetas = compute_theta_series(prices_for_theta, windows=LR_WINDOWS)

        # make theta subplot visible and plot lines
        self.ax_theta.set_visible(True); self.ax_theta.clear()
        self.ax_theta.set_ylim(-90,90); self.ax_theta.set_ylabel('Theta (deg)')
        x_arr = np.asarray(x)
        offset = s - start_for_theta
        theta_lines=[]
        for w in LR_WINDOWS:
            arr_full = thetas.get(w, np.full(len(prices_for_theta), np.nan))
            vis = arr_full[offset: offset + len(prices)]
            if len(vis) < len(prices):
                vis = np.concatenate([vis, np.full(len(prices)-len(vis), np.nan)])
            view[f"theta{w}"] = vis
            color = VAL_COLORS.get(w, 'tab:blue')
            line, = self.ax_theta.plot(x_arr, vis, color=color, linewidth=1.2, label=THETA_LABELS.get(w, f'theta{w}'))
            theta_lines.append(line)
        # FIXED: CORRECTED THRESHOLDS - Red at +9.01°, Green at -9.01°
        self.ax_theta.axhline(9.01, color='red', linestyle='--', linewidth=0.8)
        self.ax_theta.axhline(-9.01, color='green', linestyle='--', linewidth=0.8)

        # Build masks: four cases (above/below) × (increasing/decreasing across windows)
        vals_matrix = np.vstack([view[f"theta{w}"].to_numpy(dtype=float) for w in LR_WINDOWS]).T  # shape (N,4) order: [144,233,377,610]
        finite = np.isfinite(vals_matrix).all(axis=1)
        above = finite & (vals_matrix.min(axis=1) >= POS_THETA_DEG)
        below = finite & (vals_matrix.max(axis=1) <= NEG_THETA_DEG)
        inc = np.all(np.diff(vals_matrix, axis=1) > 0, axis=1)   # strictly increasing across windows (144->610)
        dec = np.all(np.diff(vals_matrix, axis=1) < 0, axis=1)   # strictly decreasing across windows

        m_above_inc = above & inc
        m_above_dec = above & dec
        m_below_inc = below & inc
        m_below_dec = below & dec

        # helper to shade contiguous spans on theta axis
        def shade_mask(mask, color, alpha=0.12):
            spans = spans_from_mask(mask, x_arr)
            for xs,wid in spans:
                try: self.ax_theta.axvspan(xs, xs+wid, color=color, alpha=alpha, zorder=1)
                except Exception: pass

        shade_mask(m_above_inc, 'salmon', 0.18)   # ABOVE + increasing (610 greatest)
        shade_mask(m_above_dec, 'red', 0.12)  # ABOVE + decreasing
        shade_mask(m_below_inc, 'lime', 0.18) # BELOW + increasing
        shade_mask(m_below_dec, 'green', 0.12)    # BELOW + decreasing

        # legend for theta
        try:
            if theta_lines:
                labels = [ln.get_label() for ln in theta_lines]
                self.ax_theta.legend(theta_lines, labels, loc='upper left', fontsize='small')
        except Exception:
            pass

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

