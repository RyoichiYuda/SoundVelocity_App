"""音速測定を繰り返し実行するためのTkinter UI本体。

このファイルがアプリケーションの起動入口であり、主に次の役割を持つ。

- 数値入力欄、ファイル選択、実行ボタンを配置する
- 日常計測用の基本タブと、全機能を持つアドバンスドタブを提供する
- 入力設定をJSONへ保存し、起動時に自動で復元する
- CSVまたはPicoScopeから参照波形と実測波形を個別に取得する
- 入力値をSI単位へ変換して ``analysis_pipeline.py`` へ渡す
- 解析を別スレッドで実行し、計算中も画面が固まらないようにする
- 取得波形、フィルター波形、窓付き参照波形、照合スコアを描画する
- 一致時間、最良スコア、音速を画面へ表示する

信号処理の計算式はこのファイルへ書かず、UIと解析を分離している。
"""

from __future__ import annotations

import math
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
import pypicosdk as psdk
from matplotlib import font_manager, rcParams
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# このフォルダーだけを別の場所へコピーしても、同梱モジュールを読み込めるように
# main.py自身の場所をimport検索対象にする。起動時のカレントディレクトリには依存しない。
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from analysis_pipeline import (
    MATCH_CORRELATION,
    MATCH_SQUARED_ERROR,
    AnalysisParameters,
    AnalysisResult,
    butter_bandpass_filter,
    run_analysis,
)
from app_settings import ApplicationSettings, load_settings, save_settings
from data_sources import CsvDataSource, SignalData
from picoscope_acquisition import acquire_waveform
from window_functions import WindowParameters, generate_window, save_window_csv


# 既定CSVもこのフォルダー内へ同梱するため、リポジトリの構造に依存しない。
# UIの「選択」ボタンから、同梱外のCSVへ変更することもできる。
DEFAULT_MEASUREMENT_PATH = APP_DIR / "data" / "hand-0001.csv"
DEFAULT_REFERENCE_PATH = APP_DIR / "data" / "zero_ref.csv"
SETTINGS_PATH = APP_DIR / "settings.json"


def resolve_input_path(path: str | Path) -> Path:
    """入力パスを絶対パスへ変換し、相対指定はアプリフォルダーを基準にする。

    GUIをアプリフォルダー外のカレントディレクトリから起動しても、
    ``data/...`` のような入力が同じファイルを指すようにする。
    """

    resolved_path = Path(path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = APP_DIR / resolved_path
    return resolved_path.resolve()


def initial_directory_for_input(path: str | Path) -> Path:
    """ファイル選択の開始フォルダーを、必ず存在する場所から選ぶ。"""

    if not str(path).strip():
        return APP_DIR

    resolved_path = resolve_input_path(path)
    initial_directory = (
        resolved_path if resolved_path.is_dir() else resolved_path.parent
    )
    if initial_directory.is_dir():
        return initial_directory
    return APP_DIR


# コンボボックスに表示する文字列。内部の計算識別子とは分けておくことで、
# 表示名を変更しても解析モジュールへ影響しないようにしている。
SOURCE_CSV = "CSV"
SOURCE_PICOSCOPE = "PicoScope"
METHOD_CORRELATION = "正規化相互相関"
METHOD_SQUARED_ERROR = "二乗誤差（従来方式）"

# 設定ファイルには、画面表示の文言に左右されない安定した識別子を保存する。
# ここでのみJSON側の値とUI表示を相互変換する。
SETTINGS_SOURCE_MODES = {
    "csv": SOURCE_CSV,
    "picoscope": SOURCE_PICOSCOPE,
}
SETTINGS_TRIGGER_DIRECTIONS = {
    "falling": "立ち下がり",
    "rising": "立ち上がり",
    "above": "しきい値より上",
    "below": "しきい値より下",
    "either": "立ち上がりまたは立ち下がり",
}
SETTINGS_RESOLUTIONS = {
    8: "8 bit",
    12: "12 bit",
    14: "14 bit",
    15: "15 bit",
    16: "16 bit",
}
SETTINGS_RANGES_MV = {
    10: "±10 mV",
    20: "±20 mV",
    50: "±50 mV",
    100: "±100 mV",
    200: "±200 mV",
    500: "±500 mV",
    1000: "±1 V",
    2000: "±2 V",
    5000: "±5 V",
    10000: "±10 V",
    20000: "±20 V",
}
RANGE_FULL_SCALE_V = {
    displayed_value: range_mv * 1e-3
    for range_mv, displayed_value in SETTINGS_RANGES_MV.items()
}
SETTINGS_MATCHING_METHODS = {
    MATCH_CORRELATION: METHOD_CORRELATION,
    MATCH_SQUARED_ERROR: METHOD_SQUARED_ERROR,
}

TRIGGER_DIRECTIONS = {
    "立ち下がり": psdk.TRIGGER_DIR.FALLING,
    "立ち上がり": psdk.TRIGGER_DIR.RISING,
    "しきい値より上": psdk.TRIGGER_DIR.ABOVE,
    "しきい値より下": psdk.TRIGGER_DIR.BELOW,
    "立ち上がりまたは立ち下がり": psdk.TRIGGER_DIR.RISING_OR_FALLING,
}
PICOSCOPE_RESOLUTIONS = {
    "8 bit": psdk.RESOLUTION.BIT_8,
    "12 bit": psdk.RESOLUTION.BIT_12,
    "14 bit": psdk.RESOLUTION.BIT_14,
    "15 bit": psdk.RESOLUTION.BIT_15,
    "16 bit": psdk.RESOLUTION.BIT_16,
}
PICOSCOPE_RANGES = {
    "±10 mV": psdk.RANGE.mV10,
    "±20 mV": psdk.RANGE.mV20,
    "±50 mV": psdk.RANGE.mV50,
    "±100 mV": psdk.RANGE.mV100,
    "±200 mV": psdk.RANGE.mV200,
    "±500 mV": psdk.RANGE.mV500,
    "±1 V": psdk.RANGE.V1,
    "±2 V": psdk.RANGE.V2,
    "±5 V": psdk.RANGE.V5,
    "±10 V": psdk.RANGE.V10,
    "±20 V": psdk.RANGE.V20,
}

# 基本タブからの取得ではChannel Aを±20 Vに固定する。アドバンスドタブでは
# 従来どおりプルダウンの選択値を使うため、UI変数とは分けて定数化する。
CHANNEL_A_FIXED_RANGE = "±20 V"
CHANNEL_B_RANGE_VALUES = tuple(PICOSCOPE_RANGES)
BASIC_SPEED_BORDER_COLOR = "#0B6E99"
BASIC_SPEED_BACKGROUND_COLOR = "#E8F6FB"
BASIC_SPEED_TEXT_COLOR = "#073B4C"
# 「参照波形を取得」ボタンの文字色はここで変更する。
BASIC_REFERENCE_BUTTON_TEXT_COLOR = "#00FE33"
# 基本タブの−／＋ボタンの一辺（ピクセル）。
BASIC_RANGE_STEP_BUTTON_SIZE_PX = 64
# 基本タブの参照／実測ボタンの高さ（ピクセル）。
BASIC_ACQUISITION_BUTTON_HEIGHT_PX = 96
BASIC_DISPLAY_NORMAL_US = 40.0
BASIC_DISPLAY_EXPANDED_US = 80.0


@dataclass(frozen=True)
class PicoScopeSettings:
    """UIで指定された1回分のPicoScope取得条件。"""

    duration_s: float
    trigger_a_threshold_mv: float
    trigger_a_direction: int
    resolution: int
    channel_a_range: int
    channel_b_range: int


@dataclass(frozen=True)
class FilteredWaveformPreview:
    """Bレンジ確認用に取得・フィルター処理した実測波形。"""

    measurement: SignalData
    filtered_channel_1: np.ndarray
    filtered_channel_2: np.ndarray


def acquire_picoscope_signal(
    settings: PicoScopeSettings,
    role: str,
) -> tuple[SignalData, float]:
    """取得モジュールを呼び、既存解析用のSignalDataへ変換する。"""

    capture = acquire_waveform(
        duration_s=settings.duration_s,
        trigger_a_threshold_mv=settings.trigger_a_threshold_mv,
        trigger_a_direction=settings.trigger_a_direction,
        resolution=settings.resolution,
        channel_a_range=settings.channel_a_range,
        channel_b_range=settings.channel_b_range,
    )
    return (
        SignalData(
            time_s=capture.time_s,
            # 取得モジュールはmV、既存UIとグラフはVを前提とするため変換する。
            channel_1=capture.channel_a_mv * 1e-3,
            channel_2=capture.channel_b_mv * 1e-3,
            source_name=f"PicoScope {capture.device_variant}（{role}）",
        ),
        capture.sample_interval_s,
    )


def sample_interval_from_signal(signal: SignalData) -> float:
    """波形の時間軸から実サンプリング間隔を求める。

    CSVでもUIの旧設定値に依存せず、実際の時間列を解析へ使う。
    現在のフィルターと照合は等間隔の時間軸を前提とするため、
    単調増加でない場合や不等間の場合は受け付けない。
    """

    intervals = np.diff(signal.time_s)
    if intervals.size == 0 or not np.all(np.isfinite(intervals)):
        raise ValueError("時間軸からサンプリング間隔を求められません。")
    if np.any(intervals <= 0):
        raise ValueError("時間軸は古い時刻から順に並べてください。")

    sample_interval_s = float(np.median(intervals))
    absolute_tolerance = max(
        abs(sample_interval_s) * 1e-9,
        np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(signal.time_s))))
        * 8.0,
    )
    if not np.allclose(
        intervals,
        sample_interval_s,
        # PicoScopeのCSVは時刻を小数点以下8桁[µs]で保存するため、
        # 8 ns間隔が丸めにより最大約1.25 ppmぶれることを許容する。
        rtol=1e-5,
        atol=absolute_tolerance,
    ):
        raise ValueError(
            "時間軸の間隔が一定ではありません。"
            "等間隔の波形データを使用してください。"
        )
    return sample_interval_s


class TimeResolutionMismatchError(ValueError):
    """参照波形と実測波形の時間分解能が一致しないことを表す。"""


def require_matching_time_resolution(
    reference_sample_interval_s: float,
    measurement_sample_interval_s: float,
) -> None:
    """2波形の実サンプリング間隔が解析可能な範囲で一致するか検査する。"""

    if not math.isclose(
        measurement_sample_interval_s,
        reference_sample_interval_s,
        rel_tol=1e-5,
        abs_tol=1e-15,
    ):
        raise TimeResolutionMismatchError(
            "参照波形と実測波形の時間分解能（サンプリング間隔）が一致しません。"
            "現在の時間分解能で参照波形を取り直してください。"
        )


def configure_plot_font() -> None:
    """実行PCに存在する日本語フォントをMatplotlibへ設定する。

    Windows、macOS、Linuxで一般的な候補を順番に探す。clone先に特定フォントが
    なくても起動でき、利用可能な候補が見つかった場合だけ設定する。
    """

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "Hiragino Sans",
        "IPAexGothic",
    ):
        if candidate in available_fonts:
            rcParams["font.family"] = candidate
            break
    rcParams["axes.unicode_minus"] = False


def enlarge_interface_fonts(root: tk.Misc) -> None:
    """グラフ以外のTk標準フォントを1段階大きくする。"""

    for font_name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkFixedFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
        "TkTooltipFont",
    ):
        try:
            interface_font = tkfont.nametofont(font_name, root=root)
            current_size = int(interface_font.cget("size"))
        except tk.TclError:
            continue
        interface_font.configure(
            size=current_size + 1 if current_size >= 0 else current_size - 1
        )


class ScrollableControls(ttk.Frame):
    """画面左側の入力欄を縦スクロール可能にする部品。

    小さい画面でも全パラメーターへ到達できるよう、Canvasの中にttk.Frameを置き、
    Frameの高さに合わせてスクロール範囲を更新する。
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)

        # Canvasがスクロール可能な表示領域、innerが実際の入力部品の配置先になる。
        self.canvas = tk.Canvas(self, highlightthickness=0, width=390)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=(8, 8, 12, 8))
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 入力部品の増減でinnerの高さが変わったら、スクロール範囲を再計算する。
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        # ウィンドウ幅の変更時はinnerもCanvasと同じ幅にし、入力欄を追従させる。
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(
                self._window_id, width=event.width
            ),
        )


class MeasurementApplication:
    """画面状態、データ取得、解析実行、グラフ更新をまとめるアプリ本体。"""

    def __init__(self, root: tk.Tk) -> None:
        """ウィンドウを初期化し、ユーザーの取得操作を待つ。"""

        self.root = root
        self.root.title("音速測定・波形照合")
        self.root.geometry("1500x900")
        self.root.minsize(1120, 700)

        # _busy中は多重実行を防ぐため、実行関連ボタンを無効にする。
        self._busy = False

        # 最新データを保持する。「現在データで再計算」ではCSVを読み直さず、
        # ここに保存した配列へ新しいパラメーターを適用する。
        self.current_measurement: SignalData | None = None
        self.current_reference: SignalData | None = None
        self.current_result: AnalysisResult | None = None
        self.current_filtered_preview: FilteredWaveformPreview | None = None
        self.current_measurement_sample_interval_s: float | None = None
        self.current_reference_sample_interval_s: float | None = None
        self.current_reference_source_mode: str | None = None
        self.current_reference_channel_a_range: int | None = None

        self._create_variables()
        self._build_ui()
        self._load_startup_settings()
        self._draw_empty_plots()

        # 距離欄でのEnterは新規取得＋計算、その他の欄では再計算にする。
        self.root.bind("<Return>", self._enter_pressed)
        self.root.bind("<KP_Enter>", self._enter_pressed)

    def _create_variables(self) -> None:
        """入力欄・結果表示と結び付くTkinter変数を作成する。

        StringVarを使うことで、入力欄の内容変更とラベル表示をTkinterが自動同期する。
        数値への変換は実行ボタンを押した時点で行う。
        """

        # --- データ取得元、CSVパス、PicoScope設定 ---
        self.source_mode_var = tk.StringVar(value=SOURCE_PICOSCOPE)
        self.measurement_path_var = tk.StringVar(
            value=str(DEFAULT_MEASUREMENT_PATH)
        )
        self.reference_path_var = tk.StringVar(value=str(DEFAULT_REFERENCE_PATH))
        self.capture_duration_us_var = tk.StringVar(value="20.0")
        self.trigger_a_threshold_mv_var = tk.StringVar(value="-2000")
        self.trigger_a_direction_var = tk.StringVar(value="立ち下がり")
        self.picoscope_resolution_var = tk.StringVar(value="8 bit")
        self.channel_a_range_var = tk.StringVar(value="±20 V")
        self.channel_b_range_var = tk.StringVar(value="±10 V")

        # --- バンドパスフィルター設定 ---
        # UIでは読みやすいns・MHzを使い、解析直前に秒・Hzへ変換する。
        self.sample_interval_ns_var = tk.StringVar(value="8.0")
        self.filter_low_mhz_var = tk.StringVar(value="1.0")
        self.filter_high_mhz_var = tk.StringVar(value="5.0")
        self.filter_order_var = tk.StringVar(value="4")
        self.filter_passes_var = tk.StringVar(value="2")

        # --- 窓関数と参照波形の切り出し設定（単位はµs） ---
        self.window_initial_us_var = tk.StringVar(value="0.75")
        self.window_rise_us_var = tk.StringVar(value="0.10")
        self.window_flat_us_var = tk.StringVar(value="0.50")
        self.window_fall_us_var = tk.StringVar(value="0.50")
        self.reference_start_us_var = tk.StringVar(value="0.0")

        # --- 照合方式、音速用距離、グラフ表示範囲 ---
        self.measurement_channel_var = tk.StringVar(value="2")
        self.reference_channel_var = tk.StringVar(value="2")
        self.matching_method_var = tk.StringVar(value=METHOD_SQUARED_ERROR)
        self.distance_mm_var = tk.StringVar(value="")
        self.display_min_us_var = tk.StringVar(value="0.0")
        self.display_max_us_var = tk.StringVar(value="20.0")

        # --- 解析後に更新する結果とステータス表示 ---
        self.result_time_var = tk.StringVar(value="—")
        self.result_score_var = tk.StringVar(value="—")
        self.result_speed_var = tk.StringVar(value="距離未入力")
        self.result_window_var = tk.StringVar(value="—")
        self.status_var = tk.StringVar(value="先に参照データを取得してください。")

    def _build_ui(self) -> None:
        """基本・アドバンスドの2タブと、それぞれの計測画面を組み立てる。"""

        # Matplotlibのグラフ文字は変えず、Tkinter側の文字だけを大きくする。
        enlarge_interface_fonts(self.root)

        # ttkの共通見た目を設定する。解析結果だけ少し太字で強調する。
        style = ttk.Style(self.root)
        style.configure("TButton", padding=(8, 5))
        style.configure("Result.TLabel", font=("TkDefaultFont", 12, "bold"))
        style.configure("BasicResult.TLabel", font=("TkDefaultFont", 14, "bold"))
        style.configure("TNotebook.Tab", padding=(18, 8))
        style.configure(
            "RangeStep.TButton",
            font=("TkDefaultFont", 19, "bold"),
            padding=(0, 0),
        )
        style.configure(
            "RangeValue.TLabel",
            font=("TkDefaultFont", 14, "bold"),
        )
        style.configure(
            "BasicAcquisition.TButton",
            font=("TkDefaultFont", 12, "bold"),
            padding=(16, 8),
        )
        style.configure(
            "BasicReference.TButton",
            font=("TkDefaultFont", 13, "bold"),
            foreground=BASIC_REFERENCE_BUTTON_TEXT_COLOR,
            padding=(16, 8),
        )

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.basic_tab = ttk.Frame(self.notebook)
        self.advanced_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.basic_tab, text="基本")
        self.notebook.add(self.advanced_tab, text="アドバンスド")

        self._build_basic_tab(self.basic_tab)
        self._build_advanced_tab(self.advanced_tab)
        # 2つのEntryは同じStringVarを共有し、どちらからでもEnter計測できる。
        self.distance_entries = (self.basic_distance_entry, self.distance_entry)
        self._update_channel_b_range_buttons()

    def _build_basic_tab(self, parent: ttk.Frame) -> None:
        """日常計測に必要な操作、主要結果、2グラフだけを配置する。"""

        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        operation_frame = ttk.LabelFrame(parent, text="計測", padding=10)
        operation_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=10,
            pady=(10, 6),
        )
        # 取得領域と、3つの要素間隔だけで余白を分ける。
        # 5:3:3:3にすることで、標準画面で取得ボタンを従来の
        # 約60%幅にしつつ、狭い画面にも追従させる。
        operation_frame.columnconfigure(0, weight=5)
        for gap_column in (1, 3, 5):
            operation_frame.columnconfigure(
                gap_column,
                weight=3,
                uniform="basic_operation_gaps",
            )

        basic_acquisition_frame = ttk.Frame(operation_frame)
        basic_acquisition_frame.grid(
            row=0,
            column=0,
            sticky="ew",
        )
        basic_acquisition_frame.columnconfigure(0, weight=1)
        basic_acquisition_frame.rowconfigure(
            (0, 2),
            minsize=BASIC_ACQUISITION_BUTTON_HEIGHT_PX,
        )
        basic_acquisition_frame.rowconfigure(1, minsize=6)

        self.basic_reference_button = ttk.Button(
            basic_acquisition_frame,
            text="参照波形を取得",
            command=self._basic_acquire_reference,
            style="BasicReference.TButton",
        )
        self.basic_reference_button.grid(row=0, column=0, sticky="nsew")
        self.basic_acquire_button = ttk.Button(
            basic_acquisition_frame,
            text="実測波形を取得・計算",
            command=self._basic_acquire_measurement,
            style="BasicAcquisition.TButton",
        )
        self.basic_acquire_button.grid(
            row=2,
            column=0,
            sticky="nsew",
        )
        range_frame = ttk.LabelFrame(
            operation_frame,
            text="入力レンジ",
            padding=(8, 4),
        )
        range_frame.grid(row=0, column=2, sticky="ew")
        ttk.Label(
            range_frame,
            text=f"A  {CHANNEL_A_FIXED_RANGE}（固定）",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 3))
        ttk.Label(range_frame, text="B").grid(row=1, column=0, padx=(0, 6))
        self._build_channel_b_range_stepper(range_frame, "basic", row=1, column=1)
        display_time_frame = ttk.LabelFrame(
            operation_frame,
            text="表示時間",
            padding=(8, 6),
        )
        display_time_frame.grid(
            row=0,
            column=4,
            sticky="ew",
        )
        display_time_frame.columnconfigure(0, weight=1)
        self.basic_display_normal_button = ttk.Button(
            display_time_frame,
            text="通常（40 µs）",
            command=lambda: self._set_basic_display_range(
                "通常",
                BASIC_DISPLAY_NORMAL_US,
            ),
        )
        self.basic_display_normal_button.grid(
            row=0,
            column=0,
            sticky="ew",
        )
        self.basic_display_expanded_button = ttk.Button(
            display_time_frame,
            text="拡大（80 µs）",
            command=lambda: self._set_basic_display_range(
                "拡大",
                BASIC_DISPLAY_EXPANDED_US,
            ),
        )
        self.basic_display_expanded_button.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(6, 0),
        )

        distance_frame = ttk.Frame(operation_frame)
        distance_frame.grid(row=0, column=6, sticky="e")
        ttk.Label(distance_frame, text="距離 [mm]").grid(
            row=0,
            column=0,
            padx=(0, 6),
            sticky="e",
        )
        self.basic_distance_entry = ttk.Entry(
            distance_frame,
            textvariable=self.distance_mm_var,
            width=10,
        )
        self.basic_distance_entry.grid(row=0, column=1, sticky="e")

        result_frame = ttk.LabelFrame(parent, text="計算結果", padding=10)
        result_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        result_frame.rowconfigure(0, weight=1)
        compact_results = (
            ("一致時間", self.result_time_var),
            ("最良スコア", self.result_score_var),
        )
        for column, (label, variable) in enumerate(compact_results):
            result_frame.columnconfigure(
                column,
                weight=1,
                uniform="basic_results",
            )
            card = ttk.LabelFrame(result_frame, text=label, padding=(14, 8))
            card.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 4, 4),
            )
            ttk.Label(
                card,
                textvariable=variable,
                style="BasicResult.TLabel",
                anchor="center",
            ).pack(fill="both", expand=True)

        # 音速は日常計測で最も確認しやすくするため、広い色付きカードで強調する。
        result_frame.columnconfigure(2, weight=2, uniform="basic_results")
        self.basic_speed_card = tk.Frame(
            result_frame,
            background=BASIC_SPEED_BORDER_COLOR,
            borderwidth=0,
            padx=3,
            pady=3,
        )
        self.basic_speed_card.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=(4, 0),
        )
        speed_content = tk.Frame(
            self.basic_speed_card,
            background=BASIC_SPEED_BACKGROUND_COLOR,
            borderwidth=0,
            padx=18,
            pady=8,
        )
        speed_content.pack(fill="both", expand=True)
        tk.Label(
            speed_content,
            text="音速",
            background=BASIC_SPEED_BACKGROUND_COLOR,
            foreground=BASIC_SPEED_TEXT_COLOR,
            font=("TkDefaultFont", 14, "bold"),
            anchor="w",
        ).pack(fill="x")
        self.basic_speed_value_label = tk.Label(
            speed_content,
            textvariable=self.result_speed_var,
            background=BASIC_SPEED_BACKGROUND_COLOR,
            foreground=BASIC_SPEED_TEXT_COLOR,
            font=("TkDefaultFont", 27, "bold"),
            anchor="center",
        )
        self.basic_speed_value_label.pack(fill="both", expand=True, pady=(2, 0))

        plot_frame = ttk.Frame(parent, padding=(8, 2, 8, 4))
        plot_frame.grid(row=2, column=0, sticky="nsew")
        self._build_basic_plots(plot_frame)
        self._add_status_bar(parent, row=3)

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        """従来の全設定・全グラフ画面を、そのままアドバンスドへ配置する。"""

        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # Panedwindowにより、ユーザーが操作パネルとグラフの境界を左右へ動かせる。
        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        controls = ScrollableControls(paned)
        plot_frame = ttk.Frame(paned, padding=(4, 4, 8, 4))
        paned.add(controls, weight=0)
        paned.add(plot_frame, weight=1)

        self._build_controls(controls.inner)
        self._build_plots(plot_frame)
        self._add_status_bar(parent, row=1)

    def _add_status_bar(self, parent: ttk.Frame, row: int) -> None:
        """共有ステータスを各タブの最下部へ表示する。"""

        # 最下部には「解析中」「完了」「エラー」など現在状態を常時表示する。
        status = ttk.Label(
            parent,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            padding=(8, 4),
        )
        status.grid(row=row, column=0, sticky="ew")

    def _build_channel_b_range_stepper(
        self,
        parent: ttk.Widget,
        prefix: str,
        *,
        row: int,
        column: int,
    ) -> None:
        """Channel Bレンジを一段ずつ変える大きな−/＋ボタンを配置する。"""

        stepper = ttk.Frame(parent)
        stepper.grid(row=row, column=column, sticky="ew")
        stepper.columnconfigure(1, weight=1)

        decrease_holder = ttk.Frame(
            stepper,
            width=BASIC_RANGE_STEP_BUTTON_SIZE_PX,
            height=BASIC_RANGE_STEP_BUTTON_SIZE_PX,
        )
        decrease_holder.grid(row=0, column=0, padx=(0, 8))
        decrease_holder.grid_propagate(False)
        decrease_holder.rowconfigure(0, weight=1)
        decrease_holder.columnconfigure(0, weight=1)
        decrease_button = ttk.Button(
            decrease_holder,
            text="−",
            style="RangeStep.TButton",
            command=self._basic_range_minus_pressed,
        )
        decrease_button.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            stepper,
            textvariable=self.channel_b_range_var,
            style="RangeValue.TLabel",
            anchor="center",
            width=10,
        ).grid(row=0, column=1, sticky="ew")
        increase_holder = ttk.Frame(
            stepper,
            width=BASIC_RANGE_STEP_BUTTON_SIZE_PX,
            height=BASIC_RANGE_STEP_BUTTON_SIZE_PX,
        )
        increase_holder.grid(row=0, column=2, padx=(8, 0))
        increase_holder.grid_propagate(False)
        increase_holder.rowconfigure(0, weight=1)
        increase_holder.columnconfigure(0, weight=1)
        increase_button = ttk.Button(
            increase_holder,
            text="＋",
            style="RangeStep.TButton",
            command=self._basic_range_plus_pressed,
        )
        increase_button.grid(row=0, column=0, sticky="nsew")

        setattr(self, f"{prefix}_range_decrease_button", decrease_button)
        setattr(self, f"{prefix}_range_increase_button", increase_button)

    def _build_controls(self, parent: ttk.Frame) -> None:
        """入力項目を用途別のグループに分けて左パネルへ配置する。"""

        parent.columnconfigure(0, weight=1)

        # --- データ取得グループ ---
        source_frame = ttk.LabelFrame(parent, text="データ取得", padding=8)
        source_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        source_frame.columnconfigure(1, weight=1)

        ttk.Label(source_frame, text="取得元").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.source_combobox = ttk.Combobox(
            source_frame,
            textvariable=self.source_mode_var,
            values=(SOURCE_CSV, SOURCE_PICOSCOPE),
            state="readonly",
            width=20,
        )
        self.source_combobox.grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=3
        )
        self.source_combobox.bind("<<ComboboxSelected>>", self._source_mode_changed)

        self._add_file_row(
            source_frame,
            row=1,
            label="測定CSV",
            variable=self.measurement_path_var,
            command=self._select_measurement_file,
        )
        self._add_file_row(
            source_frame,
            row=2,
            label="参照CSV",
            variable=self.reference_path_var,
            command=self._select_reference_file,
        )

        # --- PicoScope取得条件 ---
        picoscope_frame = ttk.LabelFrame(parent, text="PicoScope設定", padding=8)
        picoscope_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        picoscope_frame.columnconfigure(1, weight=1)
        self._add_entry(
            picoscope_frame,
            0,
            "取得時間 [µs]",
            self.capture_duration_us_var,
        )
        self._add_entry(
            picoscope_frame,
            1,
            "Trigger A threshold [mV]",
            self.trigger_a_threshold_mv_var,
        )
        self._add_combobox(
            picoscope_frame,
            2,
            "Trigger A direction",
            self.trigger_a_direction_var,
            tuple(TRIGGER_DIRECTIONS),
        )
        self._add_combobox(
            picoscope_frame,
            3,
            "ADC分解能",
            self.picoscope_resolution_var,
            tuple(PICOSCOPE_RESOLUTIONS),
        )
        self.channel_a_range_combobox = self._add_combobox(
            picoscope_frame,
            4,
            "Channel Aレンジ",
            self.channel_a_range_var,
            tuple(PICOSCOPE_RANGES),
        )
        self.channel_b_range_combobox = self._add_combobox(
            picoscope_frame,
            5,
            "Channel Bレンジ",
            self.channel_b_range_var,
            tuple(PICOSCOPE_RANGES),
        )
        self.channel_a_range_combobox.bind(
            "<<ComboboxSelected>>",
            self._advanced_channel_a_range_changed,
        )
        self.channel_b_range_combobox.bind(
            "<<ComboboxSelected>>",
            self._advanced_channel_b_range_changed,
        )

        # --- フィルターグループ ---
        filter_frame = ttk.LabelFrame(parent, text="フィルター", padding=8)
        filter_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        filter_frame.columnconfigure(1, weight=1)
        self._add_entry(
            filter_frame,
            0,
            "サンプリング間隔 [ns]（取得後自動）",
            self.sample_interval_ns_var,
        )
        self._add_entry(filter_frame, 1, "下限周波数 [MHz]", self.filter_low_mhz_var)
        self._add_entry(filter_frame, 2, "上限周波数 [MHz]", self.filter_high_mhz_var)
        self._add_entry(filter_frame, 3, "フィルター次数", self.filter_order_var)
        self._add_entry(filter_frame, 4, "フィルター回数", self.filter_passes_var)

        # --- 窓関数グループ ---
        # 既存window_2.csvと同じ既定値から、毎回メモリ上で窓を作り直す。
        window_frame = ttk.LabelFrame(parent, text="窓関数・参照波形", padding=8)
        window_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        window_frame.columnconfigure(1, weight=1)
        self._add_entry(window_frame, 0, "初期ゼロ [µs]", self.window_initial_us_var)
        self._add_entry(window_frame, 1, "立ち上がり [µs]", self.window_rise_us_var)
        self._add_entry(window_frame, 2, "フラット [µs]", self.window_flat_us_var)
        self._add_entry(window_frame, 3, "立ち下がり [µs]", self.window_fall_us_var)
        self._add_entry(window_frame, 4, "参照開始時間 [µs]", self.reference_start_us_var)

        # --- 照合方法・音速・表示範囲グループ ---
        matching_frame = ttk.LabelFrame(parent, text="照合・表示", padding=8)
        matching_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        matching_frame.columnconfigure(1, weight=1)
        self._add_combobox(
            matching_frame,
            0,
            "測定チャンネル",
            self.measurement_channel_var,
            ("1", "2"),
        )
        self._add_combobox(
            matching_frame,
            1,
            "参照チャンネル",
            self.reference_channel_var,
            ("1", "2"),
        )
        self._add_combobox(
            matching_frame,
            2,
            "照合方法",
            self.matching_method_var,
            (METHOD_CORRELATION, METHOD_SQUARED_ERROR),
        )
        self.distance_entry = self._add_entry(
            matching_frame,
            3,
            "距離 [mm]（任意）",
            self.distance_mm_var,
        )
        self._add_entry(matching_frame, 4, "表示開始 [µs]", self.display_min_us_var)
        self._add_entry(matching_frame, 5, "表示終了 [µs]", self.display_max_us_var)

        # --- 設定ファイル ---
        settings_frame = ttk.LabelFrame(parent, text="設定ファイル", padding=8)
        settings_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure((0, 1), weight=1)
        ttk.Label(
            settings_frame,
            text=f"{SETTINGS_PATH.name}（起動時に自動読込）",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.save_settings_button = ttk.Button(
            settings_frame,
            text="設定を保存",
            command=self._save_settings,
        )
        self.save_settings_button.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.load_settings_button = ttk.Button(
            settings_frame,
            text="設定を読み込む",
            command=self._load_settings,
        )
        self.load_settings_button.grid(row=1, column=1, sticky="ew", padx=(3, 0))

        # --- 実行ボタン ---
        action_frame = ttk.LabelFrame(parent, text="実行", padding=8)
        action_frame.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        action_frame.columnconfigure((0, 1), weight=1)

        # 参照と実測を別操作にし、試料を切り替えてから次の取得を開始できるようにする。
        self.acquire_reference_button = ttk.Button(
            action_frame,
            text="参照データを取得",
            command=self.acquire_reference,
        )
        self.acquire_reference_button.grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )

        self.acquire_button = ttk.Button(
            action_frame,
            text="実測データを取得・計算",
            command=self.acquire_and_analyze,
        )
        self.acquire_button.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )

        # 取得済み配列を使うため、窓・フィルター値の調整を高速に繰り返せる。
        self.recalculate_button = ttk.Button(
            action_frame,
            text="現在データで再計算",
            command=self.recalculate,
        )
        self.recalculate_button.grid(row=2, column=0, sticky="ew", padx=(0, 3))

        self.save_window_button = ttk.Button(
            action_frame,
            text="窓CSVを保存",
            command=self._save_window,
        )
        self.save_window_button.grid(row=2, column=1, sticky="ew", padx=(3, 0))

        # --- 最新の数値結果 ---
        result_frame = ttk.LabelFrame(parent, text="計算結果", padding=8)
        result_frame.grid(row=7, column=0, sticky="ew")
        result_frame.columnconfigure(1, weight=1)
        self._add_result_row(result_frame, 0, "一致時間", self.result_time_var)
        self._add_result_row(result_frame, 1, "最良スコア", self.result_score_var)
        self._add_result_row(result_frame, 2, "音速", self.result_speed_var)
        self._add_result_row(result_frame, 3, "窓関数", self.result_window_var)

    def _build_plots(self, parent: ttk.Frame) -> None:
        """Matplotlibの2×2グラフと操作ツールバーをTkinterへ埋め込む。"""

        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # constrained_layoutにより、画面サイズ変更時にも軸ラベルの重なりを抑える。
        self.figure = Figure(figsize=(11, 8), dpi=100, constrained_layout=True)
        axes = self.figure.subplots(2, 2)
        self.raw_axis = axes[0, 0]
        self.filtered_axis = axes[0, 1]
        self.window_axis = axes[1, 0]
        self.matching_axis = axes[1, 1]

        # FigureCanvasTkAggがMatplotlib FigureとTkinterウィジェットを橋渡しする。
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        # 拡大、移動、表示リセット、画像保存などMatplotlib標準操作を提供する。
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left")

    def _build_basic_plots(self, parent: ttk.Frame) -> None:
        """基本タブ用のフィルター後波形・評価関数グラフを埋め込む。"""

        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.basic_figure = Figure(
            figsize=(12, 6),
            dpi=100,
            constrained_layout=True,
        )
        axes = self.basic_figure.subplots(1, 2)
        self.basic_filtered_axis = axes[0]
        self.basic_matching_axis = axes[1]
        self.basic_canvas = FigureCanvasTkAgg(self.basic_figure, master=parent)
        self.basic_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    @staticmethod
    def _add_entry(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
    ) -> ttk.Entry:
        """同じ配置規則で「項目名＋数値入力欄」を追加する補助関数。"""

        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=3
        )
        entry = ttk.Entry(parent, textvariable=variable, width=14)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        return entry

    @staticmethod
    def _add_combobox(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ) -> ttk.Combobox:
        """チャンネルや照合方式など、選択式入力欄を追加する。"""

        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=3
        )
        combobox = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=20,
        )
        combobox.grid(row=row, column=1, sticky="ew", pady=3)
        return combobox

    @staticmethod
    def _add_file_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: object,
    ) -> None:
        """ファイルパス入力欄とファイル選択ボタンを1行に配置する。"""

        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Button(parent, text="選択", command=command).grid(
            row=row, column=2, padx=(5, 0), pady=3
        )

    @staticmethod
    def _add_result_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        """計算結果の名前と、右寄せした値ラベルを配置する。"""

        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )
        ttk.Label(parent, textvariable=variable, style="Result.TLabel").grid(
            row=row, column=1, sticky="e", pady=3
        )

    def _select_measurement_file(self) -> None:
        """ファイル選択ダイアログで測定CSVのパスを更新する。"""

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="測定データCSVを選択",
            initialdir=str(
                initial_directory_for_input(self.measurement_path_var.get())
            ),
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            self.measurement_path_var.set(selected)

    def _select_reference_file(self) -> None:
        """ファイル選択ダイアログで参照CSVのパスを更新する。"""

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="参照データCSVを選択",
            initialdir=str(
                initial_directory_for_input(self.reference_path_var.get())
            ),
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            self.reference_path_var.set(selected)

    def _change_channel_b_range(self, step: int) -> bool:
        """Channel Bレンジを対応レンジ一覧の隣へ一段だけ変更する。"""

        if self._busy:
            return False
        if step not in (-1, 1):
            raise ValueError("レンジの変更量は−1または＋1で指定してください。")

        try:
            current_index = CHANNEL_B_RANGE_VALUES.index(
                self.channel_b_range_var.get()
            )
        except ValueError as exc:
            raise ValueError("対応していないChannel B入力レンジです。") from exc

        next_index = min(
            max(current_index + step, 0),
            len(CHANNEL_B_RANGE_VALUES) - 1,
        )
        if next_index == current_index:
            self._update_channel_b_range_buttons()
            return False

        channel_b = CHANNEL_B_RANGE_VALUES[next_index]
        self.channel_b_range_var.set(channel_b)
        self._update_channel_b_range_buttons()
        # 表示軸もChannel Bレンジに連動するため、取得元を問わず古い実測結果を
        # 消去し、新しいレンジの待機グラフへ直ちに更新する。参照波形は保持する。
        self._discard_measurement_data()

        reference_status = (
            "。参照波形を保持し、次の実測取得で時間分解能を確認します"
            if self.current_reference is not None
            else ""
        )
        self.status_var.set(
            f"Channel Bレンジを変更しました: {channel_b}{reference_status}"
        )
        self._select_distance_for_next_measurement()
        return True

    def _basic_range_minus_pressed(self) -> None:
        """基本タブの「−」でChannel Bレンジを一段大きくする。"""

        if self._change_channel_b_range(1):
            self._start_basic_range_preview()

    def _basic_range_plus_pressed(self) -> None:
        """基本タブの「＋」でChannel Bレンジを一段小さくする。"""

        if self._change_channel_b_range(-1):
            self._start_basic_range_preview()

    def _set_basic_display_range(
        self,
        mode: str,
        display_max_us: float,
    ) -> None:
        """基本タブのプリセットから0 µs始まりの表示時間へ切り替える。"""

        if self._busy:
            return
        if display_max_us not in (
            BASIC_DISPLAY_NORMAL_US,
            BASIC_DISPLAY_EXPANDED_US,
        ):
            raise ValueError("基本タブの表示時間は40 µsまたは80 µsです。")

        display_range = (0.0, display_max_us)
        self.display_min_us_var.set("0.0")
        self.display_max_us_var.set(str(display_max_us))
        if self.current_result is not None:
            self._draw_result(self.current_result, display_range)
        elif self.current_filtered_preview is not None:
            self._draw_basic_range_preview(
                self.current_filtered_preview,
                display_range,
            )
        self.status_var.set(
            f"表示時間を{mode}（0–{display_max_us:g} µs）に変更しました。"
        )
        self._select_distance_for_next_measurement()

    def _update_channel_b_range_buttons(self) -> None:
        """実行中とレンジ上下限を考慮し、基本タブの−/＋を更新する。"""

        try:
            current_index = CHANNEL_B_RANGE_VALUES.index(
                self.channel_b_range_var.get()
            )
        except ValueError:
            current_index = -1

        minus_state = (
            "disabled"
            if self._busy
            or current_index < 0
            or current_index >= len(CHANNEL_B_RANGE_VALUES) - 1
            else "normal"
        )
        plus_state = (
            "disabled" if self._busy or current_index <= 0 else "normal"
        )
        minus_button = getattr(self, "basic_range_decrease_button", None)
        plus_button = getattr(self, "basic_range_increase_button", None)
        if minus_button is not None:
            minus_button.configure(state=minus_state)
        if plus_button is not None:
            plus_button.configure(state=plus_state)

    def _advanced_channel_a_range_changed(self, _event: object = None) -> None:
        """AdvancedでChannel Aレンジを変更し、参照再取得を要求する。"""

        ranges = (
            f"Channel A {self.channel_a_range_var.get()} / "
            f"Channel B {self.channel_b_range_var.get()}"
        )
        if (
            self.source_mode_var.get() == SOURCE_PICOSCOPE
            and self.current_reference is not None
        ):
            self._discard_acquired_data()
            self.status_var.set(
                "Channel Aレンジを変更しました。"
                f"参照波形を取り直してください: {ranges}"
            )
            return
        self.status_var.set(f"Channel Aレンジを変更しました: {ranges}")

    def _advanced_channel_b_range_changed(self, _event: object = None) -> None:
        """AdvancedのChannel B変更を基本表示へ反映し、参照は保持する。"""

        self._update_channel_b_range_buttons()
        self._discard_measurement_data()
        ranges = (
            f"Channel A {self.channel_a_range_var.get()} / "
            f"Channel B {self.channel_b_range_var.get()}"
        )
        reference_status = (
            "。参照波形を保持し、次の実測取得で時間分解能を確認します"
            if self.current_reference is not None
            else ""
        )
        self.status_var.set(
            f"Channel Bレンジを変更しました: {ranges}{reference_status}"
        )

    def _source_mode_changed(self, _event: object = None) -> None:
        """取得元を切り替えたとき、異なる取得元の波形が混ざるのを防ぐ。"""

        self._discard_acquired_data()
        self.status_var.set("取得元を変更しました。先に参照データを取得してください。")

    def _discard_acquired_data(self) -> None:
        """設定変更後に再利用できない取得済み波形と解析結果を破棄する。"""

        self.current_measurement = None
        self.current_reference = None
        self.current_result = None
        self.current_filtered_preview = None
        self.current_measurement_sample_interval_s = None
        self.current_reference_sample_interval_s = None
        self.current_reference_source_mode = None
        self.current_reference_channel_a_range = None
        self._clear_result_display()

    def _discard_measurement_data(self) -> None:
        """参照波形を残し、以前の実測波形と解析結果だけを破棄する。"""

        self.current_measurement = None
        self.current_measurement_sample_interval_s = None
        self.current_result = None
        self.current_filtered_preview = None
        self._clear_result_display()

    @staticmethod
    def _setting_key(
        mapping: dict[object, str],
        displayed_value: str,
        label: str,
    ) -> object:
        """UIの表示値に対応する設定ファイル用の安定IDを返す。"""

        for setting_value, display_value in mapping.items():
            if display_value == displayed_value:
                return setting_value
        raise ValueError(f"「{label}」の選択値が不正です: {displayed_value}")

    @staticmethod
    def _portable_path_for_settings(path: str) -> str:
        """アプリ内のCSVパスを、フォルダー移動に強い相対表記へ変換する。"""

        if not path.strip():
            return ""
        resolved_path = resolve_input_path(path)
        try:
            return resolved_path.relative_to(APP_DIR).as_posix()
        except ValueError:
            return str(resolved_path)

    def _settings_from_ui(self) -> ApplicationSettings:
        """25個の入力欄を、検証可能な型付き設定へまとめる。"""

        distance_text = self.distance_mm_var.get().strip()
        distance_mm = (
            self._parse_float(self.distance_mm_var, "距離")
            if distance_text
            else None
        )
        settings = ApplicationSettings(
            source_mode=str(
                self._setting_key(
                    SETTINGS_SOURCE_MODES,
                    self.source_mode_var.get(),
                    "取得元",
                )
            ),
            measurement_path=self._portable_path_for_settings(
                self.measurement_path_var.get()
            ),
            reference_path=self._portable_path_for_settings(
                self.reference_path_var.get()
            ),
            capture_duration_us=self._parse_float(
                self.capture_duration_us_var,
                "取得時間",
            ),
            trigger_a_threshold_mv=self._parse_float(
                self.trigger_a_threshold_mv_var,
                "Trigger A threshold",
            ),
            trigger_a_direction=str(
                self._setting_key(
                    SETTINGS_TRIGGER_DIRECTIONS,
                    self.trigger_a_direction_var.get(),
                    "Trigger A direction",
                )
            ),
            resolution_bits=int(
                self._setting_key(
                    SETTINGS_RESOLUTIONS,
                    self.picoscope_resolution_var.get(),
                    "ADC分解能",
                )
            ),
            channel_a_range_mv=int(
                self._setting_key(
                    SETTINGS_RANGES_MV,
                    self.channel_a_range_var.get(),
                    "Channel Aレンジ",
                )
            ),
            channel_b_range_mv=int(
                self._setting_key(
                    SETTINGS_RANGES_MV,
                    self.channel_b_range_var.get(),
                    "Channel Bレンジ",
                )
            ),
            sample_interval_ns=self._parse_float(
                self.sample_interval_ns_var,
                "サンプリング間隔",
            ),
            filter_low_mhz=self._parse_float(
                self.filter_low_mhz_var,
                "下限周波数",
            ),
            filter_high_mhz=self._parse_float(
                self.filter_high_mhz_var,
                "上限周波数",
            ),
            filter_order=self._parse_int(self.filter_order_var, "フィルター次数"),
            filter_passes=self._parse_int(self.filter_passes_var, "フィルター回数"),
            window_initial_us=self._parse_float(
                self.window_initial_us_var,
                "初期ゼロ",
            ),
            window_rise_us=self._parse_float(
                self.window_rise_us_var,
                "立ち上がり",
            ),
            window_flat_us=self._parse_float(
                self.window_flat_us_var,
                "フラット",
            ),
            window_fall_us=self._parse_float(
                self.window_fall_us_var,
                "立ち下がり",
            ),
            reference_start_us=self._parse_float(
                self.reference_start_us_var,
                "参照開始時間",
            ),
            measurement_channel=self._parse_int(
                self.measurement_channel_var,
                "測定チャンネル",
            ),
            reference_channel=self._parse_int(
                self.reference_channel_var,
                "参照チャンネル",
            ),
            matching_method=str(
                self._setting_key(
                    SETTINGS_MATCHING_METHODS,
                    self.matching_method_var.get(),
                    "照合方法",
                )
            ),
            distance_mm=distance_mm,
            display_min_us=self._parse_float(self.display_min_us_var, "表示開始"),
            display_max_us=self._parse_float(self.display_max_us_var, "表示終了"),
        )
        settings.validate()
        return settings

    def _apply_settings(self, settings: ApplicationSettings) -> None:
        """検証済み設定の全項目をUIへ一括反映する。"""

        # ApplicationSettings.validate() 済みなので、各辞書の参照は必ず成功する。
        self.source_mode_var.set(SETTINGS_SOURCE_MODES[settings.source_mode])
        self.measurement_path_var.set(settings.measurement_path)
        self.reference_path_var.set(settings.reference_path)
        self.capture_duration_us_var.set(str(settings.capture_duration_us))
        self.trigger_a_threshold_mv_var.set(str(settings.trigger_a_threshold_mv))
        self.trigger_a_direction_var.set(
            SETTINGS_TRIGGER_DIRECTIONS[settings.trigger_a_direction]
        )
        self.picoscope_resolution_var.set(
            SETTINGS_RESOLUTIONS[settings.resolution_bits]
        )
        self.channel_a_range_var.set(
            SETTINGS_RANGES_MV[settings.channel_a_range_mv]
        )
        self.channel_b_range_var.set(
            SETTINGS_RANGES_MV[settings.channel_b_range_mv]
        )
        self.sample_interval_ns_var.set(str(settings.sample_interval_ns))
        self.filter_low_mhz_var.set(str(settings.filter_low_mhz))
        self.filter_high_mhz_var.set(str(settings.filter_high_mhz))
        self.filter_order_var.set(str(settings.filter_order))
        self.filter_passes_var.set(str(settings.filter_passes))
        self.window_initial_us_var.set(str(settings.window_initial_us))
        self.window_rise_us_var.set(str(settings.window_rise_us))
        self.window_flat_us_var.set(str(settings.window_flat_us))
        self.window_fall_us_var.set(str(settings.window_fall_us))
        self.reference_start_us_var.set(str(settings.reference_start_us))
        self.measurement_channel_var.set(str(settings.measurement_channel))
        self.reference_channel_var.set(str(settings.reference_channel))
        self.matching_method_var.set(
            SETTINGS_MATCHING_METHODS[settings.matching_method]
        )
        self.distance_mm_var.set(
            "" if settings.distance_mm is None else str(settings.distance_mm)
        )
        self.display_min_us_var.set(str(settings.display_min_us))
        self.display_max_us_var.set(str(settings.display_max_us))
        self._update_channel_b_range_buttons()

    def _load_startup_settings(self) -> None:
        """起動時に固定の設定ファイルを読み、失敗時は既定値で継続する。"""

        if not SETTINGS_PATH.is_file():
            self.status_var.set(
                f"{SETTINGS_PATH.name} がないため既定値で起動しました。"
                "先に参照データを取得してください。"
            )
            return

        try:
            settings = load_settings(SETTINGS_PATH)
        except Exception as exc:
            self.status_var.set(
                "設定ファイルを読み込めなかったため、既定値で起動しました。"
            )
            messagebox.showwarning(
                "設定ファイル",
                f"{SETTINGS_PATH.name} を読み込めませんでした。\n"
                f"既定値で起動します。\n\n{exc}",
                parent=self.root,
            )
            return

        self._apply_settings(settings)
        self.status_var.set(
            f"設定を読み込みました: {SETTINGS_PATH.name}。"
            "先に参照データを取得してください。"
        )

    def _save_settings(self) -> None:
        """現在の全入力値を、次回起動時に読む固定JSONへ保存する。"""

        if self._busy:
            return
        try:
            settings = self._settings_from_ui()
            save_settings(settings, SETTINGS_PATH)
        except Exception as exc:
            self._show_error(exc)
            return
        self.status_var.set(f"設定を保存しました: {SETTINGS_PATH}")

    def _load_settings(self) -> None:
        """固定JSONを手動で再読込し、古い取得データを破棄する。"""

        if self._busy:
            return
        try:
            settings = load_settings(SETTINGS_PATH)
        except Exception as exc:
            self._show_error(exc)
            return

        self._apply_settings(settings)
        # 取得元が同じでもCSVパスや取得条件が変わる可能性があるため必ず破棄する。
        self._discard_acquired_data()
        self.status_var.set(
            f"設定を読み込みました: {SETTINGS_PATH.name}。"
            "参照データを取り直してください。"
        )

    @staticmethod
    def _parse_float(variable: tk.StringVar, label: str) -> float:
        """入力文字列を小数へ変換し、失敗時は項目名付きエラーにする。"""

        try:
            return float(variable.get().strip())
        except ValueError as exc:
            raise ValueError(f"「{label}」には数値を入力してください。") from exc

    @staticmethod
    def _parse_int(variable: tk.StringVar, label: str) -> int:
        """入力文字列を整数へ変換し、失敗時は項目名付きエラーにする。"""

        try:
            return int(variable.get().strip())
        except ValueError as exc:
            raise ValueError(f"「{label}」には整数を入力してください。") from exc

    def _read_picoscope_settings(
        self,
        channel_a_range_override: str | None = None,
    ) -> PicoScopeSettings:
        """PicoScope設定欄を読み、取得モジュールへ渡す値へ変換する。"""

        duration_s = self._parse_float(
            self.capture_duration_us_var,
            "取得時間",
        ) * 1e-6
        if duration_s <= 0:
            raise ValueError("取得時間は0より大きくしてください。")

        channel_a_range = (
            channel_a_range_override
            if channel_a_range_override is not None
            else self.channel_a_range_var.get()
        )
        if channel_a_range not in PICOSCOPE_RANGES:
            raise ValueError("対応していないChannel A入力レンジです。")
        if self.channel_b_range_var.get() not in PICOSCOPE_RANGES:
            raise ValueError("対応していないChannel B入力レンジです。")

        trigger_a_threshold_mv = self._parse_float(
            self.trigger_a_threshold_mv_var,
            "Trigger A threshold",
        )
        channel_a_range_mv = int(
            self._setting_key(
                SETTINGS_RANGES_MV,
                channel_a_range,
                "Channel Aレンジ",
            )
        )
        if abs(trigger_a_threshold_mv) > channel_a_range_mv:
            raise ValueError(
                "Trigger A thresholdをChannel Aの入力レンジ内にしてください。"
            )

        return PicoScopeSettings(
            duration_s=duration_s,
            trigger_a_threshold_mv=trigger_a_threshold_mv,
            trigger_a_direction=TRIGGER_DIRECTIONS[
                self.trigger_a_direction_var.get()
            ],
            resolution=PICOSCOPE_RESOLUTIONS[
                self.picoscope_resolution_var.get()
            ],
            channel_a_range=PICOSCOPE_RANGES[channel_a_range],
            channel_b_range=PICOSCOPE_RANGES[self.channel_b_range_var.get()],
        )

    def _read_window_parameters(self) -> WindowParameters:
        """窓関数の入力欄を読み、µs・nsから秒単位へ変換する。"""

        sample_interval_s = self._parse_float(
            self.sample_interval_ns_var, "サンプリング間隔"
        ) * 1e-9
        return WindowParameters(
            sample_interval_s=sample_interval_s,
            initial_zero_s=self._parse_float(
                self.window_initial_us_var, "初期ゼロ"
            )
            * 1e-6,
            rise_s=self._parse_float(self.window_rise_us_var, "立ち上がり")
            * 1e-6,
            flat_s=self._parse_float(self.window_flat_us_var, "フラット")
            * 1e-6,
            fall_s=self._parse_float(self.window_fall_us_var, "立ち下がり")
            * 1e-6,
        )

    def _read_inputs(
        self,
    ) -> tuple[AnalysisParameters, WindowParameters, tuple[float, float]]:
        """全入力欄を読み、解析用パラメーターへ変換・検査する。

        Returns:
            解析設定、窓設定、グラフ表示範囲[µs]の3要素。
        """

        window_parameters = self._read_window_parameters()

        # 距離は任意入力。空欄なら音速を計算せず、一致時間だけを表示する。
        distance_text = self.distance_mm_var.get().strip()
        distance_mm = None
        if distance_text:
            distance_mm = self._parse_float(self.distance_mm_var, "距離")

        # UIの日本語表示を、解析モジュールが使う内部識別子へ変換する。
        method = {
            METHOD_CORRELATION: MATCH_CORRELATION,
            METHOD_SQUARED_ERROR: MATCH_SQUARED_ERROR,
        }[self.matching_method_var.get()]

        # UI入力のMHz、µsをHz、秒へ変換して解析モジュールへ渡す。
        parameters = AnalysisParameters(
            sample_interval_s=window_parameters.sample_interval_s,
            filter_low_hz=self._parse_float(
                self.filter_low_mhz_var, "下限周波数"
            )
            * 1e6,
            filter_high_hz=self._parse_float(
                self.filter_high_mhz_var, "上限周波数"
            )
            * 1e6,
            filter_order=self._parse_int(self.filter_order_var, "フィルター次数"),
            filter_passes=self._parse_int(self.filter_passes_var, "フィルター回数"),
            reference_start_s=self._parse_float(
                self.reference_start_us_var, "参照開始時間"
            )
            * 1e-6,
            measurement_channel=self._parse_int(
                self.measurement_channel_var, "測定チャンネル"
            ),
            reference_channel=self._parse_int(
                self.reference_channel_var, "参照チャンネル"
            ),
            matching_method=method,
            distance_mm=distance_mm,
        )

        display_min_us = self._parse_float(self.display_min_us_var, "表示開始")
        display_max_us = self._parse_float(self.display_max_us_var, "表示終了")
        if display_max_us <= display_min_us:
            raise ValueError("表示終了時間は表示開始時間より大きくしてください。")

        # スレッドを開始する前に検査し、入力ミスはすぐダイアログ表示する。
        parameters.validate()
        window_parameters.validate()
        return parameters, window_parameters, (display_min_us, display_max_us)

    def _read_filtered_preview_inputs(
        self,
    ) -> tuple[AnalysisParameters, tuple[float, float]]:
        """レンジ確認用にフィルター設定とグラフ表示範囲だけを読む。"""

        parameters = AnalysisParameters(
            sample_interval_s=self._parse_float(
                self.sample_interval_ns_var,
                "サンプリング間隔",
            )
            * 1e-9,
            filter_low_hz=self._parse_float(
                self.filter_low_mhz_var,
                "下限周波数",
            )
            * 1e6,
            filter_high_hz=self._parse_float(
                self.filter_high_mhz_var,
                "上限周波数",
            )
            * 1e6,
            filter_order=self._parse_int(self.filter_order_var, "フィルター次数"),
            filter_passes=self._parse_int(
                self.filter_passes_var,
                "フィルター回数",
            ),
            distance_mm=None,
        )
        display_min_us = self._parse_float(self.display_min_us_var, "表示開始")
        display_max_us = self._parse_float(self.display_max_us_var, "表示終了")
        if display_max_us <= display_min_us:
            raise ValueError("表示終了時間は表示開始時間より大きくしてください。")
        parameters.validate()
        return parameters, (display_min_us, display_max_us)

    def _basic_acquire_reference(self) -> None:
        """基本タブの固定Channel Aレンジで参照波形を取得する。"""

        self.acquire_reference(
            channel_a_range_override=CHANNEL_A_FIXED_RANGE,
        )

    def _basic_acquire_measurement(self) -> None:
        """基本タブの固定Channel Aレンジで実測波形を取得・解析する。"""

        self.acquire_and_analyze(
            channel_a_range_override=CHANNEL_A_FIXED_RANGE,
        )

    def _start_basic_range_preview(self) -> None:
        """変更後のBレンジで実測波形を取得し、フィルター表示だけを作る。"""

        if self._busy:
            return

        try:
            filter_parameters, display_range = self._read_filtered_preview_inputs()
            source_mode = self.source_mode_var.get()
            if source_mode == SOURCE_CSV:
                measurement_source = CsvDataSource(
                    resolve_input_path(self.measurement_path_var.get())
                )
                picoscope_settings = None
            else:
                measurement_source = None
                picoscope_settings = self._read_picoscope_settings(
                    CHANNEL_A_FIXED_RANGE,
                )
            reference_sample_interval_s = (
                self.current_reference_sample_interval_s
                if self.current_reference is not None
                and self.current_reference_source_mode == source_mode
                else None
            )
        except Exception as exc:
            self.result_speed_var.set("—")
            self.status_var.set(
                "Bレンジ確認用の波形取得を開始できませんでした。"
            )
            self._show_error(exc)
            return

        self.current_measurement = None
        self.current_measurement_sample_interval_s = None
        self.current_result = None
        self.current_filtered_preview = None
        self._clear_result_display()
        self.result_speed_var.set("—")
        self._set_busy(True)
        self.status_var.set("Bレンジ変更後の波形を取得しています…")

        def worker() -> None:
            try:
                if source_mode == SOURCE_CSV:
                    assert measurement_source is not None
                    measurement = measurement_source.acquire()
                    actual_sample_interval_s = sample_interval_from_signal(measurement)
                else:
                    assert picoscope_settings is not None
                    measurement, actual_sample_interval_s = acquire_picoscope_signal(
                        picoscope_settings,
                        "実測",
                    )

                if reference_sample_interval_s is not None:
                    require_matching_time_resolution(
                        reference_sample_interval_s,
                        actual_sample_interval_s,
                    )

                actual_parameters = replace(
                    filter_parameters,
                    sample_interval_s=actual_sample_interval_s,
                )
                actual_parameters.validate()
                filter_arguments = (
                    actual_parameters.filter_low_hz,
                    actual_parameters.filter_high_hz,
                    1.0 / actual_sample_interval_s,
                    actual_parameters.filter_order,
                    actual_parameters.filter_passes,
                )
                preview = FilteredWaveformPreview(
                    measurement=measurement,
                    filtered_channel_1=butter_bandpass_filter(
                        measurement.channel_1,
                        *filter_arguments,
                    ),
                    filtered_channel_2=butter_bandpass_filter(
                        measurement.channel_2,
                        *filter_arguments,
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda error=exc: self._basic_range_preview_failed(error),
                )
                return

            self.root.after(
                0,
                lambda: self._basic_range_preview_completed(
                    preview,
                    actual_sample_interval_s,
                    display_range,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _basic_range_preview_completed(
        self,
        preview: FilteredWaveformPreview,
        sample_interval_s: float,
        display_range: tuple[float, float],
    ) -> None:
        """レンジ確認波形を保存し、評価関数と音速を出さずに描画する。"""

        self.current_measurement = preview.measurement
        self.current_measurement_sample_interval_s = sample_interval_s
        self.current_result = None
        self.current_filtered_preview = preview
        self.sample_interval_ns_var.set(f"{sample_interval_s * 1e9:.9g}")
        self.result_speed_var.set("—")
        self._draw_basic_range_preview(preview, display_range)
        self.status_var.set(
            f"Bレンジ確認用の波形取得完了: {preview.measurement.source_name}"
        )
        self._set_busy(False)
        self._select_distance_for_next_measurement()

    def _basic_range_preview_failed(self, error: Exception) -> None:
        """レンジ確認用の取得・フィルター処理失敗を画面へ反映する。"""

        self._set_busy(False)
        self.current_result = None
        self.current_filtered_preview = None
        if isinstance(error, TimeResolutionMismatchError):
            self._discard_acquired_data()
            self.status_var.set(
                "時間分解能が変わりました。参照波形を取り直してください。"
            )
        else:
            self.status_var.set("Bレンジ確認用の波形取得に失敗しました。")
        self.result_speed_var.set("—")
        self._show_error(error)
        self._select_distance_for_next_measurement()

    def acquire_reference(
        self,
        channel_a_range_override: str | None = None,
    ) -> None:
        """参照波形を取得して保持し、次の実測取得を待つ。"""

        if self._busy:
            return

        try:
            source_mode = self.source_mode_var.get()
            if source_mode == SOURCE_CSV:
                csv_source = CsvDataSource(
                    resolve_input_path(self.reference_path_var.get())
                )
                picoscope_settings = None
            else:
                csv_source = None
                picoscope_settings = self._read_picoscope_settings(
                    channel_a_range_override,
                )
        except Exception as exc:
            self._show_error(exc)
            return

        # 再取得が失敗した場合に以前の参照を誤使用しないよう、開始時点で破棄する。
        self.current_reference = None
        self.current_reference_sample_interval_s = None
        self.current_reference_source_mode = None
        self.current_reference_channel_a_range = (
            None
            if picoscope_settings is None
            else picoscope_settings.channel_a_range
        )
        self.current_measurement = None
        self.current_measurement_sample_interval_s = None
        self.current_result = None
        self.current_filtered_preview = None
        self._clear_result_display()
        self._set_busy(True)
        self.status_var.set("参照データを取得しています…")

        def worker() -> None:
            try:
                if source_mode == SOURCE_CSV:
                    assert csv_source is not None
                    reference = csv_source.acquire()
                    sample_interval_s = sample_interval_from_signal(reference)
                else:
                    assert picoscope_settings is not None
                    reference, sample_interval_s = acquire_picoscope_signal(
                        picoscope_settings,
                        "参照",
                    )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda error=exc: self._reference_acquisition_failed(error),
                )
                return

            self.root.after(
                0,
                lambda: self._reference_acquisition_completed(
                    reference,
                    sample_interval_s,
                    source_mode,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _reference_acquisition_completed(
        self,
        reference: SignalData,
        sample_interval_s: float | None,
        source_mode: str,
    ) -> None:
        """参照取得成功時に波形と実サンプリング間隔を保存する。"""

        self.current_reference = reference
        self.current_reference_sample_interval_s = sample_interval_s
        self.current_reference_source_mode = source_mode
        # 新しい参照に対する実測を行うまでは、以前の実測結果を再利用しない。
        self.current_measurement = None
        self.current_measurement_sample_interval_s = None
        self.current_result = None
        self.current_filtered_preview = None
        if sample_interval_s is not None:
            self.sample_interval_ns_var.set(f"{sample_interval_s * 1e9:.9g}")
        self.status_var.set(
            f"参照取得完了: {reference.source_name}。次に実測データを取得してください。"
        )
        self._set_busy(False)
        self._select_distance_for_next_measurement()

    def _reference_acquisition_failed(self, error: Exception) -> None:
        """参照取得失敗時にUI操作を戻してエラーを表示する。"""

        self.current_reference_channel_a_range = None
        self._set_busy(False)
        self.status_var.set("参照データの取得に失敗しました。")
        self._show_error(error)

    def acquire_and_analyze(
        self,
        channel_a_range_override: str | None = None,
    ) -> None:
        """実測波形を取得し、保持済みの参照波形を使って解析する。"""

        self._start_analysis(
            load_new_data=True,
            channel_a_range_override=channel_a_range_override,
        )

    def _enter_pressed(self, event: tk.Event) -> str:
        """Enterを押した欄に応じて新規取得または再計算する。"""

        if event.widget in self.distance_entries:
            # 距離が空のときは、誤ったEnterで実機取得を開始しない。
            if self.distance_mm_var.get().strip():
                if event.widget is self.basic_distance_entry:
                    self.acquire_and_analyze(
                        channel_a_range_override=CHANNEL_A_FIXED_RANGE,
                    )
                else:
                    # Advancedの「実測データを取得・計算」と同じ処理。
                    self.acquire_and_analyze()
        else:
            self.recalculate()
        # Tkのbindtags伝播を止め、1回のEnterで二重実行されるのを防ぐ。
        return "break"

    def recalculate(self) -> None:
        """取得済み波形へ現在のパラメーターを適用して再解析する。"""

        if self.current_measurement is None or self.current_reference is None:
            self._show_error(
                ValueError("参照データと実測データを取得してから再計算してください。")
            )
            return
        self._start_analysis(load_new_data=False)

    def _start_analysis(
        self,
        load_new_data: bool,
        channel_a_range_override: str | None = None,
    ) -> None:
        """入力値を確定し、データ取得と解析をバックグラウンドで実行する。"""

        # 連打によって複数解析が同時に走ると結果の表示順が入れ替わるため抑止する。
        if self._busy:
            return

        try:
            # Tkinter変数はメインスレッドで読み取り、通常のPython値に変換しておく。
            parameters, window_parameters, display_range = self._read_inputs()
            source_mode = self.source_mode_var.get()
            if self.current_reference is None:
                raise ValueError("先に「参照データを取得」を実行してください。")
            if self.current_reference_source_mode != source_mode:
                raise ValueError("現在の取得元で参照データを取り直してください。")
            reference_sample_interval_s = self.current_reference_sample_interval_s
            if reference_sample_interval_s is None:
                raise ValueError("参照データのサンプリング間隔を取得できません。")

            if load_new_data:
                if source_mode == SOURCE_CSV:
                    measurement_source = CsvDataSource(
                        resolve_input_path(self.measurement_path_var.get())
                    )
                    picoscope_settings = None
                else:
                    measurement_source = None
                    picoscope_settings = self._read_picoscope_settings(
                        channel_a_range_override,
                    )
                    if (
                        self.current_reference_channel_a_range
                        != picoscope_settings.channel_a_range
                    ):
                        raise ValueError(
                            "参照波形と実測波形のChannel Aレンジを"
                            "同じ設定にしてください。参照波形を取り直してください。"
                        )
                measurement = None
                reference = self.current_reference
                sample_interval_s = None
            else:
                # 再計算ではメモリに保持した最新データを使い、ファイルI/Oを省く。
                measurement_source = None
                picoscope_settings = None
                measurement = self.current_measurement
                reference = self.current_reference
                sample_interval_s = self.current_measurement_sample_interval_s
        except Exception as exc:
            self._show_error(exc)
            return

        if load_new_data:
            # 新規取得に失敗した後、以前の実測を今回の結果と誤認しないよう破棄する。
            self.current_measurement = None
            self.current_measurement_sample_interval_s = None
        # 新規取得と再計算のどちらでも、今回の処理中に旧結果を表示しない。
        self.current_result = None
        self.current_filtered_preview = None
        self._clear_result_display()
        self._set_busy(True)
        action = "データを取得して解析しています…" if load_new_data else "再計算しています…"
        self.status_var.set(action)

        def worker() -> None:
            """時間の掛かる取得・フィルター・相関処理を行う作業スレッド。"""

            acquired_measurement = measurement
            acquired_reference = reference
            actual_sample_interval_s = sample_interval_s
            measurement_is_reusable = False
            try:
                if load_new_data:
                    if source_mode == SOURCE_CSV:
                        assert measurement_source is not None
                        acquired_measurement = measurement_source.acquire()
                        actual_sample_interval_s = sample_interval_from_signal(
                            acquired_measurement
                        )
                    else:
                        assert picoscope_settings is not None
                        acquired_measurement, actual_sample_interval_s = (
                            acquire_picoscope_signal(
                                picoscope_settings,
                                "実測",
                            )
                        )
                assert acquired_measurement is not None
                assert acquired_reference is not None
                if actual_sample_interval_s is None:
                    actual_sample_interval_s = sample_interval_from_signal(
                        acquired_measurement
                    )
                require_matching_time_resolution(
                    reference_sample_interval_s,
                    actual_sample_interval_s,
                )
                measurement_is_reusable = True

                # フィルター、窓、照合時間にはUIの旧値ではなく実機/時間列の値を使用する。
                analysis_parameters = replace(
                    parameters,
                    sample_interval_s=actual_sample_interval_s,
                )
                analysis_window_parameters = replace(
                    window_parameters,
                    sample_interval_s=actual_sample_interval_s,
                )

                # UI部品を触らず、解析モジュールへ通常の配列と設定値だけを渡す。
                result = run_analysis(
                    acquired_measurement,
                    acquired_reference,
                    analysis_parameters,
                    analysis_window_parameters,
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda error=exc,
                    acquired=(
                        acquired_measurement if measurement_is_reusable else None
                    ),
                    interval=(
                        actual_sample_interval_s if measurement_is_reusable else None
                    ): self._analysis_failed(error, acquired, interval),
                )
                return

            # TkinterとMatplotlibの更新は必ずメインスレッドで行う必要があるため、
            # root.after(0, ...)で完了処理をメインイベントループへ戻す。
            self.root.after(
                0,
                lambda: self._analysis_completed(
                    result,
                    display_range,
                    actual_sample_interval_s,
                ),
            )

        # daemon=Trueにより、UIを閉じたとき作業スレッドだけが残るのを防ぐ。
        threading.Thread(target=worker, daemon=True).start()

    def _analysis_completed(
        self,
        result: AnalysisResult,
        display_range: tuple[float, float],
        sample_interval_s: float | None,
    ) -> None:
        """解析成功時に最新データを保存し、グラフと結果ラベルを更新する。"""

        self.current_measurement = result.measurement
        self.current_measurement_sample_interval_s = sample_interval_s
        self.current_reference = result.reference
        self.current_result = result
        self.current_filtered_preview = None
        if sample_interval_s is not None:
            self.sample_interval_ns_var.set(f"{sample_interval_s * 1e9:.9g}")
        self._draw_result(result, display_range)
        self._update_result_labels(result)
        self.status_var.set(
            f"完了: {result.measurement.source_name} / {result.reference.source_name}"
        )
        self._set_busy(False)
        self._select_distance_for_next_measurement()

    def _analysis_failed(
        self,
        error: Exception,
        measurement: SignalData | None = None,
        sample_interval_s: float | None = None,
    ) -> None:
        """解析失敗時に取得済み波形を保持し、UI操作を再開する。"""

        self._set_busy(False)
        self.current_result = None
        if isinstance(error, TimeResolutionMismatchError):
            # 新しい時間分解能の波形とは照合できないため、参照を明示的に
            # 無効化して基本タブの次操作を参照取得へ戻す。
            self._discard_acquired_data()
            self.status_var.set(
                "時間分解能が変わりました。参照波形を取り直してください。"
            )
            self._show_error(error)
            return
        if measurement is None or sample_interval_s is None:
            self.status_var.set(
                "実測データの取得または解析に失敗しました。"
                "入力値と取得条件を確認してください。"
            )
        else:
            # 取得自体が成功した場合は、実機から取り直さず再計算できる。
            self.current_measurement = measurement
            self.current_measurement_sample_interval_s = sample_interval_s
            self.sample_interval_ns_var.set(f"{sample_interval_s * 1e9:.9g}")
            self.status_var.set(
                "実測データは取得済みですが、解析に失敗しました。"
                "設定を修正して「現在データで再計算」を押してください。"
            )
        self._show_error(error)

    def _show_error(self, error: Exception) -> None:
        """例外メッセージを統一形式のエラーダイアログで表示する。"""

        messagebox.showerror("エラー", str(error), parent=self.root)

    def _set_busy(self, busy: bool) -> None:
        """解析中の多重操作を防ぐため、実行関連ボタンの有効状態を切り替える。"""

        self._busy = busy
        state = "disabled" if busy else "normal"
        self.acquire_reference_button.configure(state=state)
        self.acquire_button.configure(state=state)
        self.recalculate_button.configure(state=state)
        self.save_window_button.configure(state=state)
        self.save_settings_button.configure(state=state)
        self.load_settings_button.configure(state=state)
        self.basic_reference_button.configure(state=state)
        self.basic_acquire_button.configure(state=state)
        self.basic_display_normal_button.configure(state=state)
        self.basic_display_expanded_button.configure(state=state)
        self._update_channel_b_range_buttons()
        range_combobox_state = "disabled" if busy else "readonly"
        self.channel_a_range_combobox.configure(state=range_combobox_state)
        self.channel_b_range_combobox.configure(state=range_combobox_state)
        self.source_combobox.configure(state="disabled" if busy else "readonly")

    def _select_distance_for_next_measurement(self) -> None:
        """次の数値入力で現在の距離をそのまま置き換えられるようにする。"""

        distance_entry = self._active_distance_entry()
        distance_entry.focus_set()
        distance_entry.icursor(tk.END)
        distance_entry.selection_range(0, tk.END)

    def _active_distance_entry(self) -> ttk.Entry:
        """現在表示中のタブにある距離入力欄を返す。"""

        if self.notebook.select() == str(self.basic_tab):
            return self.basic_distance_entry
        return self.distance_entry

    def _clear_result_display(self) -> None:
        """取得条件が変わったとき、以前の解析表示を消去する。"""

        self.result_time_var.set("—")
        self.result_score_var.set("—")
        self.result_speed_var.set("距離未入力")
        self.result_window_var.set("—")
        self._draw_empty_plots()

    def _draw_empty_plots(self) -> None:
        """両タブの全グラフへデータ取得前の待機表示を描く。"""

        titles = (
            (self.raw_axis, "取得データ"),
            (self.filtered_axis, "フィルター後データ"),
            (self.window_axis, "窓関数・参照波形"),
            (self.matching_axis, "照合結果"),
            (self.basic_filtered_axis, "データフィルタ後"),
            (self.basic_matching_axis, "評価関数"),
        )
        for axis, title in titles:
            axis.clear()
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "データ読み込み待ち",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="0.45",
            )
            axis.grid(True, alpha=0.25)
        channel_b_range_v = self._channel_b_range_v()
        for axis in (self.filtered_axis, self.basic_filtered_axis):
            self._configure_filtered_voltage_axis(axis, channel_b_range_v)
        self.canvas.draw_idle()
        self.basic_canvas.draw_idle()

    def _channel_b_range_v(self) -> float:
        """現在のChannel Bレンジの片側フルスケール値をV単位で返す。"""

        try:
            return RANGE_FULL_SCALE_V[self.channel_b_range_var.get()]
        except KeyError as exc:
            raise ValueError("対応していないChannel B入力レンジです。") from exc

    @staticmethod
    def _configure_filtered_voltage_axis(
        axis: Axes,
        channel_b_range_v: float,
    ) -> None:
        """フィルター後グラフへ固定Y軸と入力目安線を設定する。"""

        guide_voltage_v = channel_b_range_v * 0.60
        display_voltage_v = channel_b_range_v * 1.20
        for voltage_v in (-guide_voltage_v, guide_voltage_v):
            axis.axhline(
                voltage_v,
                color="#d55e00",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=1,
            )
        axis.set_ylim(-display_voltage_v, display_voltage_v)

    @staticmethod
    def _draw_filtered_axis(
        axis: Axes,
        result: AnalysisResult | FilteredWaveformPreview,
        display_range: tuple[float, float],
        title: str,
        channel_b_range_v: float,
    ) -> None:
        """指定軸へCh1・Ch2のフィルター後波形を描く。"""

        display_min_us, display_max_us = display_range
        measurement_time_us = result.measurement.time_s * 1e6
        axis.clear()
        axis.plot(
            measurement_time_us,
            result.filtered_channel_1,
            label="Ch1 filtered",
            linewidth=0.9,
        )
        axis.plot(
            measurement_time_us,
            result.filtered_channel_2,
            label="Ch2 filtered",
            linewidth=0.9,
        )
        axis.set_title(title)
        axis.set_xlabel("時間 [µs]")
        axis.set_ylabel("電圧 [V]")
        axis.set_xlim(display_min_us, display_max_us)
        MeasurementApplication._configure_filtered_voltage_axis(
            axis,
            channel_b_range_v,
        )
        axis.legend(loc="upper right")

    def _draw_basic_range_preview(
        self,
        preview: FilteredWaveformPreview,
        display_range: tuple[float, float],
    ) -> None:
        """Bレンジ確認のフィルター後波形だけをBasicへ描画する。"""

        self._draw_filtered_axis(
            self.basic_filtered_axis,
            preview,
            display_range,
            "データフィルタ後",
            self._channel_b_range_v(),
        )
        self.basic_filtered_axis.grid(True, alpha=0.25)

        # レンジ確認では照合を行わない。前回の評価関数も残さず、
        # 距離欄から通常計測するまで案内のみを表示する。
        display_min_us, display_max_us = display_range
        self.basic_matching_axis.clear()
        self.basic_matching_axis.set_title("評価関数")
        self.basic_matching_axis.set_xlabel("時間 [µs]")
        self.basic_matching_axis.set_xlim(
            max(0.0, display_min_us),
            display_max_us,
        )
        self.basic_matching_axis.text(
            0.5,
            0.5,
            "距離入力から計測すると表示されます",
            ha="center",
            va="center",
            transform=self.basic_matching_axis.transAxes,
            color="0.45",
        )
        self.basic_matching_axis.grid(True, alpha=0.25)
        self.basic_canvas.draw_idle()

    @staticmethod
    def _draw_matching_axis(
        axis: Axes,
        result: AnalysisResult,
        display_range: tuple[float, float],
        title_prefix: str = "",
    ) -> None:
        """指定軸へ評価関数と最良一致位置を描く。"""

        display_min_us, display_max_us = display_range
        matching_time_us = result.matching_time_s * 1e6
        best_time_us = result.best_time_s * 1e6
        axis.clear()
        axis.plot(
            matching_time_us,
            result.matching_score,
            label="照合スコア",
            linewidth=1.0,
        )
        axis.axvline(
            best_time_us,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"一致位置 {best_time_us:.3f} µs",
        )
        axis.scatter(
            [best_time_us],
            [result.best_score],
            color="red",
            s=24,
            zorder=3,
        )
        if result.matching_method == MATCH_CORRELATION:
            axis.set_title(f"{title_prefix}正規化相互相関（最大位置を採用）")
            axis.set_ylabel("相関係数")
        else:
            axis.set_title(f"{title_prefix}二乗誤差（最小位置を採用）")
            axis.set_ylabel("二乗誤差")
        axis.set_xlabel("時間 [µs]")
        axis.set_xlim(max(0.0, display_min_us), display_max_us)
        axis.legend(loc="upper right", fontsize=8)

    def _draw_result(
        self,
        result: AnalysisResult,
        display_range: tuple[float, float],
    ) -> None:
        """1回分の解析結果をAdvancedの4枚とBasicの2枚へ描画する。

        左上: 取得したCh1・Ch2の生波形
        右上: Ch1・Ch2のバンドパスフィルター後波形
        左下: 窓関数、参照元、フィルター後参照、窓適用後参照
        右下: 全候補位置の相関係数または二乗誤差と最良位置
        """

        display_min_us, display_max_us = display_range
        # 内部では秒で保持している時間軸を、グラフ用にµsへ変換する。
        measurement_time_us = result.measurement.time_s * 1e6
        channel_b_range_v = self._channel_b_range_v()

        # --- 左上: PicoScope相当の取得生データ ---
        self.raw_axis.clear()
        self.raw_axis.plot(
            measurement_time_us,
            result.measurement.channel_1,
            label="Ch1",
            linewidth=0.9,
        )
        self.raw_axis.plot(
            measurement_time_us,
            result.measurement.channel_2,
            label="Ch2",
            linewidth=0.9,
        )
        self.raw_axis.set_title("取得データ（フィルター前）")
        self.raw_axis.set_xlabel("時間 [µs]")
        self.raw_axis.set_ylabel("電圧 [V]")
        self.raw_axis.set_xlim(display_min_us, display_max_us)
        self.raw_axis.legend(loc="upper right")

        # --- 右上: 同じ全波形へフィルターを掛けた結果 ---
        self._draw_filtered_axis(
            self.filtered_axis,
            result,
            display_range,
            "バンドパスフィルター後",
            channel_b_range_v,
        )

        # --- 左下: 参照波形が窓関数によってどのように切り出されるかを表示 ---
        window_time_us = result.window.time_s * 1e6
        self.window_axis.clear()
        self.window_axis.plot(
            window_time_us,
            result.window.gain,
            label="Window gain",
            color="black",
            linewidth=1.6,
        )
        self.window_axis.plot(
            window_time_us,
            result.reference_raw_normalized,
            label="Reference raw",
            alpha=0.45,
        )
        self.window_axis.plot(
            window_time_us,
            result.reference_filtered_normalized,
            label="Reference filtered",
            alpha=0.75,
        )
        self.window_axis.plot(
            window_time_us,
            result.reference_windowed,
            label="Reference × window",
            linewidth=1.5,
        )
        self.window_axis.set_title("窓関数と参照波形")
        self.window_axis.set_xlabel("窓内時間 [µs]")
        self.window_axis.set_ylabel("正規化振幅 / ゲイン")
        self.window_axis.legend(loc="upper right", fontsize=8)

        # --- 右下: 参照波形を移動させた各位置の照合スコア ---
        self._draw_matching_axis(self.matching_axis, result, display_range)

        # 4グラフへ共通して薄いグリッドを表示し、値を読み取りやすくする。
        for axis in (
            self.raw_axis,
            self.filtered_axis,
            self.window_axis,
            self.matching_axis,
        ):
            axis.grid(True, alpha=0.25)

        # Basicでは、同じデータからフィルター後波形と評価関数だけを表示する。
        self._draw_filtered_axis(
            self.basic_filtered_axis,
            result,
            display_range,
            "データフィルタ後",
            channel_b_range_v,
        )
        self._draw_matching_axis(
            self.basic_matching_axis,
            result,
            display_range,
            title_prefix="評価関数：",
        )
        self.basic_filtered_axis.grid(True, alpha=0.25)
        self.basic_matching_axis.grid(True, alpha=0.25)

        self.canvas.draw_idle()
        self.basic_canvas.draw_idle()

    def _update_result_labels(self, result: AnalysisResult) -> None:
        """解析結果を読みやすい単位・桁数へ整形して左パネルへ表示する。"""

        self.result_time_var.set(f"{result.best_time_s * 1e6:.4f} µs")
        self.result_score_var.set(f"{result.best_score:.6g}")
        if result.sound_speed_m_s is None:
            self.result_speed_var.set("距離未入力")
        else:
            self.result_speed_var.set(f"{result.sound_speed_m_s:.2f} m/s")

        # 窓のサンプル数と、先頭から最終サンプルまでの時間を併記する。
        duration_us = result.window.time_s[-1] * 1e6 if result.window.time_s.size else 0.0
        self.result_window_var.set(
            f"{result.window.gain.size} samples / {duration_us:.3f} µs"
        )

    def _save_window(self) -> None:
        """現在の入力値で窓を生成し、ユーザーが選んだCSVへ保存する。"""

        try:
            window = generate_window(self._read_window_parameters())
        except Exception as exc:
            self._show_error(exc)
            return

        # 保存ボタンを押した場合だけダイアログを開く。自動解析ではファイルを書かない。
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="窓関数CSVを保存",
            initialdir=str(APP_DIR),
            initialfile="window_generated.csv",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not destination:
            return

        try:
            save_window_csv(window, Path(destination))
        except Exception as exc:
            self._show_error(exc)
            return
        self.status_var.set(f"窓関数を保存しました: {destination}")


def main() -> None:
    """フォント設定、Tkルート作成、イベントループ開始を行う起動関数。"""

    configure_plot_font()
    root = tk.Tk()
    MeasurementApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
