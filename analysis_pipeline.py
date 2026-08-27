"""フィルター、窓処理、波形照合、音速計算を行う解析モジュール。

TkinterやCSV読み込みには依存せず、すでに取得済みの :class:`SignalData` を受け取り、
次の順番で解析する。

1. 測定波形のCh1・Ch2へバンドパスフィルターを適用する
2. 0秒以降の測定対象チャンネルを切り出して正規化する
3. 参照データの指定位置から、窓関数と同じ長さの波形を切り出す
4. 参照波形をフィルター・正規化し、窓関数を掛ける
5. 測定波形上で参照波形を1サンプルずつ移動させて照合スコアを計算する
6. 最大相関または最小二乗誤差の位置を到達時間へ変換する
7. 距離が入力されていれば「距離÷時間」で音速を計算する

UI以外からも単体で呼び出せるため、将来PicoScope取得処理を追加しても解析部分を
そのまま再利用できる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from data_sources import SignalData
from window_functions import GeneratedWindow, WindowParameters, generate_window


# 照合方式をUI表示文字列から独立させるための内部識別子。
MATCH_CORRELATION = "normalized_correlation"  # 正規化相互相関の最大値を採用
MATCH_SQUARED_ERROR = "squared_error"  # 既存コードと同じ二乗誤差の最小値を採用


@dataclass(frozen=True)
class AnalysisParameters:
    """1回の解析に使用する設定値。

    UI上では見やすいns、µs、MHz、mmで入力するが、このクラスでは計算しやすい
    SI単位（秒、Hz）を基本として保持する。
    """

    sample_interval_s: float = 8e-9  # サンプリング間隔 [s]
    filter_low_hz: float = 1e6  # バンドパス下限 [Hz]
    filter_high_hz: float = 5e6  # バンドパス上限 [Hz]
    filter_order: int = 4  # Butterworthフィルター次数
    filter_passes: int = 2  # sosfiltfiltを繰り返す回数（既存コードは2回）
    reference_start_s: float = 0.0  # 参照データの切り出し開始時刻 [s]
    measurement_channel: int = 2  # 照合対象にする測定チャンネル
    reference_channel: int = 2  # テンプレートにする参照チャンネル
    matching_method: str = MATCH_CORRELATION  # 相関または二乗誤差
    distance_mm: float | None = None  # 音速計算用の測定距離 [mm]

    def validate(self) -> None:
        """フィルター設計や照合を始める前に、全設定値を検査する。"""

        if self.sample_interval_s <= 0:
            raise ValueError("サンプリング間隔は0より大きくしてください。")
        if self.filter_low_hz <= 0:
            raise ValueError("フィルター下限は0より大きくしてください。")
        if self.filter_high_hz <= self.filter_low_hz:
            raise ValueError("フィルター上限は下限より大きくしてください。")

        # デジタルフィルターの上限は、サンプリング周波数の半分である
        # ナイキスト周波数より低くなければならない。
        nyquist_hz = 0.5 / self.sample_interval_s
        if self.filter_high_hz >= nyquist_hz:
            raise ValueError(
                f"フィルター上限はナイキスト周波数 {nyquist_hz / 1e6:.3f} MHz "
                "未満にしてください。"
            )
        if self.filter_order < 1:
            raise ValueError("フィルター次数は1以上にしてください。")
        if self.filter_passes < 1:
            raise ValueError("フィルター回数は1以上にしてください。")
        if self.measurement_channel not in (1, 2):
            raise ValueError("測定チャンネルは1または2にしてください。")
        if self.reference_channel not in (1, 2):
            raise ValueError("参照チャンネルは1または2にしてください。")
        if self.matching_method not in (MATCH_CORRELATION, MATCH_SQUARED_ERROR):
            raise ValueError("対応していない照合方法です。")
        if self.distance_mm is not None and self.distance_mm < 0:
            raise ValueError("距離は0以上にしてください。")


@dataclass(frozen=True)
class AnalysisResult:
    """解析結果と、UIで4枚のグラフを描くための途中データ一式。

    最良値だけでなく、フィルター波形、参照波形、窓関数、全照合スコアを保持する。
    UIは計算をやり直さず、このオブジェクトだけから全グラフを更新できる。
    """

    measurement: SignalData
    reference: SignalData  # 参照用CSVの元データ
    filtered_channel_1: np.ndarray  # 表示用Ch1フィルター波形
    filtered_channel_2: np.ndarray  # 表示用Ch2フィルター波形
    selected_measurement_time_s: np.ndarray  # 0秒以降の相対時間軸
    selected_measurement_filtered: np.ndarray  # 照合に使った正規化測定波形
    window: GeneratedWindow  # 生成した窓関数
    reference_raw_normalized: np.ndarray  # 切り出した参照元波形
    reference_filtered_normalized: np.ndarray  # フィルター後参照波形
    reference_windowed: np.ndarray  # フィルター後参照波形×窓
    matching_time_s: np.ndarray  # 各照合位置の相対時間 [s]
    matching_score: np.ndarray  # 相関係数または二乗誤差
    matching_method: str  # 今回使用した照合方式
    best_index: int  # 最大相関または最小誤差になった配列位置
    best_time_s: float  # best_indexを時間へ変換した値 [s]
    best_score: float  # 最良位置での相関係数または二乗誤差
    sound_speed_m_s: float | None  # 距離が入力された場合の音速 [m/s]


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut_hz: float,
    highcut_hz: float,
    sample_rate_hz: float,
    order: int,
    passes: int,
) -> np.ndarray:
    """Butterworthバンドパスフィルターを指定回数適用する。

    ``output="sos"`` の2次セクション形式は、高次フィルターでも数値的に安定しやすい。
    ``sosfiltfilt`` は波形を順方向・逆方向の両方から処理するため、フィルターによる
    位相ずれを抑えられる。既存の ``main_2filterd_with_window.py`` に合わせ、
    既定ではこのゼロ位相フィルターを2回繰り返す。
    """

    # フィルター係数はデータごとではなく、設定値から一度だけ作る。
    sos = butter(
        order,
        [lowcut_hz, highcut_hz],
        btype="band",
        fs=sample_rate_hz,
        output="sos",
    )
    # 入力配列そのものを書き換えないよう、新しいfloat配列として扱う。
    filtered = np.asarray(data, dtype=float)
    try:
        for _ in range(passes):
            filtered = sosfiltfilt(sos, filtered)
    except ValueError as exc:
        raise ValueError(
            "フィルター処理に対してデータが短すぎるか、設定値が不正です。"
        ) from exc
    return filtered


def _normalize_peak(data: np.ndarray, description: str) -> np.ndarray:
    """最大絶対振幅が1になるよう波形を正規化する。"""

    # 振幅の単位やPicoScopeレンジが異なっても、波形形状を中心に比較できるようにする。
    peak = float(np.max(np.abs(data)))
    if not np.isfinite(peak) or peak <= np.finfo(float).eps:
        raise ValueError(f"{description}の振幅が0のため正規化できません。")
    return np.asarray(data, dtype=float) / peak


def _matching_scores(
    measurement: np.ndarray,
    reference: np.ndarray,
    window_gain: np.ndarray,
    method: str,
) -> np.ndarray:
    """測定波形上で参照波形を移動させ、各位置の照合スコアを返す。

    ``mode="valid"`` により、参照波形が測定波形からはみ出さない位置だけを計算する。

    正規化相互相関:
        窓付き測定区間と窓付き参照波形の内積を、両者のエネルギーで割る。
        波形形状がよく一致するほど1へ近づき、最大位置を採用する。

    二乗誤差（従来方式）:
        ``sum((測定区間×窓 - 参照波形×窓)^2)`` を計算する。
        波形がよく一致するほど0へ近づき、最小位置を採用する。
    """

    if not (reference.size == window_gain.size):
        raise ValueError("参照波形と窓関数の長さが一致しません。")
    if measurement.size < reference.size:
        raise ValueError("測定波形が参照波形より短いため照合できません。")

    # 測定区間と参照波形の両方に窓を掛けるため、内積の式では窓が2乗になる。
    window_squared = np.square(window_gain)

    # np.correlateを使うことで、Pythonのforループで1位置ずつ計算するより高速に
    # 全候補位置の内積をまとめて求められる。
    cross = np.correlate(measurement, reference * window_squared, mode="valid")

    # 各測定区間について、窓適用後の二乗和（エネルギー）を求める。
    measurement_energy = np.correlate(
        np.square(measurement), window_squared, mode="valid"
    )
    # 参照波形側は移動しないため、エネルギーは1つのスカラー値になる。
    reference_energy = float(np.sum(np.square(reference) * window_squared))

    if method == MATCH_CORRELATION:
        # 内積を両波形の大きさで割り、振幅差に影響されにくい相関係数にする。
        denominator = np.sqrt(np.maximum(measurement_energy, 0.0) * reference_energy)
        scores = np.zeros_like(cross, dtype=float)
        # エネルギー0の区間ではゼロ除算せず、相関スコア0のままとする。
        np.divide(cross, denominator, out=scores, where=denominator > 0)
        return scores

    # (a-b)^2 = a^2 - 2ab + b^2 を利用し、従来のforループと同じ
    # 二乗誤差をベクトル演算で計算する。
    scores = measurement_energy - 2.0 * cross + reference_energy
    # 浮動小数点の丸め誤差で完全一致時に微小な負値が出る場合だけ0へ丸める。
    return np.maximum(scores, 0.0)


def run_analysis(
    measurement: SignalData,
    reference: SignalData,
    parameters: AnalysisParameters,
    window_parameters: WindowParameters,
) -> AnalysisResult:
    """取得済み波形に対する一連の解析を実行する中心関数。

    Args:
        measurement: 測定対象の時間・Ch1・Ch2配列。
        reference: ゼロ基準など、送信波形テンプレートを含む参照データ。
        parameters: フィルター・照合・距離などの解析条件。
        window_parameters: 窓関数の各区間時間。

    Returns:
        UI表示に必要な途中波形と最終計算値をまとめた :class:`AnalysisResult`。
    """

    # 配列処理に入る前に、UIからの設定値をまとめて検査する。
    parameters.validate()
    window_parameters.validate()
    if not np.isclose(
        parameters.sample_interval_s,
        window_parameters.sample_interval_s,
        rtol=0.0,
        atol=np.finfo(float).eps,
    ):
        raise ValueError("解析と窓関数のサンプリング間隔が一致していません。")

    # 現在の入力値から毎回窓を作り直すため、窓時間を変更して「再計算」すると
    # すぐにグラフと照合結果へ反映される。
    window = generate_window(window_parameters)
    sample_rate_hz = 1.0 / parameters.sample_interval_s

    # 同じフィルター条件をCh1、Ch2、測定対象、参照波形へ共通適用する。
    filter_arguments = (
        parameters.filter_low_hz,
        parameters.filter_high_hz,
        sample_rate_hz,
        parameters.filter_order,
        parameters.filter_passes,
    )
    filtered_channel_1 = butter_bandpass_filter(
        measurement.channel_1, *filter_arguments
    )
    filtered_channel_2 = butter_bandpass_filter(
        measurement.channel_2, *filter_arguments
    )

    # CSVにはトリガー前の負時間も含まれる。0秒に最も近いサンプルを探し、
    # 照合対象はそこから後ろだけにする。
    measurement_start = int(np.abs(measurement.time_s - 0.0).argmin())
    selected_measurement_time_s = (
        np.arange(measurement.time_s.size - measurement_start, dtype=float)
        * parameters.sample_interval_s
    )
    # UIで選んだ測定チャンネルだけを、照合用として別途フィルター・正規化する。
    selected_measurement_raw = measurement.channel(parameters.measurement_channel)[
        measurement_start:
    ]
    selected_measurement_filtered = butter_bandpass_filter(
        selected_measurement_raw, *filter_arguments
    )
    selected_measurement_filtered = _normalize_peak(
        selected_measurement_filtered, "測定波形"
    )

    # 参照CSVから、指定開始時刻に最も近い位置を探す。
    reference_start = int(
        np.abs(reference.time_s - parameters.reference_start_s).argmin()
    )
    reference_end = reference_start + window.gain.size
    if reference_end > reference.time_s.size:
        raise ValueError(
            "参照開始位置から窓関数の長さだけデータを取得できません。"
            "参照開始時間または窓の長さを調整してください。"
        )

    # 窓関数と同じ長さだけ参照波形を切り出す。長さを揃えることで、
    # 要素ごとの乗算と照合計算が可能になる。
    reference_raw = reference.channel(parameters.reference_channel)[
        reference_start:reference_end
    ]
    reference_raw_normalized = _normalize_peak(reference_raw, "参照元波形")
    reference_filtered = butter_bandpass_filter(reference_raw, *filter_arguments)
    reference_filtered_normalized = _normalize_peak(reference_filtered, "参照波形")
    # 窓関数により、参照波形のうち照合へ強く反映する時間範囲を限定する。
    reference_windowed = reference_filtered_normalized * window.gain

    # 測定波形の先頭から末尾まで参照波形を移動し、全候補位置を評価する。
    matching_score = _matching_scores(
        selected_measurement_filtered,
        reference_filtered_normalized,
        window.gain,
        parameters.matching_method,
    )
    matching_time_s = (
        np.arange(matching_score.size, dtype=float) * parameters.sample_interval_s
    )

    # 相関は大きいほど一致、二乗誤差は小さいほど一致なので最良値の探し方が異なる。
    if parameters.matching_method == MATCH_CORRELATION:
        best_index = int(np.argmax(matching_score))
    else:
        best_index = int(np.argmin(matching_score))
    best_time_s = float(matching_time_s[best_index])
    best_score = float(matching_score[best_index])

    # 距離が空欄なら一致時間だけを表示する。距離が入力されている場合だけ
    # 距離[m] / 伝搬時間[s] で音速を計算する。
    sound_speed_m_s = None
    if parameters.distance_mm is not None and parameters.distance_mm > 0:
        if best_time_s <= 0:
            raise ValueError("一致時間が0秒のため音速を計算できません。")
        sound_speed_m_s = parameters.distance_mm * 1e-3 / best_time_s

    # UIが追加計算なしで全グラフを描けるよう、途中結果もまとめて返す。
    return AnalysisResult(
        measurement=measurement,
        reference=reference,
        filtered_channel_1=filtered_channel_1,
        filtered_channel_2=filtered_channel_2,
        selected_measurement_time_s=selected_measurement_time_s,
        selected_measurement_filtered=selected_measurement_filtered,
        window=window,
        reference_raw_normalized=reference_raw_normalized,
        reference_filtered_normalized=reference_filtered_normalized,
        reference_windowed=reference_windowed,
        matching_time_s=matching_time_s,
        matching_score=matching_score,
        matching_method=parameters.matching_method,
        best_index=best_index,
        best_time_s=best_time_s,
        best_score=best_score,
        sound_speed_m_s=sound_speed_m_s,
    )
