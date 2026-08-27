"""PicoScope 5000DからA/B波形を取得して呼び出し元へ返すモジュール。

使用例::

    import pypicosdk as psdk

    from picoscope_acquisition import acquire_waveform

    capture = acquire_waveform(
        duration_s=20e-6,
        trigger_a_threshold_mv=-2000,
        trigger_a_direction=psdk.TRIGGER_DIR.FALLING,
        resolution=psdk.RESOLUTION.BIT_8,
        channel_a_range=psdk.RANGE.V20,
        channel_b_range=psdk.RANGE.V10,
    )
    time_s = capture.time_s
    channel_a_mv = capture.channel_a_mv
    channel_b_mv = capture.channel_b_mv

``auto_trigger=0`` のため、Channel Aで指定したトリガーが発生するまで呼び出しは待機する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pypicosdk as psdk


DEFAULT_CHANNEL_A_RANGE = psdk.RANGE.V20
DEFAULT_CHANNEL_B_RANGE = psdk.RANGE.V10
PRE_TRIGGER_PERCENT = 0
MINIMUM_SAMPLES = 8

_VALID_TRIGGER_DIRECTIONS = frozenset(
    {
        psdk.TRIGGER_DIR.ABOVE,
        psdk.TRIGGER_DIR.BELOW,
        psdk.TRIGGER_DIR.RISING,
        psdk.TRIGGER_DIR.FALLING,
        psdk.TRIGGER_DIR.RISING_OR_FALLING,
    }
)
_VALID_RESOLUTIONS = frozenset(
    {
        psdk.RESOLUTION.BIT_8,
        psdk.RESOLUTION.BIT_12,
        psdk.RESOLUTION.BIT_14,
        psdk.RESOLUTION.BIT_15,
        psdk.RESOLUTION.BIT_16,
    }
)
_RANGE_FULL_SCALE_MV = {
    psdk.RANGE.mV10: 10.0,
    psdk.RANGE.mV20: 20.0,
    psdk.RANGE.mV50: 50.0,
    psdk.RANGE.mV100: 100.0,
    psdk.RANGE.mV200: 200.0,
    psdk.RANGE.mV500: 500.0,
    psdk.RANGE.V1: 1_000.0,
    psdk.RANGE.V2: 2_000.0,
    psdk.RANGE.V5: 5_000.0,
    psdk.RANGE.V10: 10_000.0,
    psdk.RANGE.V20: 20_000.0,
}
_VALID_RANGES = frozenset(_RANGE_FULL_SCALE_MV)


class WaveformAcquisitionError(RuntimeError):
    """PicoScopeの設定または波形取得に失敗したことを表す例外。"""


@dataclass(frozen=True)
class WaveformCapture:
    """1回の取得で得られた波形と実際の取得条件。

    Attributes:
        time_s: トリガー時刻を0とする時間軸 [s]。
        channel_a_mv: Channel Aの電圧 [mV]。
        channel_b_mv: Channel Bの電圧 [mV]。
        sample_interval_s: PicoScopeが設定したサンプリング間隔 [s]。
        timebase: PicoScope SDKへ渡したタイムベース番号。
        resolution: 取得時のADC分解能。
        device_variant: 接続したPicoScopeのモデル名。
        channel_a_range: Channel Aに設定した入力レンジ。
        channel_b_range: Channel Bに設定した入力レンジ。
    """

    time_s: np.ndarray
    channel_a_mv: np.ndarray
    channel_b_mv: np.ndarray
    sample_interval_s: float
    timebase: int
    resolution: int
    device_variant: str
    channel_a_range: int
    channel_b_range: int

    @property
    def sample_rate_hz(self) -> float:
        """実際のサンプリングレート [samples/s] を返す。"""

        return 1.0 / self.sample_interval_s

    @property
    def actual_duration_s(self) -> float:
        """返された波形の先頭から末尾までの時間 [s] を返す。"""

        if self.time_s.size < 2:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """時間、Channel A、Channel Bの順で配列を返す。"""

        return self.time_s, self.channel_a_mv, self.channel_b_mv


def _validate_arguments(
    duration_s: float,
    trigger_a_threshold_mv: float,
    trigger_a_direction: int,
    resolution: int,
    channel_a_range: int,
    channel_b_range: int,
) -> None:
    """実機を開く前に呼び出し引数を検査する。"""

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_sには0より大きい有限値を指定してください。")
    if not math.isfinite(trigger_a_threshold_mv):
        raise ValueError("trigger_a_threshold_mvには有限値を指定してください。")
    if trigger_a_direction not in _VALID_TRIGGER_DIRECTIONS:
        raise ValueError("trigger_a_directionに有効なTRIGGER_DIRを指定してください。")
    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError("resolutionに5000Dで使用可能なRESOLUTIONを指定してください。")
    if channel_a_range not in _VALID_RANGES:
        raise ValueError("channel_a_rangeに有効なRANGEを指定してください。")
    if channel_b_range not in _VALID_RANGES:
        raise ValueError("channel_b_rangeに有効なRANGEを指定してください。")
    channel_a_full_scale_mv = _RANGE_FULL_SCALE_MV[channel_a_range]
    if abs(trigger_a_threshold_mv) > channel_a_full_scale_mv:
        raise ValueError(
            "trigger_a_threshold_mvはChannel Aの入力レンジ内にしてください: "
            f"しきい値 {trigger_a_threshold_mv:g} mV / "
            f"レンジ ±{channel_a_full_scale_mv:g} mV"
        )


def _select_fastest_capture(
    scope: psdk.ps5000a,
    duration_s: float,
) -> tuple[int, int, float]:
    """現在のチャンネル構成で使用できる最短タイムベースを選択する。"""

    fastest = scope.get_minimum_timebase_stateless()
    timebase = int(fastest["timebase"])
    fastest_interval_s = float(fastest["time_interval"])
    if not math.isfinite(fastest_interval_s) or fastest_interval_s <= 0:
        raise WaveformAcquisitionError(
            "PicoScopeが有効なサンプリング間隔を返しませんでした。"
        )

    # N点の時間軸が持つ区間はN-1個なので、指定時間を必ず
    # 含むように終点の1点を追加する。
    samples = max(
        MINIMUM_SAMPLES,
        math.ceil(duration_s / fastest_interval_s) + 1,
    )

    # get_timebase()はps5000aGetTimebase2を呼び、選択したタイムベースの
    # 実サンプリング間隔と、現在のメモリセグメントで取得可能な点数を返す。
    timebase_info = scope.get_timebase(timebase, samples)
    sample_interval_s = float(timebase_info["Interval(ns)"]) * 1e-9
    max_samples = int(timebase_info["Samples"])

    if not math.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise WaveformAcquisitionError(
            "PicoScopeがタイムベースを検証できませんでした。"
        )
    if samples > max_samples:
        raise WaveformAcquisitionError(
            "最高時間分解能で指定時間を取得するためのメモリが不足しています: "
            f"必要 {samples:,} samples / 使用可能 {max_samples:,} samples"
        )

    return timebase, samples, sample_interval_s


def acquire_waveform(
    duration_s: float,
    trigger_a_threshold_mv: float,
    trigger_a_direction: int,
    resolution: int,
    channel_a_range: int = DEFAULT_CHANNEL_A_RANGE,
    channel_b_range: int = DEFAULT_CHANNEL_B_RANGE,
) -> WaveformCapture:
    """PicoScopeから1回分のChannel A/B波形を取得する。

    Args:
        duration_s: 取得する時間長さ [s]。
        trigger_a_threshold_mv: Channel Aのトリガーしきい値 [mV]。
        trigger_a_direction: ``psdk.TRIGGER_DIR`` のいずれか。
        resolution: ``psdk.RESOLUTION`` のいずれか。
        channel_a_range: Channel Aの入力レンジ。既定値は ``psdk.RANGE.V20``。
        channel_b_range: Channel Bの入力レンジ。既定値は ``psdk.RANGE.V10``。

    Returns:
        時間軸、A/B波形および実際の取得条件を格納した ``WaveformCapture``。

    Raises:
        ValueError: 引数が不正な場合。
        WaveformAcquisitionError: PicoScopeの設定または取得に失敗した場合。

    Notes:
        Channel A/Bの2チャンネルを常に有効にする。機種によって2チャンネル取得に
        対応しない分解能を指定した場合は、SDKのエラーをWaveformAcquisitionError
        として通知する。トリガーの自動タイムアウトは設定しない。
    """

    _validate_arguments(
        duration_s,
        trigger_a_threshold_mv,
        trigger_a_direction,
        resolution,
        channel_a_range,
        channel_b_range,
    )

    scope = psdk.ps5000a()
    opened = False
    try:
        scope.open_unit(resolution=resolution)
        opened = True

        scope.set_channel(channel=psdk.CHANNEL.A, range=channel_a_range)
        scope.set_channel(channel=psdk.CHANNEL.B, range=channel_b_range)
        scope.set_simple_trigger(
            channel=psdk.CHANNEL.A,
            threshold=trigger_a_threshold_mv,
            direction=trigger_a_direction,
            auto_trigger=0,
        )

        timebase, samples, sample_interval_s = _select_fastest_capture(
            scope,
            duration_s,
        )
        device_variant = scope.get_unit_info(psdk.UNIT_INFO.PICO_VARIANT_INFO)

        channel_buffer, time_axis = scope.run_simple_block_capture(
            timebase=timebase,
            samples=samples,
            output_unit="mv",
            time_unit="s",
            ratio=1,
            ratio_mode=psdk.RATIO_MODE.RAW,
            pre_trig_percent=PRE_TRIGGER_PERCENT,
        )

        time_s = np.asarray(time_axis, dtype=float)
        channel_a_mv = np.asarray(channel_buffer[psdk.CHANNEL.A], dtype=float)
        channel_b_mv = np.asarray(channel_buffer[psdk.CHANNEL.B], dtype=float)
        if not (time_s.size == channel_a_mv.size == channel_b_mv.size):
            raise WaveformAcquisitionError(
                "時間軸とChannel A/Bのデータ数が一致しません。"
            )
        if time_s.size < MINIMUM_SAMPLES:
            raise WaveformAcquisitionError(
                f"PicoScopeが{time_s.size}点しか返しませんでした。"
            )
        if not all(
            np.all(np.isfinite(values))
            for values in (time_s, channel_a_mv, channel_b_mv)
        ):
            raise WaveformAcquisitionError(
                "PicoScopeの取得データにNaNまたは無限大が含まれています。"
            )

        returned_intervals_s = np.diff(time_s)
        if np.any(returned_intervals_s <= 0):
            raise WaveformAcquisitionError(
                "PicoScopeが単調増加でない時間軸を返しました。"
            )
        returned_interval_s = float(np.median(returned_intervals_s))
        if not math.isclose(
            returned_interval_s,
            sample_interval_s,
            rel_tol=1e-6,
            abs_tol=1e-15,
        ):
            raise WaveformAcquisitionError(
                "PicoScopeの時間軸とSDKが返したサンプリング間隔が"
                "一致しません。"
            )

        actual_duration_s = float(time_s[-1] - time_s[0])
        duration_tolerance_s = max(
            sample_interval_s * 1e-9,
            np.finfo(float).eps * max(1.0, abs(duration_s)) * 8.0,
        )
        if actual_duration_s + duration_tolerance_s < duration_s:
            raise WaveformAcquisitionError(
                "PicoScopeが指定時間分のデータを返しませんでした: "
                f"指定 {duration_s * 1e6:.9g} µs / "
                f"実際 {actual_duration_s * 1e6:.9g} µs"
            )

        return WaveformCapture(
            time_s=time_s,
            channel_a_mv=channel_a_mv,
            channel_b_mv=channel_b_mv,
            sample_interval_s=sample_interval_s,
            timebase=timebase,
            resolution=resolution,
            device_variant=device_variant,
            channel_a_range=channel_a_range,
            channel_b_range=channel_b_range,
        )
    except WaveformAcquisitionError:
        raise
    except Exception as exc:
        raise WaveformAcquisitionError(f"PicoScopeの波形取得に失敗しました: {exc}") from exc
    finally:
        if opened:
            scope.close_unit()


__all__ = ["WaveformAcquisitionError", "WaveformCapture", "acquire_waveform"]
