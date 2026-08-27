"""音速測定UIで使用する窓関数の生成・保存処理。

既存の ``calculate/window_fcn.py`` と同じ考え方で、次の4区間を連結して
1本の窓関数を作る。

1. 初期ゼロ区間: 不要な先頭波形を完全に無効化する
2. 立ち上がり区間: ハニング窓で0付近から滑らかにゲインを上げる
3. フラット区間: ゲイン1で波形をそのまま使用する
4. 立ち下がり区間: ハニング窓で滑らかにゲインを下げる

窓の時間条件をUIから変更できるよう、計算処理を独立した関数としてまとめている。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowParameters:
    """窓関数を生成するための時間パラメーター。

    すべての値は秒単位で保持する。UIではns・µs単位で入力された値を、
    ``main.py`` 側で秒へ変換してからこのクラスへ渡す。
    ``frozen=True`` により、解析途中で設定値が書き換わらないようにしている。
    """

    sample_interval_s: float = 8e-9  # 1サンプルの時間間隔 [s]
    initial_zero_s: float = 0.75e-6  # 先頭をゲイン0にする時間 [s]
    rise_s: float = 0.10e-6  # ゲインを0から1へ近づける時間 [s]
    flat_s: float = 0.50e-6  # ゲイン1を維持する時間 [s]
    fall_s: float = 0.50e-6  # ゲインを1から0へ近づける時間 [s]

    def validate(self) -> None:
        """設定値が窓関数として成立するか確認する。

        不正値を早い段階で日本語のエラーにすることで、NumPy内で分かりにくい
        例外が発生するのを防ぐ。
        """

        if self.sample_interval_s <= 0:
            raise ValueError("サンプリング間隔は0より大きくしてください。")

        durations = {
            "初期ゼロ時間": self.initial_zero_s,
            "立ち上がり時間": self.rise_s,
            "フラット時間": self.flat_s,
            "立ち下がり時間": self.fall_s,
        }
        for name, value in durations.items():
            if value < 0:
                raise ValueError(f"{name}は0以上にしてください。")

        if sum(durations.values()) <= 0:
            raise ValueError("窓関数の全長は0より大きくしてください。")


@dataclass(frozen=True)
class GeneratedWindow:
    """生成済み窓関数と、その各サンプルに対応する相対時間軸。"""

    time_s: np.ndarray
    gain: np.ndarray


def _duration_to_samples(duration_s: float, sample_interval_s: float) -> int:
    """指定時間をサンプル数へ変換する。端数は既存処理と同じく切り捨てる。"""

    # 例: 0.75 µs / 8 ns = 93.75なので、既存コードと同じ93サンプルになる。
    return int(duration_s / sample_interval_s)


def _hann_rise(sample_count: int) -> np.ndarray:
    """ハニング窓の前半を使い、滑らかな立ち上がり部分を作る。"""

    if sample_count <= 0:
        return np.empty(0, dtype=float)
    if sample_count == 1:
        return np.ones(1, dtype=float)
    return np.hanning(2 * sample_count)[:sample_count]


def _hann_fall(sample_count: int) -> np.ndarray:
    """ハニング窓の後半を使い、滑らかな立ち下がり部分を作る。"""

    if sample_count <= 0:
        return np.empty(0, dtype=float)
    if sample_count == 1:
        return np.ones(1, dtype=float)
    return np.hanning(2 * sample_count)[sample_count:]


def generate_window(parameters: WindowParameters) -> GeneratedWindow:
    """UIで指定された時間条件から窓関数を生成する。

    Returns:
        ``time_s`` と ``gain`` を持つ :class:`GeneratedWindow`。
        既定値では既存の ``window_2.csv`` と同じ229サンプルになる。
    """

    parameters.validate()

    # 各区間の時間を、現在のサンプリング間隔に対応するサンプル数へ変換する。
    initial_count = _duration_to_samples(
        parameters.initial_zero_s, parameters.sample_interval_s
    )
    rise_count = _duration_to_samples(parameters.rise_s, parameters.sample_interval_s)
    flat_count = _duration_to_samples(parameters.flat_s, parameters.sample_interval_s)
    fall_count = _duration_to_samples(parameters.fall_s, parameters.sample_interval_s)

    # 4区間を時系列順に結合する。以降の相関処理では、このgainを参照波形と
    # 測定波形の両方へ掛けることで、比較対象とする時間範囲を揃える。
    gain = np.concatenate(
        [
            np.zeros(initial_count, dtype=float),
            _hann_rise(rise_count),
            np.ones(flat_count, dtype=float),
            _hann_fall(fall_count),
        ]
    )
    if gain.size == 0:
        raise ValueError(
            "窓関数が0サンプルになりました。時間をサンプリング間隔以上にしてください。"
        )

    # 窓の先頭を0秒とした相対時間軸を作る。
    time_s = np.arange(gain.size, dtype=float) * parameters.sample_interval_s
    return GeneratedWindow(time_s=time_s, gain=gain)


def save_window_csv(window: GeneratedWindow, destination: Path) -> None:
    """生成した窓関数を既存 ``window_2.csv`` と同じ2列形式で保存する。

    保存列は ``Time(s)`` と ``Gain``。保存先フォルダーが存在しない場合は
    自動作成する。この関数はUIで「窓CSVを保存」を押したときだけ呼ばれる。
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Time(s)": window.time_s, "Gain": window.gain}).to_csv(
        destination, index=False
    )
