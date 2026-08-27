"""測定波形を取得する処理を、解析処理から切り離すためのモジュール。

現在はPicoScope本体が手元にないため :class:`CsvDataSource` がCSVを読み込む。
将来は :class:`PicoScopeDataSource.acquire` の中身をSDK呼び出しへ置き換えるだけで、
UIやフィルター・相関処理を変更せずリアルタイム取得へ移行できる構造にしている。

どの取得方法でも、最終的に「時間、Ch1、Ch2」を :class:`SignalData` として返す。
解析側は元データがCSVか実機かを意識しない。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class DataAcquisitionError(RuntimeError):
    """データ取得失敗やCSV形式不正を、UIへ分かりやすく通知する例外。"""


@dataclass
class SignalData:
    """1回の測定で得られる時間軸と2チャンネルの波形。

    Attributes:
        time_s: 各サンプルの時刻 [s]。
        channel_1: PicoScope Channel 1相当の電圧配列 [V]。
        channel_2: PicoScope Channel 2相当の電圧配列 [V]。
        source_name: グラフ下部の状態表示に使用する取得元名。
    """

    time_s: np.ndarray
    channel_1: np.ndarray
    channel_2: np.ndarray
    source_name: str = ""

    def __post_init__(self) -> None:
        """入力配列をfloat型へ統一し、解析可能なデータか検査する。"""

        # リストやpandas Seriesが渡されても、以降をNumPy配列として扱えるようにする。
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.channel_1 = np.asarray(self.channel_1, dtype=float)
        self.channel_2 = np.asarray(self.channel_2, dtype=float)

        # 3配列の長さが異なると、時間と電圧の対応関係が崩れるため受け付けない。
        lengths = {self.time_s.size, self.channel_1.size, self.channel_2.size}
        if len(lengths) != 1:
            raise DataAcquisitionError("時間軸とチャンネルのデータ数が一致しません。")
        if self.time_s.size < 8:
            raise DataAcquisitionError("解析には8サンプル以上のデータが必要です。")
        # NaNやinfがあるとフィルター・正規化・相関の結果がすべてNaNになるため、
        # データ取得直後に検出する。
        if not all(
            np.all(np.isfinite(values))
            for values in (self.time_s, self.channel_1, self.channel_2)
        ):
            raise DataAcquisitionError("データに数値でない値または無限大が含まれています。")

    def channel(self, number: int) -> np.ndarray:
        """UIで選択されたチャンネル番号に対応する電圧配列を返す。"""

        if number == 1:
            return self.channel_1
        if number == 2:
            return self.channel_2
        raise ValueError("チャンネル番号は1または2を指定してください。")


class MeasurementDataSource(Protocol):
    """UIがデータ取得元へ要求する共通インターフェース。

    ``Protocol`` なので継承は必須ではない。``acquire()`` を実装したクラスなら、
    CSV、PicoScope、テスト用配列などを同じ方法でUIへ渡せる。
    """

    def acquire(self) -> SignalData:
        """1回分の新しい波形を取得して返す。"""


@dataclass
class CsvDataSource:
    """現在使用しているPicoScope出力相当の3列CSVを読み込む取得元。

    CSVの列順は「時間、Ch1、Ch2」を想定する。現在のファイルでは時間列が
    マイクロ秒なので、既定の ``time_scale_to_seconds=1e-6`` で秒へ変換する。
    """

    path: Path
    time_scale_to_seconds: float = 1e-6

    def acquire(self) -> SignalData:
        """指定CSVを毎回読み直し、最新内容を :class:`SignalData` で返す。

        「新規測定・計算」ボタンを押すたびにこの関数が呼ばれるため、現段階でも
        同じパスのCSVを外部で更新すれば、新しい内容を続けて解析できる。
        """

        # 相対パスが渡された場合も、ここで絶対パスへ確定してから存在確認する。
        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise DataAcquisitionError(f"CSVファイルが見つかりません: {path}")

        try:
            # header=Noneで一度すべてをデータとして読む。これにより現在の
            # ヘッダーなしCSVと、一般的なヘッダー付きCSVの両方を判定できる。
            frame = pd.read_csv(path, header=None, comment="#")
        except Exception as exc:
            raise DataAcquisitionError(f"CSVを読み込めませんでした: {exc}") from exc

        if frame.shape[1] < 3:
            raise DataAcquisitionError(
                f"CSVには時間・Ch1・Ch2の3列が必要です: {path}"
            )

        # 最初の3列を数値へ変換する。変換できない文字列は一旦NaNにして、
        # 次の処理で「列名」なのか「途中の壊れたデータ」なのかを区別する。
        numeric = frame.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
        valid_rows = numeric.notna().all(axis=1)

        # 先頭行だけが非数値なら列名として除外する。それ以外の位置に非数値行が
        # あれば、データ欠損として行番号を示してエラーにする。
        if not bool(valid_rows.iloc[0]) and bool(valid_rows.iloc[1:].all()):
            numeric = numeric.iloc[1:]
        elif not bool(valid_rows.all()):
            invalid_row = int(np.flatnonzero(~valid_rows.to_numpy())[0]) + 1
            raise DataAcquisitionError(
                f"CSVの{invalid_row}行目に数値でないデータがあります: {path}"
            )

        values = numeric.to_numpy(dtype=float)
        return SignalData(
            # CSV時間列はµsなので秒へ変換する。PicoScope実装では最初から秒で返す。
            time_s=values[:, 0] * self.time_scale_to_seconds,
            channel_1=values[:, 1],
            channel_2=values[:, 2],
            source_name=path.name,
        )


@dataclass
class ArrayDataSource:
    """配列を返す任意の関数を、共通取得インターフェースへ変換するアダプター。

    PicoScope SDKを別関数で試作するとき、その関数を ``acquire_callback`` に渡せば
    UIへすぐ接続できる。コールバックは ``(time_s, ch1, ch2)`` を返す必要がある。
    """

    acquire_callback: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]
    source_name: str = "PicoScope"

    def acquire(self) -> SignalData:
        """コールバックを1回実行し、返された3配列を検査して格納する。"""

        try:
            time_s, channel_1, channel_2 = self.acquire_callback()
        except Exception as exc:
            raise DataAcquisitionError(f"データ取得に失敗しました: {exc}") from exc
        return SignalData(time_s, channel_1, channel_2, self.source_name)


class PicoScopeDataSource:
    """将来のPicoScope SDK実装場所を明示するための仮クラス。

    現在は意図的にエラーを返す。実機接続時は ``acquire()`` 内でPicoScopeを設定し、
    取得した時間[s]、Ch1[V]、Ch2[V]から ``SignalData`` を作って返す。
    ``main.py`` 側のボタン処理や ``analysis_pipeline.py`` は変更不要。
    """

    def acquire(self) -> SignalData:
        """PicoScope未接続であることをUIへ通知する。"""

        raise DataAcquisitionError(
            "PicoScope取得はまだ未実装です。data_sources.py の "
            "PicoScopeDataSource.acquire() にSDKの取得処理を実装してください。"
        )
