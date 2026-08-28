"""Application settings serialization and validation.

The UI uses localized labels and strings, but settings files use stable identifiers
and numeric values so that a saved file does not depend on display wording. Missing
known fields receive the same defaults as the application UI. Unknown fields are
rejected so that misspelled keys cannot silently reset a value to its default.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SETTINGS_VERSION = 1

SOURCE_MODES = frozenset({"csv", "picoscope"})
TRIGGER_DIRECTIONS = frozenset(
    {"falling", "rising", "above", "below", "either"}
)
MATCHING_METHODS = frozenset({"normalized_correlation", "squared_error"})
PICOSCOPE_RESOLUTIONS_BITS = frozenset({8, 12, 14, 15, 16})
PICOSCOPE_RANGES_MV = frozenset(
    {10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000}
)

_SECTION_KEYS = {
    "data_source": frozenset({"mode", "measurement_path", "reference_path"}),
    "picoscope": frozenset(
        {
            "capture_duration_us",
            "trigger_a_threshold_mv",
            "trigger_a_direction",
            "resolution_bits",
            "channel_a_range_mv",
            "channel_b_range_mv",
        }
    ),
    "filter": frozenset(
        {"sample_interval_ns", "low_mhz", "high_mhz", "order", "passes"}
    ),
    "window": frozenset(
        {
            "initial_zero_us",
            "rise_us",
            "flat_us",
            "fall_us",
            "reference_start_us",
        }
    ),
    "matching": frozenset(
        {"measurement_channel", "reference_channel", "method", "distance_mm"}
    ),
    "display": frozenset({"min_us", "max_us"}),
}
_DOCUMENT_KEYS = frozenset({"version", *_SECTION_KEYS})


class SettingsError(ValueError):
    """設定ファイルまたは設定値が不正なことを表す例外。"""


_FIELD_LABELS = {
    "source_mode": "取得元",
    "measurement_path": "測定CSVパス",
    "reference_path": "参照CSVパス",
    "capture_duration_us": "取得時間 [µs]",
    "trigger_a_threshold_mv": "Trigger A threshold [mV]",
    "trigger_a_direction": "Trigger A direction",
    "resolution_bits": "ADC分解能 [bit]",
    "channel_a_range_mv": "Channel Aレンジ [mV]",
    "channel_b_range_mv": "Channel Bレンジ [mV]",
    "sample_interval_ns": "サンプリング間隔 [ns]",
    "filter_low_mhz": "フィルター下限周波数 [MHz]",
    "filter_high_mhz": "フィルター上限周波数 [MHz]",
    "filter_order": "フィルター次数",
    "filter_passes": "フィルター回数",
    "window_initial_us": "窓関数の初期ゼロ [µs]",
    "window_rise_us": "窓関数の立ち上がり [µs]",
    "window_flat_us": "窓関数のフラット [µs]",
    "window_fall_us": "窓関数の立ち下がり [µs]",
    "reference_start_us": "参照開始時間 [µs]",
    "measurement_channel": "測定チャンネル",
    "reference_channel": "参照チャンネル",
    "matching_method": "照合方法",
    "distance_mm": "距離 [mm]",
    "display_min_us": "表示開始 [µs]",
    "display_max_us": "表示終了 [µs]",
}

_FLOAT_FIELDS = (
    "capture_duration_us",
    "trigger_a_threshold_mv",
    "sample_interval_ns",
    "filter_low_mhz",
    "filter_high_mhz",
    "window_initial_us",
    "window_rise_us",
    "window_flat_us",
    "window_fall_us",
    "reference_start_us",
    "display_min_us",
    "display_max_us",
)


def _label(field_name: str) -> str:
    return _FIELD_LABELS[field_name]


def _require_string(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise SettingsError(f"「{_label(field_name)}」は文字列で指定してください。")
    return value


def _require_number(value: Any, field_name: str) -> float:
    # bool is an int subclass in Python, but must not be accepted as a JSON number.
    if type(value) not in (int, float):
        raise SettingsError(f"「{_label(field_name)}」は数値で指定してください。")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise SettingsError(
            f"「{_label(field_name)}」は有限の数値にしてください。"
        ) from exc
    if not math.isfinite(result):
        raise SettingsError(f"「{_label(field_name)}」は有限の数値にしてください。")
    return result


def _require_integer(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise SettingsError(f"「{_label(field_name)}」は整数で指定してください。")
    return value


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    """UIに表示する25個の設定項目。

    時間や周波数はUIと同じ単位で保持する。選択項目は翻訳された表示文言では
    なく、設定ファイルでも共通して使う安定IDを保持する。
    """

    # データ取得元
    source_mode: str = "picoscope"
    measurement_path: str = "data/hand-0001.csv"
    reference_path: str = "data/zero_ref.csv"

    # PicoScope
    capture_duration_us: float = 20.0
    trigger_a_threshold_mv: float = -2_000.0
    trigger_a_direction: str = "falling"
    resolution_bits: int = 8
    channel_a_range_mv: int = 20_000
    channel_b_range_mv: int = 10_000

    # フィルター
    sample_interval_ns: float = 8.0
    filter_low_mhz: float = 1.0
    filter_high_mhz: float = 5.0
    filter_order: int = 4
    filter_passes: int = 2

    # 窓関数・参照波形
    window_initial_us: float = 0.75
    window_rise_us: float = 0.10
    window_flat_us: float = 0.50
    window_fall_us: float = 0.50
    reference_start_us: float = 0.0

    # 照合・表示
    measurement_channel: int = 2
    reference_channel: int = 2
    matching_method: str = "squared_error"
    distance_mm: float | None = None
    display_min_us: float = 0.0
    display_max_us: float = 20.0

    def __post_init__(self) -> None:
        """型を正規化した後、UIと解析で成立する範囲か検証する。"""

        for field_name in _FLOAT_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _require_number(getattr(self, field_name), field_name),
            )

        for field_name in (
            "resolution_bits",
            "channel_a_range_mv",
            "channel_b_range_mv",
            "filter_order",
            "filter_passes",
            "measurement_channel",
            "reference_channel",
        ):
            _require_integer(getattr(self, field_name), field_name)

        for field_name in (
            "source_mode",
            "measurement_path",
            "reference_path",
            "trigger_a_direction",
            "matching_method",
        ):
            _require_string(getattr(self, field_name), field_name)

        if self.distance_mm is not None:
            object.__setattr__(
                self,
                "distance_mm",
                _require_number(self.distance_mm, "distance_mm"),
            )

        self.validate()

    def validate(self) -> None:
        """全設定値の選択肢、範囲、項目間の整合性を検証する。"""

        if self.source_mode not in SOURCE_MODES:
            choices = ", ".join(sorted(SOURCE_MODES))
            raise SettingsError(
                f"「{_label('source_mode')}」は {choices} のいずれかにしてください。"
            )

        for field_name in ("measurement_path", "reference_path"):
            value = getattr(self, field_name)
            if not value.strip():
                raise SettingsError(f"「{_label(field_name)}」を空欄にはできません。")
            if "\x00" in value:
                raise SettingsError(
                    f"「{_label(field_name)}」に使用できない文字が含まれています。"
                )

        if self.capture_duration_us <= 0:
            raise SettingsError("「取得時間 [µs]」は0より大きくしてください。")
        if self.trigger_a_direction not in TRIGGER_DIRECTIONS:
            choices = ", ".join(sorted(TRIGGER_DIRECTIONS))
            raise SettingsError(
                "「Trigger A direction」は "
                f"{choices} のいずれかにしてください。"
            )
        if self.resolution_bits not in PICOSCOPE_RESOLUTIONS_BITS:
            choices = ", ".join(
                str(value) for value in sorted(PICOSCOPE_RESOLUTIONS_BITS)
            )
            raise SettingsError(
                f"「ADC分解能 [bit]」は {choices} のいずれかにしてください。"
            )

        for field_name in ("channel_a_range_mv", "channel_b_range_mv"):
            value = getattr(self, field_name)
            if value not in PICOSCOPE_RANGES_MV:
                choices = ", ".join(
                    str(item) for item in sorted(PICOSCOPE_RANGES_MV)
                )
                raise SettingsError(
                    f"「{_label(field_name)}」は {choices} のいずれかにしてください。"
                )
        if abs(self.trigger_a_threshold_mv) > self.channel_a_range_mv:
            raise SettingsError(
                "「Trigger A threshold [mV]」はChannel Aレンジ内にしてください。"
            )

        if self.sample_interval_ns <= 0:
            raise SettingsError("「サンプリング間隔 [ns]」は0より大きくしてください。")
        if self.filter_low_mhz <= 0:
            raise SettingsError("「フィルター下限周波数 [MHz]」は0より大きくしてください。")
        if self.filter_high_mhz <= self.filter_low_mhz:
            raise SettingsError(
                "「フィルター上限周波数 [MHz]」は下限より大きくしてください。"
            )
        nyquist_mhz = 500.0 / self.sample_interval_ns
        if self.filter_high_mhz >= nyquist_mhz:
            raise SettingsError(
                "「フィルター上限周波数 [MHz]」はナイキスト周波数 "
                f"{nyquist_mhz:g} MHz未満にしてください。"
            )
        if self.filter_order < 1:
            raise SettingsError("「フィルター次数」は1以上にしてください。")
        if self.filter_passes < 1:
            raise SettingsError("「フィルター回数」は1以上にしてください。")

        window_durations = (
            self.window_initial_us,
            self.window_rise_us,
            self.window_flat_us,
            self.window_fall_us,
        )
        window_fields = (
            "window_initial_us",
            "window_rise_us",
            "window_flat_us",
            "window_fall_us",
        )
        for field_name, duration in zip(window_fields, window_durations, strict=True):
            if duration < 0:
                raise SettingsError(f"「{_label(field_name)}」は0以上にしてください。")
        if sum(window_durations) <= 0:
            raise SettingsError("窓関数の全長は0より大きくしてください。")
        if not any(
            duration_us * 1_000.0 >= self.sample_interval_ns
            for duration_us in window_durations
        ):
            raise SettingsError(
                "窓関数が0サンプルになります。少なくとも1区間を"
                "サンプリング間隔以上にしてください。"
            )

        if self.measurement_channel not in (1, 2):
            raise SettingsError("「測定チャンネル」は1または2にしてください。")
        if self.reference_channel not in (1, 2):
            raise SettingsError("「参照チャンネル」は1または2にしてください。")
        if self.matching_method not in MATCHING_METHODS:
            choices = ", ".join(sorted(MATCHING_METHODS))
            raise SettingsError(
                f"「照合方法」は {choices} のいずれかにしてください。"
            )
        if self.distance_mm is not None and self.distance_mm < 0:
            raise SettingsError("「距離 [mm]」は0以上にしてください。")
        if self.display_max_us <= self.display_min_us:
            raise SettingsError("表示終了時間は表示開始時間より大きくしてください。")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SettingsError(f"設定ファイルに同じ項目「{key}」が複数あります。")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise SettingsError(f"設定ファイルに使用できない数値「{value}」があります。")


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name, {})
    if type(value) is not dict:
        raise SettingsError(f"設定ファイルの「{name}」はオブジェクトにしてください。")
    unknown_keys = set(value) - _SECTION_KEYS[name]
    if unknown_keys:
        unknown_names = ", ".join(sorted(unknown_keys))
        raise SettingsError(
            f"設定ファイルの「{name}」に未対応の項目があります: {unknown_names}"
        )
    return value


def _value(section: dict[str, Any], key: str, default: Any) -> Any:
    return section[key] if key in section else default


def _settings_from_document(document: Any) -> ApplicationSettings:
    if type(document) is not dict:
        raise SettingsError("設定ファイルの最上位はJSONオブジェクトにしてください。")

    unknown_keys = set(document) - _DOCUMENT_KEYS
    if unknown_keys:
        unknown_names = ", ".join(sorted(unknown_keys))
        raise SettingsError(f"設定ファイルに未対応の項目があります: {unknown_names}")
    if "version" not in document:
        raise SettingsError("設定ファイルに必須項目「version」がありません。")

    version = document["version"]
    if type(version) is not int:
        raise SettingsError("設定ファイルの「version」は整数で指定してください。")
    if version != SETTINGS_VERSION:
        raise SettingsError(
            f"設定ファイルのversion {version}には対応していません。"
            f"対応versionは{SETTINGS_VERSION}です。"
        )

    defaults = ApplicationSettings()
    data_source = _section(document, "data_source")
    picoscope = _section(document, "picoscope")
    filter_settings = _section(document, "filter")
    window = _section(document, "window")
    matching = _section(document, "matching")
    display = _section(document, "display")

    return ApplicationSettings(
        source_mode=_value(data_source, "mode", defaults.source_mode),
        measurement_path=_value(
            data_source, "measurement_path", defaults.measurement_path
        ),
        reference_path=_value(data_source, "reference_path", defaults.reference_path),
        capture_duration_us=_value(
            picoscope, "capture_duration_us", defaults.capture_duration_us
        ),
        trigger_a_threshold_mv=_value(
            picoscope,
            "trigger_a_threshold_mv",
            defaults.trigger_a_threshold_mv,
        ),
        trigger_a_direction=_value(
            picoscope, "trigger_a_direction", defaults.trigger_a_direction
        ),
        resolution_bits=_value(
            picoscope, "resolution_bits", defaults.resolution_bits
        ),
        channel_a_range_mv=_value(
            picoscope, "channel_a_range_mv", defaults.channel_a_range_mv
        ),
        channel_b_range_mv=_value(
            picoscope, "channel_b_range_mv", defaults.channel_b_range_mv
        ),
        sample_interval_ns=_value(
            filter_settings, "sample_interval_ns", defaults.sample_interval_ns
        ),
        filter_low_mhz=_value(
            filter_settings, "low_mhz", defaults.filter_low_mhz
        ),
        filter_high_mhz=_value(
            filter_settings, "high_mhz", defaults.filter_high_mhz
        ),
        filter_order=_value(filter_settings, "order", defaults.filter_order),
        filter_passes=_value(filter_settings, "passes", defaults.filter_passes),
        window_initial_us=_value(
            window, "initial_zero_us", defaults.window_initial_us
        ),
        window_rise_us=_value(window, "rise_us", defaults.window_rise_us),
        window_flat_us=_value(window, "flat_us", defaults.window_flat_us),
        window_fall_us=_value(window, "fall_us", defaults.window_fall_us),
        reference_start_us=_value(
            window, "reference_start_us", defaults.reference_start_us
        ),
        measurement_channel=_value(
            matching, "measurement_channel", defaults.measurement_channel
        ),
        reference_channel=_value(
            matching, "reference_channel", defaults.reference_channel
        ),
        matching_method=_value(matching, "method", defaults.matching_method),
        distance_mm=_value(matching, "distance_mm", defaults.distance_mm),
        display_min_us=_value(display, "min_us", defaults.display_min_us),
        display_max_us=_value(display, "max_us", defaults.display_max_us),
    )


def _settings_to_document(settings: ApplicationSettings) -> dict[str, Any]:
    return {
        "version": SETTINGS_VERSION,
        "data_source": {
            "mode": settings.source_mode,
            "measurement_path": settings.measurement_path,
            "reference_path": settings.reference_path,
        },
        "picoscope": {
            "capture_duration_us": settings.capture_duration_us,
            "trigger_a_threshold_mv": settings.trigger_a_threshold_mv,
            "trigger_a_direction": settings.trigger_a_direction,
            "resolution_bits": settings.resolution_bits,
            "channel_a_range_mv": settings.channel_a_range_mv,
            "channel_b_range_mv": settings.channel_b_range_mv,
        },
        "filter": {
            "sample_interval_ns": settings.sample_interval_ns,
            "low_mhz": settings.filter_low_mhz,
            "high_mhz": settings.filter_high_mhz,
            "order": settings.filter_order,
            "passes": settings.filter_passes,
        },
        "window": {
            "initial_zero_us": settings.window_initial_us,
            "rise_us": settings.window_rise_us,
            "flat_us": settings.window_flat_us,
            "fall_us": settings.window_fall_us,
            "reference_start_us": settings.reference_start_us,
        },
        "matching": {
            "measurement_channel": settings.measurement_channel,
            "reference_channel": settings.reference_channel,
            "method": settings.matching_method,
            "distance_mm": settings.distance_mm,
        },
        "display": {
            "min_us": settings.display_min_us,
            "max_us": settings.display_max_us,
        },
    }


def load_settings(path: str | Path) -> ApplicationSettings:
    """UTF-8 JSON設定ファイルを読み込み、検証済み設定を返す。"""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_number,
            )
    except SettingsError:
        raise
    except FileNotFoundError as exc:
        raise SettingsError(f"設定ファイルが見つかりません: {source}") from exc
    except UnicodeDecodeError as exc:
        raise SettingsError(
            f"設定ファイルをUTF-8として読み込めません: {source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SettingsError(
            f"設定ファイルのJSON形式が不正です（{exc.lineno}行目、"
            f"{exc.colno}列目）: {source}"
        ) from exc
    except OSError as exc:
        raise SettingsError(f"設定ファイルを読み込めません: {source}（{exc}）") from exc

    return _settings_from_document(document)


def save_settings(settings: ApplicationSettings, path: str | Path) -> None:
    """設定をUTF-8 JSONで、保存先と同じディレクトリから原子的に置換する。"""

    if not isinstance(settings, ApplicationSettings):
        raise SettingsError("保存する設定はApplicationSettingsで指定してください。")
    settings.validate()

    destination = Path(path)
    parent = destination.parent
    temporary_path: Path | None = None
    file_descriptor: int | None = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            file_descriptor = None
            json.dump(
                _settings_to_document(settings),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as exc:
        raise SettingsError(f"設定ファイルを保存できません: {destination}（{exc}）") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


__all__ = ["ApplicationSettings", "load_settings", "save_settings"]
