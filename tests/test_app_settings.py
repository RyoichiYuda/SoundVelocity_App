from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

from app_settings import (
    ApplicationSettings,
    load_settings,
    save_settings,
)


class ApplicationSettingsTests(unittest.TestCase):
    def test_dataclass_has_all_27_ui_settings(self) -> None:
        self.assertEqual(len(fields(ApplicationSettings)), 27)
        self.assertEqual(
            {field.name for field in fields(ApplicationSettings)},
            {
                "source_mode",
                "measurement_path",
                "reference_path",
                "capture_duration_us",
                "trigger_a_threshold_mv",
                "trigger_a_direction",
                "resolution_bits",
                "channel_a_range_mv",
                "channel_b_range_mv",
                "sample_interval_ns",
                "filter_low_mhz",
                "filter_high_mhz",
                "filter_order",
                "filter_passes",
                "window_initial_us",
                "window_rise_us",
                "window_flat_us",
                "window_fall_us",
                "reference_start_us",
                "measurement_channel",
                "reference_channel",
                "matching_method",
                "distance_mm",
                "display_min_us",
                "display_max_us",
                "continuous_measurement_enabled",
                "continuous_measurement_count",
            },
        )

    def test_defaults_match_current_ui_and_use_stable_ids(self) -> None:
        settings = ApplicationSettings()

        self.assertEqual(settings.source_mode, "picoscope")
        self.assertEqual(settings.trigger_a_direction, "falling")
        self.assertEqual(settings.resolution_bits, 8)
        self.assertEqual(settings.channel_a_range_mv, 20_000)
        self.assertEqual(settings.channel_b_range_mv, 10_000)
        self.assertEqual(settings.matching_method, "squared_error")
        self.assertIsNone(settings.distance_mm)
        self.assertFalse(settings.continuous_measurement_enabled)
        self.assertEqual(settings.continuous_measurement_count, 5)

    def test_direct_construction_normalizes_json_numbers_to_float(self) -> None:
        settings = ApplicationSettings(
            capture_duration_us=20,
            distance_mm=10,
        )

        self.assertIs(type(settings.capture_duration_us), float)
        self.assertIs(type(settings.distance_mm), float)

    def test_rejects_non_numeric_and_non_integer_types(self) -> None:
        invalid_arguments = (
            {"capture_duration_us": "20"},
            {"capture_duration_us": True},
            {"resolution_bits": 8.0},
            {"filter_order": False},
            {"distance_mm": "10"},
            {"measurement_channel": "2"},
            {"continuous_measurement_enabled": 0},
            {"continuous_measurement_enabled": "false"},
            {"continuous_measurement_count": 5.0},
            {"continuous_measurement_count": False},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ValueError, "指定してください"
            ):
                ApplicationSettings(**arguments)

    def test_rejects_invalid_choices(self) -> None:
        invalid_arguments = (
            {"source_mode": "CSV"},
            {"trigger_a_direction": "立ち下がり"},
            {"resolution_bits": 10},
            {"channel_a_range_mv": 12_345},
            {"channel_b_range_mv": 12_345},
            {"matching_method": "相互相関"},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ValueError, "いずれか"
            ):
                ApplicationSettings(**arguments)

    def test_rejects_invalid_numeric_ranges_and_relations(self) -> None:
        invalid_arguments = (
            {"capture_duration_us": 0},
            {"trigger_a_threshold_mv": -20_001},
            {"sample_interval_ns": 0},
            {"filter_low_mhz": 0},
            {"filter_low_mhz": 5, "filter_high_mhz": 5},
            {"sample_interval_ns": 8, "filter_high_mhz": 62.5},
            {"filter_order": 0},
            {"filter_passes": 0},
            {"window_initial_us": -0.1},
            {
                "window_initial_us": 0,
                "window_rise_us": 0,
                "window_flat_us": 0,
                "window_fall_us": 0,
            },
            {
                "sample_interval_ns": 8,
                "window_initial_us": 0.001,
                "window_rise_us": 0.001,
                "window_flat_us": 0.001,
                "window_fall_us": 0.001,
            },
            {"measurement_channel": 3},
            {"reference_channel": 0},
            {"distance_mm": -0.1},
            {"display_min_us": 10, "display_max_us": 10},
            {"continuous_measurement_count": 1},
            {"continuous_measurement_count": 101},
            {"capture_duration_us": float("nan")},
            {"display_max_us": float("inf")},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ApplicationSettings(**arguments)

    def test_rejects_empty_or_invalid_paths(self) -> None:
        for arguments in (
            {"measurement_path": "   "},
            {"reference_path": "data/zero\x00ref.csv"},
            {"measurement_path": 123},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ApplicationSettings(**arguments)


class SettingsFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.path = self.directory / "settings.json"

    def _write_document(self, document: object) -> None:
        self.path.write_text(
            json.dumps(document, ensure_ascii=False),
            encoding="utf-8",
        )

    def _default_document(self) -> dict[str, object]:
        save_settings(ApplicationSettings(), self.path)
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_bundled_settings_file_contains_valid_application_settings(self) -> None:
        bundled_path = Path(__file__).resolve().parents[1] / "settings.json"

        settings = load_settings(bundled_path)

        self.assertIsInstance(settings, ApplicationSettings)
        settings.validate()

    def test_save_writes_readable_nested_version_1_utf8_json(self) -> None:
        settings = ApplicationSettings(
            source_mode="csv",
            measurement_path="データ/実測.csv",
            reference_path="データ/参照.csv",
        )

        save_settings(settings, self.path)

        raw = self.path.read_bytes()
        text = raw.decode("utf-8")
        document = json.loads(text)
        self.assertIn("データ/実測.csv", text)
        self.assertIn("\n  \"data_source\": {", text)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(document["version"], 1)
        self.assertEqual(
            set(document),
            {
                "version",
                "data_source",
                "picoscope",
                "filter",
                "window",
                "matching",
                "display",
                "continuous_measurement",
            },
        )
        self.assertEqual(document["data_source"]["mode"], "csv")
        self.assertEqual(document["picoscope"]["trigger_a_direction"], "falling")
        self.assertEqual(document["matching"]["method"], "squared_error")
        self.assertEqual(
            document["continuous_measurement"],
            {"enabled": False, "count": 5},
        )

    def test_round_trip_preserves_all_settings(self) -> None:
        settings = ApplicationSettings(
            source_mode="csv",
            measurement_path="input/測定.csv",
            reference_path="input/参照.csv",
            capture_duration_us=35.5,
            trigger_a_threshold_mv=-400,
            trigger_a_direction="rising",
            resolution_bits=12,
            channel_a_range_mv=500,
            channel_b_range_mv=200,
            sample_interval_ns=4,
            filter_low_mhz=2,
            filter_high_mhz=8,
            filter_order=6,
            filter_passes=3,
            window_initial_us=0.5,
            window_rise_us=0.2,
            window_flat_us=0.7,
            window_fall_us=0.3,
            reference_start_us=-1.25,
            measurement_channel=1,
            reference_channel=1,
            matching_method="normalized_correlation",
            distance_mm=12.34,
            display_min_us=-2,
            display_max_us=30,
            continuous_measurement_enabled=True,
            continuous_measurement_count=12,
        )

        save_settings(settings, self.path)

        self.assertEqual(load_settings(self.path), settings)

    def test_missing_known_fields_receive_defaults(self) -> None:
        self._write_document(
            {
                "version": 1,
                "data_source": {"mode": "csv"},
                "filter": {"order": 7},
            }
        )

        loaded = load_settings(self.path)

        defaults = ApplicationSettings()
        self.assertEqual(loaded.source_mode, "csv")
        self.assertEqual(loaded.filter_order, 7)
        self.assertEqual(loaded.capture_duration_us, defaults.capture_duration_us)
        self.assertEqual(loaded.reference_path, defaults.reference_path)
        self.assertEqual(loaded.distance_mm, defaults.distance_mm)
        self.assertEqual(
            loaded.continuous_measurement_enabled,
            defaults.continuous_measurement_enabled,
        )
        self.assertEqual(
            loaded.continuous_measurement_count,
            defaults.continuous_measurement_count,
        )

    def test_missing_version_is_rejected(self) -> None:
        self._write_document({})

        with self.assertRaisesRegex(ValueError, "version"):
            load_settings(self.path)

    def test_unknown_top_level_or_nested_fields_are_rejected(self) -> None:
        documents = (
            {"version": 1, "future_section": {}},
            {"version": 1, "filter": {"oder": 9}},
            {"version": 1, "continuous_measurement": {"times": 5}},
        )

        for document in documents:
            with self.subTest(document=document):
                self._write_document(document)
                with self.assertRaisesRegex(ValueError, "未対応"):
                    load_settings(self.path)

    def test_rejects_unsupported_or_wrongly_typed_version(self) -> None:
        for version in (2, "1", 1.0, True, None):
            with self.subTest(version=version):
                self._write_document({"version": version})
                with self.assertRaisesRegex(ValueError, "version"):
                    load_settings(self.path)

    def test_rejects_non_object_root_or_known_section(self) -> None:
        for document in ([], None, {"version": 1, "filter": []}):
            with self.subTest(document=document):
                self._write_document(document)
                with self.assertRaisesRegex(ValueError, "オブジェクト"):
                    load_settings(self.path)

    def test_rejects_wrong_json_value_types(self) -> None:
        invalid_values = (
            ("picoscope", "capture_duration_us", "20"),
            ("picoscope", "resolution_bits", 8.0),
            ("filter", "order", True),
            ("matching", "distance_mm", False),
            ("data_source", "mode", 1),
            ("continuous_measurement", "enabled", 1),
            ("continuous_measurement", "count", 5.0),
        )
        base_document = self._default_document()

        for section, key, value in invalid_values:
            with self.subTest(section=section, key=key, value=value):
                document = copy.deepcopy(base_document)
                document[section][key] = value
                self._write_document(document)
                with self.assertRaises(ValueError):
                    load_settings(self.path)

    def test_rejects_malformed_json_duplicate_keys_and_nonfinite_numbers(self) -> None:
        invalid_json_texts = (
            "{",
            '{"version": 1, "version": 1}',
            '{"version": 1, "display": {"max_us": NaN}}',
            '{"version": 1, "display": {"max_us": Infinity}}',
        )

        for text in invalid_json_texts:
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_settings(self.path)

    def test_rejects_non_utf8_file_with_japanese_error(self) -> None:
        self.path.write_bytes(b'\xff{"version": 1}')

        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_settings(self.path)

    def test_missing_file_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "見つかりません"):
            load_settings(self.directory / "missing.json")

    def test_save_creates_parent_and_uses_atomic_replace_in_same_directory(self) -> None:
        destination = self.directory / "new" / "nested" / "settings.json"
        replace_calls: list[tuple[Path, Path]] = []
        real_replace = os.replace

        def tracking_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
            replace_calls.append((Path(source), Path(target)))
            real_replace(source, target)

        with mock.patch("app_settings.os.replace", side_effect=tracking_replace):
            save_settings(ApplicationSettings(), destination)

        self.assertTrue(destination.is_file())
        self.assertEqual(len(replace_calls), 1)
        temporary_path, target_path = replace_calls[0]
        self.assertEqual(temporary_path.parent, destination.parent)
        self.assertEqual(target_path, destination)
        self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_failed_atomic_replace_preserves_old_file_and_removes_temporary(self) -> None:
        original = b"old settings\n"
        self.path.write_bytes(original)

        with mock.patch(
            "app_settings.os.replace",
            side_effect=OSError("replace failed"),
        ):
            with self.assertRaisesRegex(ValueError, "保存できません"):
                save_settings(ApplicationSettings(), self.path)

        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_save_rejects_non_settings_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "ApplicationSettings"):
            save_settings({}, self.path)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
