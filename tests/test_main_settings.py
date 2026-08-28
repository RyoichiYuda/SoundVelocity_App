"""Tk画面を開かずに設定値とUI変数の連携を検証する。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main
from app_settings import ApplicationSettings, save_settings


class FakeStringVar:
    """設定連携テストに必要なStringVarの最小代替。"""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: object) -> None:
        self.value = str(value)


SETTING_VARIABLE_NAMES = (
    "source_mode_var",
    "measurement_path_var",
    "reference_path_var",
    "capture_duration_us_var",
    "trigger_a_threshold_mv_var",
    "trigger_a_direction_var",
    "picoscope_resolution_var",
    "channel_a_range_var",
    "channel_b_range_var",
    "sample_interval_ns_var",
    "filter_low_mhz_var",
    "filter_high_mhz_var",
    "filter_order_var",
    "filter_passes_var",
    "window_initial_us_var",
    "window_rise_us_var",
    "window_flat_us_var",
    "window_fall_us_var",
    "reference_start_us_var",
    "measurement_channel_var",
    "reference_channel_var",
    "matching_method_var",
    "distance_mm_var",
    "display_min_us_var",
    "display_max_us_var",
)


def make_application_without_tk() -> main.MeasurementApplication:
    application = object.__new__(main.MeasurementApplication)
    for variable_name in SETTING_VARIABLE_NAMES:
        setattr(application, variable_name, FakeStringVar())
    application.status_var = FakeStringVar()
    application._busy = False
    return application


class MainSettingsIntegrationTests(unittest.TestCase):
    def test_all_ui_values_round_trip_through_stable_setting_ids(self) -> None:
        application = make_application_without_tk()
        expected = ApplicationSettings(
            source_mode="csv",
            measurement_path="data/hand-0001.csv",
            reference_path="data/zero_ref.csv",
            capture_duration_us=31.5,
            trigger_a_threshold_mv=10.0,
            trigger_a_direction="either",
            resolution_bits=16,
            channel_a_range_mv=500,
            channel_b_range_mv=2000,
            sample_interval_ns=10.0,
            filter_low_mhz=2.0,
            filter_high_mhz=10.0,
            filter_order=3,
            filter_passes=1,
            window_initial_us=0.2,
            window_rise_us=0.3,
            window_flat_us=0.4,
            window_fall_us=0.5,
            reference_start_us=0.6,
            measurement_channel=1,
            reference_channel=1,
            matching_method="normalized_correlation",
            distance_mm=12.3,
            display_min_us=-2.0,
            display_max_us=25.0,
        )

        application._apply_settings(expected)

        self.assertEqual(application._settings_from_ui(), expected)
        self.assertEqual(application.source_mode_var.get(), main.SOURCE_CSV)
        self.assertEqual(
            application.trigger_a_direction_var.get(),
            "立ち上がりまたは立ち下がり",
        )
        self.assertEqual(
            application.channel_a_range_var.get(),
            "±500 mV",
        )
        self.assertEqual(application.matching_method_var.get(), main.METHOD_CORRELATION)

    def test_startup_load_applies_file_without_starting_acquisition(self) -> None:
        application = make_application_without_tk()
        application._apply_settings = Mock()
        application.root = object()

        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            expected = ApplicationSettings(source_mode="csv")
            save_settings(expected, settings_path)

            with patch.object(main, "SETTINGS_PATH", settings_path):
                application._load_startup_settings()

        application._apply_settings.assert_called_once_with(expected)
        self.assertIn("設定を読み込みました", application.status_var.get())

    def test_manual_load_discards_all_data_even_when_source_mode_is_same(self) -> None:
        application = make_application_without_tk()
        application._apply_settings(ApplicationSettings())
        application.current_measurement = object()
        application.current_reference = object()
        application.current_result = object()
        application.current_measurement_sample_interval_s = 1.0
        application.current_reference_sample_interval_s = 1.0
        application.current_reference_source_mode = main.SOURCE_PICOSCOPE
        application.current_reference_channel_a_range = 10
        application._clear_result_display = Mock()

        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            save_settings(ApplicationSettings(filter_order=5), settings_path)

            with patch.object(main, "SETTINGS_PATH", settings_path):
                application._load_settings()

        self.assertIsNone(application.current_measurement)
        self.assertIsNone(application.current_reference)
        self.assertIsNone(application.current_result)
        self.assertIsNone(application.current_measurement_sample_interval_s)
        self.assertIsNone(application.current_reference_sample_interval_s)
        self.assertIsNone(application.current_reference_source_mode)
        self.assertIsNone(application.current_reference_channel_a_range)
        application._clear_result_display.assert_called_once_with()
        self.assertEqual(application.filter_order_var.get(), "5")
        self.assertIn("参照データを取り直してください", application.status_var.get())

    def test_invalid_manual_load_keeps_ui_and_acquired_data_unchanged(self) -> None:
        application = make_application_without_tk()
        expected = ApplicationSettings(filter_order=6)
        application._apply_settings(expected)
        measurement = object()
        application.current_measurement = measurement
        application._show_error = Mock()

        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(
                json.dumps({"version": 1, "filter": {"oder": 9}}),
                encoding="utf-8",
            )

            with patch.object(main, "SETTINGS_PATH", settings_path):
                application._load_settings()

        self.assertEqual(application._settings_from_ui(), expected)
        self.assertIs(application.current_measurement, measurement)
        application._show_error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
