"""基本・アドバンスド両タブで共有する操作と描画のテスト。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import tkinter as tk
from matplotlib.figure import Figure

import main
from tests.test_main_settings import FakeBooleanVar, FakeStringVar


class FakeNotebook:
    def __init__(self, selected: str) -> None:
        self.selected = selected

    def select(self) -> str:
        return self.selected


def make_signal(source_name: str, marker: float = 0.0) -> main.SignalData:
    """連続測定テスト用の小さな等間隔波形を作る。"""

    time_s = np.arange(8, dtype=float) * 8e-9
    channel_1 = np.arange(8, dtype=float) + marker
    channel_2 = -channel_1
    return main.SignalData(time_s, channel_1, channel_2, source_name)


def make_analysis_result(
    measurement: main.SignalData,
    reference: main.SignalData,
    sound_speed_m_s: float | None,
    marker: float,
) -> main.AnalysisResult:
    """平均値と最終回の波形を識別できる解析結果を作る。"""

    values = np.arange(8, dtype=float) + marker
    return main.AnalysisResult(
        measurement=measurement,
        reference=reference,
        filtered_channel_1=values,
        filtered_channel_2=-values,
        selected_measurement_time_s=measurement.time_s,
        selected_measurement_filtered=values,
        window=SimpleNamespace(
            time_s=np.array([0.0, 8e-9]),
            gain=np.array([0.0, 1.0]),
        ),
        reference_raw_normalized=values,
        reference_filtered_normalized=values,
        reference_windowed=values,
        matching_time_s=np.array([0.0, 8e-9]),
        matching_score=np.array([marker + 1.0, marker]),
        matching_method=main.MATCH_SQUARED_ERROR,
        best_index=1,
        best_time_s=8e-9,
        best_score=marker,
        sound_speed_m_s=sound_speed_m_s,
    )


def make_ready_picoscope_application() -> tuple[
    main.MeasurementApplication,
    main.SignalData,
    main.PicoScopeSettings,
]:
    """参照取得済みのPicoScope解析状態をGUIなしで作る。"""

    application = object.__new__(main.MeasurementApplication)
    application._busy = False
    parameters = main.AnalysisParameters(
        sample_interval_s=8e-9,
        filter_low_hz=1e6,
        filter_high_hz=5e6,
        filter_order=4,
        filter_passes=2,
        distance_mm=10.0,
    )
    window_parameters = main.WindowParameters(sample_interval_s=8e-9)
    application._read_inputs = Mock(
        return_value=(parameters, window_parameters, (0.0, 20.0))
    )
    application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
    reference = make_signal("reference")
    application.current_reference = reference
    application.current_reference_source_mode = main.SOURCE_PICOSCOPE
    application.current_reference_sample_interval_s = 8e-9
    picoscope_settings = main.PicoScopeSettings(
        duration_s=20e-6,
        trigger_a_threshold_mv=-2000.0,
        trigger_a_direction=main.TRIGGER_DIRECTIONS["立ち下がり"],
        resolution=main.PICOSCOPE_RESOLUTIONS["8 bit"],
        channel_a_range=main.PICOSCOPE_RANGES["±20 V"],
        channel_b_range=main.PICOSCOPE_RANGES["±10 V"],
    )
    application.current_reference_channel_a_range = picoscope_settings.channel_a_range
    application._read_picoscope_settings = Mock(return_value=picoscope_settings)
    application.current_measurement = None
    application.current_measurement_sample_interval_s = None
    application.current_result = None
    application.current_filtered_preview = None
    application.sample_interval_ns_var = FakeStringVar("8")
    application.status_var = FakeStringVar()
    application._clear_result_display = Mock()
    application._set_busy = Mock()
    application._draw_result = Mock()
    application._update_result_labels = Mock()
    application._show_error = Mock()
    application._discard_acquired_data = Mock()
    application._select_distance_for_next_measurement = Mock()
    application.root = SimpleNamespace(after=lambda _delay, callback: callback())
    return application, reference, picoscope_settings


class TabBehaviorTests(unittest.TestCase):
    def test_interface_fonts_grow_one_step_without_changing_graph_font(self) -> None:
        point_font = Mock()
        point_font.cget.return_value = "10"
        pixel_font = Mock()
        pixel_font.cget.return_value = "-12"

        def named_font(font_name: str, *, root: object) -> Mock:
            del root
            if font_name == "TkDefaultFont":
                return point_font
            if font_name == "TkTextFont":
                return pixel_font
            raise tk.TclError("font not available")

        graph_font_size = main.rcParams["font.size"]
        with patch("main.tkfont.nametofont", side_effect=named_font):
            main.enlarge_interface_fonts(object())

        point_font.configure.assert_called_once_with(size=11)
        pixel_font.configure.assert_called_once_with(size=-13)
        self.assertEqual(main.rcParams["font.size"], graph_font_size)

    def test_basic_reference_button_always_acquires_reference(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.acquire_reference = Mock()
        application.acquire_and_analyze = Mock()
        application.current_reference = object()

        application._basic_acquire_reference()

        application.acquire_reference.assert_called_once_with(
            channel_a_range_override=main.CHANNEL_A_FIXED_RANGE,
        )
        application.acquire_and_analyze.assert_not_called()

    def test_basic_measurement_button_always_acquires_measurement(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.acquire_reference = Mock()
        application.acquire_and_analyze = Mock()
        application.current_reference = None
        application.continuous_measurement_enabled_var = FakeBooleanVar(True)
        application.continuous_measurement_count_var = FakeStringVar("5")

        application._basic_acquire_measurement()

        application.acquire_reference.assert_not_called()
        application.acquire_and_analyze.assert_called_once_with(
            channel_a_range_override=main.CHANNEL_A_FIXED_RANGE,
        )

    def test_advanced_continuous_on_uses_selected_measurement_count(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.continuous_measurement_enabled_var = FakeBooleanVar(True)
        application.continuous_measurement_count_var = FakeStringVar("4")
        application.acquire_and_analyze = Mock()
        application._show_error = Mock()

        application._advanced_acquire_and_analyze()

        application.acquire_and_analyze.assert_called_once_with(measurement_count=4)
        application._show_error.assert_not_called()

    def test_advanced_continuous_off_is_one_measurement_and_ignores_count(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.continuous_measurement_enabled_var = FakeBooleanVar(False)
        # OFF時は回数欄を解釈せず、従来の単発を守る。
        application.continuous_measurement_count_var = FakeStringVar("invalid")
        application.acquire_and_analyze = Mock()
        application._show_error = Mock()

        application._advanced_acquire_and_analyze()

        application.acquire_and_analyze.assert_called_once_with()
        application._show_error.assert_not_called()

    def test_advanced_continuous_rejects_invalid_count_before_acquisition(
        self,
    ) -> None:
        for invalid_count in ("1", "101", "not-a-number"):
            with self.subTest(invalid_count=invalid_count):
                application = object.__new__(main.MeasurementApplication)
                application._busy = False
                application.continuous_measurement_enabled_var = FakeBooleanVar(True)
                application.continuous_measurement_count_var = FakeStringVar(
                    invalid_count
                )
                application.acquire_and_analyze = Mock()
                application._show_error = Mock()

                application._advanced_acquire_and_analyze()

                application.acquire_and_analyze.assert_not_called()
                application._show_error.assert_called_once()

    def test_advanced_measurement_command_is_ignored_while_busy(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = True
        application.continuous_measurement_enabled_var = Mock()
        application.acquire_and_analyze = Mock()
        application._show_error = Mock()

        application._advanced_acquire_and_analyze()

        application.continuous_measurement_enabled_var.get.assert_not_called()
        application.acquire_and_analyze.assert_not_called()
        application._show_error.assert_not_called()

    def test_distance_enter_uses_basic_single_or_advanced_measurement_command(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        basic_entry = object()
        advanced_entry = object()
        application.basic_distance_entry = basic_entry
        application.distance_entries = (basic_entry, advanced_entry)
        application.distance_mm_var = FakeStringVar("12.5")
        application.acquire_and_analyze = Mock()
        application._advanced_acquire_and_analyze = Mock()
        application.recalculate = Mock()

        for entry in application.distance_entries:
            with self.subTest(entry=entry):
                result = application._enter_pressed(SimpleNamespace(widget=entry))

                self.assertEqual(result, "break")

        application.acquire_and_analyze.assert_called_once_with(
            channel_a_range_override=main.CHANNEL_A_FIXED_RANGE,
        )
        application._advanced_acquire_and_analyze.assert_called_once_with()
        application.recalculate.assert_not_called()

    def test_empty_distance_does_not_acquire_and_advanced_input_recalculates(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        distance_entry = object()
        application.basic_distance_entry = distance_entry
        application.distance_entries = (distance_entry, object())
        application.distance_mm_var = FakeStringVar("")
        application.acquire_and_analyze = Mock()
        application._advanced_acquire_and_analyze = Mock()
        application.recalculate = Mock()

        application._enter_pressed(SimpleNamespace(widget=distance_entry))
        application._enter_pressed(SimpleNamespace(widget=object()))

        application.acquire_and_analyze.assert_not_called()
        application._advanced_acquire_and_analyze.assert_not_called()
        application.recalculate.assert_called_once_with()

    def test_result_focus_returns_to_distance_entry_on_selected_tab(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.basic_tab = "basic-tab"
        application.advanced_tab = "advanced-tab"
        application.basic_distance_entry = Mock()
        application.distance_entry = Mock()

        for selected, expected, other in (
            ("basic-tab", application.basic_distance_entry, application.distance_entry),
            ("advanced-tab", application.distance_entry, application.basic_distance_entry),
        ):
            with self.subTest(selected=selected):
                expected.reset_mock()
                other.reset_mock()
                application.notebook = FakeNotebook(selected)

                application._select_distance_for_next_measurement()

                expected.focus_set.assert_called_once_with()
                expected.icursor.assert_called_once_with(tk.END)
                expected.selection_range.assert_called_once_with(0, tk.END)
                other.focus_set.assert_not_called()

    def test_continuous_measurement_controls_follow_on_off_and_busy_state(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.continuous_measurement_enabled_var = FakeBooleanVar(False)
        application.continuous_measurement_checkbutton = Mock()
        application.continuous_measurement_count_combobox = Mock()

        application._update_continuous_measurement_controls()

        application.continuous_measurement_checkbutton.configure.assert_called_with(
            state="normal"
        )
        application.continuous_measurement_count_combobox.configure.assert_called_with(
            state="disabled"
        )

        application.continuous_measurement_enabled_var.set(True)
        application._update_continuous_measurement_controls()

        application.continuous_measurement_count_combobox.configure.assert_called_with(
            state="readonly"
        )

        application._busy = True
        application._update_continuous_measurement_controls()

        application.continuous_measurement_checkbutton.configure.assert_called_with(
            state="disabled"
        )
        application.continuous_measurement_count_combobox.configure.assert_called_with(
            state="disabled"
        )

    def test_basic_channel_b_buttons_change_one_step_without_changing_advanced_a(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.channel_a_range_var = FakeStringVar("±5 V")
        application.channel_b_range_var = FakeStringVar("±10 V")
        application.source_mode_var = FakeStringVar(main.SOURCE_CSV)
        application.current_reference = None
        application.status_var = FakeStringVar()
        application._discard_measurement_data = Mock()
        application._select_distance_for_next_measurement = Mock()

        for current, expected in zip(
            main.CHANNEL_B_RANGE_VALUES[:-1],
            main.CHANNEL_B_RANGE_VALUES[1:],
            strict=True,
        ):
            with self.subTest(direction="increase", current=current):
                application.channel_b_range_var.set(current)
                application._change_channel_b_range(1)
                self.assertEqual(application.channel_b_range_var.get(), expected)

        for current, expected in zip(
            main.CHANNEL_B_RANGE_VALUES[1:],
            main.CHANNEL_B_RANGE_VALUES[:-1],
            strict=True,
        ):
            with self.subTest(direction="decrease", current=current):
                application.channel_b_range_var.set(current)
                application._change_channel_b_range(-1)
                self.assertEqual(application.channel_b_range_var.get(), expected)

        self.assertEqual(
            application.channel_a_range_var.get(),
            "±5 V",
        )
        self.assertIn("Channel Bレンジを変更しました", application.status_var.get())
        self.assertEqual(application._discard_measurement_data.call_count, 20)
        self.assertEqual(
            application._select_distance_for_next_measurement.call_count,
            20,
        )

    def test_basic_channel_b_button_actions_are_reversed_and_start_preview(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._change_channel_b_range = Mock(side_effect=(True, True))
        application._start_basic_range_preview = Mock()

        application._basic_range_minus_pressed()
        application._basic_range_plus_pressed()

        self.assertEqual(
            application._change_channel_b_range.call_args_list,
            [call(1), call(-1)],
        )
        self.assertEqual(application._start_basic_range_preview.call_count, 2)

    def test_basic_range_preview_acquires_and_filters_with_basic_ranges(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        filter_parameters = main.AnalysisParameters(
            sample_interval_s=8e-9,
            filter_low_hz=1e6,
            filter_high_hz=5e6,
            filter_order=4,
            filter_passes=2,
            distance_mm=None,
        )
        application._read_filtered_preview_inputs = Mock(
            return_value=(filter_parameters, (0.0, 40.0))
        )
        application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
        application.current_reference = object()
        application.current_reference_sample_interval_s = 8e-9
        application.current_reference_source_mode = main.SOURCE_PICOSCOPE
        picoscope_settings = main.PicoScopeSettings(
            duration_s=40e-6,
            trigger_a_threshold_mv=-2000.0,
            trigger_a_direction=main.TRIGGER_DIRECTIONS["立ち下がり"],
            resolution=main.PICOSCOPE_RESOLUTIONS["8 bit"],
            channel_a_range=main.PICOSCOPE_RANGES[main.CHANNEL_A_FIXED_RANGE],
            channel_b_range=main.PICOSCOPE_RANGES["±5 V"],
        )
        application._read_picoscope_settings = Mock(
            return_value=picoscope_settings
        )
        application.current_measurement = object()
        application.current_measurement_sample_interval_s = 16e-9
        application.current_result = object()
        application.current_filtered_preview = object()
        application.result_speed_var = FakeStringVar("1234.5 m/s")
        application.status_var = FakeStringVar()
        application._clear_result_display = Mock()
        application._set_busy = Mock()
        application._show_error = Mock()
        application._basic_range_preview_completed = Mock()
        application.root = SimpleNamespace(
            after=lambda _delay, callback: callback(),
        )
        measurement = main.SignalData(
            time_s=np.arange(8, dtype=float) * 8e-9,
            channel_1=np.arange(1.0, 9.0),
            channel_2=-np.arange(1.0, 9.0),
            source_name="PicoScope test（実測）",
        )
        filtered_1 = np.arange(0.1, 0.9, 0.1)
        filtered_2 = -np.arange(0.1, 0.9, 0.1)

        with (
            patch("main.threading.Thread") as thread_class,
            patch(
                "main.acquire_picoscope_signal",
                return_value=(measurement, 8e-9),
            ) as acquire_signal,
            patch(
                "main.butter_bandpass_filter",
                side_effect=(filtered_1, filtered_2),
            ) as bandpass_filter,
            patch(
                "main.require_matching_time_resolution",
            ) as require_same_resolution,
            patch("main.run_analysis") as run_analysis,
        ):
            application._start_basic_range_preview()
            worker = thread_class.call_args.kwargs["target"]
            worker()

        application._read_picoscope_settings.assert_called_once_with(
            main.CHANNEL_A_FIXED_RANGE,
        )
        acquire_signal.assert_called_once_with(picoscope_settings, "実測")
        require_same_resolution.assert_called_once_with(8e-9, 8e-9)
        self.assertEqual(bandpass_filter.call_count, 2)
        run_analysis.assert_not_called()
        completed_args = application._basic_range_preview_completed.call_args.args
        preview = completed_args[0]
        self.assertIs(preview.measurement, measurement)
        np.testing.assert_array_equal(preview.filtered_channel_1, filtered_1)
        np.testing.assert_array_equal(preview.filtered_channel_2, filtered_2)
        self.assertEqual(completed_args[1:], (8e-9, (0.0, 40.0)))
        self.assertIsNone(application.current_result)
        self.assertEqual(application.result_speed_var.get(), "—")
        application._clear_result_display.assert_called_once_with()
        application._set_busy.assert_called_once_with(True)
        thread_class.return_value.start.assert_called_once_with()

    def test_basic_display_time_presets_update_shared_range_and_redraw(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.display_min_us_var = FakeStringVar("12.0")
        application.display_max_us_var = FakeStringVar("60.0")
        application.capture_duration_us_var = FakeStringVar("40.0")
        application.distance_mm_var = FakeStringVar("10.61")
        application.current_result = None
        application.current_filtered_preview = None
        application.status_var = FakeStringVar()
        application._draw_result = Mock()
        application._draw_basic_range_preview = Mock()
        application.acquire_reference = Mock()
        application.acquire_and_analyze = Mock()
        application.recalculate = Mock()
        application._select_distance_for_next_measurement = Mock()

        application._set_basic_display_range(
            "通常",
            main.BASIC_DISPLAY_NORMAL_US,
        )

        self.assertEqual(application.display_min_us_var.get(), "0.0")
        self.assertEqual(application.display_max_us_var.get(), "40.0")
        application._draw_result.assert_not_called()
        self.assertIn("通常", application.status_var.get())

        result = object()
        application.current_result = result
        application._set_basic_display_range(
            "拡大",
            main.BASIC_DISPLAY_EXPANDED_US,
        )

        self.assertEqual(application.display_min_us_var.get(), "0.0")
        self.assertEqual(application.display_max_us_var.get(), "80.0")
        self.assertEqual(application.capture_duration_us_var.get(), "40.0")
        self.assertEqual(application.distance_mm_var.get(), "10.61")
        application._draw_result.assert_called_once_with(result, (0.0, 80.0))
        application.acquire_reference.assert_not_called()
        application.acquire_and_analyze.assert_not_called()
        application.recalculate.assert_not_called()
        self.assertEqual(
            application._select_distance_for_next_measurement.call_count,
            2,
        )
        self.assertIn("拡大", application.status_var.get())

    def test_basic_display_time_preset_redraws_preview_without_matching_result(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.display_min_us_var = FakeStringVar("0.0")
        application.display_max_us_var = FakeStringVar("40.0")
        application.current_result = None
        preview = object()
        application.current_filtered_preview = preview
        application.result_speed_var = FakeStringVar("—")
        application.status_var = FakeStringVar()
        application._draw_result = Mock()
        application._draw_basic_range_preview = Mock()
        application._select_distance_for_next_measurement = Mock()

        application._set_basic_display_range(
            "拡大",
            main.BASIC_DISPLAY_EXPANDED_US,
        )

        application._draw_basic_range_preview.assert_called_once_with(
            preview,
            (0.0, 80.0),
        )
        application._draw_result.assert_not_called()
        self.assertEqual(application.result_speed_var.get(), "—")
        application._select_distance_for_next_measurement.assert_called_once_with()

    def test_basic_display_time_preset_is_ignored_while_busy(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = True
        application.display_min_us_var = FakeStringVar("5.0")
        application.display_max_us_var = FakeStringVar("60.0")
        application.current_result = object()
        application.status_var = FakeStringVar("busy")
        application._draw_result = Mock()
        application._select_distance_for_next_measurement = Mock()

        application._set_basic_display_range(
            "通常",
            main.BASIC_DISPLAY_NORMAL_US,
        )

        self.assertEqual(application.display_min_us_var.get(), "5.0")
        self.assertEqual(application.display_max_us_var.get(), "60.0")
        application._draw_result.assert_not_called()
        application._select_distance_for_next_measurement.assert_not_called()
        self.assertEqual(application.status_var.get(), "busy")

    def test_channel_b_range_stops_at_supported_minimum_and_maximum(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.channel_a_range_var = FakeStringVar(main.CHANNEL_A_FIXED_RANGE)
        application.channel_b_range_var = FakeStringVar("±10 mV")
        application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
        application.current_reference = object()
        application.status_var = FakeStringVar("unchanged")
        application._discard_acquired_data = Mock()
        application._start_basic_range_preview = Mock()

        application._basic_range_plus_pressed()

        self.assertEqual(application.channel_b_range_var.get(), "±10 mV")
        application._discard_acquired_data.assert_not_called()
        self.assertEqual(application.status_var.get(), "unchanged")

        application.channel_b_range_var.set("±20 V")
        application._basic_range_minus_pressed()

        self.assertEqual(application.channel_b_range_var.get(), "±20 V")
        application._discard_acquired_data.assert_not_called()
        application._start_basic_range_preview.assert_not_called()

        with self.assertRaisesRegex(ValueError, "−1または＋1"):
            application._change_channel_b_range(2)

    def test_basic_range_button_does_not_acquire_while_busy(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = True
        application.channel_b_range_var = FakeStringVar("±5 V")
        application._start_basic_range_preview = Mock()

        application._basic_range_plus_pressed()

        self.assertEqual(application.channel_b_range_var.get(), "±5 V")
        application._start_basic_range_preview.assert_not_called()

    def test_channel_b_range_buttons_disable_at_each_endpoint(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.basic_range_decrease_button = Mock()
        application.basic_range_increase_button = Mock()
        application.channel_b_range_var = FakeStringVar(
            main.CHANNEL_B_RANGE_VALUES[0]
        )

        application._update_channel_b_range_buttons()

        self.assertEqual(
            application.basic_range_decrease_button.configure.call_args.kwargs[
                "state"
            ],
            "normal",
        )
        self.assertEqual(
            application.basic_range_increase_button.configure.call_args.kwargs[
                "state"
            ],
            "disabled",
        )

        application.channel_b_range_var.set(main.CHANNEL_B_RANGE_VALUES[-1])
        application._update_channel_b_range_buttons()

        self.assertEqual(
            application.basic_range_decrease_button.configure.call_args.kwargs[
                "state"
            ],
            "disabled",
        )
        self.assertEqual(
            application.basic_range_increase_button.configure.call_args.kwargs[
                "state"
            ],
            "normal",
        )

    def test_picoscope_channel_b_range_change_keeps_reference(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.channel_a_range_var = FakeStringVar("±20 V")
        application.channel_b_range_var = FakeStringVar("±10 V")
        application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
        reference = object()
        application.current_reference = reference
        application.status_var = FakeStringVar()
        application._discard_acquired_data = Mock()
        application._discard_measurement_data = Mock()
        application._select_distance_for_next_measurement = Mock()

        application._change_channel_b_range(-1)

        application._discard_acquired_data.assert_not_called()
        application._discard_measurement_data.assert_called_once_with()
        application._select_distance_for_next_measurement.assert_called_once_with()
        self.assertIs(application.current_reference, reference)
        self.assertIn("参照波形を保持", application.status_var.get())
        self.assertIn("時間分解能を確認", application.status_var.get())
        self.assertNotIn("取り直してください", application.status_var.get())

    def test_discard_measurement_data_preserves_reference(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        reference = object()
        application.current_reference = reference
        application.current_measurement = object()
        application.current_measurement_sample_interval_s = 8e-9
        application.current_result = object()
        application._clear_result_display = Mock()

        application._discard_measurement_data()

        self.assertIs(application.current_reference, reference)
        self.assertIsNone(application.current_measurement)
        self.assertIsNone(application.current_measurement_sample_interval_s)
        self.assertIsNone(application.current_result)
        application._clear_result_display.assert_called_once_with()

    def test_advanced_channel_b_range_change_keeps_reference(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.channel_a_range_var = FakeStringVar("±5 V")
        application.channel_b_range_var = FakeStringVar("±2 V")
        application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
        reference = object()
        application.current_reference = reference
        application.status_var = FakeStringVar()
        application._discard_acquired_data = Mock()
        application._discard_measurement_data = Mock()

        application._advanced_channel_b_range_changed()

        application._discard_acquired_data.assert_not_called()
        application._discard_measurement_data.assert_called_once_with()
        self.assertIs(application.current_reference, reference)
        self.assertIn("Channel A ±5 V", application.status_var.get())
        self.assertIn("Channel B ±2 V", application.status_var.get())
        self.assertIn("参照波形を保持", application.status_var.get())

    def test_advanced_channel_a_range_change_requires_reference_reacquisition(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application.channel_a_range_var = FakeStringVar("±5 V")
        application.channel_b_range_var = FakeStringVar("±2 V")
        application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
        application.current_reference = object()
        application.status_var = FakeStringVar()
        application._discard_acquired_data = Mock()

        application._advanced_channel_a_range_changed()

        application._discard_acquired_data.assert_called_once_with()
        self.assertIn("参照波形を取り直してください", application.status_var.get())

    def test_picoscope_settings_use_advanced_a_or_basic_fixed_override(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.capture_duration_us_var = FakeStringVar("20")
        application.trigger_a_threshold_mv_var = FakeStringVar("-2000")
        application.trigger_a_direction_var = FakeStringVar("立ち下がり")
        application.picoscope_resolution_var = FakeStringVar("8 bit")
        application.channel_a_range_var = FakeStringVar("±5 V")
        application.channel_b_range_var = FakeStringVar("±5 V")

        advanced_settings = application._read_picoscope_settings()
        basic_settings = application._read_picoscope_settings(
            main.CHANNEL_A_FIXED_RANGE
        )

        self.assertEqual(
            advanced_settings.channel_a_range,
            main.PICOSCOPE_RANGES["±5 V"],
        )
        self.assertEqual(
            basic_settings.channel_a_range,
            main.PICOSCOPE_RANGES[main.CHANNEL_A_FIXED_RANGE],
        )
        self.assertEqual(
            advanced_settings.channel_b_range,
            main.PICOSCOPE_RANGES["±5 V"],
        )

    def test_continuous_measurement_runs_serially_averages_speed_and_draws_last(
        self,
    ) -> None:
        application, reference, picoscope_settings = (
            make_ready_picoscope_application()
        )
        measurements = [
            make_signal(f"measurement-{index}", marker=float(index))
            for index in range(1, 4)
        ]
        speeds = (1000.0, 1200.0, 1700.0)
        results = [
            make_analysis_result(measurement, reference, speed, float(index * 10))
            for index, (measurement, speed) in enumerate(
                zip(measurements, speeds, strict=True),
                start=1,
            )
        ]
        events: list[str] = []
        measurement_indexes = {id(item): index for index, item in enumerate(measurements, 1)}
        acquisition_iterator = iter(measurements)

        def acquire_signal(
            settings: main.PicoScopeSettings,
            role: str,
        ) -> tuple[main.SignalData, float]:
            self.assertIs(settings, picoscope_settings)
            self.assertEqual(role, "実測")
            measurement = next(acquisition_iterator)
            events.append(f"acquire-{measurement_indexes[id(measurement)]}")
            return measurement, 8e-9

        def analyze(
            measurement: main.SignalData,
            acquired_reference: main.SignalData,
            _parameters: main.AnalysisParameters,
            _window_parameters: main.WindowParameters,
        ) -> main.AnalysisResult:
            index = measurement_indexes[id(measurement)]
            events.append(f"analyze-{index}")
            self.assertIs(acquired_reference, reference)
            return results[index - 1]

        application._draw_result.side_effect = lambda *_args: events.append("draw")
        with (
            patch("main.threading.Thread") as thread_class,
            patch("main.acquire_picoscope_signal", side_effect=acquire_signal),
            patch("main.run_analysis", side_effect=analyze) as run_analysis,
        ):
            application._start_analysis(load_new_data=True, measurement_count=3)
            thread_class.call_args.kwargs["target"]()

        self.assertEqual(
            events,
            [
                "acquire-1",
                "analyze-1",
                "acquire-2",
                "analyze-2",
                "acquire-3",
                "analyze-3",
                "draw",
            ],
        )
        self.assertEqual(run_analysis.call_count, 3)
        application._draw_result.assert_called_once()
        displayed_result, displayed_range = application._draw_result.call_args.args
        self.assertIs(displayed_result.measurement, measurements[-1])
        np.testing.assert_array_equal(
            displayed_result.filtered_channel_1,
            results[-1].filtered_channel_1,
        )
        self.assertAlmostEqual(displayed_result.sound_speed_m_s, 1300.0)
        self.assertEqual(displayed_range, (0.0, 20.0))
        self.assertIs(application.current_measurement, measurements[-1])
        self.assertIs(application.current_result, displayed_result)
        application._update_result_labels.assert_called_once_with(displayed_result)
        self.assertEqual(
            application._set_busy.call_args_list,
            [call(True), call(False)],
        )
        self.assertIn("3回の音速を平均", application.status_var.get())
        thread_class.return_value.start.assert_called_once_with()

    def test_basic_measurement_stays_single_when_advanced_continuous_is_on(
        self,
    ) -> None:
        application, reference, _picoscope_settings = (
            make_ready_picoscope_application()
        )
        application.continuous_measurement_enabled_var = FakeBooleanVar(True)
        application.continuous_measurement_count_var = FakeStringVar("5")
        measurement = make_signal("basic measurement", marker=1.0)
        result = make_analysis_result(measurement, reference, 1250.0, 1.0)

        with (
            patch("main.threading.Thread") as thread_class,
            patch(
                "main.acquire_picoscope_signal",
                return_value=(measurement, 8e-9),
            ) as acquire_signal,
            patch("main.run_analysis", return_value=result) as run_analysis,
        ):
            application._basic_acquire_measurement()
            thread_class.call_args.kwargs["target"]()

        acquire_signal.assert_called_once()
        run_analysis.assert_called_once()
        application._draw_result.assert_called_once()
        displayed_result = application._draw_result.call_args.args[0]
        self.assertIs(displayed_result, result)
        self.assertEqual(displayed_result.sound_speed_m_s, 1250.0)
        self.assertNotIn("回の音速を平均", application.status_var.get())

    def test_continuous_analysis_failure_stops_and_keeps_latest_reusable_waveform(
        self,
    ) -> None:
        application, reference, _picoscope_settings = (
            make_ready_picoscope_application()
        )
        measurements = [
            make_signal("measurement-1", marker=1.0),
            make_signal("measurement-2", marker=2.0),
            make_signal("measurement-3", marker=3.0),
        ]
        first_result = make_analysis_result(
            measurements[0],
            reference,
            1000.0,
            1.0,
        )
        error = RuntimeError("second analysis failed")

        with (
            patch("main.threading.Thread") as thread_class,
            patch(
                "main.acquire_picoscope_signal",
                side_effect=[(item, 8e-9) for item in measurements],
            ) as acquire_signal,
            patch(
                "main.run_analysis",
                side_effect=[first_result, error],
            ) as run_analysis,
        ):
            application._start_analysis(load_new_data=True, measurement_count=3)
            thread_class.call_args.kwargs["target"]()

        self.assertEqual(acquire_signal.call_count, 2)
        self.assertEqual(run_analysis.call_count, 2)
        application._draw_result.assert_not_called()
        self.assertIsNone(application.current_result)
        self.assertIs(application.current_measurement, measurements[1])
        self.assertEqual(application.current_measurement_sample_interval_s, 8e-9)
        self.assertIn("実測データは取得済み", application.status_var.get())
        application._show_error.assert_called_once_with(error)
        self.assertEqual(
            application._set_busy.call_args_list,
            [call(True), call(False)],
        )

    def test_later_acquisition_failure_does_not_reuse_previous_iteration(
        self,
    ) -> None:
        application, reference, _picoscope_settings = (
            make_ready_picoscope_application()
        )
        first_measurement = make_signal("measurement-1", marker=1.0)
        first_result = make_analysis_result(
            first_measurement,
            reference,
            1000.0,
            1.0,
        )
        error = RuntimeError("second acquisition failed")

        with (
            patch("main.threading.Thread") as thread_class,
            patch(
                "main.acquire_picoscope_signal",
                side_effect=[(first_measurement, 8e-9), error],
            ) as acquire_signal,
            patch("main.run_analysis", return_value=first_result) as run_analysis,
        ):
            application._start_analysis(load_new_data=True, measurement_count=3)
            thread_class.call_args.kwargs["target"]()

        self.assertEqual(acquire_signal.call_count, 2)
        run_analysis.assert_called_once()
        application._draw_result.assert_not_called()
        self.assertIsNone(application.current_measurement)
        self.assertIsNone(application.current_measurement_sample_interval_s)
        self.assertIn("取得または解析に失敗", application.status_var.get())
        application._show_error.assert_called_once_with(error)

    def test_continuous_time_resolution_change_stops_and_discards_reference(
        self,
    ) -> None:
        application, reference, _picoscope_settings = (
            make_ready_picoscope_application()
        )
        measurements = [
            make_signal("measurement-1", marker=1.0),
            make_signal("measurement-2", marker=2.0),
            make_signal("measurement-3", marker=3.0),
        ]
        first_result = make_analysis_result(
            measurements[0],
            reference,
            1000.0,
            1.0,
        )

        with (
            patch("main.threading.Thread") as thread_class,
            patch(
                "main.acquire_picoscope_signal",
                side_effect=[
                    (measurements[0], 8e-9),
                    (measurements[1], 16e-9),
                    (measurements[2], 8e-9),
                ],
            ) as acquire_signal,
            patch("main.run_analysis", return_value=first_result) as run_analysis,
        ):
            application._start_analysis(load_new_data=True, measurement_count=3)
            thread_class.call_args.kwargs["target"]()

        self.assertEqual(acquire_signal.call_count, 2)
        run_analysis.assert_called_once()
        application._draw_result.assert_not_called()
        application._discard_acquired_data.assert_called_once_with()
        error = application._show_error.call_args.args[0]
        self.assertIsInstance(error, main.TimeResolutionMismatchError)
        self.assertIn("参照波形を取り直してください", application.status_var.get())
        self.assertEqual(
            application._set_busy.call_args_list,
            [call(True), call(False)],
        )

    def test_continuous_measurement_requires_distance_before_starting_thread(
        self,
    ) -> None:
        for distance_mm in (None, 0.0):
            with self.subTest(distance_mm=distance_mm):
                application, _reference, _picoscope_settings = (
                    make_ready_picoscope_application()
                )
                parameters, window_parameters, display_range = (
                    application._read_inputs.return_value
                )
                application._read_inputs.return_value = (
                    main.AnalysisParameters(
                        sample_interval_s=parameters.sample_interval_s,
                        filter_low_hz=parameters.filter_low_hz,
                        filter_high_hz=parameters.filter_high_hz,
                        filter_order=parameters.filter_order,
                        filter_passes=parameters.filter_passes,
                        distance_mm=distance_mm,
                    ),
                    window_parameters,
                    display_range,
                )

                with patch("main.threading.Thread") as thread_class:
                    application._start_analysis(
                        load_new_data=True,
                        measurement_count=2,
                    )

                thread_class.assert_not_called()
                application._set_busy.assert_not_called()
                application._show_error.assert_called_once()
                self.assertIn(
                    "距離",
                    str(application._show_error.call_args.args[0]),
                )

    def test_start_analysis_ignores_new_request_while_busy(self) -> None:
        application, _reference, _picoscope_settings = (
            make_ready_picoscope_application()
        )
        application._busy = True

        with patch("main.threading.Thread") as thread_class:
            application._start_analysis(load_new_data=True, measurement_count=3)

        application._read_inputs.assert_not_called()
        application._set_busy.assert_not_called()
        thread_class.assert_not_called()

    def test_analysis_can_start_after_channel_b_range_change(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application._read_inputs = Mock(
            return_value=(object(), object(), (0.0, 20.0))
        )
        application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
        application.current_reference = object()
        application.current_reference_source_mode = main.SOURCE_PICOSCOPE
        application.current_reference_sample_interval_s = 8e-9
        application.current_reference_channel_a_range = main.PICOSCOPE_RANGES[
            main.CHANNEL_A_FIXED_RANGE
        ]
        application._read_picoscope_settings = Mock(
            return_value=main.PicoScopeSettings(
                duration_s=20e-6,
                trigger_a_threshold_mv=-2000.0,
                trigger_a_direction=main.TRIGGER_DIRECTIONS["立ち下がり"],
                resolution=main.PICOSCOPE_RESOLUTIONS["8 bit"],
                channel_a_range=main.PICOSCOPE_RANGES[
                    main.CHANNEL_A_FIXED_RANGE
                ],
                # 参照取得後に変更されたBレンジを想定する。
                channel_b_range=main.PICOSCOPE_RANGES["±5 V"],
            )
        )
        application.current_measurement = object()
        application.current_measurement_sample_interval_s = 8e-9
        application.current_result = object()
        application._clear_result_display = Mock()
        application._set_busy = Mock()
        application._show_error = Mock()
        application.status_var = FakeStringVar()

        with patch("main.threading.Thread") as thread_class:
            application._start_analysis(
                load_new_data=True,
                channel_a_range_override=main.CHANNEL_A_FIXED_RANGE,
            )

        application._show_error.assert_not_called()
        thread_class.assert_called_once()
        thread_class.return_value.start.assert_called_once_with()

    def test_analysis_still_rejects_changed_channel_a_range(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        application._busy = False
        application._read_inputs = Mock(
            return_value=(object(), object(), (0.0, 20.0))
        )
        application.source_mode_var = FakeStringVar(main.SOURCE_PICOSCOPE)
        application.current_reference = object()
        application.current_reference_source_mode = main.SOURCE_PICOSCOPE
        application.current_reference_sample_interval_s = 8e-9
        application.current_reference_channel_a_range = main.PICOSCOPE_RANGES[
            "±5 V"
        ]
        application._read_picoscope_settings = Mock(
            return_value=main.PicoScopeSettings(
                duration_s=20e-6,
                trigger_a_threshold_mv=-2000.0,
                trigger_a_direction=main.TRIGGER_DIRECTIONS["立ち下がり"],
                resolution=main.PICOSCOPE_RESOLUTIONS["8 bit"],
                channel_a_range=main.PICOSCOPE_RANGES[
                    main.CHANNEL_A_FIXED_RANGE
                ],
                channel_b_range=main.PICOSCOPE_RANGES["±5 V"],
            )
        )
        application._show_error = Mock()

        with patch("main.threading.Thread") as thread_class:
            application._start_analysis(
                load_new_data=True,
                channel_a_range_override=main.CHANNEL_A_FIXED_RANGE,
            )

        thread_class.assert_not_called()
        error = application._show_error.call_args.args[0]
        self.assertIn("Channel Aレンジ", str(error))
        self.assertIn("参照波形を取り直してください", str(error))

    def test_time_resolution_comparison_accepts_same_interval_only(self) -> None:
        main.require_matching_time_resolution(8e-9, 8e-9)
        main.require_matching_time_resolution(8e-9, 8e-9 * (1.0 + 5e-6))

        with self.assertRaisesRegex(
            main.TimeResolutionMismatchError,
            "時間分解能",
        ):
            main.require_matching_time_resolution(8e-9, 16e-9)

    def test_time_resolution_change_discards_reference_and_requests_reacquisition(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.current_result = object()
        application.status_var = FakeStringVar()
        application._set_busy = Mock()
        application._discard_acquired_data = Mock()
        application._show_error = Mock()
        error = main.TimeResolutionMismatchError("時間分解能が一致しません")

        application._analysis_failed(error)

        application._set_busy.assert_called_once_with(False)
        application._discard_acquired_data.assert_called_once_with()
        self.assertIn("参照波形を取り直してください", application.status_var.get())
        application._show_error.assert_called_once_with(error)

    def test_basic_range_preview_time_resolution_change_discards_reference(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.current_result = object()
        application.current_filtered_preview = object()
        application.result_speed_var = FakeStringVar("1234.5 m/s")
        application.status_var = FakeStringVar()
        application._set_busy = Mock()
        application._discard_acquired_data = Mock()
        application._show_error = Mock()
        application._select_distance_for_next_measurement = Mock()
        error = main.TimeResolutionMismatchError("時間分解能が一致しません")

        application._basic_range_preview_failed(error)

        application._set_busy.assert_called_once_with(False)
        application._discard_acquired_data.assert_called_once_with()
        self.assertIsNone(application.current_result)
        self.assertIsNone(application.current_filtered_preview)
        self.assertEqual(application.result_speed_var.get(), "—")
        self.assertIn("参照波形を取り直してください", application.status_var.get())
        application._show_error.assert_called_once_with(error)
        application._select_distance_for_next_measurement.assert_called_once_with()

    def test_basic_range_preview_completion_keeps_only_filtered_result(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        reference = object()
        application.current_reference = reference
        application.current_result = object()
        application.current_filtered_preview = None
        application.sample_interval_ns_var = FakeStringVar("16")
        application.result_speed_var = FakeStringVar("1234.5 m/s")
        application.status_var = FakeStringVar()
        application._draw_basic_range_preview = Mock()
        application._draw_result = Mock()
        application._update_result_labels = Mock()
        application._set_busy = Mock()
        application._select_distance_for_next_measurement = Mock()
        measurement = main.SignalData(
            time_s=np.arange(8, dtype=float) * 8e-9,
            channel_1=np.arange(1.0, 9.0),
            channel_2=-np.arange(1.0, 9.0),
            source_name="range preview",
        )
        preview = main.FilteredWaveformPreview(
            measurement=measurement,
            filtered_channel_1=np.arange(0.1, 0.9, 0.1),
            filtered_channel_2=-np.arange(0.1, 0.9, 0.1),
        )

        application._basic_range_preview_completed(
            preview,
            8e-9,
            (0.0, 40.0),
        )

        self.assertIs(application.current_measurement, measurement)
        self.assertEqual(application.current_measurement_sample_interval_s, 8e-9)
        self.assertIsNone(application.current_result)
        self.assertIs(application.current_filtered_preview, preview)
        self.assertIs(application.current_reference, reference)
        self.assertEqual(application.sample_interval_ns_var.get(), "8")
        self.assertEqual(application.result_speed_var.get(), "—")
        application._draw_basic_range_preview.assert_called_once_with(
            preview,
            (0.0, 40.0),
        )
        application._draw_result.assert_not_called()
        application._update_result_labels.assert_not_called()
        application._set_busy.assert_called_once_with(False)
        application._select_distance_for_next_measurement.assert_called_once_with()

    def test_basic_range_preview_draws_filtered_data_and_clears_matching_graph(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.basic_filtered_axis, application.basic_matching_axis = (
            Figure().subplots(1, 2)
        )
        application.basic_matching_axis.plot(
            np.array([0.0, 1.0]),
            np.array([10.0, 20.0]),
            label="old matching score",
        )
        application.basic_canvas = Mock()
        application.canvas = Mock()
        application.channel_b_range_var = FakeStringVar("±5 V")
        preview = main.FilteredWaveformPreview(
            measurement=main.SignalData(
                time_s=np.linspace(0.0, 80e-6, 8),
                channel_1=np.arange(1.0, 9.0),
                channel_2=-np.arange(1.0, 9.0),
                source_name="range preview",
            ),
            filtered_channel_1=np.arange(0.1, 0.9, 0.1),
            filtered_channel_2=-np.arange(0.1, 0.9, 0.1),
        )

        application._draw_basic_range_preview(preview, (0.0, 40.0))

        filtered_lines = application.basic_filtered_axis.lines
        self.assertEqual(
            [line.get_label() for line in filtered_lines[:2]],
            ["Ch1 filtered", "Ch2 filtered"],
        )
        np.testing.assert_array_equal(
            filtered_lines[0].get_ydata(),
            preview.filtered_channel_1,
        )
        np.testing.assert_array_equal(
            filtered_lines[1].get_ydata(),
            preview.filtered_channel_2,
        )
        self.assertEqual(application.basic_filtered_axis.get_xlim(), (0.0, 40.0))
        self.assertEqual(application.basic_filtered_axis.get_ylim(), (-6.0, 6.0))
        self.assertEqual(application.basic_matching_axis.get_title(), "評価関数")
        self.assertEqual(len(application.basic_matching_axis.lines), 0)
        self.assertGreaterEqual(len(application.basic_matching_axis.texts), 1)
        application.basic_canvas.draw_idle.assert_called_once_with()
        application.canvas.draw_idle.assert_not_called()

    def test_full_analysis_after_preview_restores_matching_and_speed_updates(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        measurement = SimpleNamespace(source_name="measurement")
        reference = SimpleNamespace(source_name="reference")
        result = SimpleNamespace(measurement=measurement, reference=reference)
        application.current_filtered_preview = object()
        application.sample_interval_ns_var = FakeStringVar("8")
        application.status_var = FakeStringVar()
        application._draw_result = Mock()
        application._update_result_labels = Mock()
        application._set_busy = Mock()
        application._select_distance_for_next_measurement = Mock()

        application._analysis_completed(result, (0.0, 80.0), 8e-9)

        self.assertIs(application.current_result, result)
        self.assertIsNone(application.current_filtered_preview)
        application._draw_result.assert_called_once_with(result, (0.0, 80.0))
        application._update_result_labels.assert_called_once_with(result)
        application._set_busy.assert_called_once_with(False)
        application._select_distance_for_next_measurement.assert_called_once_with()

    def test_filtered_and_matching_helpers_draw_same_data_for_both_tabs(self) -> None:
        result = SimpleNamespace(
            measurement=SimpleNamespace(
                time_s=np.array([0.0, 1e-6, 2e-6]),
            ),
            filtered_channel_1=np.array([1.0, 2.0, 3.0]),
            filtered_channel_2=np.array([-1.0, -2.0, -3.0]),
            matching_time_s=np.array([0.0, 1e-6]),
            matching_score=np.array([4.0, 1.0]),
            best_time_s=1e-6,
            best_score=1.0,
            matching_method=main.MATCH_SQUARED_ERROR,
        )
        axes = Figure().subplots(2, 2)
        advanced_filtered, basic_filtered = axes[0]
        advanced_matching, basic_matching = axes[1]

        main.MeasurementApplication._draw_filtered_axis(
            advanced_filtered,
            result,
            (0.0, 2.0),
            "バンドパスフィルター後",
            5.0,
        )
        main.MeasurementApplication._draw_filtered_axis(
            basic_filtered,
            result,
            (0.0, 2.0),
            "データフィルタ後",
            5.0,
        )
        main.MeasurementApplication._draw_matching_axis(
            advanced_matching,
            result,
            (0.0, 2.0),
        )
        main.MeasurementApplication._draw_matching_axis(
            basic_matching,
            result,
            (0.0, 2.0),
            title_prefix="評価関数：",
        )

        for advanced_line, basic_line in zip(
            advanced_filtered.lines,
            basic_filtered.lines,
            strict=True,
        ):
            np.testing.assert_array_equal(advanced_line.get_xdata(), basic_line.get_xdata())
            np.testing.assert_array_equal(advanced_line.get_ydata(), basic_line.get_ydata())
        np.testing.assert_array_equal(
            advanced_matching.lines[0].get_ydata(),
            basic_matching.lines[0].get_ydata(),
        )
        self.assertEqual(advanced_filtered.get_xlim(), basic_filtered.get_xlim())
        self.assertEqual(advanced_matching.get_xlim(), basic_matching.get_xlim())
        self.assertEqual(basic_filtered.get_title(), "データフィルタ後")
        self.assertTrue(basic_matching.get_title().startswith("評価関数："))
        self.assertEqual(advanced_filtered.get_ylim(), (-6.0, 6.0))
        self.assertEqual(basic_filtered.get_ylim(), (-6.0, 6.0))
        self.assertEqual(
            [line.get_ydata()[0] for line in advanced_filtered.lines[-2:]],
            [-3.0, 3.0],
        )
        self.assertEqual(
            [line.get_ydata()[0] for line in basic_filtered.lines[-2:]],
            [-3.0, 3.0],
        )

    def test_channel_b_range_voltage_conversion_supports_mv_and_v_ranges(
        self,
    ) -> None:
        application = object.__new__(main.MeasurementApplication)
        application.channel_b_range_var = FakeStringVar()

        for displayed_range, expected_voltage in (
            ("±10 mV", 0.01),
            ("±500 mV", 0.5),
            ("±1 V", 1.0),
            ("±20 V", 20.0),
        ):
            with self.subTest(displayed_range=displayed_range):
                application.channel_b_range_var.set(displayed_range)
                self.assertEqual(
                    application._channel_b_range_v(),
                    expected_voltage,
                )

        application.channel_b_range_var.set("unknown")
        with self.assertRaisesRegex(ValueError, "Channel B"):
            application._channel_b_range_v()

    def test_empty_plot_and_busy_updates_include_both_tabs(self) -> None:
        application = object.__new__(main.MeasurementApplication)
        axes = Figure().subplots(3, 2).flat
        (
            application.raw_axis,
            application.filtered_axis,
            application.window_axis,
            application.matching_axis,
            application.basic_filtered_axis,
            application.basic_matching_axis,
        ) = axes
        application.canvas = Mock()
        application.basic_canvas = Mock()
        application.channel_b_range_var = FakeStringVar("±5 V")

        application._draw_empty_plots()

        self.assertEqual(application.basic_filtered_axis.get_title(), "データフィルタ後")
        self.assertEqual(application.basic_matching_axis.get_title(), "評価関数")
        self.assertEqual(application.filtered_axis.get_ylim(), (-6.0, 6.0))
        self.assertEqual(application.basic_filtered_axis.get_ylim(), (-6.0, 6.0))
        self.assertEqual(
            [line.get_ydata()[0] for line in application.filtered_axis.lines],
            [-3.0, 3.0],
        )
        application.canvas.draw_idle.assert_called_once_with()
        application.basic_canvas.draw_idle.assert_called_once_with()

        widget_names = (
            "acquire_reference_button",
            "acquire_button",
            "recalculate_button",
            "save_window_button",
            "save_settings_button",
            "load_settings_button",
            "basic_reference_button",
            "basic_acquire_button",
            "basic_display_normal_button",
            "basic_display_expanded_button",
            "source_combobox",
            "channel_a_range_combobox",
            "channel_b_range_combobox",
        )
        for name in widget_names:
            setattr(application, name, Mock())
        range_button_names = (
            "basic_range_decrease_button",
            "basic_range_increase_button",
        )
        for name in range_button_names:
            setattr(application, name, Mock())
        application.continuous_measurement_enabled_var = FakeBooleanVar(True)
        application.continuous_measurement_checkbutton = Mock()
        application.continuous_measurement_count_combobox = Mock()
        application.channel_b_range_var.set("±10 V")

        application._set_busy(True)
        application._set_busy(False)

        for name in widget_names:
            widget = getattr(application, name)
            self.assertEqual(
                widget.configure.call_args_list[-2].kwargs["state"],
                "disabled",
            )
            expected_normal = (
                "readonly" if name.endswith("combobox") else "normal"
            )
            self.assertEqual(
                widget.configure.call_args_list[-1].kwargs["state"],
                expected_normal,
            )
        for name in range_button_names:
            widget = getattr(application, name)
            self.assertEqual(
                widget.configure.call_args_list[-2].kwargs["state"],
                "disabled",
            )
            self.assertEqual(
                widget.configure.call_args_list[-1].kwargs["state"],
                "normal",
            )
        self.assertEqual(
            application.continuous_measurement_checkbutton.configure.call_args_list[
                -2
            ].kwargs["state"],
            "disabled",
        )
        self.assertEqual(
            application.continuous_measurement_checkbutton.configure.call_args_list[
                -1
            ].kwargs["state"],
            "normal",
        )
        self.assertEqual(
            application.continuous_measurement_count_combobox.configure.call_args_list[
                -2
            ].kwargs["state"],
            "disabled",
        )
        self.assertEqual(
            application.continuous_measurement_count_combobox.configure.call_args_list[
                -1
            ].kwargs["state"],
            "readonly",
        )


if __name__ == "__main__":
    unittest.main()
