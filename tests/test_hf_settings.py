"""Input/resource validation only. These tests do not establish model inference."""
import math

import pytest

from backend.models import PRESETS, validate_automatic_settings


@pytest.mark.parametrize("name", ["pred_iou_thresh", "stability_score_thresh", "box_nms_thresh"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -.01, 1.01, True, "0.5"])
def test_rejects_invalid_thresholds(name, value):
    with pytest.raises(ValueError, match="finite number"):
        validate_automatic_settings({name: value}, 1_000_000)


@pytest.mark.parametrize("value", [0, 1])
def test_accepts_threshold_endpoints(value):
    values = validate_automatic_settings({"pred_iou_thresh": value, "stability_score_thresh": value, "box_nms_thresh": value}, 1_000_000)
    assert values["pred_iou_thresh"] == value


@pytest.mark.parametrize("settings", [{"points_per_side": 33}, {"points_per_side": 1}, {"points_per_side": 8.5},
                                      {"points_per_batch": 0}, {"points_per_batch": 17}, {"points_per_batch": True},
                                      {"crop_n_layers": 1}, {"crop_nms_thresh": .7}, {"max_hole_area": 1}])
def test_rejects_unsupported_controls(settings):
    with pytest.raises(ValueError):
        validate_automatic_settings(settings, 1_000_000)


def test_caps_large_image_batch_and_records_request_without_mutating_preset():
    preset = dict(PRESETS["Balanced"])
    values = validate_automatic_settings(preset, 8_000_000)
    assert values["points_per_batch"] == 1
    assert values["requested_points_per_batch"] == 4
    assert preset == PRESETS["Balanced"]
    assert "crop_nms_thresh" not in preset


@pytest.mark.parametrize("pixels", [0, -1, 8_000_001])
def test_rejects_invalid_or_oversized_automatic_images(pixels):
    with pytest.raises(ValueError):
        validate_automatic_settings({}, pixels)
