"""Deterministic, reversible whole-mask suppression on canonical pixels."""
import math
import numpy as np

DEFAULTS = dict(remove_overlap=False, overlap_threshold=80.0, overlap_metric="smaller")


def validate_overlap(settings):
    values = {**DEFAULTS, **{k: settings[k] for k in DEFAULTS if k in settings}}
    if not isinstance(values['remove_overlap'], bool):
        raise ValueError('remove_overlap must be true or false.')
    value = values['overlap_threshold']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('overlap_threshold must be a finite percentage from 0 to 100.')
    if values['overlap_metric'] not in ('smaller', 'iou'):
        raise ValueError('overlap_metric must be smaller or iou.')
    return values


def suppress_overlaps(results, settings, check_cancel=lambda: None):
    """Return audit records; do not delete proposals or change mask membership.

    Higher predicted quality wins; ties preserve original model order. Compare
    only against retained proposals. A zero threshold still requires shared pixels.
    Packed masks bound temporary memory; bounding rectangles skip disjoint pairs.
    """
    options = validate_overlap(settings)
    records = [None] * len(results)
    if not options['remove_overlap']:
        return records
    packed, areas, boxes = [], [], []
    for result in results:
        check_cancel()
        mask = np.asarray(result['mask'], dtype=bool)
        yy, xx = np.nonzero(mask)
        packed.append(np.packbits(mask, axis=None))
        areas.append(int(mask.sum()))
        boxes.append((int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())) if len(xx) else None)
    def score(index):
        value = results[index].get('score')
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else -math.inf
    kept = []
    for index in sorted(range(len(results)), key=lambda i: (-score(i), i)):
        check_cancel()
        for winner in kept:
            a, b = boxes[index], boxes[winner]
            if a is None or b is None or a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
                continue
            intersection = int(np.bitwise_count(packed[index] & packed[winner]).sum())
            denominator = min(areas[index], areas[winner]) if options['overlap_metric'] == 'smaller' else areas[index] + areas[winner] - intersection
            percent = 100 * intersection / denominator if denominator else 0
            if intersection and percent >= options['overlap_threshold']:
                records[index] = dict(suppressed_by_index=winner + 1, overlap_percent=percent,
                                      metric=options['overlap_metric'], threshold=options['overlap_threshold'])
                break
        if records[index] is None:
            kept.append(index)
    return records
