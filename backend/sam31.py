"""Experimental image-only SAM3.1 detector layout bridge for Transformers 5.16.1.

Meta's multiplex detector has three FPN scales [4,2,1]; HF Sam3Model always
slices off its final fourth level. Append an alias that that exact slice discards,
so all three genuine features enter the unchanged detector. No synthetic weights,
video model, tracking state, missing-parameter initialization, or remote inference.
"""
def detector_neck_output(module, arguments, output):
    features, positions = output
    if len(features) != 3 or len(positions) != 3:
        raise RuntimeError('SAM3.1 detector bridge expects exactly three feature levels.')
    return (*features, features[-1]), (*positions, positions[-1])
