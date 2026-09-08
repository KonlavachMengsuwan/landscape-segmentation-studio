# Future thermal analysis

This release performs image segmentation only. It does not infer temperatures from photograph colors, false-color thermal screenshots, labels, or SAM prompts.

A future module can link a source image identity and canonical pixel grid to a separately selected radiometric matrix. It must record units, missing-data markers, calibration, source hash, and the explicit transformation between sensor and canonical coordinates. Equal dimensions do not establish alignment; rescaling does not resolve camera parallax.

Before applying a mask to temperatures, the user must review alignment using paired landmarks and an overlay. Report valid-pixel count, median, interquartile range, missing-data fraction, and units for the verified overlap. Keep transformations and analysis results separate from the immutable source and original model mask. Pixel counts alone are not surface areas in square metres.

No doctoral datasets or manuscript analyses were searched, imported, modified, or rerun for this application. Synthetic fixtures used for software tests are labeled and are not radiometric evidence.
