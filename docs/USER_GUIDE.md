# Landscape Segmentation Studio user guide

The application works with still images and runs downloaded models on this Mac.
Use the launch and shutdown commands in [MAC_SETUP.md](MAC_SETUP.md). Normal
image inference uses local model files and local browser assets. The supplied
SAM 3.1 checkpoint now supports experimentally verified local image text
segmentation; see [SAM3_ACCESS.md](SAM3_ACCESS.md) for its compatibility limits.
Windows setup is documented separately in [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

## Start an image project

1. Click **Open your first image**, **Import images**, or drop selected images
   onto the application. Multiple selection is supported, with up to 16 images
   per project. **Explore the public sample** opens the attributed example.
2. Use the project name in the top bar to open, create, rename, or restore a
   project. Switching projects first saves the current project's pending changes.
3. Select a thumbnail to make that image active. The image caption and zoom
   indicator describe the canonical image, not a reduced inference input.

JPEG, PNG, and ordinary single-page 8-bit RGB/grayscale TIFF are supported.
Scientific, high-bit-depth, radiometric, and multipage TIFF data are rejected.
The import limit is 24 megapixels and 100 MiB per file. Automatic segmentation
has a separate, more conservative 8-megapixel input limit.

The application copies and hashes the exact original bytes. EXIF orientation
is applied to a separate full-resolution canonical PNG; mirrored orientations
also retain their coordinate mapping. Grayscale becomes equal RGB channels.
Transparent pixels are composited on white for the canonical image and
inference. Original alpha, metadata, profile, and source bytes are retained.
This RGB decoding is not scientific calibration or radiometric conversion.

If a copied image asset is missing, choose **Relink exact original…** under the
image browser, or the relink action shown on the failed image. Select the exact
original file. Its SHA-256 must match the stored identity; a same-named,
different file is rejected. See [IMAGE_RECOVERY.md](IMAGE_RECOVERY.md).

## Make a real segmentation run

In **Segment**, select a verified **SAM 2.1 Tiny** or **SAM 2.1 Small** model.
The setup dialog shows installation and verification state separately. Model
selection loads one model at a time when a job starts. **Unload model from
memory** in Model setup releases it through the worker after active work ends.

For **Point & box**:

1. Choose **Include** and click inside the desired object. Choose **Exclude**
   for negative points, or **Box** and drag a rectangle around a region.
2. Drag an existing point to move it; right-click it to remove it. Drag a box
   edge to move the box, or either marked corner to resize it. **Clear prompts**
   removes current points and box without altering an existing run.
3. **Exact source coordinates** provides numeric point entry and editing, as
   well as the box's x minimum, y minimum, x maximum, and y maximum.
4. Click **Run segmentation**. This creates a new run with its own original
   masks, model identity, settings, prompts, and measured timings.

Coordinates refer to the canonical full-resolution image: x increases right
and y increases down from the upper-left pixel. Pan, zoom, Retina scaling,
and model resizing do not change this coordinate grid.

For **Automatic**, choose a conservative preset or adjust **Inference controls**:

| Preset | Points per side | Points per batch | Quality cutoff | Stability cutoff | Crop layers | Box NMS cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast preview | 8 | 4 | 0.80 | 0.90 | 0 | 0.70 |
| Balanced | 16 | 4 | 0.88 | 0.95 | 0 | 0.70 |
| Detailed | 24 | 4 | 0.88 | 0.95 | 0 | 0.70 |

The grid has the square of the points-per-side value: 8 means 64 prompts;
24 means 576, or nine times the preview grid. Dense settings show a warning
and ask before starting the larger run. These presets are starting points,
not universal quality recommendations. Custom inference presets are saved
with the project. Reset restores Fast preview values.

Threshold changes require an explicit new run. Higher box NMS thresholds
retain more overlapping boxes. Predicted quality and stability are diagnostics,
not measured accuracy. CPU postprocessing is explicit and included in the
end-to-end timing. Crop layers and minimum-region/hole/sprinkle cleanup remain
disabled with their implementation limitations stated in the controls.

The status reports queued, running, completed, failed, or cancelled; there is
no estimated progress percentage. **Cancel run** asks the worker to stop at a
safe boundary. A current model operation may have to complete first, and its
cancelled output is discarded. Changing image, prompts, model, or inference
settings prevents an older result from replacing the newer context. A completed
result kept separate appears in **Runs** with an explicit apply action enabled
only for its matching project and image. Failed jobs never create dummy masks.

## Inspect and correct masks

The **Masks** tab lists stable instance IDs, labels, pixel counts, visibility,
review status, and available model scores. Search accepts a label, ID, or status.
Click a row or the canvas to select a mask. Shift-click or the checkboxes select
several masks. Alt/Option-click cycles through overlapping visible masks at that
pixel. Hover inspection reports the topmost visible instance.

- **Accept** and **Reject** record review state. They do not erase membership.
- Edit **Surface label** for the selected mask. Labels remain human reviewable
  proposals. There is no automatic land-cover correctness guarantee.
- Eye controls show or hide masks. **Solo** shows selected masks; **Show all**
  restores visibility. These changes do not alter pixel counts.
- **Duplicate** creates a separate instance retaining its source membership and
  derivation reference. **Delete** hides a soft-deleted instance from the editable
  list while retaining it in audit data; Undo restores it.
- **Merge** creates a new manual union of selected instances. The input instances
  remain intact. Its manual provenance names its source IDs; it is not another
  prediction from the model.
- Select exactly one mask, then choose **Brush correction** or **Erase correction**.
  Set the brush diameter in canonical source pixels and paint. One completed
  stroke is one undo operation. Edits populate a separate derivative mask;
  **Restore original model mask** discards that derivative in the editable view.

The canvas draws later list instances over earlier ones. This display order
does not assign pixels exclusively to one mask. Holes, disconnected components,
and overlaps retain exact membership. Automatic masks do not necessarily form
a complete or nonoverlapping surface map. Area is measured in pixels, never in
square meters without a separate valid calibration.

Undo/redo keeps up to 30 recent in-memory operations. Saved JSON revisions on
disk provide a separate persistence history; the in-memory undo stack is not
restored after a browser restart.

## Explore visualization styles

The quick menu over the canvas and the **Style** tab switch among:

| View | Behavior |
| --- | --- |
| Original image | Photograph, with optional prompt markers. |
| Outlines only | Mask boundaries without fill. |
| Filled masks | Colored fill without normal mask outlines. |
| Fill + outlines | Independent fill and outline controls. |
| Mask-only canvas | Masks over explicit white, black, or transparent/checkerboard background. |
| Selected-object spotlight | Selected masks' exact union stays visible while its surroundings are dimmed. |
| Reviewed categories | Accepted masks with the same exact surface label share a palette color. |
| Comparison divider | Original or another saved run on the left; current result on the right. |

**Subtle overlay**, **Bold boundaries**, **Publication on white**, and
**Monochrome inspection** are editable collections of the visible settings.
**Save current style** stores custom presets in the project. Applying a style,
changing a palette, or switching theme never reruns the model.

Ten palettes include Landscape, Colorblind-friendly, Botanical, Coastal, Earth & clay, Soft pastels, Jewel tones, Slate & amber, Monochrome, and Deterministic distinct colors. The color strip previews the selected palette. The decorative palettes are not scientific scalar colormaps or a guarantee of color-vision accessibility. Shuffle updates the saved seed. IDs and labels
do not change. Monochrome exposes an explicit color control. A per-mask fill
color overrides that mask's palette assignment until cleared.

Enable **Index numbers at centroids** in Style to center each number on the mean of that segment's member pixel centers. Manual corrections update the centroid; holes and disconnected components remain part of the exact calculation. A true centroid can lie in a hole or outside a curved region. Empty masks have no centroid badge.

Numbers are one-based positions in the complete saved run. Hiding, filtering, accepting, or deleting a mask does not renumber the remaining entries. Duplicates and merged masks receive new numbers at the end. The mask list shows the same numbers, and search for `#12` to select exactly number 12. Numbers are local to each run; stable UUIDs remain the audit identity. Both comparison sides use their own run numbering, with run numbers in their view titles.

**Index number size** uses screen pixels and stays readable while zooming. The export dialog has a separate **Include centroid numbers** switch and **Export number size** in output pixels, with a live preview. Number badges are included in styled PNG and transparent overlay when enabled; individual binary masks remain unchanged. Centroid badges do not move to avoid collisions, so overlapping centroids may overlap visually. The per-mask label-visibility override also controls its centroid badge.

Use global opacity, boundary width/color, and optional high-contrast double
stroke. The advanced sections expose mask IDs, labels, available scores,
bounding boxes, prompt visibility, and label size. Per-mask controls override
fill color/opacity, outline color/opacity/width, label visibility, and visibility.
Brightness, saturation, and background dimming are display-only adjustments.

Interactive boundary width is in **screen pixels**, independent of zoom.
Export has a separate **output-pixel** width and preview. Contours follow exact
pixel edges; cosmetic contour smoothing is not applied.

Light, Dark, and System themes are in the top bar and persist in the browser.
They change interface colors without changing mask colors, source data, or
export backgrounds. Sidebars can be collapsed; a collapsed image browser can
be restored using the folder control above the canvas. The panel resize grips
allow width adjustments within their desktop limits.

## Compare runs

**Runs** shows results for the selected image. **Open run** restores that run's
prompt/settings context without rewriting the run's immutable provenance.
The details reveal checkpoint, revision, implementation, device, dtype,
settings, prompts, and measured timing fields.

Choose **Compare** on a different run, or choose Comparison divider in Style.
Select the original photograph or a saved run as the comparison source.
Both sides share exactly one pan/zoom transformation. Drag the divider, or
focus its range control and use the keyboard. Comparison does not report an
accuracy ranking in the absence of reference annotations.

## Save, export, and reopen

**Save** or Command/Ctrl-S commits a new version on the verified SSD. The top
bar distinguishes unsaved changes from a successful local save. Save captures
the current image, active run, prompts, settings, style, selection, and viewport
camera. A browser reload needs the project to be reopened through its name menu.

A failed save leaves current edits in memory and reports the error. Closing or
reloading a tab with unsaved state triggers the browser's unsaved-work warning.
If the SSD or server disappears, the application shows a recoverable banner
and disables new saves, exports, imports, and model runs. Keep the window open,
reconnect the same SSD, and retry after the status recheck. Never disconnect
the drive during an active write. See [STORAGE_CONTRACT.md](STORAGE_CONTRACT.md)
for version integrity and ExFAT durability limits.

**Export bundle** opens a canonical-resolution preview. Choose the explicit export background (photograph, white, black, or transparent mask canvas), set export outline
width in output pixels, then **Save export bundle to SSD**. Export first saves
the project. A download link appears only after the saved bundle is returned.
The bundle contains a styled PNG, RGBA transparent overlay, exact individual
binary masks, original masks, CSV, run audit JSON, a manifest, and portable
project/source assets. It does not contain model weights or credentials.

The export renderer excludes canvas interaction highlights and prompt markers.
Comparison is a workspace view: its export explicitly renders the active result
over the whole source image, not a captured divider. Transparent overlay excludes
the photograph. In Original image mode this overlay is transparent because no
masks are being drawn. Mask-only canvas background is explicit and independent
of the interface theme.

Visible masks are included in individual binary PNGs and CSV. A visible rejected
mask remains included; hide it to omit those outputs. Hidden/deleted instances
remain in the project and audit JSON. No overlap-flattening label raster is used.
See [EXPORT_FORMATS.md](EXPORT_FORMATS.md) for precise output and import semantics.

Use the project menu's **Restore bundle** to verify and restore an exported ZIP
as a new project. Existing projects remain unchanged. Original source identity,
lossless memberships, labels, styles, and run provenance are preserved; new
project/instance identifiers retain references to their source bundle identities.

## Keyboard and canvas controls

| Action | Shortcut |
| --- | --- |
| Select | V |
| Pan | H, or hold Space and drag |
| Positive / negative point | P / N |
| Box prompt | B |
| Brush / eraser | E / R |
| Fit to window / 100% view | F / 1 |
| Zoom around cursor | Scroll |
| Select multiple / cycle overlaps | Shift-click / Alt-click |
| Remove prompt point | Right-click marker |
| Clear selection | Escape |
| Undo / redo | Command/Ctrl-Z / Shift-Command/Ctrl-Z |
| Save | Command/Ctrl-S |

Canvas shortcuts pause while typing in a text, numeric, or select control.
**Shortcuts** below the canvas opens this reference. All major actions have
labeled buttons, and prompts support numeric entry without precise pointer use.

No thermal measurements are inferred from RGB colors, image dimensions, or
text prompt phrases. A later thermal module requires an explicitly paired,
verified temperature grid; see [THERMAL_EXTENSION.md](THERMAL_EXTENSION.md).


## Additional models, text, and overlap filtering (0.2)

The model selector includes the four sizes of SAM 2 and SAM 2.1, original SAM ViT-H, and an experimental SAM3.1 image detector when their local files and device checks are present. The file `sam_vit_h_4b8939.pth` belongs to original SAM, even if stored in a folder named SAM2. All inference remains local; only one model runs at a time.

Select **SAM 3.1 image detector · experimental** to reveal Text prompts. Enter a phrase such as `mountain`, `tree crowns`, or `road`; set detection confidence and mask probability, each 0–1; then Run text segmentation. Saved runs retain the text; Previous text prompts recalls saved phrases. These are model proposals, not verified surface classifications. This adapter has no point/box or video workflow.

Before automatic processing, expand **Advanced inference controls** and enable **Remove overlapping masks**. Choose the overlap measure and threshold (0–100%). Smaller-mask coverage catches nested masks; IoU compares the shared pixels with the union. For example, a small mask fully inside a larger mask has 100% smaller-mask overlap but may have low IoU. At 80%, the lower-scoring proposal is hidden once at least 80% overlaps a retained proposal. Equal scores preserve original proposal order. At zero, some shared pixels are still required. At 100%, complete overlap is required for the chosen measure.

Filtering changes visibility, never pixel membership. The Masks panel shows visible/filtered counts and recoverable filtered entries. Use an entry's Show button or Show all to recover it; undo/redo works. The export contains visible mask PNGs and all proposal metadata, including the suppression winner and measured percentage. Existing saved runs do not change when controls are adjusted. Bounding-box NMS is a separate earlier model postprocessing step: set it to 1 to retain all box candidates, subject to result limits.

| Parameter | Min | Max | Meaning |
| --- | ---: | ---: | --- |
| `points_per_side` | 2 | 32 | Square prompt-grid width; total prompts is its square |
| `points_per_batch` | 1 | 16 | Simultaneous prompt groups; effective batch may be lowered to cap memory |
| `pred_iou_thresh` | 0 | 1 | Minimum model-predicted mask quality; higher rejects more |
| `stability_score_thresh` | 0 | 1 | Minimum agreement of pixels under shifted mask cutoffs |
| `box_nms_thresh` | 0 | 1 | Bounding-box IoU suppression threshold; higher retains more |
| `crop_n_layers` | 0 | 0 | Disabled: no verified crop-pyramid implementation |
| `overlap_threshold` | 0% | 100% | Exact-pixel overlap suppression threshold, optional |
| `detection_threshold` (SAM3.1) | 0 | 1 | Combined detection/presence confidence |
| `mask_threshold` (SAM3.1) | 0 | 1 | Pixel probability needed for mask membership |

Crop cleanup, minimum-region cleanup, holes, and sprinkles remain unavailable and are labeled as such. Settings require a new run; visualization colors, opacity, and centroid labels never rerun a model.
