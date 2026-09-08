# Export formats

Exports are written inside `exports/<project-id>/` on the verified SSD. They
never replace imported originals. The export bundle selects one run and its
source image; every saved run for that same image remains available in its
portable project JSON. No weights, virtual environments, credentials, or
unrelated files are included.

## Bundle contents

| Path | Contents |
| --- | --- |
| `styled.png` | Optional full-resolution styled result supplied by the export renderer, using the explicitly selected view and background. |
| `overlay.png` | Optional RGBA PNG supplied by the renderer, with masks, boundaries, and optional labels on a transparent background. |
| `masks/<mask-id>.png` | Individual visible masks, using the edited derivative when one exists. |
| `masks-original/<mask-id>.png` | The corresponding immutable model mask before manual corrections. |
| `metadata.csv` | Visible masks' labels, review state, measured pixel counts, original pixel counts, available model scores, run identity, and edit state. |
| `project.json` | Portable project state, the selected image identity, and all runs for that image, including hidden masks and exact original/edited RLE. |
| `audit/run.json` | Complete selected run, including hidden masks, source identity, prompts, settings, scores, and original/edited RLE. |
| `images/<image-id>/original.*` | Exact imported source bytes, verified against their SHA-256 identity. |
| `images/<image-id>/canonical.png` | Full-resolution orientation-corrected RGB source used by prompts and masks. |
| `manifest.json` | Bundle version, source hash, canonical dimensions, selected mask IDs, overlap policy, and SHA-256 hashes/byte counts of every other file. |

All raster dimensions equal the canonical image grid. Binary masks are 8-bit
grayscale PNGs containing only **0 outside and 255 inside**. Holes and
disconnected components remain exact. Each instance has its own file, so
overlaps are never flattened. The final visible mask in the run's layer order
is topmost for display only. No instance-label raster or overlap-based
membership reassignment is performed.

Visibility and deletion state control which individual masks and CSV rows are exported. A
visible mask marked rejected is still included because it is part of the
chosen visible view. Hide it before export to omit its binary/CSV outputs.
Hidden, deleted, and rejected masks always remain in machine-readable audit data.

Pixel counts are derived from exact stored mask membership, independent of
display opacity, outlines, theme, layer order, or stale cached area fields.
They are pixel counts, not square meters. Scores retain their model-specific
meaning and are not scientific accuracy. Missing scores produce blank CSV
cells. Formula-like text beginning with `=`, `+`, `-`, or `@` after leading
whitespace receives an apostrophe in CSV to prevent spreadsheet execution.
Exact labels remain unchanged in the JSON records. The `segment_index` CSV column matches the centroid number and mask-list number: the one-based position in the complete run, including hidden/deleted entries. UUIDs remain the stable audit identifiers. Centroid numbers and their separately chosen output-pixel size are rendered into styled PNG and overlay when enabled, without modifying binary membership.

Styled and overlay PNG bytes are rendered by the frontend at canonical
resolution. The backend verifies their dimensions and the overlay's RGBA
channel. The renderer is responsible for matching the selected style,
background, label placement, and output-pixel outline width. An overlay may
have fully opaque regions; its alpha channel remains explicit. Interface
theme does not select an export background.

## Reopening and bounded import

`import_bundle(storage, bytes, name=None)` imports into a **new project** and
never overwrites an existing one. It validates ZIP paths, entry count,
declared and actual expanded byte counts, CRC integrity, manifest hashes,
source hashes, source-to-canonical pixel mapping, each RLE's dimensions, and
binary-mask agreement with the audit data before beginning project writes.
ZIP members are read in memory; raw archive extraction is never used.
Symlinks, duplicate paths including case collisions, traversal, absolute or
Windows-style paths, encrypted entries, and unsupported compression are
rejected. Limits are 200 MiB compressed or uncompressed and 1000 entries.

Original source bytes pass through the normal immutable importer. The new
project receives new project/image/run/mask IDs; provenance records retain the
old IDs and bundle hash. Model metadata, prompts, labels, settings, styles,
original RLE, and edited RLE remain intact. New source paths are relative to
the newly created project. Selection fields supported by the current schema
are remapped to their new IDs.

If an SSD write fails after validation, earlier projects remain unchanged.
A partially committed new project may appear as `Import in progress` and can
be inspected or reimported later. A failed import is never reported as saved.
Stored render PNGs are audit outputs; editing a reopened project generates
new views from its canonical source and masks.

For the RLE encoding, pixel transforms, save behavior, and format limitations,
see [STORAGE_CONTRACT.md](STORAGE_CONTRACT.md).

## Backend interface

`build_export(storage, project, run_id, styled_png=None, overlay_png=None)`
returns the saved ZIP path. `import_bundle(storage, bundle_bytes, name=None)`
returns the saved document of the new project. Call both off the interface
thread; they perform image encoding, validation, and guarded disk writes.

## Backend verification

On 7 September 2026, the project-local ARM64 Python environment passed
`app/tests/test_exports.py`: 12 tests and 15 subtests. Tests cover exact
original/effective binary pixels, holes and overlaps, RGBA alpha preservation,
CSV formula neutralization, hash verification, fresh-project roundtrip with
audit and selection remapping, canonical dimension mismatch, unsafe/colliding
IDs, source/canonical/binary tampering, traversal, ZIP symlinks, duplicate
paths, decompression limits, and malformed archives. Fixtures are explicitly
synthetic and remain under the workspace's `tmp` directory. These tests
verify backend formats; they do not establish visual agreement with the
frontend renderer or publication-quality label placement.


Overlap filtering (0.2) retains all source proposal RLEs and the per-mask `overlap_suppression` audit record: winning segment index, metric, threshold, and measured percentage. Filtered proposals initially have `visible=false`. Styled output and individual mask PNGs follow visibility; project metadata preserves the full run and can recover hidden proposals. Segment numbers therefore can contain gaps. Text runs retain `prompts.text`, the exact local checkpoint hash, experimental adapter name, and effective detection/mask thresholds.
