# Storage and pixel-grid contract

`backend.storage.Storage(workspace_root)` reads the locally recorded mount path
and volume UUID from `setup-notes/volume.json`. This machine-specific setup file
stays outside the application Git repository. Before writes, the guard checks
that the existing mount directory and macOS `diskutil` metadata still identify
the same volume. It never recreates a missing workspace or mount directory.
Symlink components, absolute paths, and parent traversal are rejected. The
application should call `storage.check()` before starting inference too.

## Python interface

- `create_project(name) -> document`
- `list_projects() -> summaries`
- `load_project(project_id) -> document`
- `save_project(project_id, document, expected_revision=None) -> saved_document`
- `import_image(project_id, filename, bytes) -> image_record`. This operation
  saves a new project revision. Reload the document before merging a later UI
  save so that the imported image and current revision are retained.
- `image_path(project_id, image_id) -> Path` resolves and hash-checks the
  canonical PNG. `original_path(...)` does the same for immutable source bytes.
- `write_export(project_id, filename, bytes) -> Path` uses the same volume guard
  and atomic replacement. `export_path` only resolves a path; callers must not
  bypass the guarded writer with direct writes.
- `encode_mask(array)`, `decode_mask(rle)`, `binary_mask_png(rle)` preserve exact
  mask membership. `StorageError`, `VolumeUnavailable`, and `RevisionConflict`
  carry recoverable error messages.

The backend is a single writer with an in-process lock. Callers should pass the
revision they last loaded. A stale save raises `RevisionConflict` instead of
replacing more recent work. The launcher must not run multiple backend instances
against the same workspace.

## Document and image records

A project has `schema_version: 1`, `id`, `name`, `revision`, `created_at`,
`updated_at`, `images`, `runs`, and `style`. Additional JSON fields are retained
without reinterpretation. Runs should record their original model masks and
edited derivatives separately, along with prompts, settings, model identity,
device/dtype, implementation versions, timestamps, and review state.

Each image record has `id`, `name`, `width`, `height`, `original_width`,
`original_height`, `sha256` of original bytes, `canonical_sha256`, `format`,
`original_mode`, `original_path`, `canonical_path`, `imported_at`, and
`transforms`. These identity fields cannot be modified by a project save.
Additional UI fields can be stored alongside them. Paths are relative to the
project directory; no imported source's external absolute path is recorded.
Removing an image from the browser does not delete its immutable disk assets.

The canonical image is full-resolution, orientation-corrected RGB PNG. EXIF
orientations 1 through 8, including mirrored orientations, have explicit
`original_to_canonical` and `canonical_to_original` 3 by 3 affine matrices for
zero-based pixel indices. The x axis points right and the y axis down. Model
adapters must record their model-input transform separately and return masks in
this canonical grid. The viewport transforms this grid without changing it.

Transparent PNG pixels are composited on white for inference and the canonical
display image; `alpha_background` records this choice. Grayscale pixels expand
to equal RGB channels. Pillow's RGB conversion is used without ICC profile
transformation; the original retains its profile. Display color management is
not a radiometric conversion. Original bytes retain all metadata, while the
canonical PNG omits source metadata. Supported input is single-image JPEG/PNG
and ordinary 8-bit RGB/grayscale TIFF. Scientific/high-bit-depth or multiframe
inputs are rejected. The release limit is 24 megapixels and 100 MiB per upload.

## Lossless mask format

```json
{"size": [2, 3], "counts": [1, 2, 2, 1]}
```

`size` is `[height, width]`. `counts` alternates background and foreground run
lengths, starting with background, over pixels flattened in **row-major (C)
order**. Thus the example is `[[0,1,1],[0,0,1]]`. An all-foreground mask begins
with a zero background count. Counts sum to width times height. This is
uncompressed run-length encoding and is not COCO's column-major ordering.
Every instance retains its own RLE, including overlaps, holes, and disconnected
components. Display order never changes membership. Binary PNG exports use
8-bit grayscale 0 outside and 255 inside in the same canonical dimensions.

## Versioning, recovery, and portability

Each save writes a uniquely named immutable JSON under
`projects/<id>/history/`, flushes it, then atomically replaces the small
`current.json` pointer containing a hash. Previous JSON versions remain intact.
If committing the pointer fails, reopening still loads the prior committed
version; a newer orphan history file is never silently adopted. Hash failures
are reported rather than substituting an unrelated image or another run.

Temporary files are confined to the destination directory. File `fsync` and
atomic replacement are used. Directory `fsync` is attempted and unsupported
filesystem responses are tolerated. ExFAT physical unplug/power-loss durability
is not guaranteed. Normal save failure tests use injected errors, never an
actual drive disconnection. Unsaved UI state must remain in memory until a
successful save response. A fully committed but interrupted HTTP response can
require reloading the project's current revision before saving again.

Copy a complete `projects/<id>/` directory to preserve its relative assets and
history. A new installation must explicitly configure the destination SSD's
identity. If an asset is missing or has a changed hash, the current release
offers an error and explicit reimport as a new image entry; it does not perform
automatic filename-based relinking. Recovery of a damaged current pointer is a
manual selection of an earlier intact history version; the application does
not guess which orphan version the user intended to commit.

The schema reserves room for a future explicit thermal pairing record; RGB
colors and equal image dimensions never establish temperatures or alignment.
