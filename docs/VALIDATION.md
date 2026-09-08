# Verification report

Verified on 7 September 2026, on the actual Mac mini M4 with 16 GiB unified memory, macOS 26.6.2, and the connected SanDisk Extreme USB SSD. Source, environment, models, generated project data, and evidence are inside the dedicated SSD root. No research dataset, other computer, global network setting, or unrelated file was used.

## Real model inference

- MPS built/available checks and synchronized tensor computation passed.
- Official SAM 2.1 Tiny and Small, Transformers 5.16.1, PyTorch 2.12.1, MPS float32 SDPA: each passed combined point/box, point-only, box-only, positive/negative prompts, and Fast preview automatic inference on an attributed 768 × 512 Yosemite derivative. The final automatic runs produced 5 and 9 masks in 3.902 and 3.975 seconds respectively.
- Small automatic inference on the original 3072 × 2048 canonical grid produced 14 proposals in 14.307 seconds. The output saved and reopened exactly. Requested batch four was explicitly reduced to one to bound memory. Candidate membership is losslessly bit-packed before NMS.
- A real Tiny point run through the browser, at canonical coordinate (1536, 800), returned a 429,192-pixel mask on the 3072 × 2048 image. The browser displayed the real worker result, model score, and saved run.
- One model and one image embedding are retained. Mask interpolation/filtering/NMS execute explicitly on CPU; model forward runs on MPS. No implicit operation fallback or memory-safeguard override was used.
- A separate actual inference run passed with offline loading enforced and a per-process guard denying external socket connections. No external connection attempts were observed. This did not change Mac networking.
- SAM 3 configuration/weights returned HTTP 401 GatedRepo. Imports and the bounded candidate test's failure handling were checked; real text inference, empty text results, and CPU fallback remain unverified. No fabricated result stands in for that missing test.

Measurements are individual execution checks, not speed promises or accuracy comparisons. See [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) and [SAM3_ACCESS.md](SAM3_ACCESS.md).

## Automated correctness tests

The final Python suite passed **113 tests and 41 subtests in 13.09 seconds**. The frontend Node suite passed **9 tests**. The TypeScript check and Vite production build passed. Two upstream deprecation warnings came from the test client (httpx and AnyIO aliases), without application failures.

Covered risks include all eight EXIF orientations in deliberately asymmetric fixtures; exact original bytes and canonical hashes; grayscale and transparency; invalid/scientific/oversized inputs; row-major RLE with holes, disconnected components and overlaps; original-versus-edited membership; manual union provenance; 0/255 binary exports; CSV formula neutralization; bundle tampering, decompression bounds and identity remapping; stale revisions; failed atomic commits preserving the previous valid revision; missing assets and exact-original relinking; path escapes and simulated missing/wrong volumes; queue/cancel/stale-output behavior; unavailable models/MPS and resource errors; loopback request validation; bounded request bodies; multipart temporary storage; and launcher process identity.

The frontend's actual coordinate conversion is tested at several pans and zooms, two viewport widths, and 1×/2×/3× backing-store scales using an asymmetric source point. Mask membership and style independence are tested separately. Synthetic fixtures are clearly tests, not model demonstrations.

From the dedicated root, with the documented runtime available:

```sh
PYTHONPATH=app PYTHONPYCACHEPREFIX="$PWD/runtime/pycache" TMPDIR="$PWD/tmp" \
  runtime/venv/bin/python -m pytest app/tests \
  --basetemp=tmp/pytest-validation -o cache_dir=tmp/pytest-cache -q
cd app/frontend
node --test tests/masks.test.mjs
npm run build
```

## Browser and export inspection

The implemented application was inspected in the Codex in-app browser at 1280 × 720, 1100 × 720, and 1440 × 900. Temporary viewport overrides were reset. Light and Dark remained readable, and the theme preference survived reload. The smaller layout retains visible save state. Numeric prompt entry, keyboard range controls, fit/zoom, toolbar selection, and shared-pan comparison were exercised.

The browser's real point result was manually labeled and accepted. An 80-source-pixel brush stroke changed only its derivative from 429,192 to 443,755 pixels. Undo returned 429,192; redo returned 443,755. A save/reload/reopen cycle restored the label, acceptance, original RLE, and derivative. Changing palette, fill opacity, boundary width, brightness, and comparison state left all run objects, mask arrays, pixel counts, and source metadata exactly unchanged, verified against saved API documents.

Two saved runs were compared on the same photograph. Pan, zoom, resizing, and the before/after control retained the same source alignment. The comparison is an inspection view; the export dialog explicitly exports the active result across the full image.

A bundle created through the browser export dialog was inspected independently with Pillow and NumPy. Styled PNG and overlay were exactly 3072 × 2048. The selected white background had RGBA (255,255,255,255) at an outside pixel; the transparent overlay alpha ranged from 0 to 244. Individual binary files contained only 0 and 255 and matched the edited RLE exactly. Separate original binary files matched immutable model membership. The source SHA-256 was unchanged. Both saved runs remained in portable JSON. The bundle was then restored through the browser as a separate project. A real Small automatic job was started and cancelled through the interface; it reached Cancelled, explicitly reported that no result was applied or saved, and retained the two earlier runs.

The real browser captured no warning/error console entries during this workflow. Normal page assets and API calls use the loopback backend. Offline model execution is established by the separate guarded inference test; no OS-level packet capture was performed.

Real application screenshots, using the attributed public sample:

- [Light workspace](screenshots/light.png)
- [Dark workspace](screenshots/dark.png)
- [Saved-run comparison](screenshots/comparison.png)
- [Compact desktop](screenshots/compact-desktop.png)
- [Export preview](screenshots/export-preview.png)

## Persistence, launch, and evidence locations

The documented launcher was tested from an empty environment with only the system PATH, without shell activation. Startup confirmed the matching backend instance at 127.0.0.1:8765; shutdown stopped only its own identified process. Saved projects remained available after backend restart. The exact Start/Stop command files also passed an immediate restart from an empty environment. The final check found and fixed a TIME_WAIT port-probe false positive by matching normal SO_REUSEADDR restart semantics. Local socket regression tests cover immediate reuse and a real live listener, which is rejected and left responsive. Process-identity tests also reject unrelated processes.

Evidence outside Git, relative to the dedicated root:

| File | Evidence |
| --- | --- |
| `setup-notes/volume.json` | Verified local SSD identity |
| `setup-notes/mps-tensor.json` | Actual MPS probe |
| `setup-notes/compatibility.json` | Tiny/Small synchronized timings and memberships |
| `setup-notes/independent-prompts.json` | Separate point, box, positive/negative runs |
| `setup-notes/full-resolution-api-smoke.json` | Full-resolution automatic and save/reopen |
| `setup-notes/offline-smoke.json` | Real inference with external connections blocked |
| `setup-notes/browser-style-independence.json` | Saved masks and source unchanged by styling |
| `setup-notes/browser-export-verification.json` | Actual frontend-rendered bundle membership/alpha checks |
| `setup-notes/browser-roundtrip.json` | Restored original/edited masks, labels, and retained runs after cancellation |
| `setup-notes/release-checks.json` | Final suite and immediate launcher restart |
| `projects/` and `exports/` | Saved real sample, restored project, and portable bundle |

## Explicit verification limits

SAM 3 remains blocked. Balanced and Detailed settings are validated but have not been exhaustively benchmarked. The browser exposed a 1× backing scale; 2×/3× coordinate behavior was tested with production math, not a physical Retina browser capture. Visual comparison and export inspection used the public landscape; all EXIF rotations and mirrored orientations were covered by asymmetric automated fixtures rather than eight browser sessions.

No physical unplug during writing, power-loss simulation, exhaustive accessibility audit, or scientific accuracy evaluation was performed. Missing/wrong-drive and failed-save behavior use controlled fixtures. ExFAT cannot create native symlinks, so the symlink-component rejection branch uses a controlled injected symlink report; native-symlink filesystem integration remains untested. The 24-megapixel maximum is a guarded limit, not a guarantee that every image and mask collection will fit memory. Crop pyramids and hole/sprinkle cleanup remain disabled. Scientific/radiometric TIFF, video, training, and thermal measurements are outside this release.


## Centroid and palette update, 7 September 2026

Added exact pixel-centroid index badges, separate screen/output number sizes, six decorative palettes (ten choices total), palette swatches, stable mask-list numbering and number search, CSV `segment_index`, and numbered saved-run comparison captions. Existing saved styles load with centroid badges off until enabled. Numbers use complete run order, including hidden/deleted entries; they do not replace UUIDs. Exact centroid badges may overlap when instance centroids nearly coincide. They are not shifted away from the measured centroid.

The updated frontend passed **14 Node tests**, including asymmetric/holey/disconnected centroid calculations, empty masks, derivative centroid updates without original mutation, stable numbering, exact `#number` search, and all ten deterministic palettes. The export suite passed **13 tests and 15 subtests**. TypeScript and the production build passed. Model inference code was unchanged, so earlier real inference evidence remains applicable; this update did not rerun the models.

Real browser checks exercised the centroid toggle, keyboard number sizing, Coastal and Jewel palettes, Light/System and Dark presentation, and separate export controls at a 1440 × 900 test viewport. The temporary viewport override was reset. Numbered and unnumbered 3072 × 2048 export bundles were generated through the interface and compared independently. Every original/binary mask and source byte was identical, all run data was identical, CSV numbers matched complete run positions, and changed overlay pixels were present at all 14 computed centroid neighborhoods. Exactly 39,006 rendered overlay pixels differed; those differences were annotations, not membership. See `setup-notes/centroid-export-verification.json` outside Git.

The user's previously unsaved third sample run was saved before the update. Visual/export tests used the separate development sample project. [Centroid numbers](screenshots/centroid-numbers.png) and [Jewel palette in Dark mode](screenshots/centroid-jewel-dark.png) are real screenshots. SAM 3's public access gate was rechecked; the owner's signed-in access status was not inspected. Development permission is granted, but publisher account approval and a locally available checkpoint are still needed before real compatibility testing.

## 0.2 expansion validation (7–8 September 2026)

- 125 Python tests and 41 subtests passed, covering storage, provenance, API boundaries, exports, lifecycle, resource validation, reversible overlap filtering, and portable helper paths. This includes four platform-helper tests, not native Windows certification.
- All eight official SAM2/SAM2.1 variants and original SAM ViT-H passed combined point/box, point-only, box-only, positive/negative, and automatic MPS inference. Tiny additionally passed CPU combined point/box and automatic inference on this M4.
- SAM3.1 detector strictly converted all 1,464 required image tensors and completed real local text inference on both 768×512 and 3072×2048 sample grids. The full-resolution result was created and saved through the actual browser UI. The adapter remains experimental; no upstream CUDA numerical-parity claim is made.
- The browser ran Small automatic segmentation at 3072×2048 with 80% smaller-mask overlap suppression: 14 proposals, 8 visible, 6 filtered. All original RLEs exactly matched the existing unfiltered run. Show recovered a filtered proposal; Undo restored filtering. Saved source/run data survived reopening. The user's original three-run project was compared with the prior snapshot and was unchanged.
- Windows code uses local volume serial checks, junction rejection, UTF-8 settings, Windows process identity inspection, file locking, and separate launcher/setup scripts. All 49 pinned packages have candidate compatible Windows wheels; native execution, driver support, and Windows installers still need target-machine verification. No other machine was accessed.
- Final frontend build and 14 Node tests passed. The Mac launcher was stopped and started from a clean environment with a minimal PATH. The refreshed interface showed all ten installed models, correct text history, eight visible/six filtered masks, and stable centroid numbers.
- The actual filtered export restored into a new project with four runs, text settings, all original RLEs, and overlap audit records intact. Styled/overlay images retained 3072×2048 dimensions. SAM3.1 repeated text inference reused its image embedding and produced a valid empty result at confidence 1.0 while all network socket connections were blocked; no connection was attempted.

## 0.2.1 publication preparation, 8 September 2026

The complete Python suite passed **137 tests plus 41 subtests** (including 12 publication-boundary cases); **14 frontend tests** and the production interface build passed. Model arithmetic and inference settings were not changed. Earlier real model evidence remains the basis for M4 model compatibility, not this documentation/build check. Two existing Starlette deprecation warnings remain in tests.

The README was rewritten for fresh recipients with Windows/Mac ZIP setup, source-build differences, model files, Python versions, tutorial screenshots, legal conditions and upload instructions. Added a real first-launch screenshot from the running app; reviewed public-sample screenshots remain in the guide. Browser inspection confirmed recipient-neutral SAM3.1 setup language. New source/setup guards reject missing prebuilt frontend before dependency downloads and report missing Python clearly. New local conversions retain the model's license file.

Official model licenses and complete notices for bundled frontend runtime code are included. Python3.13/3.14 checks cover package metadata only; neither interpreter was installed or executed. Source publication packaging includes committed application files without Git history; model weights, user data, environments, credentials and local reports are excluded. Platform ZIPs include the built interface. The archive builder enforces these boundaries and checks ZIP integrity and SHA256; generated manifests identify each source revision. Windows/Intel/NVIDIA native testing and definitive legal interpretation of the SAM3 conversion clause remain unresolved as documented. No GitHub repository was created and nothing was pushed or published.

Publication preflight also verified that a fresh source-only checkout reports the missing frontend before creating an environment or drive record. All 83 local documentation/image links resolved. The final browser model-access link points to the SAM3.1 publisher; no browser warnings/errors were recorded in that check.
