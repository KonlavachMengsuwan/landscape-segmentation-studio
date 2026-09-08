# Implementation status, 7 September 2026

The image release is implemented and locally verified on the connected SanDisk SSD and this Mac mini M4. The application remains loopback-only; originals, model weights, environment, projects, and exports stay under the dedicated SSD root. No internal development folder was needed: prebuilt wheels and an explicit one-line ExFAT scanner patch resolved the earlier installation blocker.

Completed: real Tiny and Small MPS float32 point/box and automatic inference; bounded worker and cancellation; canonical image storage; brush/eraser and mask review; styles and three theme choices; saved-run comparison; versioned saves and exact-original recovery; portable PNG/binary/CSV/JSON bundles; launch/stop scripts; pinned setup and model integrity manifests; real browser inspection and screenshots. See docs/VALIDATION.md for measured evidence and explicit limits.

Historical initial-release limitation: the original SAM 3 checkpoint returned HTTP 401 GatedRepo. This was resolved for the supplied SAM 3.1 checkpoint in version 0.2, after the owner obtained publisher approval. The experimental image detector passed real local text inference; its scope and limits are in docs/SAM3_ACCESS.md.

The working environment depends on an existing native Python 3.12 base. The SSD venv is not a portable standalone Python distribution. Rebuild instructions preserve the previous environment. No remote repository or deployment was created.

Centroid/palette update: exact centroid numbering, separate export sizing, stable list/CSV indices, six additional palettes and live swatches, clearer comparison captions, and clarification of publisher access versus development permission are complete. Focused tests and numbered/unnumbered real exports passed; see the appended validation section.

## 0.2 checkpoint expansion and controls

Completed: centroid numbers/10 palettes (commit e53b0f9); all 8 SAM2/SAM2.1 official snapshots downloaded with integrity manifests; supplied SAM1/SAM2 copies identified as identical original ViT-H and preserved; strict local SAM1 conversion and real MPS inference; strict SAM3.1 detector conversion plus documented experimental 3-level bridge and real local text inference; UI model selection/text history/threshold controls; reversible exact-mask overlap filtering with percentage/metric and immutable audit; Mac/Windows setup and launcher code with local CPU/CUDA/MPS selection. No other machine accessed.

Verified: 125 Python tests +41 subtests; build passed; nine adapters × five prompt/generation modes on MPS; Tiny CPU inference; SAM3.1 768×512 and full-resolution browser text jobs; 14 automatic masks with 6 hidden, originals identical to unfiltered baseline; main user project's three runs unchanged. Windows wheels available for all 49 pins; native Windows/Intel/NVIDIA validation is explicitly pending.

Final verification: refreshed browser controls and sample screenshots; all four sample runs and their text/overlap provenance survived a real portable export roundtrip; SAM 3.1 passed cached-image and empty-result tests with network connections blocked; 14 frontend tests passed; the Mac launcher started from a clean terminal environment. Separate Mac/Windows setup archives are assembled by scripts/build_release.py with a checksum manifest. Model files, credentials, research projects, runtime and machine-specific test evidence stay outside those archives. Native Windows testing remains pending.

## 0.2.1 educational GitHub preparation

Detailed recipient README, START_HERE, seven embedded app screenshots, official license review/copies, Apache-2.0 original-app license and third-party notices are prepared. Python 3.13/3.14 wheel availability was reviewed without claiming runtime support; current setup stays 3.12. Setup now explains missing Python/interface prerequisites. Public packaging is committed-source-only and separates a fresh GitHub source ZIP from Mac/Windows prebuilt-interface ZIPs, with no weights, user data, environments or history. Full checks: 137 Python tests plus 41 subtests, 14 frontend tests, production build. No remote publication was performed.
