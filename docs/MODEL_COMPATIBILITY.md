# Measured model compatibility — 0.2

Verified on 7–8 September 2026 using the Mac mini M4 only: 16 GiB unified memory, macOS 26.6.2, ARM64 Python 3.12.14, PyTorch 2.12.1, torchvision 0.27.1, Transformers 5.16.1, NumPy 2.5.3, Pillow 12.3.0. The synchronized MPS tensor probe passed. See requirements.lock for all package pins.

All nine point/box adapters passed combined point-and-box, point-only, box-only, positive/negative, and automatic tests on the public Yosemite sample (768×512 canonical pixels). One model loads at a time. The table shows individual measurements, not accuracy scores or speed guarantees. Automatic timings below reuse the embedding from the preceding interactive test; load and image-encoding costs are separate.

| Model | Combined point/box inference (s) | Automatic inference with cached embedding (s) | Automatic proposals |
| --- | ---: | ---: | ---: |
| SAM 2.1 Tiny | 0.548 | 3.734 | 5 |
| SAM 2.1 Small | 0.357 | 3.766 | 9 |
| SAM 2.1 Base-Plus | 1.259 | 3.724 | 13 |
| SAM 2.1 Large | 1.247 | 3.734 | 7 |
| SAM 2 Tiny | 0.318 | 3.778 | 5 |
| SAM 2 Small | 0.358 | 3.821 | 7 |
| SAM 2 Base-Plus | 0.616 | 3.763 | 15 |
| SAM 2 Large | 1.221 | 3.718 | 8 |
| SAM 1 ViT-H (Huge) | 3.885 | 3.999 | 30 |

All these image models use float32 and SDPA on MPS. Resizing, canonical mask interpolation, stability/quality filtering, and bounding-box NMS use explicit CPU postprocessing. The model processor uses 1024×1024 input; source dimensions and full-resolution masks are retained. Crop pyramids and hole/sprinkle cleanup are disabled. The shared official SAM2 configuration identifies `sam2_video`, but only `Sam2Model` is instantiated; no video inference runs.

## Supplied original SAM checkpoint

Both supplied files, `SAM1/sam_vit_h_4b8939.pth` and `SAM2/sam_vit_h_4b8939.pth`, are identical **original SAM ViT-H**, not SAM2. Each is 2,564,550,879 bytes, SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`. Their ViT-H tensor shapes confirm the architecture. Both files remain untouched. The app uses a separately converted local safetensors copy; every expected tensor matched strictly. Public prompt coordinates are explicitly float32 before MPS placement because the SAM1 processor otherwise produces float64 coordinates.

On a fresh installation, place the same official file in `SAM1/` beside `app/`, then use the project Python to run `scripts/convert_local_checkpoints.py --model sam1-h` followed by `scripts/verify_models.py --model sam1-h`. Conversion refuses to overwrite an existing converted folder. Restricted or local checkpoint files are not bundled into releases.

## SAM3.1 text

The supplied SAM3.1 image detector passed genuine local text inference. It is labeled experimental because its detector extraction and three-level pyramid bridge are application adaptations, not official Transformers SAM3.1 support. All 1,464 required tensors load strictly. The 768×512 `mountain` test returned one mask (7.32 seconds inference/postprocessing, 1.61 seconds load). A full-resolution 3072×2048 application job returned one 443,050-pixel mask (5.35 seconds inference, 2.97 seconds load). No tracking components or remote services ran. See [SAM3_ACCESS.md](SAM3_ACCESS.md) for exact provenance, limitations, and reproduction.

## CPU, Windows, NVIDIA

SAM2.1 Tiny passed actual combined point/box and automatic inference on the M4 CPU using float32 with explicit CPU selection. This establishes the CPU adapter on this Mac, not native Intel Windows performance. Windows x64 volume/process/launcher code and separate CPU/NVIDIA setup scripts are provided. Windows candidates exist for all 49 pinned package wheels; PyTorch 2.12.1+cu130 and torchvision 0.27.1+cu130 have official Windows x64 CPython3.12 wheels. **No Windows machine, Intel CPU, or NVIDIA GPU was used for this validation.** Native Windows checks remain pending and run during setup. See [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

## Integrity and reproducibility

All eight official SAM2/SAM2.1 snapshots are revision-pinned with sizes and SHA256 hashes in `config/model-manifest.json`; each is verified before loading. The total official snapshot payload is 3,122,960,374 bytes. The source and converted local checkpoint hashes are in the private workspace record `setup-notes/local-model-manifest.json`. Converter attribution is in [THIRD_PARTY.md](THIRD_PARTY.md). No credentials were accessed or copied.

Machine-local evidence: `setup-notes/compatibility.json`, `sam31-text-compatibility.json`, `expanded-ui-verification.json`, `expanded-independent-prompts.log`, and the associated real mask PNGs. These records are not shipped to another machine as proof of its compatibility. Setup performs its own tests. Full-resolution point/box input is capped at 24 MP; automatic and text segmentation at 8 MP, with a 128-million-pixel result budget. Memory safeguards remain enabled.
