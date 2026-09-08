# Third-party model license copies

These files preserve the official SAM model license texts reviewed on **8 September 2026**:

| File | Material | Official source |
|---|---|---|
| `SAM1-APACHE-2.0.txt` | Original Segment Anything model/code | [Meta SAM 1 LICENSE](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/LICENSE) |
| `SAM2-APACHE-2.0.txt` | SAM 2 and SAM 2.1 model/code | [Meta SAM 2 LICENSE](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE) |
| `SAM3-LICENSE.txt` | SAM 3/3.1 materials and derivatives | [Meta SAM LICENSE](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE) |

`sources.json` records each downloaded file's immutable upstream revision, source URL, SHA-256 checksum, and review date. The SAM3 license matches the license accompanying the locally supplied SAM3.1 download in wording (whitespace differences only).

No model weights are included here. These licenses do not replace each dependency's terms or the application's own license. The Apache license for the retained Hugging Face converter functions is also preserved at `scripts/vendor/LICENSE-APACHE-2.0.txt`. See [licensing guidance](../docs/LICENSING.md) and [third-party conversion attribution](../docs/THIRD_PARTY.md).

The built frontend also includes `REACT-MIT.txt`, `REACT-DOM-MIT.txt`, `SCHEDULER-MIT.txt`, and `VITE-MIT.txt`, copied from the pinned installed packages. Keep these with the platform ZIPs. Vite's complete core MIT license is preserved for its generated module-preload helper. The Vite development server and its bundled development dependencies are not shipped in the interface; their full distribution retains its own additional notices when installed for development.
