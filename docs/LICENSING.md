# Licensing and public publication

Reviewed against the official sources on **8 September 2026**. This is a documented license review for this educational application, not a legal opinion or a guarantee that every possible use is lawful. The complete license texts control. No GitHub publication or model-weight redistribution was performed as part of this review.

## Can this application be public?

The reviewed model licenses do not impose a blanket prohibition on publishing an educational application that uses them. **Publishing the application source, setup scripts, documented examples, and properly attributed screenshots without model weights is the intended distribution route.** Preserve the license files and notices, keep each model's terms separate, and use images you have permission to publish. Education is the purpose of this project; it is not an exemption from license, copyright, privacy, or export requirements.

SAM 1 and SAM 2/2.1 provide the clearest permissive path. SAM 3/3.1 has a different, restrictive agreement and an interpretation issue discussed below. Do not describe the whole collection of application, dependencies, models, and outputs as uniformly Apache-licensed.

## Which license covers which material?

| Material | Applicable terms | Distribution in this project |
|---|---|---|
| Original application code, UI, documentation, and original scripts | The application license in [`LICENSE`](../LICENSE), subject to the third-party exclusions documented here | Source and setup bundles |
| Vendored Hugging Face SAM and SAM3 conversion functions | Apache License 2.0; retain the existing file headers and modification notices | Included as source in `scripts/vendor/`; see [third-party attribution](THIRD_PARTY.md) |
| Meta SAM 1 code and model checkpoints | Apache License 2.0 | Checkpoints obtained separately; no model weights included in source or setup ZIPs |
| Meta SAM 2 and SAM 2.1 code and model checkpoints | Apache License 2.0 | Official Hugging Face snapshots downloaded by setup/model tools; no model weights bundled |
| Meta SAM 3/3.1 materials, including their weights and derivatives | Custom **SAM License**, last updated 19 November 2025 | Optional local user-supplied files; no original or converted weights bundled |
| Installed Python/JavaScript dependencies | Each dependency's own license | Python packages installed separately; bundled frontend libraries need their notices preserved |
| Sample photograph and screenshots containing it | The photograph's own rights statement; application artwork/code terms apply separately | See [`samples/ATTRIBUTION.md`](../samples/ATTRIBUTION.md) |
| A user's photographs, masks, exports, or research dataset | Rights in the input material, applicable model terms, and applicable law | User-controlled local project files; excluded from public packages |

Meta expressly identifies the original SAM model as Apache-licensed. Its SAM 2 repository explicitly covers checkpoints, demo code, and training code under Apache 2.0, while identifying separate font and optional connected-component-code licenses. This application uses a Transformers image adapter and does not bundle Meta's demo fonts or its optional CUDA connected-component implementation. [SAM 1 repository](https://github.com/facebookresearch/segment-anything#license), [SAM 2 repository](https://github.com/facebookresearch/sam2#license).

The Apache headers on Hugging Face's conversion functions are separate grants for those files. They do **not** change the license of the weights being converted. [SAM converter](https://github.com/huggingface/transformers/blob/db1e6f1e7fda38af4c4d20192bc35ac647a7a21f/src/transformers/models/sam/convert_sam_to_hf.py), [SAM3 converter](https://github.com/huggingface/transformers/blob/22278df3198c4219f033f2a4b0931da3e5c21af4/src/transformers/models/sam3/convert_sam3_to_hf.py).

## SAM 1 and SAM 2/2.1: Apache 2.0 obligations

Apache 2.0 permits use, modification, and redistribution, including commercial use. When distributing covered material or derivatives:

1. Include a copy of Apache 2.0.
2. Preserve relevant copyright, patent, trademark, and attribution notices.
3. Mark modified files prominently as changed.
4. Preserve applicable upstream `NOTICE` attributions when a distribution includes them.
5. Respect the patent and trademark clauses. A model name identifies compatibility; it does not imply Meta endorses this application.

Apache 2.0 does not require making an independently written application use the same license merely because it calls the model's API. Its definition distinguishes separable/interface-linked works. A root application license must nevertheless leave upstream rights and obligations intact. See sections 1–4 and 6 of the [official SAM Apache license](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/LICENSE). Full local copies are in [`licenses/`](../licenses/).

The SA-1B and SA-V **datasets are separate** from the models; do not assume the model license also authorizes republishing those datasets. This application does not distribute them. The upstream repositories link their respective dataset terms. [SAM 1 dataset information](https://github.com/facebookresearch/segment-anything#dataset), [SAM 2 dataset information](https://github.com/facebookresearch/sam2#segment-anything-video-dataset).

## SAM 3 and SAM 3.1: additional terms

The SAM License grants rights to use, modify, and distribute covered materials. Redistribution of those materials or derivatives must remain under that agreement and include its complete text. Research publications using the materials must acknowledge their use. The agreement requires compliance with privacy law and trade controls and restricts certain military, weapons, nuclear, espionage, and related uses. It also contains a reverse-engineering restriction, termination terms, disclaimers, and an indemnity obligation. It is **not Apache 2.0**, and educational use does not remove these conditions. See the [full official license](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE), also preserved as [`SAM3-LICENSE.txt`](../licenses/SAM3-LICENSE.txt).

The license does not require a particular “Built with SAM” badge, does not state a general noncommercial-only restriction, and does not give this project an unrestricted right to relicense Meta material. This review found no unconditional ban on publishing an application source repository. These conclusions concern this exact license version; check the current terms before a later release.

### Access approval is separate from the application

The official SAM3.1 checkpoint repository is gated. Each recipient should obtain access through their own account, review the presented conditions, and obtain their own authorized copy. The publisher's accepted access request is not a transferable account credential. The project neither bundles credentials nor bypasses the publisher's gate. [Official SAM3.1 model page](https://huggingface.co/facebook/sam3.1).

Keeping weights out of GitHub is this project's distribution choice. It should not be misrepresented as a statement that the SAM License categorically forbids all redistribution: the agreement expressly allows redistribution **subject to its conditions**. A source-only release avoids passing along multi-gigabyte gated assets and lets recipients review the original publisher's terms.

### Experimental local conversion and its legal boundary

This application's SAM3.1 compatibility work uses a supplied checkpoint, an Apache-licensed Hugging Face key conversion, and the publicly documented Meta detector layout. It strictly maps existing tensors; it does not obtain unauthorized files or recover undisclosed model architecture. The small `backend/sam31.py` bridge and the SAM3.1 branch of `scripts/convert_local_checkpoints.py` are application compatibility code. Their provenance is described in [SAM3 access and compatibility](SAM3_ACCESS.md). Meta's public builder itself specifies the detector components and checkpoint loading. [Published model builder](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py).

Section 1(b)(iv) uses broad wording about reverse engineering and discovering underlying components, alongside section 1(a)'s express modification grant. Reading and adapting an openly documented format is not automatically evidence of prohibited reverse engineering. However, this technical review cannot conclusively resolve how Meta or a court would apply that wording to this particular experimental conversion. **Written clarification from Meta or qualified legal advice is needed if a publisher requires definitive clearance of that specific issue.** No claim of illegality or guaranteed clearance is made here. A publisher who wants to avoid this unresolved SAM3-specific question can distribute/use the SAM 1/2 path without the optional SAM3.1 integration.

Original application code does not acquire Meta ownership merely by calling an API. Equally, any Meta SAM3/3.1 material or derivative that is present remains governed by the SAM License regardless of its folder, format, or an Apache license elsewhere. In particular, a converted checkpoint, extracted detector weights, and Meta-supplied configuration/tokenizer assets must not be treated as newly Apache-licensed application assets.

The upstream SAM3.1 model page states that it has no official Transformers integration. This application's bridge is therefore clearly described as **experimental and unofficial**, with no claim of Meta endorsement or numerical parity with the upstream CUDA pipeline. [Official SAM3.1 model page](https://huggingface.co/facebook/sam3.1).

## Images, masks, screenshots, and research acknowledgement

Segmentation does not grant permission to republish the input photograph. Check copyright, privacy, confidentiality, and consent before sharing a source image, screenshot, or exported project. A model license does not prove that every generated mask or output is copyright-free or clear of third-party rights. For SAM3/3.1 specifically, the license discusses output responsibility and disclaims warranties; it should not be read as a blanket clearance of outputs.

The repository's tutorial uses the attributed Yosemite sample. Keep its attribution with screenshots showing it. Do not include unrelated personal project screenshots, file paths containing private information, or downloaded model-access pages containing account details.

Suggested acknowledgement for this educational application:

> Landscape Segmentation Studio is an independent educational application using Meta's Segment Anything models and Hugging Face Transformers. Meta and Hugging Face do not endorse this project. Model weights and other third-party materials retain their own licenses.

For a research publication, identify the actual model family, checkpoint, revision, prompts, and settings used, and acknowledge the relevant authors. Use the official research citations:

- Kirillov et al., **Segment Anything** (2023): [paper](https://arxiv.org/abs/2304.02643) and [upstream citation](https://github.com/facebookresearch/segment-anything#citing-segment-anything).
- Ravi et al., **SAM 2: Segment Anything in Images and Videos** (2024): [paper](https://arxiv.org/abs/2408.00714) and [upstream citation](https://github.com/facebookresearch/sam2#citing-sam-2).
- **SAM 3: Segment Anything with Concepts**: [paper](https://arxiv.org/abs/2511.16719). Identify SAM3.1 separately when using that checkpoint; its [release notes](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/RELEASE_SAM3p1.md) describe that release.

## Publication boundary

Publish the prepared **application source tree** and reviewed release bundles, retaining `LICENSE`, notices, `licenses/`, `docs/`, and sample attribution. Do not upload the whole working SSD folder. Exclude model checkpoints and caches, original user images, projects, exports, virtual environments, local setup reports, access tokens, and `.git` internals from a manual upload. Source publication and GitHub Release attachments are separate actions; the maintainer decides when to publish them.

Use the release builder and publication guide instead of manually collecting files from the runtime workspace. Recheck third-party notices if future releases bundle Python, additional JavaScript libraries, fonts, native binaries, models, or datasets. The present review is scoped to this application and the exact materials described above, not to arbitrary additions or future upstream license changes.
