# Publishing this project on GitHub

Publish the application source and documentation, then attach the separate Mac and Windows setup ZIPs to a GitHub Release. The checkpoint files, local Python environment, projects, original user images, and export bundles do not belong in the public repository.

This guide prepares an educational software release. It does not publish anything automatically. Read the model-license discussion linked from the [README](../README.md) before publishing; the application's license does not replace the separate terms that apply to models, copied third-party code, dependencies, or sample images.

## Which files to upload

The prepared source archive is named:

`LandscapeSegmentationStudio-GitHub-source-0.2.1.zip`

It is in the dedicated workspace's `publication/` folder, outside the running application's repository. Extract it into a **new folder**. Open its `landscape-segmentation-studio/` directory. That directory's **contents** become the root of the GitHub repository; do not upload the ZIP itself as the only repository file.

The first page of the repository should show `README.md`, the application's license, notices, and these source folders:

```text
README.md
LICENSE
NOTICE
THIRD_PARTY_NOTICES.md
START_HERE.md
.gitignore
licenses/
backend/
config/
docs/
  screenshots/
frontend/
  src/
  package.json
  package-lock.json
samples/
scripts/
tests/
requirements.lock
Setup Mac.command
Setup Windows.cmd
Setup Windows NVIDIA.cmd
Start Studio.command
Start Studio.cmd
Stop Studio.command
Stop Studio.cmd
```

Other reviewed documentation, license texts, and project configuration files in the prepared source archive should also be retained. In particular, do not discard `scripts/vendor/` attribution/license files or the bundled-dependency license notices. Keep the images in `docs/screenshots/`, because the README links to them by relative path.

| Item | Public Git repository | Release attachment |
| --- | --- | --- |
| Prepared source archive's extracted contents | Yes | Optional source ZIP |
| README, setup guides, source, tests, dependency locks | Yes | Included in platform ZIPs |
| Public Yosemite sample and its attribution | Yes | Included in platform ZIPs |
| Reviewed screenshots of the public sample | Yes | Included in platform ZIPs |
| Built `frontend/dist/` interface | Ignored in source Git | Yes, inside the prepared platform ZIPs |
| Mac and Windows setup ZIPs | Keep out of source Git | Yes |
| Checkpoints or converted model weights | No | No |
| Python environment or package caches | No | No |
| Personal images, projects, exports, logs, credentials | No | No |

The source archive deliberately has **no `.git` directory or development history**. It provides a clean initial public snapshot. The existing development repository remains unchanged. A scoped review of its reachable history found no common credential-shaped tokens, private-key headers, personal absolute paths, or checkpoint files; that review is not a guarantee against every possible secret pattern. Using the prepared snapshot avoids accidentally carrying unrelated local history into the new public repository.

## Create the repository and upload source

GitHub Desktop or Git preserves file modes and makes reviewing a large upload easier. Create a new repository for the prepared source contents and review the complete change list before publishing. On macOS, make sure the three `.command` launchers remain executable. With Git, the executable modes can be recorded from the source repository directory using:

```sh
git update-index --chmod=+x 'Setup Mac.command' 'Start Studio.command' 'Stop Studio.command'
```

For a browser upload:

1. Create a new repository in your own GitHub account. A name such as `landscape-segmentation-studio` is suitable. Start with an empty repository so GitHub does not create a second README or a different license.
2. Open the repository and choose **Add file → Upload files**.
3. Upload the contents of the extracted source folder. Include hidden configuration files such as `.gitignore`. In Finder, **Command–Shift–Period** toggles hidden-file visibility.
4. Check that `README.md` appears directly at the repository root. Do not create another enclosing `app/` or `landscape-segmentation-studio/` folder on GitHub.
5. Use a commit message such as `Publish educational local image segmentation studio`, review the file list, and commit.
6. Open the README on GitHub. Check the screenshots, local documentation links, license, and setup instructions.

GitHub currently limits a browser upload to 100 files at once and 25 MiB per file. Use several batches if needed, preserving directory paths; use Git or GitHub Desktop when that is easier. Browser upload does not apply `.gitattributes` processing and may not preserve executable file modes. The packaged Mac release already preserves the launcher modes; source users can invoke the documented shell commands when necessary. See [GitHub's file-upload guide](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

You choose the repository's owner, URL, visibility, and final publication action. Do not include access tokens in commands, screenshots, README examples, or repository files.

## Attach the user setup packages to a Release

The prepared platform packages are named:

- `LandscapeSegmentationStudio-Mac-Apple-Silicon-0.2.1.zip`
- `LandscapeSegmentationStudio-Windows-x64-0.2.1.zip`

Use the refreshed **0.2.1** packages with the current documentation and notices, rather than the earlier 0.2.0 files. Attach the release manifest containing their SHA256 checksums alongside them when available.

1. In the GitHub repository, open **Releases → Draft a new release**.
2. Create tag `v0.2.1` against the commit containing the reviewed publication files.
3. Use a title such as `0.2.1: educational local image segmentation preview`.
4. Explain that Python is a prerequisite, the setup phase downloads packages and baseline models, and normal inference runs locally afterward.
5. Clearly label **Windows CPU/NVIDIA as a test release awaiting native Windows validation**, and **SAM 3.1 text segmentation as experimental**. Mac M4 validation does not certify Windows, Intel, or NVIDIA hardware.
6. Attach the two platform ZIPs and their checksum manifest. They contain the prebuilt interface; users do not need Node.js to run these packages.
7. Mark the release as a **pre-release**, preview the description and assets, then publish when ready.

These steps follow [GitHub's release guide](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository). Do not replace a delivered archive under the same version after changing its contents; create a new version with new checksums.

GitHub also creates automatic **Source code (zip)** and **Source code (tar.gz)** downloads. Those contain repository source only. They are **not** the prepared platform setup packages and do not include `frontend/dist/`. Tell ordinary users to download the named Mac or Windows attachment under **Assets**. Source users must build the interface as described in the README.

## Keep the installation layout separate from the GitHub layout

The GitHub repository root is the application source root. On an installed computer, place that source inside an `app/` directory within a new dedicated workspace:

```text
LandscapeSegmentationStudio/
  app/                   ← repository contents or extracted platform package
    README.md
    backend/
    frontend/
    scripts/
  runtime/               ← created locally by setup
  model-cache/           ← downloaded or converted locally
  projects/              ← created by the user
  exports/               ← created by the user
  setup-notes/           ← local drive identity and test evidence
  tmp/                   ← local temporary files
```

The platform ZIPs already provide `LandscapeSegmentationStudio/app/`. Keep that layout after extraction. For a clone or GitHub's automatic source ZIP, create a new dedicated outer directory and clone/extract the repository as its `app/` child. Do not run a source checkout directly from a shared `Documents` or `Downloads` parent, because the application creates its runtime and data directories beside the source directory. On Mac, the current setup requires a supported connected external drive. Windows requires a supported local drive. See the OS-specific setup guides.

## What must stay private or local

Do **not** drag the complete SSD workspace into GitHub. These sibling folders and files are outside the public application source:

- `SAM1/`, `SAM2/`, `SAM3.1/`, and any other downloaded model/checkpoint folders;
- `model-cache/`, including converted `.safetensors` files;
- `runtime/`, virtual environments, pip/npm caches, and local process logs;
- `projects/`, `exports/`, and any user's source photographs or annotations;
- `setup-notes/`, which can contain drive identifiers, local paths, and machine-specific evidence;
- `tmp/`, personal notes, and the original private development prompt;
- `.env` files, account tokens, access approval correspondence, or browser/account profiles;
- Finder metadata such as `.DS_Store` and AppleDouble `._*` files.

The prepared source archive excludes these materials. `.gitignore` is an additional guard, not a substitute for reviewing the upload. It does not protect files already tracked in Git, and manually uploading through a website does not apply Git's ignore rules.

## Screenshots and attribution

The committed screenshots show the app using the bundled Yosemite sample. They are genuine captures of the local interface. Some earlier screenshots show an older arrangement of controls; the current README identifies the relevant workflow. Do not publish screenshots that expose a recipient's private photograph, project name, filesystem path, token, or account information.

The sample's [specific attribution](../samples/ATTRIBUTION.md) identifies it as an NPS Digital Image Archives photograph, public domain in the United States. The [National Park Service policy](https://www.nps.gov/aboutus/disclaimer.htm) distinguishes federal works from third-party material and protected agency marks. Retain the attribution and do not imply NPS endorsement. No claim is made to the original U.S. Government photograph; app licensing does not relicense that photograph.

## Release review

Before making the repository public, check that the application license and all third-party notices are present; that no model weights or user data are included; that source installation and platform release installation are clearly distinguished; and that SAM 3.1's separate terms and experimental status are accurately described. Preserve the distinction between a verified Mac installation and an unverified Windows test package. Educational intent is a description of the project, not an exemption from third-party license terms.
