"""Publication boundaries must reject tracked data accidentally added to a release."""
import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("studio_release", scripts / "build_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize("name", ["../projects/image.png", "/private/key", "projects/image.png", "backend/model.pth", "config/.env", "docs/._capture.png", "runtime/key.json", "SAM3.1/config.json", "scripts/key.pem", "unreviewed/image.jpg"])
def test_release_rejects_local_material(name):
    with pytest.raises(ValueError):
        release.validate_relative(name)


def test_source_and_built_release_are_distinct():
    with pytest.raises(ValueError):
        release.validate_relative("frontend/dist/index.html")
    assert str(release.validate_relative("frontend/dist/index.html", built=True)) == "frontend/dist/index.html"
    assert str(release.validate_relative("docs/screenshots/sam31-local-text.png")) == "docs/screenshots/sam31-local-text.png"
    assert str(release.validate_relative("licenses/SAM3-LICENSE.txt")) == "licenses/SAM3-LICENSE.txt"


def test_private_key_detection_without_embedded_credentials():
    assert release.SECRET_PATTERNS.search(b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----")
    assert not release.SECRET_PATTERNS.search(b"Read the SAM License before use.")
