import base64
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("update_manifest", Path(__file__).parents[1] / "scripts/update-manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ManifestTests(unittest.TestCase):
    def test_feed_uses_exact_signature_and_release_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "unsound.app.tar.gz"
            archive.write_bytes(b"archive fixture")
            signature = base64.b64encode(b"untrusted comment: signature fixture\nsignature").decode()
            Path(str(archive) + ".sig").write_text(signature + "\n")
            release = {"tag_name": "v0.9.1", "body": "Release notes\nwith quotes: \"hello\"", "created_at": "2026-10-26T00:00:00Z"}
            feed = module.manifest("0.9.1", "thisduck/unsound", archive, release)
            self.assertEqual(feed["notes"], release["body"])
            self.assertEqual(feed["platforms"]["darwin-aarch64"]["signature"], signature)
            self.assertEqual(feed["platforms"]["darwin-aarch64"]["url"], "https://github.com/thisduck/unsound/releases/download/v0.9.1/unsound.app.tar.gz")
            with self.assertRaises(ValueError): module.manifest("0.9.2", "thisduck/unsound", archive, release)
            Path(str(archive) + ".sig").write_text("")
            with self.assertRaises(ValueError): module.manifest("0.9.1", "thisduck/unsound", archive, release)
            archive.unlink()
            with self.assertRaises(ValueError): module.manifest("0.9.1", "thisduck/unsound", archive, release)

if __name__ == "__main__": unittest.main()
