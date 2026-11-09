"""Check checkpoint publication and retention with a fake Drive API."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.colab.drive_checkpoints import CHECKPOINT_FILES, FOLDER_MIME, UPLOAD_MARKER, DriveCheckpointPublisher


class TestDriveCheckpointPublisher(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "step-000002"
        self.checkpoint.mkdir()
        for name in CHECKPOINT_FILES:
            (self.checkpoint / name).write_bytes(name.encode())
        self.publisher = DriveCheckpointPublisher.__new__(DriveCheckpointPublisher)
        self.publisher.staging_dir = self.root
        self.publisher.folder_id = "run"
        self.publisher.keep_last = 1
        self.publisher.files = mock.Mock()

    def test_publishes_only_after_all_files_are_verified(self):
        publisher = self.publisher
        publisher.find_folder = mock.Mock(return_value=None)
        publisher.files.create.return_value.execute.return_value = {"id": "pending"}
        publisher.checkpoint_files = mock.Mock(return_value={})
        uploaded = []

        def upload(folder, path, existing):
            self.assertEqual(folder, "pending")
            payload = path.read_bytes()
            uploaded.append(path.name)
            return {"size": str(len(payload)), "md5Checksum": hashlib.md5(payload, usedforsecurity=False).hexdigest()}

        def finish(**kwargs):
            self.assertEqual(uploaded, list(CHECKPOINT_FILES))
            self.assertFalse((self.checkpoint / UPLOAD_MARKER).exists())
            return {"id": "pending", "name": self.checkpoint.name}

        publisher.upload_file = mock.Mock(side_effect=upload)
        publisher.files.update.return_value.execute.side_effect = finish
        publisher.publish(self.checkpoint)
        self.assertEqual((self.checkpoint / UPLOAD_MARKER).read_text(), "pending")

    def test_bad_checksum_leaves_checkpoint_pending(self):
        publisher = self.publisher
        publisher.find_folder = mock.Mock(return_value=None)
        publisher.files.create.return_value.execute.return_value = {"id": "pending"}
        publisher.checkpoint_files = mock.Mock(return_value={})
        publisher.upload_file = mock.Mock(return_value={"size": "0", "md5Checksum": "bad"})
        with self.assertRaisesRegex(OSError, "checksum or size mismatch"):
            publisher.publish(self.checkpoint)
        publisher.files.update.assert_not_called()
        self.assertFalse((self.checkpoint / UPLOAD_MARKER).exists())
        self.assertTrue((self.checkpoint / "training_state.pt").is_file())

    def test_retention_preserves_uploaded_backups_and_pending_folders(self):
        publisher = self.publisher
        publisher.children = mock.Mock(return_value=iter([
            {"name": "step-000000", "id": "manual-backup", "mimeType": FOLDER_MIME},
            {"name": "step-000001", "id": "old", "mimeType": FOLDER_MIME,
             "appProperties": {"sd_scripts_checkpoint": "verified"}},
            {"name": "step-000002", "id": "latest", "mimeType": FOLDER_MIME,
             "appProperties": {"sd_scripts_checkpoint": "verified"}},
            {"name": ".uploading-step-000003", "id": "pending", "mimeType": FOLDER_MIME,
             "appProperties": {"sd_scripts_checkpoint": "pending"}},
        ]))
        publisher.prune()
        publisher.files.update.assert_called_once_with(fileId="old", body={"trashed": True})
        self.assertTrue(self.checkpoint.is_dir())
