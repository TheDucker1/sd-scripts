"""Check checkpoint publication and retention with a fake Drive API."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools.colab.drive_checkpoints import (
    CHECKPOINT_FILES, FOLDER_MIME, RUN_FILES_MARKER, UPLOAD_MARKER,
    DriveCheckpointPublisher, is_connection_error, is_rate_limit_error,
)


class FakeHttpError(Exception):
    pass


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
        self.publisher.request_interval = 0
        self.publisher.next_request_at = 0
        self.publisher.run_files = {}
        self.publisher.run_signatures = {}
        self.publisher.run_folders = {".": "run"}
        errors = mock.patch.dict("sys.modules", {"googleapiclient.errors": SimpleNamespace(HttpError=FakeHttpError)})
        errors.start()
        self.addCleanup(errors.stop)

    def quota_error(self, status=403, reason="rateLimitExceeded"):
        error = Exception(reason)
        error.resp = SimpleNamespace(status=status)
        error.content = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
        return error

    def test_recognizes_only_request_rate_limits(self):
        self.assertTrue(is_rate_limit_error(self.quota_error()))
        self.assertTrue(is_rate_limit_error(self.quota_error(reason="userRateLimitExceeded")))
        self.assertTrue(is_rate_limit_error(self.quota_error(status=429)))
        self.assertFalse(is_rate_limit_error(self.quota_error(reason="insufficientPermissions")))
        self.assertFalse(is_rate_limit_error(self.quota_error(status=500)))
        self.assertFalse(is_rate_limit_error(ValueError("invalid checkpoint")))

    @mock.patch("tools.colab.drive_checkpoints.random.uniform", return_value=0)
    @mock.patch("tools.colab.drive_checkpoints.time.sleep")
    def test_quota_errors_retry_without_losing_the_operation(self, sleep, uniform):
        operation = mock.Mock(side_effect=[self.quota_error(), self.quota_error(), "uploaded"])
        self.assertEqual(self.publisher.request(operation), "uploaded")
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(60), mock.call(120)])

    @mock.patch("tools.colab.drive_checkpoints.random.uniform", return_value=0)
    @mock.patch("tools.colab.drive_checkpoints.time.sleep")
    def test_resumable_upload_retries_ssl_timeout(self, sleep, uniform):
        operation = mock.Mock(side_effect=[TimeoutError("The read operation timed out"), "uploaded"])
        self.assertEqual(self.publisher.request(operation, retry_timeout=True), "uploaded")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(60)

    def test_transient_http_and_connection_errors(self):
        self.assertTrue(is_connection_error(TimeoutError()))
        self.assertTrue(is_connection_error(ConnectionResetError()))
        self.assertTrue(is_connection_error(self.quota_error(status=503)))
        self.assertFalse(is_connection_error(self.quota_error(status=403)))
        self.assertFalse(is_connection_error(ValueError("invalid checkpoint")))

    def test_sdk_does_not_make_unpaced_metadata_retries(self):
        request = mock.Mock()
        self.publisher.execute(request)
        request.execute.assert_called_once_with(num_retries=0)

    @mock.patch("tools.colab.drive_checkpoints.random.uniform", return_value=0)
    @mock.patch("tools.colab.drive_checkpoints.time.sleep")
    def test_upload_keeps_resumable_request_after_timeout(self, sleep, uniform):
        publisher = self.publisher
        request = publisher.files.update.return_value
        result = {"id": "remote-file", "size": "10", "md5Checksum": "checksum"}
        request.next_chunk.side_effect = [TimeoutError("SSL read timed out"), (None, None), (None, result)]
        media = SimpleNamespace(MediaIoBaseUpload=mock.Mock())
        with mock.patch.dict("sys.modules", {"googleapiclient.http": media}):
            actual = publisher.upload_file("run", self.checkpoint / CHECKPOINT_FILES[0], {"id": "remote-file"})
        self.assertEqual(actual, result)
        publisher.files.update.assert_called_once()
        self.assertEqual(request.next_chunk.call_args_list, [mock.call(num_retries=0)] * 3)
        sleep.assert_called_once_with(60)

    def test_run_files_sync_once_per_completed_checkpoint(self):
        publisher = self.publisher
        publisher.publish = mock.Mock(side_effect=lambda path: (path / UPLOAD_MARKER).write_text("remote-id"))
        publisher.sync_run_files = mock.Mock()
        publisher.prune = mock.Mock()
        run = self.root / "local-run"
        run.mkdir()
        log = run / "console.log"
        log.write_text("first step\n")
        publisher.publish_pending(run)
        log.write_text("first step\nsecond step\n")
        publisher.publish_pending(run)
        publisher.publish.assert_called_once_with(self.checkpoint)
        publisher.sync_run_files.assert_called_once_with(run)
        publisher.prune.assert_called_once()
        self.assertTrue((self.checkpoint / RUN_FILES_MARKER).is_file())

        next_checkpoint = self.root / "step-000003"
        next_checkpoint.mkdir()
        publisher.publish_pending(run)  # Ignore a save still being written.
        self.assertEqual(publisher.sync_run_files.call_count, 1)
        for name in CHECKPOINT_FILES:
            (next_checkpoint / name).write_bytes(b"next save")
        publisher.publish_pending(run)
        self.assertEqual(publisher.publish.call_count, 2)
        self.assertEqual(publisher.sync_run_files.call_count, 2)

    def test_failed_run_sync_stays_pending_without_reuploading_checkpoint(self):
        publisher = self.publisher
        publisher.publish = mock.Mock(side_effect=lambda path: (path / UPLOAD_MARKER).write_text("remote-id"))
        publisher.sync_run_files = mock.Mock(side_effect=[TimeoutError("offline"), None])
        publisher.prune = mock.Mock()
        with self.assertRaises(TimeoutError):
            publisher.publish_pending(self.root)
        self.assertFalse((self.checkpoint / RUN_FILES_MARKER).exists())
        publisher.prune.assert_not_called()
        publisher.publish_pending(self.root)
        publisher.publish.assert_called_once()
        self.assertEqual(publisher.sync_run_files.call_count, 2)
        self.assertTrue((self.checkpoint / RUN_FILES_MARKER).is_file())

    def test_idle_watch_makes_no_drive_calls_even_with_legacy_interval(self):
        publisher = self.publisher
        (self.checkpoint / UPLOAD_MARKER).write_text("remote-id")
        (self.checkpoint / RUN_FILES_MARKER).write_text("verified")
        publisher.sync_run_files = mock.Mock()
        stop = mock.Mock()
        stop.is_set.side_effect = [False, True]
        publisher.watch(stop, self.root, sync_interval=0)
        publisher.sync_run_files.assert_not_called()
        self.assertEqual(publisher.files.mock_calls, [])

    @mock.patch("tools.colab.drive_checkpoints.time.sleep")
    def test_watch_reconciles_after_metadata_timeout_and_drains(self, sleep):
        publisher = self.publisher
        publisher.publish_pending = mock.Mock(side_effect=[TimeoutError("offline"), None, None])
        stop = mock.Mock()
        stop.is_set.side_effect = [False, False, True]
        publisher.watch(stop, self.root)
        self.assertEqual(publisher.publish_pending.call_count, 3)
        sleep.assert_called_once_with(60)

    @mock.patch("tools.colab.drive_checkpoints.time.sleep")
    def test_permission_error_still_fails(self, sleep):
        operation = mock.Mock(side_effect=self.quota_error(reason="insufficientPermissions"))
        with self.assertRaisesRegex(Exception, "insufficientPermissions"):
            self.publisher.request(operation)
        operation.assert_called_once()
        sleep.assert_not_called()

    @mock.patch("tools.colab.drive_checkpoints.time.monotonic", return_value=10)
    @mock.patch("tools.colab.drive_checkpoints.time.sleep")
    def test_paces_successful_requests(self, sleep, monotonic):
        self.publisher.request_interval = 1
        self.publisher.next_request_at = 11
        self.assertEqual(self.publisher.request(lambda: "uploaded"), "uploaded")
        sleep.assert_called_once_with(1)
        self.assertEqual(self.publisher.next_request_at, 11)

    def test_run_files_upload_stable_snapshots_and_retry_changed_logs(self):
        publisher = self.publisher
        run = self.root / "local-run"
        run.mkdir()
        log = run / "console.log"
        log.write_bytes(b"first line\n")
        sample = run / "samples" / "step-000300" / "pair-000.jpg"
        sample.parent.mkdir(parents=True)
        sample.write_bytes(b"tiny preview")
        publisher.find_folder = mock.Mock(return_value=None)
        publisher.files.create.return_value.execute.side_effect = [{"id": "samples"}, {"id": "preview-step"}]
        publisher.children = mock.Mock(return_value=iter(()))
        seen = []

        def upload(folder, snapshot, existing):
            payload = snapshot.read_bytes()
            seen.append((folder, snapshot.name, payload))
            if snapshot.name == "console.log" and payload == b"first line\n":
                log.write_bytes(b"first line\nsecond line\n")
            return dict(size=str(len(payload)), md5Checksum=hashlib.md5(payload, usedforsecurity=False).hexdigest())

        publisher.upload_file = mock.Mock(side_effect=upload)
        publisher.sync_run_files(run)
        self.assertIn(("preview-step", sample.name, b"tiny preview"), seen)
        publisher.sync_run_files(run)
        self.assertIn(("run", log.name, b"first line\nsecond line\n"), seen)
        self.assertEqual(len(seen), 3)
        self.assertEqual(publisher.children.call_count, 2)
        publisher.sync_run_files(run)
        self.assertEqual(len(seen), 3)

    def test_run_file_checksum_failure_is_retried(self):
        publisher = self.publisher
        run = self.root / "local-run"
        run.mkdir()
        (run / "metrics.jsonl").write_bytes(b"{}\n")
        publisher.children = mock.Mock(return_value=iter(()))
        publisher.upload_file = mock.Mock(return_value=dict(size="0", md5Checksum="bad"))
        with self.assertRaisesRegex(OSError, "checksum or size mismatch"):
            publisher.sync_run_files(run)
        self.assertEqual(publisher.run_signatures, {})

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
