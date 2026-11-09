"""Publish completed local checkpoints to Drive while Colab training runs."""

import hashlib
import json
import re
import shutil
import time
from pathlib import Path


CHECKPOINT_FILES = ("adapter_model.safetensors", "adapter_config.json", "training_state.pt", "complete.json")
FOLDER_MIME = "application/vnd.google-apps.folder"
FILE_FIELDS = "id,name,mimeType,size,md5Checksum,appProperties"
UPLOAD_MARKER = ".drive-confirmed"


def checkpoint_step(name):
    match = re.fullmatch(r"step-(\d+)", name)
    return int(match[1]) if match else None


def completed_checkpoints(root):
    checkpoints = (path for path in Path(root).iterdir() if checkpoint_step(path.name) is not None
                   and path.is_dir() and not path.is_symlink()
                   and all((path / name).is_file() for name in CHECKPOINT_FILES))
    return sorted(checkpoints, key=lambda path: checkpoint_step(path.name))


def file_signature(path):
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    return str(path.stat().st_size), checksum


class DriveCheckpointPublisher:
    def __init__(self, output_dir, keep_last):
        import google.auth
        from googleapiclient.discovery import build

        if not isinstance(keep_last, int) or keep_last < 0:
            raise ValueError("KEEP_LAST must be a nonnegative integer")
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        self.files = build("drive", "v3", credentials=credentials, cache_discovery=False).files()
        self.keep_last = keep_last
        relative = Path(output_dir).relative_to("/content/drive/MyDrive")
        if not relative.parts or ".." in relative.parts:
            raise ValueError("OUTPUT_DIR must be a run folder inside /content/drive/MyDrive")
        parent = "root"
        for part in relative.parts:
            # The mount can take a moment to publish a newly created run folder.
            for attempt in range(15):
                folder = self.find_folder(parent, part)
                if folder is not None:
                    break
                if attempt < 14:
                    time.sleep(2)
            else:
                raise FileNotFoundError(f"Drive cannot see folder {part!r} yet. Check OUTPUT_DIR in Drive and rerun.")
            parent = folder["id"]
        permissions = self.files.get(fileId=parent, fields="capabilities(canAddChildren)").execute(num_retries=3)
        if not permissions["capabilities"]["canAddChildren"]:
            raise PermissionError(f"The authenticated Drive account cannot upload to {output_dir}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", parent):
            raise ValueError("Unexpected Drive folder ID")
        self.folder_id = parent
        self.staging_dir = Path("/content/anima-checkpoint-staging") / self.folder_id
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def children(self, parent, name=None):
        query = f"'{parent}' in parents and trashed = false"
        if name is not None:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            query += f" and name = '{escaped}'"
        page_token = None
        while True:
            result = self.files.list(q=query, fields=f"nextPageToken,files({FILE_FIELDS})", pageSize=1000,
                                     pageToken=page_token).execute(num_retries=3)
            yield from result.get("files", ())
            page_token = result.get("nextPageToken")
            if not page_token:
                break

    def find_folder(self, parent, name):
        matches = [item for item in self.children(parent, name) if item["mimeType"] == FOLDER_MIME]
        if len(matches) > 1:
            raise ValueError(f"Drive contains multiple folders named {name!r} in the same location")
        return matches[0] if matches else None

    def checkpoint_files(self, folder_id):
        files = {}
        for item in self.children(folder_id):
            name = item["name"]
            if name in CHECKPOINT_FILES:
                if name in files:
                    raise ValueError(f"Duplicate Drive checkpoint file: {name}")
                files[name] = item
        return files

    def upload_file(self, folder_id, path, existing):
        from googleapiclient.http import MediaIoBaseUpload

        with path.open("rb") as handle:
            media = MediaIoBaseUpload(handle, mimetype="application/octet-stream", resumable=True, chunksize=16 * 1024**2)
            if existing is None:
                request = self.files.create(body={"name": path.name, "parents": [folder_id]},
                                            media_body=media, fields=FILE_FIELDS)
            else:
                request = self.files.update(fileId=existing["id"], media_body=media, fields=FILE_FIELDS)
            result = None
            while result is None:
                _, result = request.next_chunk(num_retries=5)
            return result

    def publish(self, checkpoint):
        expected = {name: file_signature(checkpoint / name) for name in CHECKPOINT_FILES}

        def matches(item, name):
            return (item.get("size"), item.get("md5Checksum")) == expected[name]

        published = self.find_folder(self.folder_id, checkpoint.name)
        if published is not None:
            files = self.checkpoint_files(published["id"])
            if not all(matches(files.get(name, {}), name) for name in CHECKPOINT_FILES):
                raise ValueError(f"A different or incomplete {checkpoint.name} exists in Drive. Move it out of the run folder.")
        else:
            fingerprint = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:16]
            name = f".uploading-{checkpoint.name}-{fingerprint}"
            pending = self.find_folder(self.folder_id, name)
            if pending is None:
                pending = self.files.create(body={"name": name, "parents": [self.folder_id], "mimeType": FOLDER_MIME,
                                                  "appProperties": {"sd_scripts_checkpoint": "pending"}}, fields=FILE_FIELDS).execute()
            elif pending.get("appProperties", {}).get("sd_scripts_checkpoint") != "pending":
                raise ValueError(f"Drive staging folder {name!r} belongs to another writer")
            files = self.checkpoint_files(pending["id"])
            for name in CHECKPOINT_FILES:
                existing = files.get(name)
                if not matches(existing or {}, name):
                    result = self.upload_file(pending["id"], checkpoint / name, existing)
                    if not matches(result, name):
                        raise IOError(f"Drive checksum or size mismatch for {checkpoint.name}/{name}")
            published = self.files.update(fileId=pending["id"], body={"name": checkpoint.name,
                "appProperties": {"sd_scripts_checkpoint": "verified"}}, fields="id,name").execute(num_retries=3)
            if published["name"] != checkpoint.name:
                raise IOError(f"Drive did not publish {checkpoint.name}")
        (checkpoint / UPLOAD_MARKER).write_text(published["id"], encoding="utf-8")
        print(f"Drive verified {checkpoint.name} (all four files)", flush=True)

    def prune(self):
        confirmed = [path for path in completed_checkpoints(self.staging_dir) if (path / UPLOAD_MARKER).is_file()]
        for old in confirmed[:-1]:
            if old.resolve().parent == self.staging_dir.resolve():
                shutil.rmtree(old)
        if self.keep_last:
            managed = [item for item in self.children(self.folder_id) if item["mimeType"] == FOLDER_MIME
                       and checkpoint_step(item["name"]) is not None
                       and item.get("appProperties", {}).get("sd_scripts_checkpoint") == "verified"]
            managed.sort(key=lambda item: checkpoint_step(item["name"]))
            for old in managed[:-self.keep_last]:
                self.files.update(fileId=old["id"], body={"trashed": True}).execute(num_retries=3)

    def publish_pending(self):
        from googleapiclient.errors import HttpError

        for checkpoint in completed_checkpoints(self.staging_dir):
            if (checkpoint / UPLOAD_MARKER).is_file():
                continue
            print(f"Uploading {checkpoint.name} from Colab SSD to Drive", flush=True)
            try:
                self.publish(checkpoint)
            except HttpError as error:
                raise RuntimeError(f"Drive upload failed for {checkpoint.name} (HTTP {error.resp.status}). "
                                   f"The local checkpoint remains in {checkpoint} for retry.") from None
            try:
                self.prune()
            except (HttpError, OSError) as error:
                print(f"Checkpoint verified. Retention will retry after the next upload: {error}", flush=True)

    def watch(self, stop):
        while not stop.is_set():
            self.publish_pending()
            stop.wait(2)
        self.publish_pending()
