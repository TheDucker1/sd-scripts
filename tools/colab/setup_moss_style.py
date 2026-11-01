"""Colab-only environment and asset preparation. Never starts model training.

Run with Colab's system Python, not the Windows/local training environment.
The script creates isolated bootstrap/training environments below /content.
No pretrained model is instantiated, custom model code imported, or dataset
shard downloaded. Training is launched separately by the notebook.
"""

import argparse
import fnmatch
import getpass
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def require_colab():
    try:
        colab_module = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        colab_module = False
    colab_environment = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))
    # The isolated venv does not contain google.colab. Only the already-verified
    # notebook-parent process supplies this marker for the download subprocess.
    verified_child = os.environ.get("_MOSS_COLAB_PARENT_VERIFIED") == "1" and "--assets-only" in sys.argv
    if sys.platform != "linux" or not Path("/content").is_dir() or not (colab_module or colab_environment or verified_child):
        raise SystemExit("REFUSING TO RUN: this setup is for Google Colab only. It does not support local execution.")


def run(arguments, cwd=None, capture=False, environment=None):
    # Arguments never include tokens. Do not print the inherited environment.
    print("Running:", " ".join(str(argument) for argument in arguments), flush=True)
    result = subprocess.run(
        [str(argument) for argument in arguments], cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE if capture else None, env=environment,
    )
    return result.stdout if capture else None


def content_path(value):
    resolved = Path(value).expanduser().resolve()
    if resolved == Path("/content") or Path("/content") not in resolved.parents:
        raise SystemExit(f"Use a dedicated directory below /content, not {resolved}")
    return resolved


def ensure_hf_token():
    """Runtime-only auth; never persist tokens with login() or Git credentials."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        if not sys.stdin.isatty():
            raise SystemExit("HF_TOKEN is required. Run the notebook's hidden token-input cell before setup.")
        token = getpass.getpass("Hugging Face read token (hidden, runtime only): ").strip()
    if not token:
        raise SystemExit("An authorized HF token is required for DINOv3")
    os.environ["HF_TOKEN"] = token
    return token


def prepare_source(args):
    """Clone the requested fork into Colab; never overwrite an existing checkout."""
    repo = content_path(args.repo)
    if not repo.exists():
        run(["git", "clone", "--depth", "1", "--branch", args.branch, "--single-branch", args.repo_url, repo])
    source = dict(repo_url=args.repo_url, requested_branch=args.branch, directory=str(repo))
    if (repo / ".git").exists():
        origin = run(["git", "remote", "get-url", "origin"], cwd=repo, capture=True).strip()
        if origin.rstrip("/").removesuffix(".git") != args.repo_url.rstrip("/").removesuffix(".git"):
            raise SystemExit("Existing checkout has a different origin; choose a fresh --repo directory or matching --repo-url.")
        branch = run(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo, capture=True).strip()
        if branch != args.branch:
            raise SystemExit(f"Existing checkout is on {branch!r}; expected {args.branch!r}. Choose another --repo directory.")
        source["commit"] = run(["git", "rev-parse", "HEAD"], cwd=repo, capture=True).strip()
        print(f"Using {args.repo_url}, branch {branch}, commit {source['commit']}. Existing checkout is not reset or pulled.")
    for relative in (
        "requirements.txt", "moss_image_train_style.py", "library/moss_image_style_train_utils.py", "library/streaming_dataset.py",
        "library/moss_image_style_pretrained.py", "configs/moss_style_colab.json",
    ):
        if not (repo / relative).is_file():
            raise SystemExit(f"Missing {relative}. Commit and push the current MOSS code to {args.repo_url} on {args.branch} first.")
    return repo, source


def prepare_environment(args, repo):
    bootstrap = Path("/content/.moss-uv-bootstrap")
    if not (bootstrap / "bin/python").exists():
        run([sys.executable, "-m", "venv", "--without-pip", bootstrap])
    uv = bootstrap / "bin/uv"
    if not uv.exists():
        run([sys.executable, "-m", "pip", "--python", bootstrap / "bin/python", "install", "uv"])
    run([uv, "--version"])
    environment = content_path(args.venv)
    python = environment / "bin/python"
    if not python.exists():
        run([uv, "venv", "--python", "3.12", environment])

    # Match the tested torch/vision pair. uv selects a wheel backend for the
    # Colab driver's capabilities; subsequent installs retain this exact pair.
    constraints = environment / "moss-torch-constraints.txt"
    constraints.write_text("torch==2.10.0\ntorchvision==0.25.0\n", encoding="utf-8")
    run([uv, "pip", "install", "--python", python, "--torch-backend", args.torch_backend,
         "torch==2.10.0", "torchvision==0.25.0"])
    run([uv, "pip", "install", "--python", python, "--constraint", constraints,
         "--requirement", repo / "requirements.txt", "timm==1.0.27"], cwd=repo)
    run([uv, "pip", "check", "--python", python])
    versions = run([uv, "pip", "freeze", "--python", python], capture=True)
    (environment / "moss-installed-packages.txt").write_text(versions, encoding="utf-8")
    report = run([python, "-c", (
        "import json, platform, torch, torchvision, transformers, diffusers, timm; "
        "ok=torch.cuda.is_available(); "
        "print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__, "
        "torchvision=torchvision.__version__, transformers=transformers.__version__, "
        "diffusers=diffusers.__version__, timm=timm.__version__, cuda_available=ok, "
        "gpu=torch.cuda.get_device_name(0) if ok else None, "
        "vram_gib=torch.cuda.get_device_properties(0).total_memory/1024**3 if ok else None, "
        "bf16=torch.cuda.is_bf16_supported() if ok else False), indent=2))"
    )], capture=True)
    (environment / "moss-runtime.json").write_text(report, encoding="utf-8")
    print(report)
    if not json.loads(report)["cuda_available"] and not args.skip_models:
        raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime before downloading training assets.")
    print("The installed environment has not been used for training or model loading.")
    return python


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_assets(args):
    # This phase runs under the isolated venv so imports never mutate Colab's
    # notebook kernel dependencies. Hugging Face credentials are inherited only.
    from huggingface_hub import HfApi, hf_hub_url, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    root = content_path(args.models_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    profiles = manifest.get("text_encoder_profiles", {})
    if args.text_encoder not in profiles:
        raise ValueError(f"No supported text-encoder download profile for {args.text_encoder!r}")
    model_specs = [*manifest["models"], profiles[args.text_encoder]]
    lock_path = root / "resolved_models.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    token = ensure_hf_token()
    api = HfApi(token=token)
    # model_info can expose public metadata for a gated repo. auth_check checks
    # actual read permission, before any large checkpoint transfer is attempted.
    content_model = next(spec for spec in model_specs if spec["name"] == "content_encoder")
    try:
        api.whoami()
        api.auth_check(repo_id=content_model["repo_id"], repo_type="model")
    except GatedRepoError:
        raise SystemExit(
            "DINOv3 access is not approved for this token's account. Request/accept access on the model page, "
            "then use a token with read permission. No model weights have been downloaded."
        ) from None
    except HfHubHTTPError as error:
        status = getattr(error.response, "status_code", "unknown")
        raise SystemExit(
            f"HF authentication/access check failed (HTTP {status}). Check token validity, read scope, and DINOv3 approval. "
            "Replace it through the hidden notebook prompt; no model weights have been downloaded."
        ) from None
    print("Hugging Face authentication and DINOv3 read access confirmed.")
    plan = []
    missing_bytes = 0
    for spec in model_specs:
        previous = lock.get(spec["name"], {})
        same_source = previous.get("repo_id") == spec["repo_id"] and previous.get("requested_revision") == spec["revision"]
        revision = previous["revision"] if same_source else spec["revision"]
        info = api.model_info(spec["repo_id"], revision=revision, files_metadata=True)
        selected = [file for file in info.siblings if any(fnmatch.fnmatchcase(file.rfilename, pattern) for pattern in spec["allow_patterns"])]
        if not selected or any(file.size is None for file in selected):
            raise RuntimeError(f"Cannot establish selected download sizes for {spec['repo_id']}")
        for pattern in spec.get("required_patterns", []):
            if not any(fnmatch.fnmatchcase(file.rfilename, pattern) for file in selected):
                raise RuntimeError(f"Required asset pattern {pattern!r} is absent from {spec['repo_id']}")
        directory = root / spec["name"]
        for file in selected:
            relative = Path(file.rfilename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unexpected path returned by Hub")
            destination = directory / relative
            if not destination.exists() or destination.stat().st_size != file.size:
                missing_bytes += file.size
        entry = dict(repo_id=spec["repo_id"], requested_revision=spec["revision"], revision=info.sha,
                     local_dir=str(directory), allow_patterns=spec["allow_patterns"],
                     files=[dict(path=file.rfilename, bytes=file.size) for file in selected])
        if "load_dtype" in spec:
            entry["load_dtype"] = spec["load_dtype"]
        for name in ("format", "loader", "loader_kwargs"):
            if name in spec:
                entry[name] = spec[name]
        lock[spec["name"]] = entry
        plan.append((spec["name"], entry))
        print(f"{spec['name']}: {sum(file.size for file in selected) / 1024**3:.2f} GiB, commit {info.sha}")

    free = shutil.disk_usage(root).free
    required = int(missing_bytes * 1.1) + 2 * 1024**3
    if free < required:
        raise RuntimeError(f"Insufficient disk: {free / 1024**3:.1f} GiB free; allow at least {required / 1024**3:.1f} GiB")
    # Save immutable revisions before transfers so a retry resumes the same assets.
    write_json(lock_path, lock)
    for name, entry in plan:
        print(f"Downloading {name} (weights remain on disk; they are not loaded).", flush=True)
        snapshot_download(repo_id=entry["repo_id"], revision=entry["revision"], local_dir=entry["local_dir"],
                          allow_patterns=entry["allow_patterns"], max_workers=2, token=token)
        for file in entry["files"]:
            destination = Path(entry["local_dir"]) / file["path"]
            if not destination.is_file() or destination.stat().st_size != file["bytes"]:
                raise RuntimeError(f"Incomplete asset: {destination}")

    # Only list dataset shards. Never request their bodies or the multi-TB dataset.
    spec = manifest["dataset"]
    dataset_lock = root / "dataset_stream.json"
    previous = json.loads(dataset_lock.read_text(encoding="utf-8")) if dataset_lock.exists() else {}
    revision = previous.get("revision", spec["revision"]) if previous.get("repo_id") == spec["repo_id"] else spec["revision"]
    info = api.dataset_info(spec["repo_id"], revision=revision)
    shards = []
    for file in api.list_repo_tree(spec["repo_id"], repo_type="dataset", revision=info.sha,
                                   path_in_repo=spec["directory"], recursive=False):
        if file.path.endswith((".tar", ".tar.gz")):
            shards.append(hf_hub_url(spec["repo_id"], file.path, repo_type="dataset", revision=info.sha))
    if not shards:
        raise RuntimeError("No dataset tar shards found")
    write_json(dataset_lock, dict(repo_id=spec["repo_id"], revision=info.sha, shards=sorted(shards)))
    print(f"Ready: {lock_path}; {len(shards)} shard URLs listed in {dataset_lock}. No dataset images downloaded.")
    print("The tagger checkpoint is unchanged on disk; its BF16 load preference is recorded in the manifest.")
    print(f"Text encoder profile: {args.text_encoder}. Loader settings are recorded, not executed.")


def main():
    require_colab()  # Before directory creation, package installation, or downloads.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/content/sd-scripts")
    parser.add_argument("--repo-url", default="https://github.com/TheDucker1/sd-scripts")
    parser.add_argument("--branch", default="style_loss")
    parser.add_argument("--venv", default="/content/moss-venv")
    parser.add_argument("--models-dir", default="/content/moss-models")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--torch-backend", default="auto", help="uv PyTorch wheel backend, e.g. auto or cu128")
    parser.add_argument("--text-encoder", choices=["int8_convrot", "bf16"], default="bf16")
    parser.add_argument("--skip-models", action="store_true", help="Prepare only the isolated environment")
    parser.add_argument("--assets-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.assets_only:
        args.manifest = args.manifest or str(Path(args.repo) / "configs/moss_colab_models.json")
        prepare_assets(args)
        return
    repo, source = prepare_source(args)
    if not args.skip_models:
        ensure_hf_token()
    args.manifest = str(Path(args.manifest).resolve()) if args.manifest else str(repo / "configs/moss_colab_models.json")
    print(f"Colab host: {platform.platform()}; source: {repo}")
    python = prepare_environment(args, repo)
    write_json(content_path(args.venv) / "moss-source.json", source)
    if not args.skip_models:
        # Keep auth in HF_TOKEN/Colab Secrets; no token command-line arguments.
        run([python, Path(__file__).resolve(), "--assets-only", "--repo", repo,
             "--venv", args.venv, "--models-dir", args.models_dir, "--manifest", args.manifest,
             "--text-encoder", args.text_encoder],
            environment=dict(os.environ, _MOSS_COLAB_PARENT_VERIFIED="1"))
    print(f"Environment Python: {python}")
    print("Preparation complete. Use the notebook's training cells to load local assets and launch training.")


if __name__ == "__main__":
    main()
