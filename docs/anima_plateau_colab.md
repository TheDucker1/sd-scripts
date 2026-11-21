# Anima refinement from step 20,000

This phase initializes both LoRA branches and the reference-mode embedding from
the native `step-020000` checkpoint. It starts a fresh optimizer and scheduler.
The new run has 20,000 additional updates, a starting learning rate of `3e-5`,
batch size 1 and gradient accumulation 2. The frozen base stays BF16 and the
trainable adapter stays FP32.

The `soft_plateau` sampler uses center `0.5`, width `0.055` and a uniform
component of `0.05`. Its sample allocation is approximately 9.7% below 50%
noise, 32.0% at 50–70%, 38.7% at 70–90%, and 19.5% at 90–100%.

## Prepare the new weights on Windows

Push the updated `style_loss` branch. Run the preparation helper in CMD from
the repository root to pack the trained adapter using CPU and disk I/O.

```cmd
cd /d C:\Users\d\playground\sd-scripts
"C:\Users\d\.conda\envs\ml\python.exe" tools\prepare_anima_colab_upload.py
```

The default source checkpoint is
`work/anima-snapref-v2a-512/run-high-noise/step-020000`. Override `--checkpoint`
or `--output` when needed.

Preparation verifies the source adapter checksum and packs the two adapter
initialization files. The matching adapter ZIP is reused when the command is
rerun. The notebook clones or fast-forwards the Git branch before setup.

## Upload to Drive

Upload `init-step-020000.zip` from `work/anima-v2a-plateau-upload` into the new
phase's input folder. The notebook uses the existing dataset folder.

```text
MyDrive/
  moss-training/
    anima-1000-p10-512-all-paired/
      images.tar
      latents.tar
      tag-cache.json
    anima-v2a-plateau/
      init-step-020000.zip
```

The preparation folder also contains `anima_style_setup.ipynb`. Open that notebook
in Google Colab, select a GPU runtime, and run the cells in order. The initial
guard restricts environment setup to Colab. Setup requests the Hugging Face
token for the runtime and fetches the base model, Qwen, tokenizer and VAE.

The notebook fetches the source from Git and copies the adapter initialization archive, image tar,
latent tar and tags to Colab SSD. Training reads NPZ members from the local
latent tar. The VAE loads temporarily for previews. The image geometry stays
at the cached 512² pixel budget with aspect bucketing.

## Run and continue

The default run name is `anima-v2a-plateau-512`. Outputs are published under
`MyDrive/moss-training/anima-v2a-plateau-512`.

- Preview every 300 updates, with three randomly selected distinct-image pairs
  and 30 denoising steps. Preview pairing is independent of the training pairs.
- Save every 300 updates and at the end. Keep three completed checkpoints after
  the Drive API confirms their sizes and checksums.
- Write logs, TensorBoard events and previews on SSD. A background API uploader
  publishes snapshots every 60 seconds while training continues.
- Initialize this phase from adapter weights at the first launch. Later launches
  restore the latest completed checkpoint of this phase with optimizer,
  scheduler and update count. Dataset sampling uses a fresh seed on resume.

`Drive verified step-XXXXXX` identifies a recoverable checkpoint. Updates after
the last completed checkpoint can be lost on interruption. A new Colab runtime
requires repeating setup and staging before launching again.

Per-block Inductor compilation is enabled. Colab starts with
`CHECKPOINT_BLOCKS=0`. Set it to 28 in the settings cell if the assigned GPU
requires additional activation-memory headroom.

## Local alternative

The local profile uses the same plateau and original step-20,000 initialization.
It retains the 8 GB configuration with CPU text encoding and activation
checkpointing. Run this command when choosing local training.

```cmd
"C:\Users\d\.conda\envs\ml\python.exe" anima_train_style.py ^
  --config "work\anima-snapref-v2a-512\train-plateau.json" ^
  --output_dir "work\anima-snapref-v2a-512\run-plateau" ^
  --device cuda --num_threads 4
```
