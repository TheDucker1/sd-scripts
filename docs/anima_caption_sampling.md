# Variable Anima tag retention

Add this top-level setting to an Anima training config:

```json
"tag_retention": {
  "distribution": "beta",
  "alpha": 5.0,
  "beta": 2.0
}
```

Each training prompt independently samples a keep probability from Beta(5, 2).
The density peaks at 0.8, with a continuous tail toward zero and one. Its mean is
5/7 (about 71.4%); the peak and mean are different because this curve gives short
prompts more room than a narrowly concentrated 80% distribution. To narrow it
while retaining an 80% peak, use alpha 9 and beta 3 (mean 75%).

The sampled probability replaces the subset's fixed `caption_tag_dropout_rate`
for that prompt. The existing caption processor independently retains each
flexible tag with this probability, so the actual fraction varies around it.
Short prompts can contain zero or all tags; there is no minimum-tag clamp.
Caption shuffling, separators, protected `keep_tokens` and prefix/suffix rules
continue to use the existing caption processor. Protected tags are unaffected
by tag dropout, so the fraction of all tags retained can exceed the sampled
probability when protected tags are configured.

Both online tagger captions and cached tags use this processing. The cached
caption files remain complete and unchanged, so neither tagging nor VAE caching
needs to be repeated. Decisions use the existing checkpointed prompt RNG.
Sampling runs before the text encoder and is not rerun during backward
recomputation. Training previews retain all tags, as before.

The default Colab Anima config enables this curve for new runs. The local
[rank-32 tag-retention profile](../configs/anima_style_svd_r32_tag_retention_local.json)
copies the previous high-noise rank-32 recipe and enables the curve. Replace its
dataset, cache and initializer paths as needed, and use a separate output
directory for a new run. The previously saved experiment profiles are unchanged.

Omit `tag_retention` or set `{"enabled": false}` to retain legacy fixed dropout.
Changing the setting on an existing run is a training-recipe change; strict
checkpoint config matching still applies. A weight-only warm start in a new
output directory uses the new setting with a fresh optimizer.
