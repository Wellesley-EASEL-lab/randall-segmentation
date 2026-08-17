# Randall marginal-image segmentation

This repository is the selective public code artifact for a research pipeline that segments and extracts decorative features from medieval manuscripts. 
It is targeted to manuscripts indexed by Lilian Randall's *Images in the Margins of Gothic Manuscripts* (1966). 
It contains a public workflow to segment pages, extract marginal images, and 
manually review candidate marginal extracts. The image segmentation is 
done by a custom model that builds on SAM (Kirillov et al. 2023) and MapSAM (Xia et al. 2025).
This repository contains code to support the full pipeline: segmentation-data preparation, MapSAM training and inference, full-page retiling, marginal extraction, two review applications, binary classification, embedding extraction, and clustering.

> **Release status (August 17, 2026):**
> `wellesley-easel/randall-predicted-segmentations` and
> `wellesley-easel/randall-segmentation-model` are not yet publicly readable.
> Their public release, suitable model redistribution terms, removal of the
> model card's private-draft warning, and an updated source-code link are
> external publication prerequisites. The other two datasets are public.

## Public resources and image rights

The supported Hugging Face resources are:

| Repository | Contents | Source images included? |
| --- | --- | --- |
| [`wellesley-easel/randall-open-segmentations`](https://huggingface.co/datasets/wellesley-easel/randall-open-segmentations) | Open image/mask pairs | Yes |
| [`wellesley-easel/randall-segmentation-masks`](https://huggingface.co/datasets/wellesley-easel/randall-segmentation-masks) | Additional ground-truth masks | No |
| [`wellesley-easel/randall-predicted-segmentations`](https://huggingface.co/datasets/wellesley-easel/randall-predicted-segmentations) | Full-page predicted masks | No |
| [`wellesley-easel/randall-segmentation-model`](https://huggingface.co/wellesley-easel/randall-segmentation-model) | DoRA adapter, configuration, labels | Not applicable |

The masks-only and predicted datasets do not grant access to their source
images. You must obtain those images independently, determine that your use is
permitted, and provide an explicit manifest. Dataset and model revisions are
configurable; resolved dataset revisions are recorded in run metadata. Pin
immutable revisions when archiving a paper release.

The public data are not the complete paper corpus. Consequently, this code can
run a methodological reproduction, but it cannot exactly regenerate the paper
checkpoint, extracted marginal corpus, review decisions, or classifier. No
private review labels are released.

## Installation

Python 3.10–3.12 is supported. Install PyTorch and torchvision separately so
that the build matches the local CPU/CUDA platform; follow the selector at
<https://pytorch.org/get-started/locally/>. A typical CUDA environment is:

```bash
conda create -n randall-public python=3.12 -y
conda activate randall-public
# Install the appropriate torch and torchvision build first.
python -m pip install -e .
```

For tests and release checks, use `python -m pip install -e '.[dev]'` after the
PyTorch installation. (`requirements-dev.txt` is also provided for tooling.)
Model training, model evaluation, page prediction, classifier training, and
embedding extraction normally require a CUDA GPU. Preprocessing, marginal
extraction, review applications, clustering, tests, and release auditing run on
CPU.

Meta's SAM ViT-L base checkpoint is not bundled in the hosted adapter. Download
it separately under Meta's terms and pass its path explicitly. The loader
rejects hosted configurations that are not the expected modified ViT-L,
eight-class, rank-8 architecture.

All commands below are run from the repository root using module syntax. Every
script also has `--help`.

## Open-data quickstart

Prepare a public split, evaluate the hosted adapter, predict local pages, and
extract marginal-image candidates:

```bash
python -m scripts.preprocess_segmentation_data \
  --dataset-repo wellesley-easel/randall-open-segmentations \
  --split train --output-dir outputs/prepared

python -m scripts.evaluate_segmentation \
  --dataset-repo wellesley-easel/randall-open-segmentations \
  --split test --base-checkpoint /path/to/sam_vit_l_0b3195.pth \
  --output-dir outputs/segmentation-evaluation

python -m scripts.predict_segmentations \
  --input_dir licensed-pages --recursive \
  --ckpt /path/to/sam_vit_l_0b3195.pth \
  --output_dir outputs/predictions

python -m scripts.extract_marginals \
  --predictions-jsonl outputs/predictions/predictions.jsonl \
  --output-dir outputs/extracted
```

Until the model repository is public, the evaluation and prediction commands
must instead receive local `--adapter-checkpoint`/`--dora_ckpt`, `--config`, and
`--id2label` files. Never commit those local artifacts here.

## Shared data contracts

Masks use integer class IDs 0–7: background, line filler, marginal images,
painted initials, miniatures, borders, pen flourishing, and pen-flourished
texts. RGB masks are recognized using the palette in
`randall_pipeline/constants.py`; marginal images are class 2.

The masks-only join manifest is CSV with unique rows:

```csv
uuid,image_path
dataset-uuid-1,licensed-images/page-1.tif
```

The predicted-segmentations join uses a unique composite key:

```csv
manuscript,archive,page,image_path
MS-1,Archive,12r,licensed-images/MS-1/12r.tif
```

Relative image paths resolve relative to the manifest. Missing files,
duplicate keys, unmatched manifest rows, and image/mask dimension differences
are fatal.

Marginal inventories and both review exports use this canonical schema:

```text
asset_id,filename,page,manuscript,archive,location,source_path,
segmentation_path,split,checked,include,query,notes
```

`asset_id` is a deterministic content-independent identifier. Paths are
relative to a declared artifact root whenever possible. Review applications
reject paths escaping that root. CSV exports are atomic, notes are escaped by
the UI frameworks, and the SQLite databases are newly initialized local state,
not release artifacts.

## Preprocessing

`preprocess_segmentation_data.py` normalizes a supported dataset to the training
layout. Pages whose largest dimension exceeds 1024 pixels are divided in
row-major order into the paper's 3-row × 2-column grid. Odd remainder pixels
are retained in the last row/column. Every tile is square-padded and resized to
1024; masks use nearest-neighbor interpolation so class IDs are preserved.

```bash
python -m scripts.preprocess_segmentation_data \
  --dataset-repo wellesley-easel/randall-segmentation-masks \
  --image-manifest manifests/masks-images.csv \
  --revision DATASET_COMMIT --split validation \
  --output-dir outputs/prepared
```

Inputs are a supported Hub split and, where required, an image manifest.
Outputs are `<split>/*.tif`, `annotation/<split>/*.tif`, `retiling.csv`,
`id2label.json`, and `preprocessing_run_<split>.json`. Run the command once per desired
split into the same output root. Use `--filter-background` only when explicitly
discarding all-background tiles. Failures usually indicate an unsupported Hub
schema, invalid class IDs, duplicate generated tile names, or a bad join.

## Image segmentation model training

`train_segmentation.py` contains the modified SAM/MapSAM architecture and full
eight-class DoRA training loop. The preserved paper recipe is: 1024-pixel
inputs, SAM ViT-L, rank-8 DoRA, depthwise self-prompt generators, batch size 6,
AdamW at `1e-4`, 750-step warmup, Dice parameter `0.6`, hint-loss weight `3.0`,
and seed 1234.

```bash
python -m scripts.train_segmentation \
  --root_path outputs/prepared \
  --ckpt /path/to/sam_vit_l_0b3195.pth \
  --output outputs/segmentation-training \
  --max_epochs 160
```

The prepared root must contain explicit training/test data expected by
the loader and a compatible `id2label.json`. Checkpoints, TensorBoard logs, and
training configuration are written below `--output`. The paper's hosted adapter
was selected after epoch 120 from a run configured for 160 epochs; public or
user-supplied data will not reproduce that model exactly. This is a large
ViT-L job: CUDA is required, and out-of-memory errors should be handled by
reducing batch size rather than changing image size or architecture when
reproducing the method.

## Evaluation

`evaluate_segmentation.py` supports either a Hub dataset directly or prepared
tiles via `--prepared-root`. Predictions and targets are reassembled at page
resolution before metrics are accumulated.

```bash
python -m scripts.evaluate_segmentation \
  --prepared-root outputs/prepared --split test \
  --base-checkpoint /path/to/sam_vit_l_0b3195.pth \
  --output-dir outputs/evaluation --write-parquet
```

The direct masks-only form adds `--dataset-repo` and `--image-manifest` as in
preprocessing. Outputs are `metrics.json`, `confusion_matrix.png`, and optional
`predictions.parquet`. Metrics include per-class IoU, Dice, precision, recall,
overall accuracy, and foreground accuracy. GPU memory, an unavailable private
model, a mismatched SAM base checkpoint, or missing join images are the common
failure modes.

## Image segmentation

`predict_segmentations.py` discovers JPEG, PNG, and TIFF inputs, optionally
recursively, batches model tiles across pages, and automatically retiles each
prediction to the original dimensions.

```bash
python -m scripts.predict_segmentations \
  --input_dir licensed-pages --recursive \
  --output_dir outputs/predictions \
  --ckpt /path/to/sam_vit_l_0b3195.pth \
  --infer_batch_size 12
```

It downloads `model.pth`, `config.json`, and `id2label.json` from the hosted
model by default; local overrides are `--dora_ckpt`, `--config`, and
`--id2label`. Outputs are class-ID PNG masks, color metadata, and portable
`predictions.jsonl` records with paths relative to the prediction root. Input
names are prefixed by a stable relative-path ID to prevent recursive-tree name
collisions. Inference requires CUDA and a compatible ViT-L base checkpoint.

## Marginal extraction

`extract_marginals.py` finds connected regions of class 2 and writes clean
crops plus `marginal_inventory.csv`. It accepts local prediction JSONL or any
supported Hub dataset.

```bash
python -m scripts.extract_marginals \
  --dataset-repo wellesley-easel/randall-predicted-segmentations \
  --image-manifest manifests/predicted-images.csv \
  --split train --output-dir outputs/extracted \
  --margin 0.05 --min-fraction 0.005
```

`--class-id` changes the extracted class. `--merge` and
`--merge-tolerance` combine nearby boxes; `--draw-bbox` creates diagnostic
crops rather than the clean default. Source images and masks are materialized
inside the output for a portable local review artifact. Empty outputs generally
mean the selected class is absent or `--min-fraction` is too high.

## Manual reviewing

The unassisted Gradio application has explicit initialize, serve, and export
steps. The database must not already exist unless `--reset` is deliberately
used.

```bash
python -m scripts.review_marginals init-db \
  --db outputs/review/reviews.sqlite3 \
  --inventory outputs/extracted/marginal_inventory.csv
python -m scripts.review_marginals serve \
  --db outputs/review/reviews.sqlite3 --asset-root outputs/extracted
python -m scripts.review_marginals export \
  --db outputs/review/reviews.sqlite3 \
  --output-csv outputs/review/reviews.csv
```

The server binds only to `127.0.0.1` by default. Reviewers set `checked`,
`include`, `query`, and per-image `notes`. Do not expose the app to a network
without adding authentication and transport security. A path traversal or an
inventory referring outside `--asset-root` is rejected.

## Binary classifier training

First edit or programmatically assign every labeled review row to `train`,
`validation`, or `test`, then build the Hugging Face `DatasetDict`:

```bash
python -m scripts.build_classifier_dataset \
  --reviews-csv outputs/review/reviews-with-splits.csv \
  --asset-root outputs/extracted \
  --output-dir outputs/classifier-dataset

python -m scripts.train_classifier \
  --dataset outputs/classifier-dataset \
  --output outputs/classifier --pooling cls

python -m scripts.evaluate_classifier \
  --dataset outputs/classifier-dataset \
  --checkpoint outputs/classifier/checkpoint-best \
  --split test --by-manuscript
```

Only rows with `checked=true` and `query=false` become labels; `include` is the
binary target. Unchecked or queried rows form the unlabeled split for later
clustering. The builder rejects missing/unknown labeled splits, empty required
splits, a training split lacking either class, missing assets, and manuscript
overlap across splits. The default
classifier is `facebook/dinov3-vits16-pretrain-lvd1689m`. Training first fits a
linear probe; `--fine-tune` optionally adds a backbone phase. The checkpoint
persists the chosen `cls`, `patch-mean`, or `cls-patch-mean` pooling mode.

Classifier training generally needs CUDA and downloads DINOv3 on first use.
Evaluation writes JSON metrics and can write per-image CSV predictions.

## Embedding extraction

`extract_embeddings.py` processes labeled and unlabeled splits using the
checkpoint's saved pooling method—not an unconditional raw CLS token.

```bash
python -m scripts.extract_embeddings \
  --dataset outputs/classifier-dataset \
  --checkpoint outputs/classifier/checkpoint-best \
  --output-prefix outputs/embeddings/marginals
```

Outputs are a NumPy matrix, a row-aligned CSV index, and JSON metadata recording
the model and pooling method. A mismatched checkpoint/config or insufficient
GPU memory is fatal; reduce batch size for the latter.

## Clustering

`cluster_embeddings.py` runs deterministic K-means with 25 clusters and seed
1234 by default. Majority labels are computed only from already labeled members
and are advisory.

```bash
python -m scripts.cluster_embeddings \
  --embeddings outputs/embeddings/marginals.npy \
  --index-csv outputs/embeddings/marginals.csv \
  --output-dir outputs/clusters
```

Outputs are `cluster_assignments.csv`, `cluster_summaries.csv`, and
`clustering_metrics.json`. The requested cluster count cannot exceed the number
of rows.

## Assisted manual reviewing

The Flask application groups candidates by cluster and displays labeled-cluster
majorities as suggestions. Suggestions are never written as human decisions.

```bash
python -m scripts.review_clusters init-db \
  --db outputs/cluster-review/reviews.sqlite3 \
  --inventory outputs/review/reviews-with-splits.csv \
  --assignments-csv outputs/clusters/cluster_assignments.csv \
  --summaries-csv outputs/clusters/cluster_summaries.csv
python -m scripts.review_clusters serve \
  --db outputs/cluster-review/reviews.sqlite3 \
  --asset-root outputs/extracted
python -m scripts.review_clusters export \
  --db outputs/cluster-review/reviews.sqlite3 \
  --output-csv outputs/cluster-review/reviews.csv
```

The server is localhost-only by default. Human changes to `checked`, `include`,
`query`, and `notes` are authoritative and survive export/import round trips.
Initialize from the latest canonical review export (including its split and
notes fields), not the original unreviewed inventory, to carry prior decisions
into assisted review.

## Testing and release checks

The CPU suite uses only synthetic imagery:

```bash
python -m pytest -q
python -m compileall -q randall_pipeline scripts segment_anything
python -m scripts.audit_public_release
```

Immediately before publication, also verify that all Hub dependencies are
public and that the model card declares redistribution terms:

```bash
python -m scripts.audit_public_release --check-hub
```

The audit rejects tracked databases, model weights, arrays, Parquet outputs,
review/inventory exports, manuscript-image formats, absolute machine paths, and
references to internal source repositories. CI runs supported Python versions,
tests, import/compile checks, README command checks, and the local audit.

A manual GPU acceptance run should execute preprocessing on a small open split,
then run `evaluate_segmentation.py`, `predict_segmentations.py`, a one-epoch
segmentation training smoke test, classifier training/evaluation, and embedding
extraction. Full ViT-L numerical acceptance is intentionally outside CPU CI.

## Reproducibility and provenance

Shared helpers under `randall_pipeline/` define the tiling, retiling, Hub
adapters, metric calculations, path validation, review schema, and classifier
pooling contract. The modified Segment Anything implementation is vendored so
the hosted adapter has no private runtime dependency. Implementation and source
provenance decisions are recorded in `agent-history.txt`.

The repository is MIT licensed. MapSAM's MIT notice and the full Segment
Anything Apache-2.0 license are retained under `licenses/`; see
`THIRD_PARTY_NOTICES.md`.

## Citation

Paper citation forthcoming. 
