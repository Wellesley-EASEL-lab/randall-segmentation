# Randall marginal-image segmentation artifact

This artifact provides the computational pipeline described in the paper:

- **Multiclass page segmentation:** preparation, training, evaluation, and
  inference for MapSAM-multiclass-final.
- **Marginal-image extraction:** conversion of page masks into candidate crops.
- **Two-stage human review:** interfaces for individual and clustered review.
- **Review acceleration:** classifier training, embeddings, and K-means
  clustering.

Together, these components support inspection and reproduction of every
computational stage between labeled manuscript pages and the final reviewed
collection of marginal images. The workflow follows the pipeline figure in the
paper and the sections **Segmenting Medieval Manuscripts** and **Identifying
Marginal Images**.

## Installation

Install PyTorch, torchvision, and this package:

```bash
python -m pip install torch torchvision
python -m pip install -e .
```

Run all commands from the repository root.

## 1. Prepare segmentation data

Prepare the labeled pages described in **Building the Labeled Dataset of
Pages**:

```bash
python -m scripts.preprocess_segmentation_data \
  --dataset-repo organization/segmentation-dataset \
  --split train \
  --output-dir outputs/prepared
```

Run the command for each required split.

## 2. Train and evaluate the segmentation model

Train MapSAM-multiclass-final as described in **Training an Image Segmentation
Model**:

```bash
python -m scripts.train_segmentation \
  --root_path outputs/prepared \
  --ckpt path/to/sam_vit_l_0b3195.pth \
  --output outputs/segmentation-training \
  --max_epochs 125
```

Evaluate the resulting checkpoint:

```bash
python -m scripts.evaluate_segmentation \
  --prepared-root outputs/prepared \
  --split test \
  --base-checkpoint path/to/sam_vit_l_0b3195.pth \
  --adapter-checkpoint path/to/model.pth \
  --config path/to/config.json \
  --id2label path/to/id2label.json \
  --output-dir outputs/segmentation-evaluation
```

## 3. Segment pages and extract candidates

Run the selected segmentation model over manuscript pages:

```bash
python -m scripts.predict_segmentations \
  --input_dir path/to/pages \
  --recursive \
  --ckpt path/to/sam_vit_l_0b3195.pth \
  --dora_ckpt path/to/model.pth \
  --config path/to/config.json \
  --id2label path/to/id2label.json \
  --output_dir outputs/predictions
```

This writes reassembled segmentation masks plus `predictions.jsonl`.

Following **Extracting Marginal Segments**, form boxes around connected
marginal-image regions, merge nearby regions, and filter very small candidates:

```bash
python -m scripts.extract_marginals \
  --predictions-jsonl outputs/predictions/predictions.jsonl \
  --output-dir outputs/extracted \
  --min-fraction 0.0005 \
  --merge
```

This writes cropped candidate images and `marginal_inventory.csv`.

## 4. Review individual candidates

For the unaided review stage in **Identifying Marginal Images**, initialize the
database, launch the interface, and export candidate decisions:

```bash
python -m scripts.review_marginals init-db \
  --db outputs/review/reviews.sqlite3 --inventory outputs/extracted/marginal_inventory.csv

python -m scripts.review_marginals serve \
  --db outputs/review/reviews.sqlite3 --asset-root outputs/extracted

python -m scripts.review_marginals export \
  --db outputs/review/reviews.sqlite3 --output-csv outputs/review/reviews.csv
```

## 5. Train the review classifier

Assign reviewed rows to `train`, `validation`, or `test`, then build the
classifier dataset:

```bash
python -m scripts.build_classifier_dataset \
  --reviews-csv outputs/review/reviews-with-splits.csv \
  --asset-root outputs/extracted \
  --output-dir outputs/classifier-dataset
```

Following **Training a Binary Classifier**, train a DINOv3 classifier for seven
linear-probe epochs and eight full fine-tuning epochs:

```bash
python -m scripts.train_classifier \
  --dataset outputs/classifier-dataset \
  --output outputs/classifier \
  --pooling cls \
  --epochs 7 \
  --fine-tune \
  --fine-tune-epochs 8
```

## 6. Cluster and batch-review candidates

Following **Expediting Manual Review**, embed the candidates and cluster them by
visual similarity:

```bash
python -m scripts.extract_embeddings \
  --dataset outputs/classifier-dataset \
  --checkpoint outputs/classifier/checkpoint-best \
  --output-prefix outputs/embeddings/marginals

python -m scripts.cluster_embeddings \
  --embeddings outputs/embeddings/marginals.npy \
  --index-csv outputs/embeddings/marginals.csv \
  --output-dir outputs/clusters \
  --n-clusters 200
```

Initialize the cluster-review interface from the latest review export so that
earlier decisions carry into batch review:

```bash
python -m scripts.review_clusters init-db \
  --db outputs/cluster-review/reviews.sqlite3 \
  --inventory outputs/review/reviews-with-splits.csv \
  --assignments-csv outputs/clusters/cluster_assignments.csv \
  --summaries-csv outputs/clusters/cluster_summaries.csv

python -m scripts.review_clusters serve \
  --db outputs/cluster-review/reviews.sqlite3 --asset-root outputs/extracted

python -m scripts.review_clusters export \
  --db outputs/cluster-review/reviews.sqlite3 --output-csv outputs/cluster-review/reviews.csv
```

The interface groups candidates by cluster and displays majority-label review
suggestions.

The repository is released under the MIT License. The retained MapSAM and
Segment Anything notices are listed in `THIRD_PARTY_NOTICES.md`.
