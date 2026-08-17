"""Shared public repository identifiers and label metadata."""

OPEN_SEGMENTATIONS = "wellesley-easel/randall-open-segmentations"
SEGMENTATION_MASKS = "wellesley-easel/randall-segmentation-masks"
PREDICTED_SEGMENTATIONS = "wellesley-easel/randall-predicted-segmentations"
SEGMENTATION_MODEL = "wellesley-easel/randall-segmentation-model"

ID2LABEL = {
    0: "background",
    1: "line filler",
    2: "marginal images",
    3: "painted initials",
    4: "miniatures",
    5: "borders",
    6: "pen flourishing",
    7: "pen flourished texts",
}

PALETTE = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (255, 128, 0),
}

REVIEW_FIELDS = (
    "asset_id", "filename", "page", "manuscript", "archive", "location",
    "source_path", "segmentation_path", "split", "checked", "include",
    "query", "notes",
)
