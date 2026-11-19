"""Constrained strongest-first merging for marginal-image components."""

import numpy as np

Box = tuple[int, int, int, int]


def bboxes_overlap(a: Box, b: Box) -> bool:
    """Check if two bounding boxes overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def expand_box(
    box: Box, tolerance: int
) -> Box:
    """Expand a box by tolerance pixels on all sides."""
    x1, y1, x2, y2 = box
    return (x1 - tolerance, y1 - tolerance, x2 + tolerance, y2 + tolerance)


def union_boxes(
    a: Box, b: Box
) -> Box:
    """Return the smallest box containing both input boxes."""
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def box_area(box: Box) -> int:
    """Return the area of a bounding box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_intersection_area(
    a: Box, b: Box
) -> int:
    """Return the intersection area of two bounding boxes."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def box_gap(a: Box, b: Box) -> int:
    """Return the largest horizontal or vertical gap between two boxes."""
    horizontal_gap = max(0, a[0] - b[2], b[0] - a[2])
    vertical_gap = max(0, a[1] - b[3], b[1] - a[3])
    return max(horizontal_gap, vertical_gap)


def mask_integral(mask: np.ndarray) -> np.ndarray:
    """Build a padded integral image for constant-time rectangular mask sums."""
    integral = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = mask.cumsum(axis=0, dtype=np.int64).cumsum(
        axis=1, dtype=np.int64
    )
    return integral


def integral_box_sum(integral: np.ndarray, box: Box) -> int:
    """Return the number of mask pixels inside ``box`` from a padded integral image."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(
        integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    )


def bridge_border_fraction(
    a: Box,
    b: Box,
    border_integral: np.ndarray,
) -> float:
    """Return the fraction of the rectangular gap between two boxes labelled border.

    Border pixels are deliberately favorable evidence: unlike background pixels,
    borders commonly join related marginal decorations.
    """
    union = union_boxes(a, b)
    overlap_area = box_intersection_area(a, b)
    bridge_area = box_area(union) - box_area(a) - box_area(b) + overlap_area
    if bridge_area <= 0:
        return 0.0

    border_pixels = (
        integral_box_sum(border_integral, union)
        - integral_box_sum(border_integral, a)
        - integral_box_sum(border_integral, b)
        + integral_box_sum(
            border_integral,
            (
                max(a[0], b[0]),
                max(a[1], b[1]),
                min(a[2], b[2]),
                min(a[3], b[3]),
            ),
        )
    )
    return max(0.0, min(1.0, border_pixels / bridge_area))


def candidate_merge_score(
    a: Box,
    b: Box,
    tolerance_pixels: int,
    border_integral: np.ndarray,
) -> float:
    """Score a candidate component merge, with border-filled bridges favored."""
    intersection = box_intersection_area(a, b)
    smaller_area = min(box_area(a), box_area(b))
    overlap_fraction = intersection / smaller_area if smaller_area else 0.0

    # A pair at the tolerance boundary is weak; an overlapping pair is strongest.
    gap = box_gap(a, b)
    if tolerance_pixels == 0:
        proximity = 1.0
    else:
        proximity = max(0.0, 1.0 - gap / (2 * tolerance_pixels))

    return 2.0 * overlap_fraction + proximity + bridge_border_fraction(
        a, b, border_integral
    )


def merge_is_plausible(
    margin_bbox: Box,
    target_integral: np.ndarray,
    image_shape: tuple[int, int],
    max_area_fraction: float,
    max_dimension_fraction: float,
    min_target_density: float,
) -> bool:
    """Check that a prospective merged output crop is not page-scale or sparse.

    A zero-valued threshold disables its corresponding check.  Checks are made
    against the output (margin) box, since that is the crop ultimately saved.
    """
    image_height, image_width = image_shape
    area = box_area(margin_bbox)
    if area == 0:
        return False

    if max_area_fraction and area / (image_width * image_height) > max_area_fraction:
        return False
    if max_dimension_fraction and (
        (margin_bbox[2] - margin_bbox[0]) / image_width > max_dimension_fraction
        or (margin_bbox[3] - margin_bbox[1]) / image_height > max_dimension_fraction
    ):
        return False
    if min_target_density and (
        integral_box_sum(target_integral, margin_bbox) / area < min_target_density
    ):
        return False
    return True


def merge_boxes_strongest_first(
    boxes: list[tuple[Box, Box]],
    tolerance_pixels: int,
    class_mask: np.ndarray,
    border_mask: np.ndarray,
    max_area_fraction: float,
    max_dimension_fraction: float,
    min_target_density: float,
) -> list[tuple[Box, Box]]:
    """Merge component boxes in confidence order while preventing runaway chains.

    Each eligible original-component pair is an edge.  Edges are processed from
    strongest to weakest, so an oversized proposed group retains its stronger
    internal connections and simply rejects the weaker edge that would extend it.
    This is a constrained maximum-spanning forest rather than the old repeated
    greedy box union.
    """
    if len(boxes) < 2:
        return boxes

    target_integral = mask_integral(class_mask)
    border_integral = mask_integral(border_mask)
    candidate_edges = []
    for left_index, (left_tight, _) in enumerate(boxes):
        for right_index in range(left_index + 1, len(boxes)):
            right_tight, _ = boxes[right_index]
            if not bboxes_overlap(
                expand_box(left_tight, tolerance_pixels),
                expand_box(right_tight, tolerance_pixels),
            ):
                continue
            candidate_edges.append(
                (
                    candidate_merge_score(
                        left_tight, right_tight, tolerance_pixels, border_integral
                    ),
                    left_index,
                    right_index,
                )
            )

    candidate_edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
    parents = list(range(len(boxes)))
    groups: dict[int, tuple[Box, Box, int]] = {
        index: (tight_bbox, margin_bbox, index)
        for index, (tight_bbox, margin_bbox) in enumerate(boxes)
    }

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for _, left_index, right_index in candidate_edges:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root == right_root:
            continue

        left_tight, left_margin, left_first = groups[left_root]
        right_tight, right_margin, right_first = groups[right_root]
        proposed_tight = union_boxes(left_tight, right_tight)
        proposed_margin = union_boxes(left_margin, right_margin)
        # A component already inside an existing output crop cannot make that
        # crop larger. Merge it even when the pre-existing crop exceeds a
        # size limit, to avoid a redundant nested crop.
        preserves_existing_crop = proposed_margin in (left_margin, right_margin)
        if not preserves_existing_crop and not merge_is_plausible(
            proposed_margin,
            target_integral,
            class_mask.shape,
            max_area_fraction,
            max_dimension_fraction,
            min_target_density,
        ):
            continue

        new_root, old_root = sorted((left_root, right_root))
        parents[old_root] = new_root
        groups[new_root] = (
            proposed_tight,
            proposed_margin,
            min(left_first, right_first),
        )
        del groups[old_root]

    return [
        (tight_bbox, margin_bbox)
        for tight_bbox, margin_bbox, _ in sorted(groups.values(), key=lambda group: group[2])
    ]
