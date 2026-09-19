from typing import Any

import numpy as np
from PIL import Image

from proj_1.types.Prediction import BoundingBox, MaskPrediction, ObjectPrediction


class ImagePreprocessingPipeline:
    """Handles image transformation and output postprocessing."""

    boxIdx = (0, 4)
    confidenceIdx = 4
    coeffIdx = (5, 37)
    target_size = (640, 640)  # Default target size for model input

    @staticmethod
    def preprocess(
        image: Image.Image,
        target_shape: list[Any] | tuple[Any, ...] | None = None,
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Converts a PIL image to RGB, resizes, and produces an NCHW float32 tensor."""
        image = image.convert("RGB")
        original_size = image.size  # (width, height)

        # Determine target width and height
        if not target_shape or len(target_shape) != 4:
            raise ValueError(
                "target_shape must be a 4D shape (batch, channels, height, width)"
            )

        # Format (batch, channels, height, width) or (batch, height, width, channels)
        _, _, h, w = target_shape

        # Resize image
        if (w, h) != image.size:
            image = image.resize((w, h), Image.Resampling.BILINEAR)

        # Convert to numpy array and scale to [0, 1]
        arr = np.array(image, dtype=np.float32) / 255.0

        # Transpose HWC -> CHW and add batch dimension (1, C, H, W)  shape become =>  (1, 3, 640, 640)
        tensor = np.transpose(arr, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        return (original_size, tensor)

    @staticmethod
    def _box_to_corners(
        raw_box: np.ndarray | list[float],
    ) -> tuple[float, float, float, float]:
        """Converts [center_x, center_y, width, height] to (x_min, y_min, x_max, y_max)."""
        cx = float(raw_box[0])
        cy = float(raw_box[1])
        w = float(raw_box[2])
        h = float(raw_box[3])
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @staticmethod
    def _scale_bounding_box(
        box: BoundingBox, original_size: tuple[int, int], target_size: tuple[int, int]
    ) -> BoundingBox:
        """Scales the bounding box to the original image size."""
        scale_x = original_size[0] / target_size[0]
        scale_y = original_size[1] / target_size[1]

        box.x_min *= scale_x
        box.x_max *= scale_x
        box.y_min *= scale_y
        box.y_max *= scale_y

        return box

    @staticmethod
    def _scale_mask(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Scales the mask to the original image size."""
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image = mask_image.resize(
            (original_size[0], original_size[1]), Image.Resampling.BILINEAR
        )
        return np.array(mask_image) / 255.0

    @classmethod
    def _extract_coefficients(cls, arr: np.ndarray) -> np.ndarray:
        """Extracts mask coefficients from the raw output."""
        return arr[cls.coeffIdx[0] : cls.coeffIdx[1]]

    @staticmethod
    def _crop_mask(
        mask: np.ndarray,
        box_corners: tuple[float, float, float, float],
        model_size: tuple[int, int],
        proto_size: tuple[int, int],
    ) -> np.ndarray:
        proto_h, proto_w = proto_size
        model_w, model_h = model_size
        scale_x, scale_y = proto_w / model_w, proto_h / model_h

        x1, y1, x2, y2 = box_corners
        px1 = max(0, round(x1 * scale_x))
        py1 = max(0, round(y1 * scale_y))
        px2 = min(proto_w, round(x2 * scale_x))
        py2 = min(proto_h, round(y2 * scale_y))

        cropped = np.zeros_like(mask)
        if px2 > px1 and py2 > py1:
            cropped[py1:py2, px1:px2] = mask[py1:py2, px1:px2]
        return cropped

    @classmethod
    def _extract_mask_prediction(
        cls,
        detection: np.ndarray,
        prototype_mask: np.ndarray,
        original_size: tuple[int, int],
        box_corners: tuple[float, float, float, float],
    ) -> MaskPrediction:
        proto_c, proto_h, proto_w = prototype_mask.shape
        protos_flat = prototype_mask.reshape(proto_c, -1)

        coeffs = cls._extract_coefficients(detection)

        mask = coeffs @ protos_flat
        mask = mask.reshape(proto_h, proto_w)
        mask = 1 / (1 + np.exp(-mask))  # sigmoid

        mask = cls._crop_mask(mask, box_corners, cls.target_size, (proto_h, proto_w))
        mask = cls._scale_mask(mask, original_size)

        return MaskPrediction(mask=mask.tolist())

    @classmethod
    def _extract_confidence(cls, detection: np.ndarray) -> float:
        """Extracts the confidence score from the raw output."""
        return float(detection[cls.confidenceIdx])

    @staticmethod
    def _compute_iou(
        box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
    ) -> float:
        """Calculates Intersection over Union (IoU) between two boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

        union = area1 + area2 - intersection
        return intersection / union if union > 0.0 else 0.0

    @classmethod
    def _nms(
        cls,
        boxes: list[tuple[float, float, float, float]],
        scores: list[float],
        iou_threshold: float = 0.5,
    ) -> list[int]:
        """Performs Non-Maximum Suppression and returns indices of surviving boxes."""
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        kept_indices: list[int] = []

        while sorted_indices:
            current = sorted_indices.pop(0)
            kept_indices.append(current)

            sorted_indices = [
                idx
                for idx in sorted_indices
                if cls._compute_iou(boxes[current], boxes[idx]) < iou_threshold
            ]

        return kept_indices

    @classmethod
    def clip(cls, box: BoundingBox, original_size: tuple[int, int]) -> BoundingBox:
        """Clips the bounding box coordinates to be within the image dimensions."""
        box.x_min = max(0.0, min(float(original_size[0]), box.x_min))
        box.y_min = max(0.0, min(float(original_size[1]), box.y_min))
        box.x_max = max(0.0, min(float(original_size[0]), box.x_max))
        box.y_max = max(0.0, min(float(original_size[1]), box.y_max))
        return box

    @classmethod
    def postprocess(
        cls,
        raw_output: np.ndarray,
        original_size: tuple[int, int],
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> list[ObjectPrediction]:
        """Postprocesses the raw model logits into class IDs, boxes, NMS, and masks."""
        prototype_mask: np.ndarray = raw_output[1].squeeze(0)
        prediction_results: np.ndarray = raw_output[0].squeeze(0)

        candidate_indices: list[int] = []
        candidate_boxes_model_space: list[tuple[float, float, float, float]] = []
        candidate_scores: list[float] = []

        num_anchors = prediction_results.shape[1]
        for i in range(num_anchors):
            detection = prediction_results[:, i]
            score = cls._extract_confidence(detection)
            if score < confidence_threshold:
                continue

            raw_box = detection[cls.boxIdx[0] : cls.boxIdx[1]]
            corners = cls._box_to_corners(raw_box)

            candidate_indices.append(i)
            candidate_boxes_model_space.append(corners)
            candidate_scores.append(score)

        if not candidate_indices:
            return []

        # Run Non-Maximum Suppression to remove duplicates
        surviving_ranks = cls._nms(
            candidate_boxes_model_space, candidate_scores, iou_threshold=iou_threshold
        )

        objects_predicted: list[ObjectPrediction] = []
        for rank in surviving_ranks:
            anchor_idx = candidate_indices[rank]
            score = candidate_scores[rank]
            box_model = candidate_boxes_model_space[rank]
            detection = prediction_results[:, anchor_idx]

            mask_pred = cls._extract_mask_prediction(
                detection, prototype_mask, original_size, box_model
            )

            box_scaled = BoundingBox(
                x_min=box_model[0],
                y_min=box_model[1],
                x_max=box_model[2],
                y_max=box_model[3],
            )

            # Scale bounding box to original image size if it differs from the model's target size
            if original_size != cls.target_size:
                box_scaled = cls._scale_bounding_box(
                    box_scaled, original_size, cls.target_size
                )

            # Clamp coordinates within image bounds
            box_scaled = cls.clip(box_scaled, original_size)

            obj = ObjectPrediction(
                class_id=0,
                confidence=score,
                box=box_scaled,
                mask=mask_pred,
            )
            objects_predicted.append(obj)

        return objects_predicted
