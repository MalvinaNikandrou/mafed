# Copyright (c) Facebook, Inc. and its affiliates.
import math
from collections.abc import Generator
from enum import IntEnum, unique
from typing import Literal, Union

import numpy as np
import torch
import torchvision.transforms.functional as F

RawBoxType = Union[list[float], tuple[float, ...], torch.Tensor, np.ndarray]  # type: ignore[type-arg]


@unique
class BoxMode(IntEnum):
    """Different ways to represent a box.

    `XYXY_ABS`:     (x0, y0, x1, y1) in absolute floating points coordinates. The coordinates in
                    range [0, width or height].
    `XYWH_ABS`:     (x0, y0, w, h) in absolute floating points coordinates.
    `XYWHA_ABS`:    (xc, yc, w, h, a) in absolute floating points coordinates. (xc, yc) is the
                    center of the rotated box, and the angle a is in degrees CCW.
    """

    XYXY_ABS = 0  # noqa: WPS115
    XYWH_ABS = 1  # noqa: WPS115
    # XYXY_REL = 2
    # XYWH_REL = 3
    XYWHA_ABS = 4  # noqa: WPS115

    @staticmethod
    def convert(box: RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> RawBoxType:  # noqa: WPS602
        """Convert box to a different mode, returning in the same type as provided.

        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5.
            from_mode (BoxMode): Mode to convert from.
            to_mode (BoxMode): Mode to convert to.

        Returns:
            The converted box of the same type.
        """
        box_mode_converter = BoxModeConverter()
        return box_mode_converter(box, from_mode, to_mode)


class BoxModeConverter:
    """Convert a box from one mode to another."""

    def __call__(self, box: RawBoxType, from_mode: BoxMode, to_mode: BoxMode) -> RawBoxType:
        """Do the actual converting.

        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5.
            from_mode (BoxMode): Mode to convert from.
            to_mode (BoxMode): Mode to convert to.

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        # if to_mode in self._unsupported_modes or from_mode in self._unsupported_modes:
        #   raise AssertionError("Relative mode is not supported.")

        original_type = type(box)
        box_as_tensor = self._convert_to_torch(box)

        converted_box = self._convert(box_as_tensor, from_mode, to_mode)

        if isinstance(box, (list, tuple)):
            return original_type(converted_box.flatten().tolist())
        if isinstance(box, np.ndarray):
            return converted_box.numpy()

        return converted_box

    # @property
    # def _unsupported_modes(self) -> list[BoxMode]:
    #   """Get a list of the unsupported modes."""
    #   return [BoxMode.XYXY_REL, BoxMode.XYWH_REL]

    def _convert(self, box: torch.Tensor, from_mode: BoxMode, to_mode: BoxMode) -> torch.Tensor:
        """Convert box to the desired mode if it's supported."""
        convert_functions = {
            BoxMode.XYWHA_ABS: {
                BoxMode.XYXY_ABS: self._convert_xywha_abs_to_xyxy_abs,
            },
            BoxMode.XYWH_ABS: {
                BoxMode.XYWHA_ABS: self._convert_xywh_abs_to_xywha_abs,
                BoxMode.XYXY_ABS: self._convert_xywh_abs_to_xyxy_abs,
            },
            BoxMode.XYXY_ABS: {
                BoxMode.XYWHA_ABS: self._convert_xyxy_abs_to_xywh_abs,
            },
        }

        try:
            converted_box = convert_functions[from_mode][to_mode](box)
        except KeyError:
            raise NotImplementedError(f"Conversion from BoxMode {from_mode} to {to_mode} is not supported.")

        return converted_box

    def _convert_xywha_abs_to_xyxy_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYWHA_ABS to XYXY_ABS."""
        if box.shape[-1] != 5:
            raise AssertionError("The last dimension of input shape must be 5 for XYWHA format")

        original_dtype = box.dtype
        box = box.double()

        width = box[:, 2]
        height = box[:, 3]
        angle = box[:, 4]
        cos = torch.abs(torch.cos(angle * math.pi / 180))  # noqa: WPS432
        sin = torch.abs(torch.sin(angle * math.pi / 180))  # noqa: WPS432

        # Compute the horizontal bounding rectangle of the rotated box
        new_width = cos * width + sin * height
        new_height = cos * height + sin * width

        # Convert center to top-left corner
        box[:, 0] -= new_width / 2
        box[:, 1] -= new_height / 2

        # Bottom-right corner
        box[:, 2] = box[:, 0] + new_width
        box[:, 3] = box[:, 1] + new_height

        box = box[:, :4].to(dtype=original_dtype)

        return box

    def _convert_xywh_abs_to_xywha_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYWH_ABS to XYWHA_ABS."""
        original_dtype = box.dtype
        box = box.double()

        box[:, 0] += box[:, 2] / 2
        box[:, 1] += box[:, 3] / 2

        angles = torch.zeros((box.size(0), 1), dtype=box.dtype)

        box = torch.cat((box, angles), dim=1).to(dtype=original_dtype)

        return box

    def _convert_xyxy_abs_to_xywh_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYXY_ABS to XYWH_ABS."""
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]

        return box

    def _convert_xywh_abs_to_xyxy_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYWH_ABS to XYXY_ABS."""
        box[:, 2] += box[:, 0]
        box[:, 3] += box[:, 1]

        return box

    def _convert_to_torch(self, box: RawBoxType) -> torch.Tensor:
        """Convert the box from whatever type it is to a torch Tensor."""
        if isinstance(box, torch.Tensor):
            return box.clone()

        if isinstance(box, (list, tuple)):
            if len(box) < 4 or len(box) > 5:
                raise AssertionError(
                    "`BoxMode.convert` takes either a k-tuple/list or an Nxk array/tensor where k == 4 or 5."
                )

            return torch.tensor(box)[None, :]

        if isinstance(box, np.ndarray):
            return torch.from_numpy(np.asarray(box)).clone()

        raise NotImplementedError


def quantize_bbox(
    bbox: Union[list[float], np.ndarray, torch.Tensor],  # type: ignore[type-arg]
    w_resize_ratio: float,
    h_resize_ratio: float,
    num_bins: int = 1000,
    max_image_size: int = 512,
    return_type: Literal["tensor", "numpy", "list"] = "tensor",
) -> Union[np.ndarray, list[float], torch.Tensor]:  # type: ignore[type-arg]
    """Create the quantized bounding box.

    The resize ratio for each coordinate is defined as `w_resize_ratio = patch_image_size / width`,
    `h_resize_ratio = patch_image_size / height`,

    where patch_image_size is the size of the image before feeding it to clip (224).
    """
    if not isinstance(bbox, torch.Tensor):
        bbox = torch.tensor(bbox)

    bbox = bbox.to(torch.float32)

    qx_min = int((bbox[0] * w_resize_ratio / max_image_size * (num_bins - 1)).round())
    qy_min = int((bbox[1] * h_resize_ratio / max_image_size * (num_bins - 1)).round())
    qx_max = int((bbox[2] * w_resize_ratio / max_image_size * (num_bins - 1)).round())
    qy_max = int((bbox[3] * h_resize_ratio / max_image_size * (num_bins - 1)).round())

    if return_type == "list":
        return [qx_min, qy_min, qx_max, qy_max]
    elif return_type == "tensor":
        return torch.tensor([qx_min, qy_min, qx_max, qy_max])
    return np.array([qx_min, qy_min, qx_max, qy_max])


def dequantize_bbox(
    bins: Union[list[float], np.ndarray, torch.Tensor],  # type: ignore[type-arg]
    w_resize_ratio: float,
    h_resize_ratio: float,
    num_bins: int = 1000,
    max_image_size: int = 512,
    return_type: Literal["tensor", "numpy", "list"] = "tensor",
) -> Union[np.ndarray, list[float], torch.Tensor]:  # type: ignore[type-arg]
    """Create the bounding box from its quantized version.

    The resize ratio for each coordinate is defined as `w_resize_ratio = patch_image_size / width`,
    `h_resize_ratio = patch_image_size / height`,

    where patch_image_size is the size of the image before feeding it to clip (224).
    """
    if not isinstance(bins, torch.Tensor):
        bins = torch.tensor(bins)

    x_min = bins[0].item() / (num_bins - 1) * max_image_size / w_resize_ratio
    y_min = bins[1].item() / (num_bins - 1) * max_image_size / h_resize_ratio
    x_max = bins[2].item() / (num_bins - 1) * max_image_size / w_resize_ratio
    y_max = bins[3].item() / (num_bins - 1) * max_image_size / h_resize_ratio
    if return_type == "list":
        return [x_min, y_min, x_max, y_max]
    elif return_type == "tensor":
        return torch.tensor([x_min, y_min, x_max, y_max])
    return np.array([x_min, y_min, x_max, y_max])


def get_bins_from_str(
    bins_str: str,
    return_type: Literal["tensor", "numpy", "list"] = "tensor",
) -> Union[np.ndarray, list[float], torch.Tensor]:  # type: ignore[type-arg]
    """Return the bins from the string representation."""
    quantized_bins = bins_str.replace("<s>", "").replace("</s>", "")
    quantized_bins = quantized_bins.replace(".", "").replace("<pad>", "")
    quantized_bins = quantized_bins.replace("<bin_", "").replace(">", " ").strip()

    if return_type == "list":
        return [float(coord) for coord in quantized_bins.split(" ")]
    elif return_type == "tensor":
        return torch.tensor([float(coord) for coord in quantized_bins.split(" ")])
    return np.array([float(coord) for coord in quantized_bins.split(" ")])


class Boxes:
    """This structure stores a list of boxes as a Nx4 torch.Tensor.

    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)

        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)

        if tensor.dim() != 2 and tensor.size(-1) != 4:
            raise AssertionError(f"Tensor shape is incorrect. Current shape is {tensor.size()}")

        self.tensor = tensor

    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self.tensor.device

    def __getitem__(self, index: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """Get a new `Boxes` by indexing.

        The following usage are allowed:
            1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
            2. `new_boxes = boxes[2:10]`: return a slice of boxes.
            3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
                with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes, subject to Pytorch's
        indexing semantics.
        """
        if isinstance(index, int):
            return Boxes(self.tensor[index].view(1, -1))

        box = self.tensor[index]

        if box.dim() != 2:
            raise AssertionError(f"Indexing on Boxes with {index} failed to return a matrix!")

        return Boxes(box)

    def __len__(self) -> int:
        """Get number of boxes."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """Get string representation of this Boxes."""
        return f"Boxes({str(self.tensor)})"

    def clone(self) -> "Boxes":
        """Create a clone."""
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device) -> "Boxes":
        """Move to another device."""
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """Computes the area of all the boxes."""
        box = self.tensor
        width_difference = box[:, 2] - box[:, 0]
        height_difference = box[:, 3] - box[:, 1]
        area = width_difference * height_difference
        return area

    def clip(self, box_size: tuple[int, int]) -> None:
        """Clip (in place) the boxes.

        This is done by limiting x coordinates to the range [0, width] and y coordinates to the
        range [0, height].
        """
        if not torch.isfinite(self.tensor).all():
            raise AssertionError("Box tensor contains infinite or NaN!")

        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)

        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0) -> torch.Tensor:
        """Find boxes that are non-empty.

        A box is considered empty, if either of its side is no larger than threshold.

        Args:
            threshold (float): Boxes larger than this threshold are considered empty.
                Defaults to 0.

        Returns:
            A torch Tensor that is a binary vector which represents whether each box is empty
            (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)  # noqa: WPS465
        return keep

    def inside_box(self, box_size: tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """Get the inside of the box.

        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)  # noqa: WPS465
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """Get the center of the box as Nx2 array of (x, y)."""
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """Scale the box with horizontal and vertical scaling factors."""
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: list["Boxes"]) -> "Boxes":
        """Concatenates a list of Boxes into a single Boxes."""
        if not isinstance(boxes_list, (list, tuple)):
            raise AssertionError("Boxes list must be a list or a tuple.")

        if boxes_list:
            return cls(torch.empty(0))

        if not all(isinstance(box, Boxes) for box in boxes_list):
            raise AssertionError("Every box in the list must be an instance of `Boxes`")

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @torch.jit.unused
    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """Yield a box as a Tensor of shape (4,) at a time."""
        yield from self.tensor  # https://github.com/pytorch/pytorch/issues/18627


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute pairwise intersection over union (IoU) of two sets of matched boxes.

    Both boxes must have the same number of boxes.

    Similar to `pairwise_iou`, but computes only diagonal elements of the matrix.

    Args:
        boxes1 (Boxes): bounding boxes, sized [N,4].
        boxes2 (Boxes): same length as boxes1

    Returns:
        Tensor: iou, sized [N].

    Raises:
        AssertionError: If Boxes do not contain same number of entries.
    """
    if len(boxes1) != len(boxes2):
        raise AssertionError(f"boxlists should have the same number of entries, got {len(boxes1)} and {len(boxes2)}")

    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou


def patchify_image(image: torch.Tensor, patch_size: dict[str, int]) -> torch.Tensor:
    """Convert an image into a tensor of patches.

    Adapted from
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/image_processing_fuyu.py#L180
    """
    patch_height, patch_width = patch_size["height"], patch_size["width"]

    # TODO refer to https://github.com/ArthurZucker/transformers/blob/0f0a3fe5ca5697ee58faeb5b53f049af720b5e98/src/transformers/models/vit_mae/modeling_vit_mae.py#L871
    # torch implementation is faster but does not handle non-squares

    batch_size, channels, _, _ = image.shape
    unfolded_along_height = image.unfold(2, patch_height, patch_height)
    patches = unfolded_along_height.unfold(3, patch_width, patch_width)
    patches = patches.contiguous()
    patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
    patches = patches.permute(0, 2, 3, 4, 1)
    patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)
    return patches


class ObjectCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bbox):
        image_width, image_height = img.size
        crop_height, crop_width = self.size

        x0 = float(bbox[0])
        y0 = float(bbox[1])
        x1 = float(bbox[2])
        y1 = float(bbox[3])

        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        crop_left = max(center_x - crop_width / 2 + min(image_width - center_x - crop_width / 2, 0), 0)
        crop_top = max(center_y - crop_height / 2 + min(image_height - center_y - crop_height / 2, 0), 0)

        return F.crop(img, crop_top, crop_left, crop_height, crop_width)
