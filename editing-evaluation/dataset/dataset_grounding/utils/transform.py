from typing import Dict, Any, Tuple, Optional
import torch
import torchvision.transforms.functional as F
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path

from torchvision.transforms import v2 as T  # Lollo: Droppa sul server
import random
import json
import numpy as np

import cv2  # Lollo: Droppa sul server


class DownUpResize:
    def __init__(self, downscale_factor):
        """
        Initializes the transform.
        Args:
            downscale_factor (float): Factor to downscale the image. E.g., 0.5 for half-size.
        """
        self.downscale_factor = downscale_factor

    def __call__(self, img):
        """
        Applies the downscale and upscale transformation.
        Args:
            img (PIL.Image.Image): Input image.
        Returns:
            PIL.Image.Image: Image downscaled and then upscaled back to original size.
        """
        original_size = img.size  # (width, height)
        new_size = (
            int(original_size[0] * self.downscale_factor),
            int(original_size[1] * self.downscale_factor),
        )

        # Downscale
        img = img.resize(new_size, Image.BILINEAR)

        # Upscale back to original size
        img = img.resize(original_size, Image.BILINEAR)

        return img


def get_mask_area(mask_path):
    """
    Calculates the area of the white region in a binary mask image.

    Parameters:
        mask_path (str): Path to the binary mask image.

    Returns:
        int: The area of the white region (number of white pixels).
    """
    # Load the binary mask image
    binary_image = Image.open(mask_path).convert("L")  # Convert to grayscale

    # Convert the image to a numpy array
    binary_array = np.array(binary_image)

    # Ensure the array is binary (0 for black, 1 for white)
    binary_array = (binary_array > 128).astype(np.uint8)  # Thresholding

    # Measure the area (count of white pixels)
    area = np.sum(binary_array)  # White pixels (1s) contribute to the sum

    return area


def augmentation_pipeline():
    """
    Define the augmentation pipeline for the images.
    Returns:
        random_transform: torchvision.transforms.Compose, the augmentation pipeline to be used for training regular couples
        transform: torchvision.transforms.Compose, the augmentation pipeline to be used for NoChange couples (same image twice)
    """
    # TODO: Gaussian Noise is not supported for PIL images, if we stick with this maybe add somewhere else T.RandomApply(torch.nn.ModuleList([T.GaussianNoise(mean=0, sigma=0.1)]),p=0.05),
    random_transform = T.Compose(
        [
            T.RandomApply(
                torch.nn.ModuleList(
                    [T.GaussianBlur(kernel_size=(5, 5), sigma=(0.75, 4.0))]
                ),
                p=0.1,
            ),
            T.RandomApply(torch.nn.ModuleList([T.JPEG(quality=(30, 75))]), p=0.1),
        ]
    )
    transform_list = [
        T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.0)),
        T.JPEG(quality=(30, 100)),
    ]
    transform = random.choice(transform_list)
    return random_transform, transform


def jpg_transform(quality=(30, 30)):
    # apply a jpg compression to the image
    transform = T.JPEG(quality=quality)
    return transform


def blur_jpg_transform(quality=25, sigma=5.0):
    # apply a jpg compression to the image
    transform = T.Compose(
        [
            T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma, 5.0)),
            T.JPEG(quality=(quality, quality)),
        ]
    )
    return transform


def blur_transform(kernel_size=(7, 7), sigma=(3.0, 3.0)):
    transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return transform


def downsize_upsize_transform(factor=0.5):
    down_up_transform = DownUpResize(downscale_factor=factor)  # Apply custom transform
    return down_up_transform


def get_transform(transform):
    if transform is None:
        return None
    if "jpg" in transform:
        if "random" in transform:
            quality = (int(transform.split("_")[-2]), int(transform.split("_")[-1]))
            return jpg_transform(quality=quality)
        else:
            quality = int(transform.split("_")[-1])
            return jpg_transform(quality=(quality, quality))
    if "blur" in transform:
        if "random" in transform:
            sigma = (float(transform.split("_")[-2]), float(transform.split("_")[-1]))
            return blur_transform(sigma=sigma)
        else:
            sigma = float(transform.split("_")[-1])
            return blur_transform(sigma=(sigma, sigma))
    if "downsize_upsize" in transform:
        factor = float(transform.split("_")[-1])
        return downsize_upsize_transform(factor=factor)
    return None


def get_center_crop_region(image):
    """
    Get the region to crop the center of the image.
    Args:
        image: PIL.Image
    Returns:
        region: Tuple[int, int, int, int] (top, left, height, width)
    """
    img_width, img_height = image.size

    # Determine if the image is landscape or portrait
    if img_width > img_height:
        # Landscape: crop width to match height
        crop_width = img_height
        crop_height = img_height
        top = 0
        left = (img_width - crop_width) // 2
    else:
        # Portrait or square: crop height to match width
        crop_width = img_width
        crop_height = img_width
        top = (img_height - crop_height) // 2
        left = 0

    # Return region as (top, left, height, width) for F.crop
    region = (top, left, crop_height, crop_width)
    return region


def overlay_mask_on_image(
    image_path, mask_path, output_path=None, alpha=0.5, mask_color=(255, 0, 0)
):
    """
    Overlay a binary mask onto an image.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the binary mask (PNG format).
        output_path (str): Path to save the resulting image. If None, the result is not saved.
        alpha (float): Transparency level of the mask overlay (0.0 = fully transparent, 1.0 = fully opaque).
        mask_color (tuple): RGB color to use for the mask overlay (default is red).

    Returns:
        PIL.Image.Image: Image with the mask overlay.
    """
    # Load the image and mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Load as grayscale (binary mask)
    mask = mask.resize(image.size)
    # Convert the mask to RGB and apply the mask color
    mask_rgb = Image.new("RGB", mask.size, mask_color)
    mask_rgb = Image.composite(mask_rgb, Image.new("RGB", mask.size, (0, 0, 0)), mask)

    # Blend the original image and the colored mask
    overlay = Image.blend(image, mask_rgb, alpha)

    # Save the result if an output path is provided
    if output_path:
        overlay.save(output_path)

    return overlay


def crop(image, region, annotations):
    """
    Crop the image and adjust the annotations accordingly.
    The code is an adaptation of original Detr https://github.com/facebookresearch/detr/blob/main/datasets/transforms.py
    Args:
        image: PIL.Image
        region: Tuple[int, int, int, int] (top, left, height, width)
        annotations: List[Dict[str, Any]] containing the annotations
    Returns:
        cropped_image: PIL.Image
        annotations: List[Dict[str, Any]] containing the adjusted annotations
    """
    cropped_image = F.crop(image, *region)

    i, j, h, w = region  # top, left, height, width

    boxes = []
    for ann in annotations:
        boxes.append(ann["bbox"])
    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    # Move the boxes and calculate the area of the boxes
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
    boxes = cropped_boxes.reshape(-1, 4)

    for ann, box in zip(annotations, boxes):
        ann["bbox"] = box.tolist()
    # Filter out annotations with zero area
    annotations = [ann for ann, a in zip(annotations, area) if a > 0]

    return cropped_image, annotations


def polygons_to_pil_mask(image_shape, polygons, apply_closing=False):
    """
    Convert a list of polygons to a binary mask compatible with PIL.

    Args:
        image_shape (tuple): Shape of the output mask (height, width).
        polygons (list): List of polygons, where each polygon is a list of (x, y) tuples.

    Returns:
        PIL.Image.Image: Binary mask as a PIL image.
    """
    # Create a blank binary mask using NumPy
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Convert polygons into the required format
    polygons = [np.array(poly, dtype=np.int32).reshape((-1, 2)) for poly in polygons]

    # Fill the polygons on the mask
    cv2.fillPoly(mask, polygons, color=1)
    if apply_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Fixed kernel size
        mask = cv2.dilate(mask, kernel, iterations=3)
    # Convert the binary mask (0s and 1s) to a PIL Image
    pil_mask = Image.fromarray(mask * 255)  # Scale to 0-255 for proper visualization

    return pil_mask


def get_mask_from_bbox(image_shape, bbox):
    mask_image = torch.zeros(image_shape, dtype=torch.uint8)
    x_min, y_min, x_max, y_max = map(int, bbox)
    mask_image[y_min:y_max, x_min:x_max] = 255  # Use 255 for binary mask visualization
    # Convert tensor to image and save
    mask_image = Image.fromarray(mask_image.numpy())
    return mask_image


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def box_xywh_to_xyxy(box, *, w=None, h=None):
    x, y, bw, bh = box
    x2 = x + bw
    y2 = y + bh
    if w is not None:
        x2 = min(x2, w)
    if h is not None:
        y2 = min(y2, h)
    box = x, y, x2, y2
    return box


def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = (
        round(norm_x1, 3),
        round(norm_y1, 3),
        round(norm_x2, 3),
        round(norm_y2, 3),
    )
    return normalized_box


def norm_point_xyxy(point, *, w, h):
    x, y = point
    norm_x = max(0.0, min(x / w, 1.0))
    norm_y = max(0.0, min(y / h, 1.0))
    point = norm_x, norm_y
    return point


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box


def point_xy_expand2square(point, *, w, h):
    pseudo_box = (point[0], point[1], point[0], point[1])
    expanded_box = box_xyxy_expand2square(box=pseudo_box, w=w, h=h)
    expanded_point = (expanded_box[0], expanded_box[1])
    return expanded_point


def plot_bboxes(
    image: Image,
    bboxes: List[List[float]],
    gt_bboxes: List[List[float]] = None,
    gt_labels: List[str] = None,
    xywh: bool = True,
    labels: Optional[List[str]] = None,
    saving_path: Optional[str] = None,
    cropped: bool = False,
    edge_color: str = "white",
    white_borders: bool = True
) -> None:
    """
    Args:
      image_file: str specifying the image file path
      bboxes: list of bounding box annotations for all the detections
      xywh: bool, if True, the bounding box annotations are specified as
        [xmin, ymin, width, height]. If False the annotations are specified as
        [xmin, ymin, xmax, ymax]. If you are unsure what the mode is try both
        and check the saved image to see which setting gives the
        correct visualization.

    """
    fig = plt.figure()

    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])

    plt.imshow(image)

    # Iterate over all the bounding boxes
    for i, bbox in enumerate(bboxes):
        if xywh:
            xmin, ymin, w, h = bbox
        else:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin

        # add bounding boxes to the image
        box = patches.Rectangle(
            (xmin, ymin), w, h, edgecolor=edge_color, facecolor="none", linewidth=3
        )

        ax.add_patch(box)

        if labels is not None:
            rx, ry = box.get_xy()
            cx = rx + box.get_width() / 2.0
            cy = ry + box.get_height() / 8.0
            l = ax.annotate(
                labels[i],
                (cx, cy),
                fontsize=8,
                fontweight="bold",
                color=edge_color,
                ha="center",
                va="center",
            )
            l.set_bbox(dict(facecolor="red", alpha=0.5, edgecolor=edge_color))
    if gt_bboxes is not None:
        for i, bbox in enumerate(gt_bboxes):
            if xywh:
                xmin, ymin, w, h = bbox
            else:
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin

            # add bounding boxes to the image
            box = patches.Rectangle(
                (xmin, ymin), w, h, edgecolor="green", facecolor="none"
            )

            ax.add_patch(box)

            if gt_labels is not None:
                rx, ry = box.get_xy()
                cx = rx + box.get_width() / 2.0
                cy = ry + box.get_height() / 8.0
                l = ax.annotate(
                    gt_labels[i],
                    (cx, cy),
                    fontsize=8,
                    fontweight="bold",
                    color=edge_color,
                    ha="center",
                    va="center",
                )
                l.set_bbox(dict(facecolor="green", alpha=0.5, edgecolor="green"))
    plt.axis("off")
    outfile = os.path.join(saving_path)
    if white_borders:
        fig.savefig(outfile)
    else:
        fig.savefig(outfile, bbox_inches="tight", pad_inches=0)
    print("Saved image with detections to %s" % outfile)


def plot_bboxes_clip(
    image: Image,
    bboxes: List[List[float]],
    gt_bboxes: List[List[float]] = None,
    gt_labels: List[str] = None,
    xywh: bool = True,
    labels: Optional[List[str]] = None,
    saving_path: Optional[str] = None,
    cropped: bool = False,
    edge_color: str = "red",
    yes_boxes: List[List[float]] = None,
    plot_boxes : bool = True,
    plot_patches : bool = True
) -> None:
    """
    Plots bounding boxes on an image, including detection boxes, ground-truth boxes, and optional "yes_boxes" in gray.
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(image)

    # Plot "yes_boxes" in gray
    if yes_boxes is not None:
        for bbox in yes_boxes:
            if xywh:
                xmin, ymin, w, h = bbox
            else:
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin
            
            yes_patch = patches.Rectangle(
                (xmin, ymin), w, h, edgecolor="gray", facecolor="gray", alpha=1
            )
            if plot_patches:
                ax.add_patch(yes_patch)

    # Plot detection bounding boxes
    for i, bbox in enumerate(bboxes):
        if xywh:
            xmin, ymin, w, h = bbox
        else:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin

        box = patches.Rectangle(
            (xmin, ymin), w, h, edgecolor=edge_color, facecolor="none", linewidth=3
        )
        if plot_boxes:
            ax.add_patch(box)

        if labels is not None:
            rx, ry = box.get_xy()
            cx = rx + box.get_width() / 2.0
            cy = ry + box.get_height() / 8.0
            l = ax.annotate(
                labels[i],
                (cx, cy),
                fontsize=8,
                fontweight="bold",
                color=edge_color,
                ha="center",
                va="center",
            )
            l.set_bbox(dict(facecolor="red", alpha=0.5, edgecolor=edge_color))

    # Plot ground-truth bounding boxes
    if gt_bboxes is not None:
        for i, bbox in enumerate(gt_bboxes):
            if xywh:
                xmin, ymin, w, h = bbox
            else:
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin

            box = patches.Rectangle(
                (xmin, ymin), w, h, edgecolor="green", facecolor="none"
            )
            ax.add_patch(box)

            if gt_labels is not None:
                rx, ry = box.get_xy()
                cx = rx + box.get_width() / 2.0
                cy = ry + box.get_height() / 8.0
                l = ax.annotate(
                    gt_labels[i],
                    (cx, cy),
                    fontsize=8,
                    fontweight="bold",
                    color="green",
                    ha="center",
                    va="center",
                )
                l.set_bbox(dict(facecolor="green", alpha=0.5, edgecolor="green"))

    plt.axis("off")
    plt.tight_layout()
    outfile = os.path.join(saving_path)
    fig.savefig(outfile, bbox_inches="tight", pad_inches=0)

    print("Saved image with detections to %s" % outfile)


# Function to debug the deserialization of GroundTruth to bbox
def convert_prompt_to_bbox(prompt: str) -> List[List[float]]:
    commands = []
    subjects = []
    bboxes = []
    try:
        predictions = json.loads(prompt)
    except:
        return "[]", "[]", "[]"
    for prediction in predictions:
        if "nochange" in prediction.lower():
            return ["NoChange"], [""], [""]
        commands.append(prediction.split(":")[0])
        subjects.append(prediction.split(":")[1].split(",")[0])
        try:
            coordinates = json.loads(
                ",".join(prediction.split(":")[1].split(",")[1:])
                .replace("(", "[")
                .replace(")", "]")
            )
            if len(coordinates) != 4:
                commands.pop()
                subjects.pop()
                break
            formaterror = False
            for coordinate in coordinates:
                # check i
                if not isinstance(coordinate, float):
                    commands.pop()
                    subjects.pop()
                    formaterror = True
                    break
                if not 0 <= coordinate <= 1:
                    commands.pop()
                    subjects.pop()
                    formaterror = True
                    break
            if not formaterror:
                bboxes.append(coordinates)
        except:
            commands.pop()
            subjects.pop()
    return commands, subjects, bboxes


# @TRANSFORMS.register_module()
class Expand2square:
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color

    def __call__(
        self, image: Image.Image, labels: Dict[str, Any] = None
    ) -> Tuple[Image.Image, Optional[Dict[str, Any]]]:
        width, height = image.size
        processed_image = expand2square(image, background_color=self.background_color)
        if labels is None:
            return processed_image, labels
        if "boxes" in labels:
            bboxes = [
                box_xyxy_expand2square(bbox, w=width, h=height)
                for bbox in labels["boxes"]
            ]
            labels["boxes"] = bboxes
        if "points" in labels:
            points = [
                point_xy_expand2square(point, w=width, h=height)
                for point in labels["points"]
            ]
            labels["points"] = points
        return processed_image, labels