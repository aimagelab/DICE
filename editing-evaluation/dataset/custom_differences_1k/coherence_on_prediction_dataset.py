from torch.utils.data import Dataset
import json
import os
from PIL import Image
import rootutils
import sys
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def center_crop_and_resize(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Crop the image to a square and resize it to ``target_size``."""
    min_side = min(image.size)
    left = (image.width - min_side) / 2
    top = (image.height - min_side) / 2
    right = (image.width + min_side) / 2
    bottom = (image.height + min_side) / 2
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.LANCZOS)

sys.path.append(os.path.join(rootutils.find_root(__file__, indicator=".project-root"), "editing-evaluation"))
from dataset.dataset_grounding.utils.transform import (
    box_xywh_to_xyxy,
    norm_box_xyxy,
    get_center_crop_region,
    crop,
    get_transform,
    plot_bboxes,
)

class CoherenceOnPrediction(Dataset):
    def __init__(self, json_path, output_dir, split_from=0, mode = "normal", white_borders=True, skip_saving=False):
        try:
            self.data = json.load(open(json_path))
            self.data = self.data[split_from:]
            self.output_dir = output_dir
            self.mode = mode
            self.color = {"ADD": "red", "EDIT": "green", "REMOVE": "blue"}
            self.white_borders = white_borders
            self.skip_saving = skip_saving
        except Exception:
            self.data = []
            print("Error loading json file")
        os.makedirs(self.output_dir, exist_ok=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.mode == "magicbrush":
            image_original = Image.open(data["original_path"]).convert("RGB").resize((512, 512))
            image_edited = Image.open(data["edited_path"]).convert("RGB").resize((512, 512))
            #center crop the image leaving cutting 10 pixels from each side
            image_original = image_original.crop((10, 10, 502, 502)).resize((512, 512))
            image_edited = image_edited.crop((10, 10, 502, 502)).resize((512, 512))
        else:
            image_original = Image.open(data["original_path"])
            image_edited = Image.open(data["edited_path"])
        image_w, image_h = image_original.size
        if image_w != image_h:
            image_original = center_crop_and_resize(image_original, (512, 512))
            image_edited = center_crop_and_resize(image_edited, (512, 512))
            image_w, image_h = image_original.size
        command = data["operation"]
        de_norm_bbox = [data["bbox"][0]*image_w, data["bbox"][1]*image_h, data["bbox"][2]*image_w, data["bbox"][3]*image_h]
        color = self.color.get(command, "green")
        if (not (os.path.exists(f"{self.output_dir}/_edited{idx}.jpg") or os.path.exists(f"{self.output_dir}/_original{idx}.jpg")) and self.skip_saving) or not self.skip_saving:
            if "ADD" in command or "EDIT" in command:
                plot_bboxes(image_edited, 
                    [de_norm_bbox], 
                    [], 
                    [], 
                    labels=["" for _ in range(len(de_norm_bbox))], 
                    xywh=False, 
                    saving_path=f"{self.output_dir}/_edited{idx}.jpg",
                    edge_color=color,
                    white_borders=self.white_borders
                )
                plot_bboxes(image_original,
                    [],
                    [],
                    [],
                    labels=[],
                    xywh=False,
                    saving_path=f"{self.output_dir}/_original{idx}.jpg",
                    edge_color=color,
                    white_borders=self.white_borders
                )
                #image_original.save(f"{self.output_dir}/_original{idx}.jpg")
            else:
                plot_bboxes(
                    image_original,
                    [de_norm_bbox],
                    [],
                    [],
                    labels=["" for _ in range(len(de_norm_bbox))],
                    xywh=False,
                    saving_path=f"{self.output_dir}/_original{idx}.jpg",
                    edge_color=color,
                    white_borders=self.white_borders
                )
                plot_bboxes(image_edited,
                    [],
                    [],
                    [],
                    labels=[],
                    xywh=False,
                    saving_path=f"{self.output_dir}/_edited{idx}.jpg",
                    edge_color=color,
                    white_borders=self.white_borders
                )
            #image_edited.save(f"{self.output_dir}/_edited{idx}.jpg")
        return {
            "image_original": f"{self.output_dir}/_original{idx}.jpg",
            "image_edited": f"{self.output_dir}/_edited{idx}.jpg",
            "bbox": data["bbox"],
            "differences": data["complete_difference"].split(", ")[0],
            "motivation": "Reason: no reason" + f"\nAnswer: NO",
            "gt": 0,
            "prompt": data["gt_prompt"],
            "image_original_dataset_path": data["original_path"],
            "image_edited_dataset_path": data["edited_path"],
            "confidence":data.get("confidence", 1)
        }