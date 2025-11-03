#!/usr/bin/env python3
"""
Simple Evaluation Example
=========================
Minimal example showing how to load and use both models for evaluation.

This script can be adapted to your specific use case.
"""

import os
import sys
import json
import copy

# Add the editing-evaluation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "editing-evaluation"))

import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    import flash_attn
    attn_implementation = 'flash_attention_2'
except ModuleNotFoundError:
    attn_implementation = 'sdpa'
    print('Cannot import flash_attn. Using SDPA attention')

def de_norm_box_xyxy(box, w, h):
    """Denormalize bounding box coordinates."""
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    return [x1, y1, x2, y2]


def plot_bboxes(image, bboxes, labels=None, xywh=True, saving_path=None, edge_color="white"):
    """Plot bounding boxes on an image."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(image)
    
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
        ax.add_patch(box)
        
        if labels is not None and i < len(labels):
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
    
    plt.axis("off")
    if saving_path:
        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        fig.savefig(saving_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved image with detections to {saving_path}")
    plt.close()


class DifferenceSaver:
    """Save detected differences to JSON."""
    def __init__(self, save_path):
        self.save_path = save_path
        self.differences = []
    
    def add_difference(self, original_path, edited_path, complete_difference, 
                      operation, bbox, gt_prompt, confidence=None):
        self.differences.append({
            "original_path": original_path,
            "edited_path": edited_path,
            "complete_difference": complete_difference,
            "operation": operation,
            "bbox": bbox,
            "gt_prompt": gt_prompt,
            "confidence": confidence
        })
    
    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.differences, f, indent=2)
        print(f"Saved {len(self.differences)} differences to {self.save_path}")


def main():
    """
    Simple example of loading both models and running evaluation.
    """

    # ============================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ============================================================================

    # Model checkpoints (LoRA weights). Download models from HF link in readme
    BASE_MODEL_DIFFERENCE = "models--aimagelab--DICE_differencedet_Idefics/snapshots/2dea6affae3772528bbbfdf11983080354a076aa/model_based_tuned_stage1/image_first_after15k_after_lvis_idefics"
    DIFFERENCE_MODEL_CHECKPOINT = "models--aimagelab--DICE_differencedet_Idefics/snapshots/2dea6affae3772528bbbfdf11983080354a076aa/lora_tuned_stage2/checkpoint-15000"
    COHERENCE_MODEL_CHECKPOINT = "models--aimagelab--DICE_coherence_Idefics/snapshots/67c98141f85a42a122f64d65eb70bfed4d2d8bf7/lora_tuned/checkpoint-550"
    BASE_IDEFICS = "models--HuggingFaceM4--Idefics3-8B-Llama3/snapshots/fddb4ff79181e55a994674777e06cd5456ce3dc3"
    # HuggingFace cache directory
    HF_CACHE = "PATH_TO_HF_CACHE"

    # Test images
    IMAGE_ORIGINAL = "original.jpg"
    IMAGE_EDITED = "edited.jpg"
    EDIT_PROMPT = "Change the color of the vase to yellow"
    OUTPUT_DIR = "./output"

    # ============================================================================
    # SETUP
    # ============================================================================

    print("=" * 80)
    print("SIMPLE EVALUATION EXAMPLE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # Import the evaluators
    try:
        from editing_evaluation.Idefics.TunedIdefics import TunedIdefics
        from editing_evaluation.prompts.prompts import (
            PRETRAIN_PROMPT_NOCHANGE,
            PROMPT_DIFFERENCE_COHERENCE_SYSTEM,
            PROMPT_DIFFERENCE_COHERENCE,
        )
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("\nMake sure you run this from the correct directory:")
        print("  cd /path/to/release-editing-eval")
        print("  python simple_evaluation_example.py")
        return

    # ============================================================================
    # STEP 1: LOAD DIFFERENCE DETECTION MODEL
    # ============================================================================

    print("Loading Difference Detection Model...")
    print("-" * 80)

    try:
        difference_model = TunedIdefics(
            model_name="DifferenceDetector",
            weights_lora=DIFFERENCE_MODEL_CHECKPOINT,
            weights_hf=HF_CACHE,
            resize_len=1456,
            merged_path=BASE_MODEL_DIFFERENCE,
            attn_implementation=attn_implementation
        )
        print("✓ Difference detection model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading difference detection model: {e}")
        print("\nPlease update DIFFERENCE_MODEL_CHECKPOINT and HF_CACHE paths in this script.\n")
        return

    # ============================================================================
    # STEP 2: LOAD COHERENCE EVALUATION MODEL
    # ============================================================================

    print("Loading Coherence Evaluation Model...")
    print("-" * 80)

    try:
        coherence_model = TunedIdefics(
            model_name="CoherenceEvaluator",
            weights_lora=COHERENCE_MODEL_CHECKPOINT,
            weights_hf=HF_CACHE,
            resize_len=1456,
            merged_path=BASE_IDEFICS,
        )
        print("✓ Coherence evaluation model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading coherence evaluation model: {e}")
        print("\nPlease update COHERENCE_MODEL_CHECKPOINT and HF_CACHE paths in this script.\n")
        return

    # ============================================================================
    # STEP 3: DETECT DIFFERENCES
    # ============================================================================

    print("=" * 80)
    print("RUNNING DIFFERENCE DETECTION")
    print("=" * 80)

    if not os.path.exists(IMAGE_ORIGINAL) or not os.path.exists(IMAGE_EDITED):
        print(f"\n✗ Test images not found:")
        print(f"  Original: {IMAGE_ORIGINAL}")
        print(f"  Edited: {IMAGE_EDITED}")
        print("\nPlease update IMAGE_ORIGINAL and IMAGE_EDITED paths in this script.")
        print("\nExample usage with your own images:")
        print_example_code()
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load images
    image_original = Image.open(IMAGE_ORIGINAL).convert("RGB") #T
    image_edited = Image.open(IMAGE_EDITED).convert("RGB")
    width_original, height_original = image_original.size
    width_edited, height_edited = image_edited.size

    print(f"\nOriginal Image: {IMAGE_ORIGINAL}")
    print(f"Edited Image: {IMAGE_EDITED}")
    print(f"Edit Prompt: {EDIT_PROMPT}\n")

    # Run difference detection
    print("Running detection...")
    (
        img_original,
        img_edited,
        generated_text,
        commands,
        subjects,
        bboxes,
        confidence,
    ) = difference_model.find_differences(
        img_original=IMAGE_ORIGINAL, 
        img_edited=IMAGE_EDITED, 
        prompt=PRETRAIN_PROMPT_NOCHANGE,
        image_first=True
    )

    # Display detection results
    print("\n" + "-" * 80)
    print("DETECTION RESULTS")
    print("-" * 80)
    print(f"\nGenerated Text:\n{generated_text}\n")
    print(f"Number of changes detected: {len(commands)}\n")

    for i, (cmd, subj, bbox, conf) in enumerate(
        zip(
            commands,
            subjects,
            bboxes,
            confidence if confidence else [1.0] * len(commands),
        ),
        1,
    ):
        print(f"Change {i}:")
        print(f"  Type: {cmd}")
        print(f"  Subject: {subj}")
        print(f"  Bounding Box: {bbox}")
        print(f"  Confidence: {conf:.3f}")
        print()

    if len(commands) == 0:
        print("No changes detected. Evaluation complete.\n")
        return

    # ============================================================================
    # STEP 4: SAVE RESULTS TO JSON
    # ============================================================================

    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80 + "\n")

    # Initialize saver
    saver = DifferenceSaver(os.path.join(OUTPUT_DIR, "predicted_differences.json"))

    # Save each detected difference
    for i, (cmd, subj, bbox, conf) in enumerate(
        zip(commands, subjects, bboxes, confidence if confidence else [1.0] * len(commands))
    ):
        complete_difference = f"{cmd}: {subj}"
        saver.add_difference(
            original_path=IMAGE_ORIGINAL,
            edited_path=IMAGE_EDITED,
            complete_difference=complete_difference,
            operation=cmd,
            bbox=bbox,
            gt_prompt=EDIT_PROMPT,
            confidence=conf
        )
    
    saver.save()

    # ============================================================================
    # STEP 5: PLOT BOUNDING BOXES
    # ============================================================================

    print("\n" + "=" * 80)
    print("PLOTTING BOUNDING BOXES")
    print("=" * 80 + "\n")

    # Separate boxes by type (ADD/EDIT go on edited image, REMOVE goes on original)
    bbox_original = []
    bbox_edited = []
    labels_original = []
    labels_edited = []

    for cmd, subj, bbox in zip(commands, subjects, bboxes):
        # Denormalize bbox if needed
        if all(coord <= 1 for coord in bbox):
            if "ADD" in cmd or "EDIT" in cmd:
                denorm_bbox = de_norm_box_xyxy(bbox, width_edited, height_edited)
                bbox_edited.append(denorm_bbox)
                labels_edited.append(f"{cmd}: {subj}")
            else:  # REMOVE
                denorm_bbox = de_norm_box_xyxy(bbox, width_original, height_original)
                bbox_original.append(denorm_bbox)
                labels_original.append(f"{cmd}: {subj}")
        else:
            if "ADD" in cmd or "EDIT" in cmd:
                bbox_edited.append(bbox)
                labels_edited.append(f"{cmd}: {subj}")
            else:
                bbox_original.append(bbox)
                labels_original.append(f"{cmd}: {subj}")

    # Plot original image with REMOVE boxes
    plot_bboxes(
        image_original,
        bbox_original,
        labels=labels_original,
        xywh=False,
        saving_path=os.path.join(OUTPUT_DIR, "original_with_boxes.jpg"),
        edge_color="blue"
    )

    # Plot edited image with ADD/EDIT boxes
    plot_bboxes(
        image_edited,
        bbox_edited,
        labels=labels_edited,
        xywh=False,
        saving_path=os.path.join(OUTPUT_DIR, "edited_with_boxes.jpg"),
        edge_color="red"
    )

    # ============================================================================
    # STEP 6: PREPARE FOR COHERENCE EVALUATION
    # ============================================================================

    print("\n" + "=" * 80)
    print("RUNNING COHERENCE EVALUATION")
    print("=" * 80 + "\n")

    # Create coherence dataset from predictions
    from dataset.custom_differences_1k.coherence_on_prediction_dataset import CoherenceOnPrediction
    
    coherence_output_dir = os.path.join(OUTPUT_DIR, "coherence_visualizations")
    os.makedirs(coherence_output_dir, exist_ok=True)
    
    try:
        coherence_dataset = CoherenceOnPrediction(
            json_path=os.path.join(OUTPUT_DIR, "predicted_differences.json"),
            output_dir=coherence_output_dir,
            mode="normal",
            white_borders=True,
            skip_saving=False
        )
        
        print(f"Created coherence dataset with {len(coherence_dataset)} items\n")
        
        # Process each item for coherence evaluation
        coherence_results = []
        
        for idx in range(len(coherence_dataset)):
            item = coherence_dataset[idx]
            
            print(f"Evaluating Change {idx + 1}/{len(coherence_dataset)}")
            print("-" * 80)
            
            detected_change = item["differences"]
            print(f"Detected Change: {detected_change}")
            print(f"Edit Prompt: {item['prompt']}\n")
            
            # Prepare coherence evaluation prompt
            
            final_prompt = PROMPT_DIFFERENCE_COHERENCE.replace(
                    "{SUBTSITUTE_PROMPT}", item["prompt"]
            ).replace("{SUBTSITUTE_CHANGE}", detected_change)

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_DIFFERENCE_COHERENCE_SYSTEM,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {"type": "image"},
                        {"type": "image"},
                    ],
                },
            ]
            # Run coherence evaluation
            print("Running coherence evaluation...")
            (_, _, coherence_result, _, _, _, _) = coherence_model.find_differneces_from_messages(
                img_original=item["image_original"],
                img_edited=item["image_edited"],
                messages=messages,
                mode="coherence",
            )
            
            print(f"\nCoherence Result:\n{coherence_result}\n")
            
            # Save coherence result
            coherence_results.append({
                "index": idx,
                "difference": detected_change,
                "prompt": item["prompt"],
                "coherence_evaluation": coherence_result,
                "bbox": item["bbox"],
                "confidence": item["confidence"]
            })
            
            print("=" * 80 + "\n")
        
        # Save all coherence results
        coherence_results_path = os.path.join(OUTPUT_DIR, "coherence_results.json")
        with open(coherence_results_path, 'w') as f:
            json.dump(coherence_results, f, indent=2)
        print(f"Saved coherence results to {coherence_results_path}\n")
        
    except Exception as e:
        print(f"Error during coherence evaluation: {e}")
        print("Skipping coherence evaluation stage.\n")

    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"  - predicted_differences.json: Detected differences")
    print(f"  - original_with_boxes.jpg: Original image with REMOVE boxes")
    print(f"  - edited_with_boxes.jpg: Edited image with ADD/EDIT boxes")
    print(f"  - coherence_results.json: Coherence evaluation results")
    print(f"  - coherence_visualizations/: Visualizations for coherence evaluation")
    print()


def print_example_code():
    """Print example code for users to adapt."""
    print("\n" + "=" * 80)
    print("EXAMPLE CODE TO USE WITH YOUR OWN IMAGES")
    print("=" * 80)
    print("""
from editing_evaluation.Idefics.TunedIdefics import TunedIdefics
from editing_evaluation.prompts.prompts import PRETRAIN_PROMPT_NOCHANGE

# Load difference detection model
difference_model = TunedIdefics(
    model_name="DifferenceDetector",
    weights_lora="/path/to/your/checkpoint",
    weights_hf="/path/to/hf/cache",
    resize_len=728
)

# Detect differences
(img_orig, img_edit, text, commands, subjects, bboxes, confidence) = \
    difference_model.find_differences(
        img_original="your_original.jpg",
        img_edited="your_edited.jpg",
        prompt=PRETRAIN_PROMPT_NOCHANGE
    )

# Results are automatically saved to JSON and visualized with bounding boxes
# Use CoherenceOnPrediction dataset for coherence evaluation

from editing_evaluation.dataset.custom_differences_1k.coherence_on_prediction_dataset import CoherenceOnPrediction

coherence_dataset = CoherenceOnPrediction(
    json_path="output/predicted_differences.json",
    output_dir="output/coherence_visualizations",
    mode="normal"
)

# Load coherence model
coherence_model = TunedIdefics(
    model_name="CoherenceEvaluator",
    weights_lora="/path/to/coherence/checkpoint",
    weights_hf="/path/to/hf/cache",
    resize_len=728
)

# Evaluate coherence for each detected change
for item in coherence_dataset:
    (_, _, coherence_result, _, _, _, _) = coherence_model.find_differences(
        img_original=item["image_original"],
        img_edited=item["image_edited"],
        prompt=full_prompt  # Prepared with PROMPT_DIFFERENCE_COHERENCE
    )
    print(f"Coherence: {coherence_result}")
""")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
