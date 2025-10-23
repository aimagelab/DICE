import copy
import json
import json_repair
import torch
from editing_evaluation.Evaluator import Evaluator
from editing_evaluation.prompts.prompts import (
    PRETRAIN_PROMPT_COMPLETE,
    PRETRAIN_PROMPT_NOCHANGE,
    PRETRAIN_PROMPT_NOEDIT,
)
from peft import PeftModel  # This is for loading the QLoRA adapter
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.image_utils import load_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predefined_token_probs(
    generated_output,
    tokenizer,
    predefined_tokens=["ADD", "EDIT", "REMOVE"],
):
    """
    Generates text using a language model and extracts the probabilities of predefined tokens,
    applying softmax only over those tokens.

    Args:
        model_name (str): Name of the Hugging Face model.
        text (str): Input text prompt.
        predefined_tokens (list): List of target tokens (e.g., ["ADD", "EDIT", "REMOVE"]).
        max_new_tokens (int): Maximum number of new tokens to generate.
        repetition_penalty (float): Repetition penalty for generation.

    Returns:
        dict: A dictionary containing token probabilities at each timestep.
    """
    # Convert predefined tokens to token IDs
    predefined_token_ids = tokenizer.convert_tokens_to_ids(predefined_tokens)

    # Extract logits for each generated token
    # Extract generated token IDs
    generated_ids = generated_output.sequences[0]  # Shape: (seq_len,)
    # Infer the number of generated tokens (length of scores matches generated tokens)
    num_generated_tokens = len(generated_output.scores)

    # Compute prompt length dynamically
    input_length = (
        generated_ids.shape[0] - num_generated_tokens
    )  # Total length - generated part

    # Extract only the generated tokens
    generated_only_ids = generated_ids[input_length:]  # Tokens after the prompt
    generated_tokens = tokenizer.convert_ids_to_tokens(
        generated_only_ids
    )  # Convert IDs to token strings

    # Extract logits for each generated token
    logits = torch.stack(
        generated_output.scores, dim=1
    )  # Shape: (batch_size, seq_len, vocab_size)

    # Extract logits only for predefined tokens
    selected_logits = logits[
        :, :, predefined_token_ids
    ]  # Shape: (batch_size, seq_len, num_predefined_tokens)

    # Apply softmax only over the predefined tokens
    selected_probs = torch.nn.functional.softmax(
        selected_logits, dim=-1
    )  # Shape: (batch_size, seq_len, num_predefined_tokens)

    token_probabilities = []

    # Iterate through generated tokens to store probability of each occurrence
    for t, token in enumerate(generated_tokens):  # Iterate over each generated token
        if token in predefined_tokens:  # If it's one of our target tokens
            token_index = predefined_tokens.index(
                token
            )  # Get its index in the predefined list
            token_prob = selected_probs[:, t, token_index].item()  # Get its probability
            token_probabilities.append(token_prob)  # Store probability

    return token_probabilities


# Example usage
# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-2-7b-hf"  # Change as needed
#     text = "The system should"
#     predefined_tokens = ["ADD", "EDIT", "REMOVE"]

#     probs = get_predefined_token_probs(model_name, text, predefined_tokens)
#     for token, prob_list in probs.items():
#         print(f"Token: {token}, Normalized Probabilities: {prob_list}")


class TunedIdefics(Evaluator):
    def __init__(self, *args, **kwargs):
        if "model_name" not in kwargs:
            raise ValueError("Editor must have a name")
        self.name = kwargs["model_name"]
        self.weights_hf = kwargs["weights_hf"]
        self.weights_lora = kwargs["weights_lora"]
        self.resize_len = kwargs["resize_len"]
        self.merged_path = kwargs.get("merged_path")
        processor_path = kwargs.get("processor_path", "HuggingFaceM4/Idefics3-8B-Llama3")
        use_4bit = True  # Activate 4-bit precision base model loading
        bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
        bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
        use_nested_quant = False  # Activate nested quantization for 4-bit base models (double quantization)
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            cache_dir=self.weights_hf,
            size={
                "longest_edge": self.resize_len
            },  # Image is resized to have the longest edge of 2*364=728 pixels. Then a number of crops of 384x384 are encoded. + an additional crop with the whole image.
        )
        if self.merged_path:
            if self.weights_lora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=use_4bit,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=use_nested_quant,
                )

            model = AutoModelForVision2Seq.from_pretrained(
                self.merged_path,  #
                attn_implementation=kwargs.get("attn_implementation", "sdpa"),
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                cache_dir=self.weights_hf
            ).to(device)

            model_noqlora = AutoModelForVision2Seq.from_pretrained(
                self.merged_path,  #
                attn_implementation=kwargs.get("attn_implementation", "sdpa"),
                torch_dtype=torch.bfloat16,
                cache_dir=self.weights_hf
            ).to(device)

            print("Transferring the not quantized vision and controller to model...")
            correct_model_device = model.model.text_model.device
            model.model.vision_model = copy.deepcopy(model_noqlora.model.vision_model)
            model.model.vision_model.to(correct_model_device)
            model.model.connector = copy.deepcopy(model_noqlora.model.connector)
            model.model.connector.to(correct_model_device)
            # if DEBUG:
            del model_noqlora
            # Load the LoRA adapters
            self.model = PeftModel.from_pretrained(
                model,
                self.weights_lora,  # Replace with the path to your fine-tuned LoRA adapters
                # torch_dtype=torch.float16
            ).to(device)
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )

            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceM4/Idefics3-8B-Llama3",  #
                attn_implementation=kwargs.get("attn_implementation", "sdpa"),
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                cache_dir=self.weights_hf,
            ).to(device)

            model_noqlora = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceM4/Idefics3-8B-Llama3",  #
                attn_implementation=kwargs.get("attn_implementation", "sdpa"),
                torch_dtype=torch.bfloat16,
                cache_dir=self.weights_hf,
            )
            print("Transferring the not quantized vision and controller to model...")
            correct_model_device = model.model.text_model.device
            model.model.vision_model = copy.deepcopy(model_noqlora.model.vision_model)
            model.model.vision_model.to(correct_model_device)
            model.model.connector = copy.deepcopy(model_noqlora.model.connector)
            model.model.connector.to(correct_model_device)
            # if DEBUG:
            del model_noqlora
            # Load the LoRA adapters
            self.model = PeftModel.from_pretrained(
                model,
                self.weights_lora,  # Replace with the path to your fine-tuned LoRA adapters
                # torch_dtype=torch.float16
            ).to(device)

    def parse_bbox(self, answer: str):
        commands = []
        subjects = []
        bboxes = []
        indexes = []
        try:
            assistant = answer.split("Assistant:")[1].split("\n")[0].replace("'", '"')
        except:
            try:
                assistant = answer.split("assistant\n")[1].split("\n")[0].replace("'", '"')
            except:
                return commands, subjects, bboxes, indexes
        try:
            predictions = json.loads(assistant)
        except:
            assistant = assistant.replace("(", "[").replace(")", "]")
            try:
                predictions = json.loads(assistant)
            except:
                try:
                    new_assistant= assistant + "]"
                    predictions = json.loads(new_assistant)
                except:
                    assistant += '"skip: error, []" ]'
                    try:
                        predictions = json.loads(assistant)
                    except:
                        try:
                            predictions=json_repair.loads(assistant)
                        except:
                            return commands, subjects, bboxes, indexes
        if type(predictions) != list:
            return commands, subjects, bboxes, indexes
        else:
            if len(predictions) == 0:
                return commands, subjects, bboxes, indexes
        if type(predictions[0]) == list:
            predictions = predictions[0]           
        for index, prediction in enumerate(predictions):
            try:
                commands.append(prediction.split(":")[0])
            except:
                continue
            if "nochange" in prediction.split(":")[0].lower():
                return [], [], []
            try:
                subjects.append(prediction.split(":")[1].split(",")[0])
            except:
                continue
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
                try:
                    try:
                        coordinates=json.loads(prediction.split("BOUNDING_BOX:")[1])
                    except:
                        coordinates=json.loads(prediction.split("BOUNDING BOX:")[1])
                    if len(coordinates) != 4:
                        commands.pop()
                        subjects.pop()
                        break
                    formaterror = False
                    for i, coordinate in enumerate(coordinates):
                        # check i
                        if not isinstance(coordinate, float):
                            commands.pop()
                            subjects.pop()
                            formaterror = True
                            break
                        #if not 0 <= coordinate <= 1:
                        #    if image is not None:
                        #        coordinates[i] = coordinate/image.size[0]
                        #    else:
                        #        commands.pop()
                        #        subjects.pop()
                        #        formaterror = True
                        #        break
                    if not formaterror:
                        bboxes.append(coordinates)
                        indexes.append(index)
                except:
                    commands.pop()
                    subjects.pop()
        return commands, subjects, bboxes, indexes

    def find_differences(
        self,
        img_original,
        img_edited,
        prompt=PRETRAIN_PROMPT_COMPLETE,
        mode="difference",
        *args,
        **kwargs,
    ):
        image_first=kwargs.get("image_first", False)
        vision_prompt = prompt
        if not image_first:
            messages = [  #
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image"},
                        {"type": "image"},
                    ],
                }
            ]
        else:
            messages = [  #
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": vision_prompt},
                    ],
                }
            ]
        return self.find_differneces_from_messages(
            img_original, img_edited, messages, mode=mode, *args, **kwargs
        )

    def find_differneces_from_messages(
        self, img_original, img_edited, messages, mode="difference", *args, **kwargs
    ):
        image_original = load_image(img_original)
        image_edited = load_image(img_edited)
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True if mode == "difference" else False
        )
        inputs = self.processor(
            text=prompt,
            images=[image_original, image_edited],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if mode == "difference":
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=500,
                repetition_penalty=1.5,
                return_dict_in_generate=True,
                output_scores=True,
            )  # repetition_penalty=1.5
            prob = get_predefined_token_probs(
                generated_ids,
                self.processor.tokenizer,
                predefined_tokens=["ADD", "EDIT", "REMOVE"],
            )
            print(prob)
            generated_texts = self.processor.batch_decode(
                generated_ids.sequences[0], skip_special_tokens=True
            )
            generated_texts = "".join(generated_texts)
        else:
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=500
            )  # repetition_penalty=1.5
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        if mode == "difference":
            commands, subjects, bboxes, indexes = self.parse_bbox(generated_texts)
            prob = [prob[i] for i in range(min(len(commands), len(prob)))]
            indexes = [index for index in indexes if index < len(prob)]
            prob = [prob[i] for i in indexes]
            return (
                image_original,
                image_edited,
                generated_texts,
                commands,
                subjects,
                bboxes,
                prob,
            )
        else:
            return (
                image_original,
                image_edited,
                generated_texts,
                None,
                None,
                None,
                None,
            )

    def evaluate_image(self, img_original, img_edited, edit_prompt, *args, **kwargs):
        pass


