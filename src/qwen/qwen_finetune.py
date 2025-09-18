# ================================
# Environment & Hyperparameters
# ================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # specify GPU device

from PIL import Image
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./datasets")
args = parser.parse_args()


'''
Here, imagemet is a pickle file containing the dataset. It should be structured as:
imagemet = {
     "train": [{"image": PIL.Image, "caption": str}, ...],
     "val": [{"image": PIL.Image, "caption": str}, ...]
     }
'''

with open(os.path.join(args.data_dir, "imagemet"), "rb") as f:
    imagemet = pickle.load(f)

# Training hyperparameters
learning_rate = 2e-4
batch_size = 4
grad_acc = 8
max_steps = 300

# Flags for layer fine-tuning (controlled by bitstrings later)
# vis_layer_active = True
# lang_layer_active = True
# attn_layer_active = True
# mlp_layer_active = True

from itertools import product

# Example: generate all 4-bit combinations for ablations
# bool_strings = ["".join(bits) for bits in product("01", repeat=4)]
# bool_strings.remove('0000')   # remove case where no layers are trained
# print(bool_strings)

# Manually set ablation config (1=ON, 0=OFF)
# Format: [vision, language, attention, mlp]

bool_strings = ["0101"] # -> only language and mlp layers are fine-tuned


def bool_string_to_list(bool_str):
    """Convert string like '0101' → (False, True, False, True)."""
    return tuple([bit == "1" for bit in bool_str])


# ================================
# Training Loop Over Configs
# ================================
for value in bool_strings:
    vis_layer_active, lang_layer_active, attn_layer_active, mlp_layer_active = bool_string_to_list(value)

    # ----------------------------
    # Model Initialization
    # ----------------------------
    from unsloth import FastVisionModel  # FastLanguageModel for LLMs
    import torch

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2-VL-7B-Instruct",
        load_in_4bit=False,  # Use 4-bit for memory efficiency; here set to False (16-bit LoRA)
        use_gradient_checkpointing="unsloth",  # Enables long context via checkpointing
    )

    # Apply LoRA with selective fine-tuning
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=vis_layer_active,
        finetune_language_layers=lang_layer_active,
        finetune_attention_modules=attn_layer_active,
        finetune_mlp_modules=mlp_layer_active,
        r=16,              # LoRA rank
        lora_alpha=16,     # Usually set equal to r
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # ----------------------------
    # Dataset Loading
    # ----------------------------
    

    # Task-specific instructions
    task_instruction = "Caption the visual metaphor in the image in a single sentence. The caption should be a simile of the form: A is as B as C."

    # ----------------------------
    # Data Formatting
    # ----------------------------
    def format_data(idx, split):
        """Return sample in chat template format for fine-tuning."""
        max_pixels = 256 * 28 * 28
        image = imagemet[split][idx]["image"]
        caption = imagemet[split][idx]["caption"]

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "max_pixels": max_pixels},
                        {"type": "text", "text": task_instruction},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": caption}]},
            ]
        }

    # Split train/val sets
    train_dataset, val_dataset = [], []
    for idx in range(len(imagemet["train"])):
        train_dataset.append(format_data(idx, "train"))
    for idx in range(len(imagemet["val"])):
        val_dataset.append(format_data(idx, "val"))

    print("len of train set:", len(train_dataset))
    print("len of val set:", len(val_dataset))


    # ----------------------------
    # Trainer Setup
    # ----------------------------
    from unsloth import is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)  # Enable training mode

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_acc,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            fp16_full_eval=True,
            per_device_eval_batch_size=batch_size,
            eval_accumulation_steps=grad_acc // 4,
            evaluation_strategy="steps",
            eval_steps=5,
            remove_unused_columns=False,  # required for vision finetuning
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=512,
        ),
    )

    # ----------------------------
    # Training
    # ----------------------------
    trainer_stats = trainer.train()

    # Save model
    save_folder = "./imagemet_lora_model"
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)

    # ----------------------------
    # Cleanup GPU memory
    # ----------------------------
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()