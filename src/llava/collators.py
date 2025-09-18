from torch.utils.data import DataLoader

def collate_fn(batch, processor, max_length=512):
    images, prompts, targets = [], [], []

    for item in batch:
        image, target = item["image"], item["caption"]

        # Build conversation using LLaVA's chat template
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Caption the visual metaphor in the image in a single sentence. The caption should be a simile of the form: A is as B as C."}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        prompts.append(prompt.strip())
        images.append(image)
        targets.append(target)

    batch = processor(
        text=prompts,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Prepare labels (ignore padding in loss)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch  # returns dict with input_ids, attention_mask, pixel_values, labels

def make_dataloader(dataset, processor, batch_size=4, shuffle=True, max_length=512, num_workers=4):
    """
    Creates a DataLoader with the custom collate_fn.
    
    Args:
        dataset: ImageMetDataset
        processor: LlavaProcessor (or similar)
        batch_size: per-GPU batch size
        shuffle: shuffle dataset order
        max_length: max sequence length for tokenizer
        num_workers: dataloader workers
    """
    def collate(batch):
        return collate_fn(batch, processor, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True
    )