import os
import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import evaluate

# Load BERTScore
bertscore = evaluate.load("bertscore")

def BERTScore(pred, ref):
    """Compute BERTScore F1 for a single prediction-reference pair."""
    result = bertscore.compute(
        predictions=[pred],
        references=[ref],
        model_type="distilbert-base-uncased"
    )
    return result["f1"][0]


class LlavaPLModule(L.LightningModule):
    def __init__(self, model, processor, train_dataset, val_dataset, batch_size=4, lr=2e-5, max_length=512):
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length

    def training_step(self, batch, _):
        ids, mask, pixels, labels = batch
        outputs = self.model(input_ids=ids, attention_mask=mask, pixel_values=pixels, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        ids, mask, pixels, answers = batch

        # Generate predictions
        generated = self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            pixel_values=pixels,
            max_new_tokens=self.max_length
        )

        # Decode predictions (skip prompt tokens)
        preds = self.processor.batch_decode(generated[:, ids.size(1):], skip_special_tokens=True)

        # Compute BERTScore for each prediction-reference pair
        scores = [BERTScore(p, a) for p, a in zip(preds, answers)]
        self.log("val_score", np.mean(scores))
        return scores

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        from collators import collate_fn
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.processor, max_length=self.max_length),
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        from collators import collate_fn
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.processor, max_length=self.max_length),
            num_workers=4,
            pin_memory=True
        )

class TrainingConfig:
    def __init__(self):
        # Hyperparameters
        self.learning_rate = 2e-4
        self.batch_size = 4
        self.grad_acc = 8
        self.max_epochs = 5
        self.precision = "16-mixed"
        self.num_workers = 4
        self.warmup_steps = 50
        self.seed = 2024

        # Wandb
        # self.wandb_project =
        # self.wandb_name =

    def to_dict(self):
        return self.__dict__


# ----------------
# Trainer Setup
# ----------------
def setup_trainer(config, checkpoint_dir="./checkpoints", early_stop_patience=2):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        save_top_k=1,
        mode="max",
        dirpath=checkpoint_dir
    )

    early_stop_callback = EarlyStopping(
        monitor="val_score",
        patience=early_stop_patience,
        mode="max"
    )

    wandb_logger = WandbLogger(project=config.wandb_project, name=config.wandb_name)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.grad_acc,
        precision=config.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger
    )
    return trainer, checkpoint_callback, early_stop_callback