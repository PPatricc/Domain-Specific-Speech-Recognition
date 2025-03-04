# src/finetune_lora.py

import logging
import torch
from torch.utils.data import DataLoader

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model

from data_preprocessing import preprocess_audio_array

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_and_prepare_dataset(dataset_path="data/generated_synthetic_data"):
    """
    Load the dataset from disk and apply a map function to
    preprocess the 'audio' field for each sample using our
    preprocess_audio_array function.

    Args:
        dataset_path (str): Path to the saved Hugging Face dataset.

    Returns:
        datasets.Dataset: The dataset, with a 'processed_audio' column added.
    """
    logging.info(f"Loading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)
    logging.info(f"Loaded dataset with {len(ds)} samples.")

    def _preprocess(example):
        try:
            processed_audio = preprocess_audio_array(
                example["audio"], 
                sr=example["sampling_rate"]
            )
            example["processed_audio"] = processed_audio
            return example
        except Exception as e:
            logging.error(f"Error preprocessing audio: {e}")
            raise

    logging.info("Applying preprocessing to the dataset...")
    ds = ds.map(_preprocess)
    logging.info("Preprocessing complete.")

    return ds

def prepare_lora_model(base_model_id="openai/whisper-small"):
    """
    Load the base Whisper model and wrap it with LoRA layers.

    Args:
        base_model_id (str): Model identifier from Hugging Face Hub.

    Returns:
        nn.Module: A Whisper model adapted for LoRA fine-tuning.
    """
    logging.info(f"Loading base model: {base_model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    
    logging.info("Wrapping model with LoRA configuration...")
    lora_model = get_peft_model(model, lora_config)
    logging.info("LoRA model created.")
    lora_model.print_trainable_parameters()

    return lora_model

def train_lora():
    """
    Main function to load the dataset, preprocess audio, then fine-tune
    the Whisper model using LoRA (Low-Rank Adaptation).

    Steps:
      1. Load the preprocessed dataset.
      2. Shuffle and split into train/validation sets.
      3. Initialize Whisper processor + LoRA model.
      4. Create DataLoaders with custom collate_fn.
      5. Train for a set number of epochs.
      6. Save the fine-tuned model and processor.
    """
    logging.info("Starting LoRA fine-tuning process...")
    dataset_path = "data/generated_synthetic_data"

    try:
        # 1. Load & preprocess the dataset
        ds = load_and_prepare_dataset(dataset_path)
        ds = ds.shuffle(seed=42)
        logging.info("Dataset shuffled. Total samples: %d", len(ds))

        # Split into training/validation sets
        train_size = int(0.8 * len(ds))
        train_ds = ds.select(range(train_size))
        eval_ds = ds.select(range(train_size, len(ds)))
        logging.info("Train set size: %d, Eval set size: %d", len(train_ds), len(eval_ds))

        # 2. Load the Whisper processor and LoRA-wrapped model
        logging.info("Loading Whisper processor and preparing LoRA model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = prepare_lora_model("openai/whisper-small")

        # 3. Define a collate function for DataLoader
        logging.info("Defining collate function for data loaders...")

        def collate_fn(batch):
            texts = [ex["text"] for ex in batch]
            audios = [ex["processed_audio"] for ex in batch]
            sampling_rates = [ex["sampling_rate"] for ex in batch]

            # Tokenize/pad text
            batch_enc = processor.tokenizer(
                texts, return_tensors="pt", 
                padding="longest", 
                truncation=True,
                max_length=448
            )

            # Gather audio into a batch
            audio_features = []
            for audio, sr in zip(audios, sampling_rates):
                feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
                audio_features.append(feats[0])  # shape (1, frames, dim) -> (frames, dim)

            audio_features = torch.stack(audio_features, dim=0)

            return {
                "input_features": audio_features, 
                "labels": batch_enc["input_ids"]  # batch-padded text
            }

        # 4. Create DataLoaders
        batch_size = 2
        logging.info(f"Creating DataLoaders with batch_size={batch_size}")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # 5. Basic training loop
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logging.info(f"Using device: {device}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        num_epochs = 10
        logging.info(f"Starting training for {num_epochs} epoch(s).")

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            logging.info(f"Epoch {epoch+1} started...")

            for step, batch in enumerate(train_loader):
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_features=input_features, labels=labels)

                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (step + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1} completed. Avg training loss: {avg_loss:.4f}")

        # 6. Save the LoRA-adapted model

        model = model.merge_and_unload()
        save_dir = "lora_whisper_model"
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

        logging.info(f"LoRA fine-tuned model saved to {save_dir}")

    except Exception as e:
        logging.error(f"An error occurred during LoRA training: {e}")
        raise

if __name__ == "__main__":
    train_lora()
