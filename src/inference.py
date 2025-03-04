#!/usr/bin/env python
import argparse
import logging
import torch
import librosa

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from data_preprocessing import preprocess_audio_array

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def transcribe_audio(model, processor, audio_path: str, sr: int = 16000) -> str:
    """
    Transcribe a single audio file using the fine-tuned Whisper model.

    Steps:
      1) Load audio via librosa.
      2) Preprocess (noise reduction, normalization, optional VAD).
      3) Generate tokens with model.generate().
      4) Decode to text.

    Args:
        model (nn.Module): A WhisperForConditionalGeneration model (fine-tuned or baseline).
        processor (WhisperProcessor): The corresponding processor.
        audio_path (str): Path to an audio file (.wav or similar).
        sr (int): Target sampling rate (default 16 kHz).

    Returns:
        str: The transcribed text.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 1) Load raw audio
    audio, orig_sr = librosa.load(audio_path, sr=sr)
    # 2) Preprocess
    processed_audio = preprocess_audio_array(audio, sr=sr)

    # 3) Tokenize into input features
    input_feats = processor(
        processed_audio, 
        sampling_rate=sr, 
        return_tensors="pt"
    ).input_features.to(device)

    # 4) Generate text
    with torch.no_grad():
        predicted_ids = model.generate(input_feats)
    # 5) Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single audio file using a fine-tuned Whisper model. Example of usage: python inference.py --model_dir lora_whisper_model --audio_path your_audio_file.wav"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="lora_whisper_model",
        help="Path to the merged fine-tuned Whisper model directory."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the input audio file (.wav, etc.)."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sampling rate to which the audio should be loaded/resampled."
    )

    args = parser.parse_args()
    logging.info(f"Loading model from: {args.model_dir}")

    try:
        model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
        processor = WhisperProcessor.from_pretrained(args.model_dir)
    except Exception as e:
        logging.error(f"Error loading model/processor from {args.model_dir}: {e}")
        return

    logging.info(f"Starting inference on {args.audio_path}...")
    try:
        text = transcribe_audio(model, processor, args.audio_path, sr=args.sr)
        logging.info(f"Transcription: {text}")
        print(f"\nTranscribed Text:\n{text}\n")
    except Exception as e:
        logging.error(f"Error during transcription: {e}")


if __name__ == "__main__":
    main()
