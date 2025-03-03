import os
import logging
import torch
import pyttsx3
import librosa
from jiwer import wer
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from data_preprocessing import preprocess_audio_array

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def create_test_audio_files(test_phrases, output_dir="data/test"):
    """
    Creates audio files from the provided test_phrases using pyttsx3 TTS,
    only if they don't already exist.

    Args:
        test_phrases (List[str]): Phrases to synthesize.
        output_dir (str): Directory to place the generated test .wav files.

    Raises:
        Exception: If pyttsx3 fails to generate audio.
    """
    logging.info("Ensuring test audio directory exists...")
    os.makedirs(output_dir, exist_ok=True)
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
    for i, phrase in enumerate(test_phrases, start=1):
        filename = os.path.join(output_dir, f"test{i}.wav")
        if not os.path.exists(filename):
            try:
                logging.info(f"Creating test audio file: {filename}")
                engine.save_to_file(phrase, filename)
                engine.runAndWait()
            except Exception as e:
                logging.error(f"Error generating TTS for phrase '{phrase}': {e}")
        else:
            logging.info(f"File already exists, skipping: {filename}")

def evaluate_model(model, processor, audio_files, references, domain_terms=None):
    """
    Evaluate a Whisper model on a list of test audio files.

    Steps:
      1. Preprocess each audio file (noise reduction, normalization, etc.).
      2. Generate predictions with model.generate(...)
      3. Compare predictions with references using WER (jiwer).
      4. (Optional) Track domain-specific term accuracy if domain_terms are provided.

    Args:
        model (nn.Module): A WhisperForConditionalGeneration model.
        processor (WhisperProcessor): The processor for tokenizing/detokenizing.
        audio_files (List[str]): Paths to the test .wav files.
        references (List[str]): Ground-truth transcripts for each audio file.
        domain_terms (List[str], optional): Domain terms to measure accuracy on.

    Returns:
        dict: {
            "avg_wer": float,
            "avg_domain_term_accuracy": float,
            "predictions_count": int,
            "predictions_with_refs": List[Tuple[str, str]]
        }
        - "predictions_with_refs" is a list of (prediction, reference) pairs.
    """
    if domain_terms is None:
        # Default domain-specific terms related to police codes, etc.
        domain_terms = [
            "code 10-31", "backup", "suspect", "officer", "bolo", 
            "perpetrator", "witness", "patrol", "dispatch", "precinct"
        ]
    
    logging.info("Starting model evaluation...")
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wers = []
    term_accuracies = []
    predictions_with_refs = []

    for idx, (audio_path, ref_text) in enumerate(zip(audio_files, references), start=1):
        logging.info(f"Evaluating file {idx}/{len(audio_files)}: {audio_path}")
        try:
            # 1. Load and preprocess
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_processed = preprocess_audio_array(audio, sr)
            
            # 2. Tokenize
            input_feats = processor(
                audio_processed, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # 3. Generate
            with torch.no_grad():
                predicted_ids = model.generate(input_feats)
            
            # 4. Decode
            prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            predictions_with_refs.append((prediction, ref_text))

            # 5. Calculate WER
            error = wer(ref_text.lower(), prediction.lower())
            wers.append(error)
            
            # 6. Domain term accuracy
            ref_terms_count = sum(1 for term in domain_terms if term.lower() in ref_text.lower())
            correct_terms = 0
            if ref_terms_count > 0:
                for term in domain_terms:
                    if term.lower() in ref_text.lower() and term.lower() in prediction.lower():
                        correct_terms += 1
                term_acc = correct_terms / ref_terms_count
                term_accuracies.append(term_acc)
            
            logging.info(f"Prediction: '{prediction}' | Reference: '{ref_text}' | WER: {error:.4f}")
        except Exception as e:
            logging.error(f"Error evaluating file {audio_path}: {e}")
            continue
    
    avg_wer = sum(wers) / len(wers) if len(wers) > 0 else 0
    avg_term_acc = sum(term_accuracies) / len(term_accuracies) if len(term_accuracies) > 0 else 0
    
    results = {
        "avg_wer": avg_wer,
        "avg_domain_term_accuracy": avg_term_acc,
        "predictions_count": len(wers),
        "predictions_with_refs": predictions_with_refs
    }
    logging.info(f"Evaluation complete. Results: {results}")
    return results

if __name__ == "__main__":
    logging.info("Preparing test phrases and audio files for evaluation...")

    test_phrases = [
        "Officer on patrol at Pine Street, no activity reported.",
        "Code ten thirty-two in progress, need immediate backup.",
        "Dispatch confirming stolen vehicle suspect heading south.",
        "Multiple witnesses reported loud altercation at West End.",
        "Perpetrator wearing a bright yellow jacket, moving quickly.",
        "Request additional units near City Hall for crowd control.",
        "Suspect fleeing scene, possibly armed, approach with caution.",
        "Officer requesting ambulance for injured civilian at Maple Avenue.",
        "Vehicle pursuit on Highway 66, watch for roadblocks.",
        "Need K-9 unit for building search at 45 Baker Street.",
        "Crowd near Riverfront Park getting unruly, require support.",
        "Patrol car spotted suspicious activity behind old warehouse.",
        "Dispatch, do we have an ETA on backup? Situation is tense.",
        "Suspect identified as a repeat offender with outstanding warrants.",
        "Code ten ninety-nine, all units stand by for emergency broadcast.",
        "Officer found evidence of forced entry at the back door.",
        "Witnesses describe suspect wearing black hoodie and red sneakers.",
        "Request additional paramedics for multiple injuries on scene.",
        "Patrol continuing downtown route, no suspicious activity observed.",
        "Code ten eighty-five, suspect in custody, requesting transport."
    ]

    # 1) Create test .wav files if needed
    try:
        create_test_audio_files(test_phrases, output_dir="data/test")
    except Exception as e:
        logging.error(f"Error during test audio creation: {e}")

    # 2) Prepare file paths and references
    test_audio_files = [f"data/test/test{i}.wav" for i in range(1, 21)]
    test_references = test_phrases

    # 3) Evaluate baseline + fine-tuned for comparison
    baseline_model_id = "openai/whisper-small"  # the official model
    tuned_model_path = "lora_whisper_model"     # your fine-tuned model

    # --- Evaluate Baseline Model ---
    logging.info(f"Loading baseline model: {baseline_model_id}")
    try:
        baseline_model = WhisperForConditionalGeneration.from_pretrained(baseline_model_id)
        baseline_processor = WhisperProcessor.from_pretrained(baseline_model_id)
        baseline_metrics = evaluate_model(baseline_model, baseline_processor, test_audio_files, test_references)
    except Exception as e:
        logging.error(f"Error loading/evaluating baseline model: {e}")
        baseline_metrics = None

    # --- Evaluate Fine-Tuned Model ---
    logging.info(f"Loading fine-tuned model from: {tuned_model_path}")
    try:
        tuned_model = WhisperForConditionalGeneration.from_pretrained(tuned_model_path)
        tuned_processor = WhisperProcessor.from_pretrained(tuned_model_path)
        tuned_metrics = evaluate_model(tuned_model, tuned_processor, test_audio_files, test_references)
    except Exception as e:
        logging.error(f"Error loading/evaluating fine-tuned model: {e}")
        tuned_metrics = None

    # 4) Compare results
    if baseline_metrics and tuned_metrics:
        print("\n=== Comparison of Baseline vs Fine-Tuned ===")
        print(f"Baseline WER: {baseline_metrics['avg_wer']:.4f}")
        print(f"Fine-Tuned WER: {tuned_metrics['avg_wer']:.4f}")
        print(f"Baseline Domain-Term Acc: {baseline_metrics['avg_domain_term_accuracy']:.4f}")
        print(f"Fine-Tuned Domain-Term Acc: {tuned_metrics['avg_domain_term_accuracy']:.4f}")
        print(f"Prediction count: {baseline_metrics['predictions_count']} baseline, {tuned_metrics['predictions_count']} tuned")

        # Optionally print side-by-side predictions for both models if desired
        # For brevity, we'll just do references for the tuned model:
        print("\n=== Fine-Tuned Model Predictions vs. References ===")
        for i, (pred, ref) in enumerate(tuned_metrics["predictions_with_refs"], start=1):
            print(f"Sample {i:02d}:")
            print(f"  Prediction: {pred}")
            print(f"  Reference:  {ref}\n")

    else:
        print("Could not compare baseline and fine-tuned results due to prior errors.")
