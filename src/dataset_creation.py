import os
import csv
import logging
import numpy as np
import torchaudio
import pyttsx3
import librosa
from datasets import Dataset
from transformers import AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_real_recordings(
    real_data_csv="data/real_recordings/real_data.csv",
    audio_dir="data/real_recordings",
    sr=16000
):
    """
    Load real voice recordings based on a CSV listing of (filename, transcript).

    Args:
        real_data_csv (str): Path to a CSV file with columns: [filename, transcript].
        audio_dir (str): Directory containing the .wav files mentioned in the CSV.
        sr (int): Sampling rate to which we resample audio.

    Returns:
        List[dict]: A list of dict entries with fields:
          "text", "audio", "sampling_rate", "noise_file", "snr_db"
    """
    entries = []
    if not os.path.exists(real_data_csv):
        logging.info(f"No CSV found at {real_data_csv}, skipping real data.")
        return entries

    logging.info(f"Loading real recordings from CSV: {real_data_csv}")
    with open(real_data_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logging.info(row)
            wav_path = os.path.join(audio_dir, row["filename"])
            if not os.path.exists(wav_path):
                logging.warning(f"Audio file not found: {wav_path}, skipping.")
                continue

            # Load the real audio
            audio, orig_sr = librosa.load(wav_path, sr=sr)
            # Convert to float32 if needed
            audio = audio.astype(np.float32)
            
            transcript = row["transcript"]
            
            # We'll store "real_data" or something for noise_file
            entries.append({
                "text": transcript,
                "audio": audio,
                "sampling_rate": sr,
                "noise_file": "real_data",
                "snr_db": 0  # or some placeholder
            })

    logging.info(f"Loaded {len(entries)} real recordings from CSV.")
    return entries

def text_to_speech(text, sr=16000):
    """
    Convert the given text to speech using pyttsx3, returning audio as a NumPy array.

    Args:
        text (str): The text to synthesize.
        sr (int, optional): The sample rate to use for the output array. Defaults to 16000.

    Returns:
        (np.ndarray, int): The synthesized audio array and sample rate.

    Raises:
        RuntimeError: If no audio is generated (file is zero length).
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
    temp_filename = "temp_tts.wav"
    try:
        engine.save_to_file(text, temp_filename)
        engine.runAndWait()
        
        if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
            raise RuntimeError(f"No audio generated for text: '{text}'")

        audio, orig_sr = librosa.load(temp_filename, sr=sr)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return audio, sr

def create_synthetic_sample(text, background_noise_path=None, snr_db=10, sr=16000):
    """
    Create a synthetic audio sample from text, optionally mixing in background noise.

    Args:
        text (str): The text to convert to speech.
        background_noise_path (str, optional): Path to a noise .wav file to mix.
        snr_db (int, optional): Desired signal-to-noise ratio in dB. Defaults to 10.
        sr (int, optional): Sample rate for both speech and noise. Defaults to 16000.

    Returns:
        (np.ndarray, int): A tuple containing the synthetic audio and the sample rate.
    """
    speech_array, sample_rate = text_to_speech(text, sr=sr)

    if background_noise_path and os.path.exists(background_noise_path):
        noise, noise_sr = torchaudio.load(background_noise_path)
        if noise_sr != sample_rate:
            noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)
        
        noise = noise.numpy().squeeze()
        
        # If noise is shorter than speech, loop it
        if len(noise) < len(speech_array):
            repeats = int(np.ceil(len(speech_array) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[:len(speech_array)]
        
        # Calculate scale factor for desired SNR
        speech_power = np.mean(speech_array ** 2)
        noise_power = np.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(speech_power / (noise_power * snr_linear))
        
        noisy_speech = speech_array + scale * noise
        return noisy_speech, sample_rate

    return speech_array, sample_rate

def generate_dataset():
    """
    Generate a dataset that includes:
      1) Synthetic TTS data with optional noise (various SNRs).
      2) Real recordings loaded from data/real_recordings/real_data.csv (if present).

    Returns:
        datasets.Dataset: The combined dataset with fields:
          text, audio, sampling_rate, noise_file, snr_db.
    """
    logging.info("Starting dataset generation...")
    
    police_phrases = [
        "Suspect detained at intersection of Main and Broadway, requesting backup.",
        "Code 10-31 in progress at 1420 Elm Street, proceeding with caution.",
        "Officer requesting 10-78, hostile crowd forming near the plaza.",
        "Vehicle matching BOLO description spotted heading eastbound on Highway 42.",
        "Witness describes the perpetrator as male, approximately six-two, wearing a dark hoodie and jeans.",
        "Requesting immediate assistance, shots fired near 210 West Pine Avenue.",
        "Dispatch, do we have any K-9 units available for a building search?",
        "Possible Code 10-46, motorist needs assistance on Route 17.",
        "Perp is fleeing on foot, heading northbound toward Central Park.",
        "Officer on scene, code 10-52, need an ambulance at 12 Mill Road.",
        "Crowd is getting hostile, requesting riot gear and additional backup.",
        "Report of a stolen vehicle, license plate Bravo-Delta-113, last seen near Lakeview.",
        "Roadblock requested at Highway 22 to intercept suspect vehicle.",
        "Officer at precinct four, clearing 10-8, ready for next assignment.",
        "Civilian reports hearing loud altercation, possible domestic disturbance at 456 Willow Lane."
    ]
    
    noise_files = [
        "data/noise_samples/wind_noise.wav",
        "data/noise_samples/traffic_noise.wav",
        "data/noise_samples/crowd_noise.wav"
    ]
    
    logging.info("Loading Whisper processor for consistency (though not strictly needed here).")
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    
    dataset_entries = []
    
    # --- 1) Synthetic TTS with optional noise
    for phrase in police_phrases:
        for noise_file in noise_files:
            for snr in [5, 10, 15]:
                try:
                    audio, sr = create_synthetic_sample(phrase, noise_file, snr_db=snr)
                    audio = audio.astype(np.float32)
                    
                    dataset_entries.append({
                        "text": phrase,
                        "audio": audio,
                        "sampling_rate": sr,
                        "noise_file": os.path.basename(noise_file),
                        "snr_db": snr
                    })
                    logging.info(
                        f"Generated sample for phrase='{phrase[:15]}...', "
                        f"noise='{noise_file}', snr_db={snr}"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to create sample for phrase='{phrase}' "
                        f"noise='{noise_file}' snr_db={snr}: {e}"
                    )

    # --- 2) Load real voice recordings (if any)
    real_entries = load_real_recordings(
        real_data_csv="data/real_recordings/real_data.csv",
        audio_dir="data/real_recordings",
        sr=16000
    )
    logging.info(f"Loaded {len(real_entries)} real recordings.")

    # Combine synthetic + real data
    dataset_entries.extend(real_entries)

    # --- Convert to a Hugging Face dataset
    dataset_dict = {
        "text": [e["text"] for e in dataset_entries],
        "audio": [e["audio"] for e in dataset_entries],
        "sampling_rate": [e["sampling_rate"] for e in dataset_entries],
        "noise_file": [e["noise_file"] for e in dataset_entries],
        "snr_db": [e["snr_db"] for e in dataset_entries],
    }
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    logging.info(f"Dataset created with {len(hf_dataset)} entries (TTS + real).")
    return hf_dataset

if __name__ == "__main__":
    try:
        ds = generate_dataset()
        
        save_dir = "data/generated_synthetic_data"
        os.makedirs(save_dir, exist_ok=True)

        ds.save_to_disk(save_dir)
        logging.info(f"Dataset saved successfully at: {save_dir}")

    except Exception as e:
        logging.error(f"Error during dataset creation or saving: {e}")
