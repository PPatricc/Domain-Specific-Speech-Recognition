import logging
import librosa
import noisereduce as nr
import numpy as np

def preprocess_audio_array(audio_array, sr=16000):
    """
    Perform basic audio preprocessing, including:
      1. Noise reduction using noisereduce (spectral gating).
      2. Normalization via librosa.util.normalize().
      3. Simple energy-based Voice Activity Detection (VAD):
         - Boosts frames with energy above a threshold.
         - Final normalization to prevent clipping.

    Args:
        audio_array (np.ndarray): The raw audio samples.
        sr (int, optional): Sampling rate, defaults to 16000.

    Returns:
        np.ndarray: The processed audio array.
    """
    try:
        logging.info("Starting audio preprocessing...")

        # 1. Noise reduction
        logging.debug(f"Audio array length before noise reduction: {len(audio_array)}")
        reduced_noise = nr.reduce_noise(y=audio_array, sr=sr)
        logging.debug("Noise reduction completed.")

        # 2. Normalization
        normalized = librosa.util.normalize(reduced_noise)
        logging.debug("Normalization completed.")

        # 3. Simple energy-based VAD
        energy = librosa.feature.rms(y=normalized, frame_length=1024, hop_length=512)[0]
        threshold = 0.02  # can be tuned
        logging.debug(f"Energy-based VAD threshold set to {threshold}")

        frame_length = 1024
        hop_length = 512
        enhanced = normalized.copy()

        for i, e in enumerate(energy):
            start = i * hop_length
            end = min(start + frame_length, len(enhanced))
            if e > threshold:
                # Slight boost
                enhanced[start:end] *= 1.1

        # Final normalization to avoid clipping
        enhanced = librosa.util.normalize(enhanced)
        logging.info("Audio preprocessing completed successfully.")

        return enhanced

    except Exception as e:
        logging.error(f"Error during audio preprocessing: {e}")
        raise e
