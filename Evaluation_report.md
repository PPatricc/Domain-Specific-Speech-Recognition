# EVALUATION_REPORT.md

## 1. Introduction

This report documents the performance and observations from a **domain-specific speech recognition pipeline** that aims to improve Whisper’s ability to recognize police-related terminology in noisy environments. The system leverages:

- **Synthetic Data Generation** with TTS + background noise (varying SNRs).  
- **Optional Real Recordings** integrated into the training dataset.  
- **LoRA Fine-Tuning** of the Whisper model.  
- **Comprehensive Evaluation** comparing baseline vs. fine-tuned performance using WER (Word Error Rate) and **domain-specific term accuracy**.

## 2. Dataset Creation

### 2.1 Synthetic Data

- **Police-related phrases** covering dispatch codes, descriptions, and typical law enforcement terminology.  
- **Noise mixing** for traffic, wind, and crowd backgrounds at 5, 10, and 15 dB SNR.  
- Integration of **TTS (pyttsx3)** for consistent synthetic samples.

- **93 total entries** (synthetic TTS plus 3 real recordings).

> **Limitation**: Very large wind noise files or a mismatch in durations can trigger huge array allocations. Might be beneficial to trimm or downsample lengthy noise files to avoid overshooting memory limits for future.

### 2.2 Real Recordings

- The system optionally loads real `.wav` files specified in a CSV (e.g., `real_data.csv`).  
- Each file is resampled to 16 kHz and appended to the dataset as additional domain-specific data.  
- This small real-data subset complements the synthetic portion and provides more authentic utterances.

## 3. Fine-Tuning Approach

### 3.1 LoRA Configuration

- **LoRA parameters**: rank \(r=16\), \(\alpha=32\), no bias, applied on \(["q_proj", "v_proj"]\).  
- **Training**: Ran for **10 epochs** on CPU hardware, given system constraints.  
- Final average training loss dropped to **0.2160** by epoch 10.

**Resource Constraints**:
- CPU-based training is feasible but slow.  
- Large arrays or lengthy noise files cause memory issues.

**Potential Remediation**:
- **Deploy on GPU** to accelerate training.  
- **Use smaller / truncated noise** or chunk-based noise mixing.

## 4. Evaluation Results

### 4.1 Test Setup

A set of **20 test audio files** was created to compare:
1. **Baseline Whisper** (`openai/whisper-small`).  
2. **LoRA Fine-Tuned Model** (`lora_whisper_model`).

**Key Metrics**:
- **Word Error Rate (WER)** via `jiwer`.  
- **Domain-specific term accuracy** for codes like “10-31,” “officer,” “suspect,” etc.

### 4.2 Baseline vs. Fine-Tuned Summary

| Metric                   | Baseline (Small) | Fine-Tuned (LoRA) |
|--------------------------|------------------|--------------------|
| **Average WER**          | 0.0681 (6.81%)   | 0.0576 (5.76%)     |
| **Domain Term Accuracy** | 1.00             | **1.00**           |
| **Predictions Count**    | 20               | 20                 |

- The **baseline** model produced a WER of **~6.81%**.  
- The **fine-tuned** model improved WER to **~5.76%**.  
- **Domain-specific term accuracy** is reported at **1.00** for both.  
  - Numeric codes (e.g., “10-32,” “ten ninety-nine”) can still be tricky, but for the tested phrases, these were recognized accurately enough to be counted as correct domain terms.

### 4.3 Detailed Observations

Below are selected transcripts and references:

- **Baseline**:  
  - “Need K-9 unit” → sometimes recognized as “Neat K9 unit.”  
  - “Patrol car” → recognized correctly in some samples, but “petrol” in others.  
  - Despite minor slip-ups, domain terms like “officer,” “suspect,” etc. are recognized fairly well.

- **Fine-Tuned**:  
  - Overall improved WER (from ~6.8% to ~5.76%).  
  - Some numeric codes are still transcribed with slight variations (“EDA” vs. “ETA”), but domain accuracy for recognized terms remains strong.  
  - By epoch 10, the model converged to a stable performance under CPU constraints.

## 5. Limitations and Future Work

### 5.1 Limitations

1. **Memory Constraints**  
   - Large background noise files cause enormous array shapes, leading to allocation errors.  
   - CPU-based training is slow and prone to stalling for big models or many epochs.

2. **Small Real Dataset**  
   - Only three real recordings reduce the capacity to generalize to actual law enforcement speech patterns.

3. **Numeric / Code Recognition**  
   - Even though domain accuracy is logged as 1.0 for these phrases, numeric codes can be misheard in other contexts not tested here. Additional data and more robust numeric coverage could help.

### 5.2 Potential Improvements

1. **More Data & Diversity**  
   - Record more real utterances in actual or simulated contexts.  
   - Vary TTS voices: male/female, different accents or speeds for “hurried speech.”

2. **Smarter Noise Handling**  
   - Consider trimming or chunk-based mixing for large wind files.  
   - Explore reverb or microphone distortion to simulate real radio transmissions.

3. **Hyperparameter Tuning**  
   - Adjust LoRA rank, dropout, or learning rates.  
   - Use a learning rate scheduler or early stopping to refine training over more epochs without overfitting.

4. **Advanced Audio Augmentation**  
   - Speed/pitch perturbation to simulate talk speed variations.  
   - Additional SNR ranges or reverb to mimic different acoustical environments.

5. **Confidence Calibration / Word-Level Scoring**  
   - Provide a threshold for uncertain words to prompt manual confirmation.  
   - Store attention or log probabilities to highlight uncertain segments.

6. **Streaming / Real-Time**  
   - If law enforcement uses wearables, implement a partial or streaming inference to handle live audio segments.  
   - Evaluate additional memory or CPU constraints if running on edge devices.

7. **Model Efficiency & Expansion**  
   - **8-bit or 4-bit quantization** to reduce model size and accelerate inference on low-resource hardware.  
   - **Knowledge Distillation**: Distill the LoRA-updated Whisper model into a smaller student model for faster inference.  
   - **Confidence-based Human-in-the-Loop**: Implement a threshold for uncertain transcriptions, prompting an officer to confirm or correct.  
   - **Integration with MLOps**: Set up automated CI/CD pipelines for continuous dataset updates, training, and evaluation.  
   - **Future UI**: A web or local interface (via Streamlit or Gradio) to enable quick audio testing and real-time feedback.

## 6. Conclusion

Despite hardware constraints and memory limitations with large wind files, the **LoRA-fine-tuned Whisper** model demonstrated improved WER—from ~6.81% to ~5.76%—over the baseline. Domain accuracy for tested phrases is **1.00**, indicating that the relevant domain terms (“suspect,” “officer,” “10-78,” etc.) were recognized fully within this evaluation set.

**Next Steps**:
- Acquire or record **more real data** for robust domain adaptation.  
- Tweak LoRA hyperparameters (rank, dropout, alpha) or adopt a learning rate scheduler.  
- Expand noise augmentation beyond standard traffic/wind/crowd.  
- Explore partial or streaming inference approaches for real-time usage scenarios.  
- Consider advanced model efficiency strategies (quantization, knowledge distillation) to facilitate on-device or edge deployment.

By addressing these points, the system can further bolster domain-specific speech recognition and handle real-world law enforcement audio more effectively.
