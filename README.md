# Domain-Specific Speech Recognition with Whisper + LoRA

This repository demonstrates how to:
1. Generate a synthetic dataset of police-related phrases with optional background noise.
2. Preprocess the audio (noise reduction, normalization, simple VAD).
3. Fine-tune a Whisper model with LoRA.
4. Evaluate on unseen test data, including logging WER and domain-specific term accuracy.

## Prerequisites

- Python 3.8+ recommended
- Install necessary libraries:

```bash
pip install -r requirements.txt
```

## Folder Structure
.
├─ data/
│  ├─ noise_samples/
│  ├─ generated_synthetic_data/  # will store the final dataset
│  └─ test/                  # TTS-generated audio for evaluation
├─ src/
│  ├─ data_preprocessing.py  # audio preprocessing
│  ├─ dataset_creation.py    # dataset generation
│  ├─ finetune_lora.py       # LoRA fine-tuning
│  └─ evaluation.py          # final evaluation script
├─ requirements.txt
└─ README.md


## Steps
1. Generate the Dataset
```bash
python src/dataset_creation.py
```

2. Fine-Tune the Model with LoRA (data preprocessing included)
```bash
python src/finetune_lora.py
```

3. Evaluate the model
```bash
python src/evaluation.py
```


### Customization
Number of epochs: Adjust num_epochs in finetune_lora.py if you want more training.
Batch size: Increase it if you have a good GPU or reduce if you run out of memory.
Noise mixing: Tweak the SNR range or add additional noise files to get more variety.
VAD threshold**: In data_preprocessing.py, you can modify threshold=0.02 to be more or less aggressive in boosting speech frames.

### Notes
On Windows, if pyttsx3 fails to generate .wav files, ensure you have SAPI5 voices installed and the correct permissions.