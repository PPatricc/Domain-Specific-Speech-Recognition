# Domain-Specific Speech Recognition with Whisper + LoRA

This repository demonstrates how to:
1. Generate a synthetic dataset of police-related phrases with optional background noise.
2. Preprocess the audio (noise reduction, normalization, simple VAD).
3. Fine-tune a Whisper model with LoRA.
4. Evaluate on unseen test data, including logging WER and domain-specific term accuracy.

 Quick Note: If you do not want to train the model yourself, you can use this one (10 epochs trained): https://drive.google.com/file/d/1dV1jqDTSRcdWmeVHzs_7esEE5Bqbqg5g/view?usp=drive_link . Please make sure to unzip the folder contents into lora_whisper_model folder.

## Prerequisites

- Python 3.8+ recommended
- Install necessary libraries:

```bash
pip install -r requirements.txt
```

## Folder Structure
.<br/>
├─ lora_whisper_model/<br/> 
├─ data/<br/> 
│  ├─ noise_samples/<br/> 
│  ├─ generated_synthetic_data/<br/> 
│  └─ test/<br/> 
├─ src/<br/> 
│  ├─ data_preprocessing.py<br/> 
│  ├─ dataset_creation.py<br/> 
│  ├─ finetune_lora.py<br/> 
│  └─ evaluation.py<br/> 
├─ requirements.txt<br/> 
└─ README.md<br/> 


## Steps
1. Generate the Dataset
```bash
python src/dataset_creation.py
```

2. Fine-Tune the Model with LoRA (data preprocessing included) - please remember to delete the placeholder in the folder and do not omit this step, as the model size is to big to be pushed to github repository.
```bash
python src/finetune_lora.py
```

3. Evaluate the model
```bash
python src/evaluation.py
```

4. (Optional) Try the inference with your own voice recording
```bash
python inference.py --model_dir lora_whisper_model --audio_path your_audio_file.wav
```

### Customization
- Number of epochs: Adjust num_epochs in finetune_lora.py if you want more training.<br/> 
- Batch size: Increase it if you have a good GPU or reduce if you run out of memory.<br/> 
- Noise mixing: Tweak the SNR range or add additional noise files to get more variety.<br/> 
- VAD threshold**: In data_preprocessing.py, you can modify threshold=0.02 to be more or less aggressive in boosting speech frames.<br/> 

### Notes
On Windows, if pyttsx3 fails to generate .wav files, ensure you have SAPI5 voices installed and the correct permissions.
