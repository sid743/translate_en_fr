# NeuralSeq: English-French Neural Machine Translation

This project implements a complete Neural Machine Translation (NMT) pipeline using a T5 transformer model. It demonstrates synthetic data generation, model fine-tuning using PyTorch/Hugging Face, and a user-friendly GUI application for real-time translation.

## 📂 Project Structure

- `gen_syn_pairs.py`: A script that procedurally generates synthetic English–French sentence pairs (grammar-aware).
- `train_en_fr.py`: The training pipeline that fine-tunes a `t5-small` model on the generated dataset.
- `translate_app.py`: A modern GUI application (built with CustomTkinter) to demonstrate the model.
- `requirements.txt`: List of Python dependencies.

## 🚀 Setup & Installation

### 1. Prerequisites

- Python 3.8 or higher
- (Optional) A GPU is recommended for training, but the app will run fine on CPU

### 2. Installation

Open your terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

## 📥 Model Setup (Required)

Due to GitHub's file size limits, the fine-tuned model is hosted externally. You must download it to run the app.

1. Download the model here: **[LINK TO YOUR GOOGLE DRIVE ZIP]** or from the .zip file uploaded on google classroom
2. Unzip the downloaded file
3. **Rename** the extracted folder to exactly: `mt_en_fr_t5_final`
4. **Move** the `mt_en_fr_t5_final` folder into this project directory  
   (it should be in the same folder as `translate_app.py`)

## 🖥️ Running the Application

To start the translation interface:

```bash
python translate_app.py
```

1. Wait for the status bar to show **"✅ Model loaded successfully"**
2. Type an English sentence (e.g., *"Please send the report tomorrow"*)
3. Click **TRANSLATE**

DONE 
Thank you, 
Regards,
Niyati, Reva and Siddharth

## 🔬 Reproducibility (Training from Scratch)

If you wish to reproduce the training process entirely instead of using the pre-trained model:

### Step 1: Generate Data

```bash
python gen_syn_pairs.py --num_pairs 40000 --output synthetic_en_fr.tsv
```

This generates **40,000 synthetic sentence pairs** based on grammatical templates.

### Step 2: Train the Model

```bash
python train_en_fr.py
```

This script fine-tunes **T5-small** and saves the artifacts to the `./mt_en_fr_t5_final` directory.

## 🛠️ Technologies Used

- **Model Architecture**: T5-Small (Transformer)
- **Frameworks**: PyTorch, Hugging Face Transformers
- **GUI**: CustomTkinter
- **Data**: Procedural Synthetic Generation
