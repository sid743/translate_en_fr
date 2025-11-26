# translate_en_fr
transformer to translate pair en&lt;->fr
Synthetic English-French Translation with T5
This project demonstrates how to generate a synthetic dataset of English/French sentence pairs and use it to fine-tune a T5-small model for bidirectional translation.

Prerequisites
Ensure you have Python installed along with the following libraries:

Bash

pip install torch transformers datasets
Usage Pipeline
Follow these three steps to generate data, train the model, and test translations.

1. Generate Data
Create the synthetic dataset. This script generates random sentence pairs based on predefined vocabulary templates.

Bash

# Generates 'en_fr_synthetic.tsv' with 80,000 pairs
python gen_syn_pairs.py --output en_fr_synthetic.tsv
2. Train Model
Fine-tune the t5-small model. This script loads the TSV, creates a bidirectional dataset (En→Fr and Fr→En), and trains the model.

Bash

# Trains the model and saves it to './mt_en_fr_t5_final'
python train_en_fr.py
Note: This may take some time depending on your hardware (GPU recommended).

3. Run Translation
Load the trained model and translate text interactively.

Bash

python translate_with_t5.py
Interactive Commands:

Enter direction: en-fr or fr-en

Enter text: e.g., "Alice likes pizza."

Press 'q' to quit.

Project Structure
gen_syn_pairs.py: Script to create synthetic English/French sentence pairs (TSV format).

train_en_fr.py: Main training script. Preprocesses data with T5 prefixes ("translate English to French: "), handles tokenization, and runs the training loop.

translate_with_t5.py: Inference script to load the saved model and run translations.

en_fr_synthetic.tsv: The generated dataset file.
