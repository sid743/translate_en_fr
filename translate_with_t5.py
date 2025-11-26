from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_dir = "./mt_en_fr_t5_final"  # same as in training script

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

def translate(text: str, direction: str = "en-fr", max_length: int = 64):
    if direction == "en-fr":
        prefix = "translate English to French: "
    elif direction == "fr-en":
        prefix = "translate French to English: "
    else:
        raise ValueError("direction must be 'en-fr' or 'fr-en'")

    input_text = prefix + text
    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        length_penalty=0.8,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    while True:
        direction = input("Direction (en-fr / fr-en, or 'q' to quit): ").strip()
        if direction == "q":
            break
        text = input("Enter text: ")
        translation = translate(text, direction=direction)
        print("Translation:", translation)
        print()
