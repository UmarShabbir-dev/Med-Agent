from datasets import load_dataset
import json
from pathlib import Path

def process_and_format_germeval(output_path:str):
    print("Reading Germeval file data")
    
    dataset = load_dataset("germeval_14", split="train", trust_remote_code=True)
    label_list = dataset.features["ner_tags"].feature.names
    formatted = []
    for example in dataset:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        sentence = " ".join(tokens)
        entities = []
        entity_tokens = []
        entity_label = None

        for token, tag in zip(tokens, ner_tags):
            label = label_list[tag]
            if label.startswith("B-"):
                if entity_tokens:
                    entities.append((entity_tokens, entity_label))
                entity_tokens = [token]
                entity_label = label[2:]
            elif label.startswith("I-") and entity_label:
                entity_tokens.append(token)
            else:
                if entity_tokens:
                    entities.append((entity_tokens, entity_label))
                    entity_tokens = []
                    entity_label = None

        if entity_tokens:
            entities.append((entity_tokens, entity_label))

        if entities:
            output = ", ".join([f"[{' '.join(ent)}] = {label}" for ent, label in entities])
        else:
            output = "Keine benannten Entitäten gefunden."

        formatted.append({
            "instruction": "Markiere die benannten Entitäten im folgenden Satz.",
            "input": sentence,
            "output": output
        })
        
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in formatted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_and_format_germeval("data/processed/germeval_alpaca.json")