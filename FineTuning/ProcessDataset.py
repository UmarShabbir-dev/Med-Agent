from datasets import load_dataset
import json
from pathlib import Path

def format_dataset_to_alpaca(output_path:str):
    print("A")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")["train"] # loading MedQA dataset from Huggingface and just selecting the training split
    print(f"{len(dataset)} loaded")
    
    
    formatted = []  # empty list to store final JSON entries
    for item in dataset:
        question = item['question']
        options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item['options'])]) # to get all the options of a Question like A,B,C ...
        answer = item['answer']
        explanation = item.get("explanation", "")
        
        formatted.append({    # Formatted the list like Alpaca Format  
            "instruction": f"Beantworte folgende medizinische Frage:\n{question}\n\nOptionen:\n{options}",
            "input": "",
            "output" : f"Richtige Antwort: {answer}\n{explanation}"
        })
        
    with open(output_path, "w", encoding="utf-8") as f:   # Saving all the formatted enteries to JSON File
        for e in formatted:
            f.write(json.dumps(e,ensure_ascii=False) + "\n") #Ascii is set to false so that German Characters are saved properly
        
        
    print(f"MedQA data formatted to Json and written to {output_path}")
        
if __name__ == "__main__":
        Path("data/processed").mkdir(parents=True,exist_ok=True)
        format_dataset_to_alpaca("data/processed/medqa_alpaca.json")