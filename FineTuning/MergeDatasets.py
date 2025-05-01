import json
from pathlib import Path

def merge_datesets(file1, file2, output_path):
    merged = []
    
    print("Reading from file 1")
    with open(file1,"r",encoding="utf-8") as f1:
        for line in f1:
            merged.append(json.loads(line.strip()))
    
    
    print("Reading from file 2")
    with open(file2,"r", encoding="utf-8") as f2:
        for line in f2:
            merged.append(json.loads(line.strip()))
            
    print(f"Writing merged jsons to {output_path}")
    with open(output_path,"w", encoding="utf-8") as out:
        for item in merged:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Merged datasets saved to {output_path}")
    
    
if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    merge_datesets("data/processed/medqa_alpaca.json",
                   "data/processed/germeval_alpaca.json",
                   "data/processed/multilingual_alpaca.json")