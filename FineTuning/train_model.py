import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

Base_Model = "mistralai/Mistral-7B-v0.1"
Data_Path = "data/processed/multilingual_alpaca.json"

def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [{"instruction": json.loads(line)["instructions"], "output": json.loads(line)["output"]} for line in lines]


def preprocess(sample):
    prompt = f"{sample['instruction']}\n\nAntwort:"
    input_text = prompt+sample["output"] 
    return tokenizer(input_text, truncation=True, padding="max_length", max_length=512)

if __name__ == "__main__":
    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(Base_Model)
    model = AutoModelForCausalLM.from_pretrained(Base_Model, load_in_4bit=True, device_map="auto")
    
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, config) 
    print("Loading and formatting dataset")
    raw_dataset = load_json_dataset(Data_Path)
    tokenized_dataset = list(map(preprocess,raw_dataset))
    
    
    print("Setting up trainer...")
    training_args = TrainingArguments(
        output_dir="./lora-mistral-medwise",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=2,
        save_strategy="no",
        fp16=True,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("🏋️‍♂️ Starting fine-tuning...")
    trainer.train()

    print("Training complete. Saving model")
    model.save_pretrained("medwise-mistral-lora")
    tokenizer.save_pretrained("medwise-mistral-lora")