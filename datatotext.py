import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import os

# Load JSON dataset
with open("cgnew.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON data into a format suitable for fine-tuning
texts = []

for module in data:
    for topic in module["topics"]:
        notes = topic["note"]
        for key, value in notes.items():  # Extracts all categories: 'most_important', 'next_important', etc.
            if isinstance(value, list):  # Ensure value is a list before extending
                texts.extend(value)

dataset = Dataset.from_dict({"text": texts})

# Load pretrained model and tokenizer
model_name = "gpt2-medium"  # Change model as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix: Assign a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding



def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512, 
        return_tensors="pt"
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define training arguments
output_dir = "./trained_chatbot_model"  # Folder for saving trained model
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,  # Increased batch size
    gradient_accumulation_steps=8,  # Helps when batch size is small
    num_train_epochs=5,  # More epochs for better learning
    warmup_steps=500,  # Helps with stable training
    weight_decay=0.01,  # Regularization to prevent overfitting
    save_strategy="epoch",
    evaluation_strategy="epoch",  # Enables evaluation per epoch
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    save_total_limit=2,  # Keeps only recent checkpoints
    remove_unused_columns=False,
    learning_rate=5e-5,  # Tuned learning rate
    lr_scheduler_type="cosine",  # Smooth learning rate decay
    do_train=True,
    do_eval=True,
)

# Initialize model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)  # ✅ Forces CPU compatibility


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ Uses GPU if available

model.to(device)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.shuffle(seed=42).select(range(len(texts) // 10)),  # Use 10% of data for evaluation
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save the trained model
trainer.save_model(output_dir)
