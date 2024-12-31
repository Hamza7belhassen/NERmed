from transformers import BertForTokenClassification, Trainer, TrainingArguments, BertTokenizer
from sklearn.model_selection import train_test_split
from datasets import DatasetDict





# Prepare data for training
def preprocess(data):
    # Tokenize the text
    encoding = tokenizer(data['text'], truncation=True, padding='max_length', max_length=512)
    tokens = encoding["input_ids"]

    # Create labels initialized to 0
    labels = [0] * len(tokens)

    # Iterate over the entities and assign token labels
    for entity in data["entities"]:
        # Get the character offsets of the entity in the text
        start_char = entity['start']
        end_char = entity['end']

        # Convert the start and end character positions to token indices
        start_token = encoding.char_to_token(start_char)
        end_token = encoding.char_to_token(end_char - 1)  # end is exclusive, so subtract 1

        if start_token is not None and end_token is not None:
            for idx in range(start_token, end_token + 1):
                labels[idx] = label_mapping.get(entity['entity_group'], 0)

    # Add the labels to the tokens dictionary
    encoding["labels"] = labels
    return encoding


processed_dataset = hf_dataset.map(preprocess)





# Convert processed dataset to a pandas DataFrame temporarily
processed_df = processed_dataset.to_pandas()

# Perform train/validation split
train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42)

# Convert back to Hugging Face Dataset
train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)

# Combine into DatasetDict
dataset = DatasetDict({"train": train_data, "validation": val_data})

# Load pre-trained BioBERT model and tokenizer
model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=29)
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="C:/Users/HassenBELHASSEN/Desktop/NER_med/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # Use training data
    eval_dataset=dataset["validation"],  # Use validation data
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("C:/Users/HassenBELHASSEN/Desktop/NER_med/final_model")
tokenizer.save_pretrained("C:/Users/HassenBELHASSEN/Desktop/NER_med/final_model")