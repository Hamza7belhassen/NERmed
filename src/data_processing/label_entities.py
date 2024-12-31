import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import DatasetDict


# Load your CSV file
df = pd.read_csv("data/final.csv")

# Extract relevant fields for NER (e.g., Abstract, Document Content)
texts = df['Abstract'].fillna('') + ' ' + df['Document Content'].fillna('')

# Combine text with a unique ID for tracking
data = [{"id": i, "text": text} for i, text in enumerate(texts)]

# Load pre-trained model and tokenizer
model_name = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Process your dataset
labeled_data = []
for entry in data:
    entities = ner_pipeline(entry['text'])
    labeled_data.append({"id": entry['id'], "text": entry['text'], "entities": entities})


labeled_df = pd.DataFrame([
    {"id": item["id"], "text": item["text"], "entities": item["entities"]}
    for item in labeled_data
])
labeled_df.to_csv("labeled_dataset.csv", index=False)

# Extract all unique entities dynamically
unique_labels = set()
for item in labeled_data:  # Replace `labeled_data` with your dataset variable
    for entity in item["entities"]:
        unique_labels.add(entity["entity_group"])

# Create a mapping dynamically
label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}


# Convert labeled data to HF dataset
hf_dataset = Dataset.from_pandas(labeled_df)


