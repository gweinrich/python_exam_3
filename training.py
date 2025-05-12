import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
from spacy.training import Example, validate_examples
import random
import pandas as pd
import os

from spacy.training import offsets_to_biluo_tags
from spacy.tokens import Doc

from spacy.training import offsets_to_biluo_tags

def check_entity_alignment(nlp, text, entities):
    doc = nlp.make_doc(text)
    # raises ValueError if misaligned
    _ = offsets_to_biluo_tags(doc, entities)

    
def clean_and_validate_entities(nlp, text, entities):
    doc = nlp.make_doc(text)
    valid_entities = []
    seen_tokens = set()

    for start, end, label in entities:
        # Strip leading and trailing spaces to ensure correct alignment
        span_text = text[start:end].strip()
        clean_start = start + len(text[start:end]) - len(span_text.lstrip())
        clean_end = end - len(span_text) + len(span_text.rstrip())

        span = doc.char_span(clean_start, clean_end, label=label)
        if span is None:
            print(f"Misaligned entity: {text[start:end]} (start: {start}, end: {end})")
            continue  # Skip misaligned entity
        span_tokens = set(range(span.start, span.end))
        if span_tokens & seen_tokens:
            continue
        seen_tokens.update(span_tokens)
        valid_entities.append((clean_start, clean_end, label))

    return valid_entities

import json

def convert_ndjson_to_json_array(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = [json.loads(line.strip()) for line in lines]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def reformat_training_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    reformatted_data = []

    for resume in data:
        content = resume.get('content', '')
        annotations = resume.get('annotation', [])

        reformatted_annotations = []
        
        for annotation in annotations:
            # Check if 'label' exists and has elements
            if 'label' in annotation and annotation['label']:
                label = annotation['label'][0]  # Assuming one label per annotation
                start = annotation['points'][0]['start']
                end = annotation['points'][0]['end']
                text = content[start:end]
                reformatted_annotations.append({
                    'text': text,
                    'start': start,
                    'end': end,
                    'label': label
                })
            else:
                print(f"Skipping annotation with no label: {annotation}")

        reformatted_data.append({
            'content': content,
            'annotations': reformatted_annotations
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reformatted_data, f, indent=2, ensure_ascii=False)




# convert_ndjson_to_json_array('./Entity Recognition in Resumes.json', './converted.json')
# reformat_training_data('./converted.json', './reformatted_data.json')


# # Optional: use GPU if available
# try:
#     spacy.require_gpu()
#     print("Using GPU")
# except Exception:
#     print("GPU not available. Falling back to CPU.")

# # Load and clean data
# df_data = pd.read_json("./reformatted_data.json")

# train_data = []

# nlp_tmp = spacy.blank("en")  # Temp pipeline for validation

# for _, row in df_data.iterrows():
#     text = row.get("content", "")
#     raw_annotations = row.get("annotations", [])  # Use .get() to avoid missing key error
    
#     if not raw_annotations:
#         print(f"No annotations found for content: {text}")
#         continue  # Skip this entry if no annotations exist
    
#     formatted_entities = []
#     for entry in raw_annotations:
#         label = entry.get("label", [])
#         if not label or not entry.get("points"):
#             continue
#         for point in entry["points"]:
#             try:
#                 start = point["start"]
#                 end = point["end"]
#                 lbl = label[0]
#                 formatted_entities.append((start, end, lbl))
#             except:
#                 continue

#     cleaned_entities = clean_and_validate_entities(nlp_tmp, text, formatted_entities)
#     train_data.append((text, {"entities": cleaned_entities}))


# # Optional: Inspect sample
# print("Sample training data:")
# for i in range(3):
#     print(train_data[i])

# # Validate examples
# # Temporary pipeline for validation
# temp_nlp = spacy.blank("en")
# temp_ner = temp_nlp.add_pipe("ner")

# # Add all labels before creating examples
# for _, annotations in train_data:
#     for ent in annotations.get("entities", []):
#         temp_ner.add_label(ent[2])

# for text, annot in train_data:
#     if not annot["entities"]:
#         print("Empty entity list for:", text[:100], "...")


# # Create and validate examples
# examples = []
# for text, ann in train_data:
#     doc = temp_nlp.make_doc(text)
#     examples.append(Example.from_dict(doc, ann))

# try:
#     validate_examples(examples, temp_nlp)
#     print("Training examples validated successfully.")
# except Exception as e:
#     print("Validation failed:", e)
#     exit(1)


# Training function
def train_ner_model(train_data, iterations=30):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)  # Add NER component to the pipeline
    
    # Add all labels to the NER pipeline
    for _, annotations in train_data:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[2])  # Add label to the NER model
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # Disable other pipes during training
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                # This is where the error occurs, nlp.update will use the examples for training
                nlp.update(examples, drop=0.5, losses=losses)
            print(f"Iteration {itn+1}/{iterations}, Losses: {losses}")
    return nlp

def remove_overlapping_entities(entities):
    # Sort entities by start index
    entities = sorted(entities, key=lambda x: x[0])
    cleaned = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:
            cleaned.append((start, end, label))
            last_end = end
        # If there's an overlap, skip the entity
    return cleaned


# # Train and save model
# print("Training spaCy NER model...")
# ner_model = train_ner_model(train_data)
# ner_model.to_disk("./domain_ner_model")
# print("Model saved to './domain_ner_model'")


# Step 1: Load JSON annotation file

# Step 2: Add the cleaning/conversion function

def load_all_json_files(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        all_data.append(record)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in {filename}")
    return all_data

def convert_annotations(data, nlp):
    cleaned = []
    for i, record in enumerate(data):
        if 'text' not in record or 'annotations' not in record:
            print(f"Skipping record {i}: Missing 'text' or 'annotations'")
            continue
        text = record['text']
        annotations = record['annotations']

        # Convert to (start, end, label) format
        entities = [(start, end, label.split(":")[-1].strip()) for start, end, label in annotations]

        try:
            check_entity_alignment(nlp, text, entities)
            cleaned.append((text, {"entities": entities}))
        except ValueError as e:
            print(f"Skipping record {i} due to alignment error: {e}")
            continue
    return cleaned

# === MAIN ===
folder_path = './ResumesJsonAnnotated/ResumesJsonAnnotated'
nlp_tmp = spacy.blank("en")
raw_data = load_all_json_files(folder_path)
cleaned_data = convert_annotations(raw_data, nlp_tmp)

# Train the model as usual
ner_model = train_ner_model(cleaned_data)
ner_model.to_disk("./domain_ner_model")
