from sklearn.metrics import classification_report

def evaluate_model(test_texts, test_labels):
    predictions = []
    for text in test_texts:
        entities = ner_pipeline(text)
        predictions.append(entities)
    print(classification_report(test_labels, predictions))

evaluate_model(test_texts, test_labels)
