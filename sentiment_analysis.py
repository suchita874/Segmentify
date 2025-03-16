### code for fine tune Model using DistlBERT and then predict value on new data

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

# Load dataset
dataset = load_dataset("imdb")

# Load pre-trained DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

#tokenized_datasets = dataset.map(tokenize_function, batched=True) # taking too long to execute

# Select only the first 1000 samples from each dataset subset
train_subset = dataset['train'].select(range(1000))
test_subset = dataset['test'].select(range(1000))
unsupervised_subset = dataset['unsupervised'].select(range(1000))

# Tokenize the subsets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Apply tokenization with a batch size of 900 (or any batch size that works)
tokenized_train = train_subset.map(tokenize_function, batched=True, batch_size=900)
tokenized_test = test_subset.map(tokenize_function, batched=True, batch_size=900)
tokenized_unsupervised = unsupervised_subset.map(tokenize_function, batched=True, batch_size=900)


tokenized_datasets = DatasetDict({
    'train': tokenized_train,
    'test': tokenized_test,
    'unsupervised': tokenized_unsupervised
})

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Test with a new review
review = "This movie was amazing! I loved it."
inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = outputs.logits
predicted_class = predictions.argmax().item()

if predicted_class == 1:
    print("Positive review!")
else:
    print("Negative review!")

# 0 - negative
