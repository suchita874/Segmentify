import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


#read csv file
df = pd.read_csv('/Users/suchitamishra/Documents/pythonProjects/Segmentify/data.csv')

#check shape of the data set (shape is tuple)
print(df.shape)

#get an overview of dataset
print(df.info())

print(df.describe().T)

#improve column name
df.columns = df.columns.str.replace('Accepted', '')
print(df.head())

#To check the null values in the dataset.
for col in df.columns:
    temp = df[col].isnull().sum()
    if temp > 0:
        print(f"Column : {col} contains {temp} null values")


#drop null values
df = df.dropna()
print("size of dataframe", len(df))

# find total nu. of unique value in each column
print(df.nunique())

# Dt_Customer contains the date column
parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
df["day"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["year"] = parts[2].astype('int')

#now drop column which are having only single values and dt_customer 
df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis = 1, inplace = True)


# Data Visualization
floats, objects = [], []
for col in df.columns:
    if df[col].dtype == object:
        objects.append(col)
    elif df[col].dtype == float:
        floats.append(col)

print(objects)
print(floats)

#To get the count plot for the columns of the datatype – object
plt.subplots(figsize=(15, 10))
for i, col in enumerate(objects):
    plt.subplot(2, 2, i + 1)
    # Use melt to transform the data to long form 
    df_melted = df.melt(id_vars=[col], value_vars=['Response'], var_name='hue')      ##lets see the comparison of the features with respect to the values of the responses
    sb.countplot(x = df[col], palette = 'Set1',  hue='value', data=df_melted)
# plt.show()

print(df['Marital_Status'].value_counts())

# use label encoder to transform catagorical value to numerical value
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


#Heatmap
plt.figure(figsize=(15, 15))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
# plt.show()


#Standardization
scaler = StandardScaler()
data = scaler.fit_transform(df)


#Segmentation
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(df)
plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
# plt.show()


#There are certainly some clusters which are clearly visual from the 2-D representation of the given data. 
# Let’s use the KMeans algorithm to find those clusters in the high dimensional plane itself

error = []
for n_clusters in range(1, 21):
    model = KMeans(init='k-means++',
                   n_clusters=n_clusters,
                   max_iter=500,
                   random_state=22)
    model.fit(df)
    error.append(model.inertia_)

#Here inertia is nothing but the sum of squared distances within the clusters.

#Plot the Inertia (Elbow Method)
plt.figure(figsize=(10, 5))
sb.lineplot(x=range(1, 21), y=error)
sb.scatterplot(x=range(1, 21), y=error)
plt.show()


#Here by using the elbow method we can say that k = 6 is the optimal number of clusters
#that should be made as after k = 6 the value of the inertia is not decreasing drastically.

# create clustering model with optimal k=5
model = KMeans(init='k-means++',
               n_clusters=5,
               max_iter=500,
               random_state=22)
segments = model.fit_predict(df)

#Scatterplot will be used to see all the 6 clusters formed by KMeans Clustering.
plt.figure(figsize=(7, 7))
df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})
sb.scatterplot(x='x', y='y', hue='segment', data=df_tsne)
plt.show()




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
