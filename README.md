## Customer Segmentation and Sentiment Analysis Using KMeans and DistilBERT
This project performs customer segmentation and sentiment analysis using a combination of KMeans clustering and DistilBERT for natural language processing (NLP). The following steps are involved:
1. Data Preprocessing and Clustering: Data is loaded, preprocessed, and KMeans clustering is applied to segment customers.
2. Fine-tuning DistilBERT: A pre-trained DistilBERT model is fine-tuned on the IMDb dataset to classify movie reviews into positive or negative sentiments.

## Data Preprocessing and Clustering
The first part of the project involves loading and preprocessing customer data from a CSV file. Then, KMeans clustering is applied to identify customer segments. The dataset includes the following preprocessing steps:
* Null Value Handling: Drop any rows with missing values.
* Feature Engineering: Extract day, month, and year from the 'Dt_Customer' column.
* Encoding Categorical Data: Categorical columns are encoded using LabelEncoder.
* Standardization: Features are standardized using StandardScaler to scale the data before applying KMeans.
* Clustering:  Using the KMeans algorithm, we create segments of customers based on the preprocessed data.

## Fine-tuning DistilBERT for Sentiment Analysis
The second part of the project involves fine-tuning a pre-trained DistilBERT model to perform sentiment analysis on IMDb movie reviews. The datasets library is used to load the IMDb dataset, and DistilBERT is fine-tuned using the transformers library.

##  Evaluating the Model
The model is evaluated using the evaluation method provided by the Trainer class. After fine-tuning, we test the model with a sample review to predict the sentiment (positive or negative).

## Results
After fine-tuning the DistilBERT model, the sentiment of reviews is predicted with high accuracy. The final results will include both evaluation metrics and a prediction for any new text input.

## Conclusion
This project provides an end-to-end workflow for customer segmentation using KMeans clustering and sentiment analysis using fine-tuned DistilBERT. By applying these techniques, businesses can better understand their customer base and classify reviews automatically for customer feedback analysis.