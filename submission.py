import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import torch

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm_notebook 

with open('mypersonality_final.csv', 'r', encoding='utf-8', errors='ignore') as file:
    data = pd.read_csv(file)


# Here we saw that if person have ['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN'] greater than 3 then it is yes for that column else no in ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
# So we keep only ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN'] columns as they show if that personality is yes or no by that user.

data_trim = data[['STATUS', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

replacement_map = {'n': 0, 'y': 1}
columns_to_replace = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
data_trim[columns_to_replace] = data_trim[columns_to_replace].replace(replacement_map)
data_trim.to_csv('data_trim.csv', index=False) # saving the table


# Here we have 5 personallity 'Openness': 'cOPN', 'Conscientiousness': 'cCON', 'Extraversion': 'cEXT','Agreeableness': 'cAGR','Neuroticism': 'cNEU'
# - Here we can see person can have multiple personality. So each one of them can be considered as a different output.
# - So we divide the data into 5 parts and train 5 different models for each personality.

# - We will do tfidf vectorization on the posts and then train the model using this vector as imput.


###########33
#USING RANDOM FOREST

def train_ml(dataset, column_name):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    X = tfidf_vectorizer.fit_transform(dataset['STATUS'])
    y = dataset[column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=50, random_state=42) 
    model.fit(X_train, y_train)
    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy



X1 = data_trim[['STATUS', 'cEXT']]
x1_accuracy = train_ml(X1,'cEXT' )
print('Accuracy wrt personality Extraversion colum name cEXT',x1_accuracy)

X2 = data_trim[['STATUS', 'cNEU']]
x2_accuracy = train_ml(X2,'cNEU' )
print('Accuracy wrt personality Neuroticism colum name cNEU',x2_accuracy)

X3 = data_trim[['STATUS', 'cAGR']]
x3_accuracy = train_ml(X3,'cAGR' )
print('Accuracy wrt personality Agreeableness colum name cAGR',x3_accuracy)

X4 = data_trim[['STATUS', 'cEXT']]
x4_accuracy = train_ml(X4,'cEXT' )
print('Accuracy wrt personality Conscientiousness colum name cCON',x4_accuracy)

X5 = data_trim[['STATUS', 'cOPN']]
x5_accuracy = train_ml(X5,'cOPN' )
print('Accuracy wrt personality Openness colum name cOPN',x5_accuracy)


##################################3
# Lets try to improve the accuracy
# WE USE BERT


def train_bert(dataset, epochs, column_name):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset['STATUS'], dataset[column_name], test_size=0.2, random_state=42)

    # Load the pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 for binary classification

    # Tokenize the text data and prepare input tensors for X_train
    max_length = 128

    input_ids_train = []
    attention_masks_train = []

    for text in X_train:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids_train.append(encoded_dict['input_ids'])
        attention_masks_train.append(encoded_dict['attention_mask'])

    input_ids_train = torch.cat(input_ids_train, dim=0)
    attention_masks_train = torch.cat(attention_masks_train, dim=0)

    # Create DataLoader for batch processing
    batch_size = 32
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, torch.tensor(y_train.tolist()))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

    # Fine-tune the BERT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = epochs  # You can adjust this based on your dataset and resources

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}")
    # Model Evaluation
    model.eval()
    y_pred = []
    for text in X_test:
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            y_pred.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


x1_accuracy_bert = train_bert(X1, 4, column_name = 'cEXT')
print('Accuracy wrt personality Extraversion colum name cEXT',x1_accuracy_bert)

X2 = data_trim[['STATUS', 'cNEU']]
x2_accuracy_bert = train_bert(X2,epochs = 4, column_name = 'cNEU' )
print('Accuracy wrt personality Neuroticism colum name cNEU',x2_accuracy_bert) 

X3 = data_trim[['STATUS', 'cAGR']]
x3_accuracy_bert = train_bert(X3,4, 'cAGR' )
print('Accuracy wrt personality Agreeableness colum name cAGR',x3_accuracy_bert)

X4 = data_trim[['STATUS', 'cEXT']]
x4_accuracy_bert = train_bert(X4,4,'cEXT' )
print('Accuracy wrt personality Conscientiousness colum name cCON',x4_accuracy_bert)

X5 = data_trim[['STATUS', 'cOPN']]
x5_accuracy_bert = train_bert(X5,4, 'cOPN' )
print('Accuracy wrt personality Openness colum name cOPN',x5_accuracy_bert)

# We can see using only 4 epochs we can achive better results than random forest. By training with more epochs we can attain better accuracy

