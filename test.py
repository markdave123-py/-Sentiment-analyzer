import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from stopwords import get_stopwords

# Download the necessary NLTK data files
nltk.download('wordnet')

# Load the datasets
df_train = pd.read_csv("train.txt", delimiter=';', names=['text', 'sentiment'])
df_val = pd.read_csv("val.txt", delimiter=';', names=['text', 'label'])

# Concatenate the dataframes and reset the index
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)
df.dropna(subset=['label'], inplace=True)

# Sample a subset of the data for easier training
df_sample, _ = train_test_split(df, test_size=0.9, stratify=df['label'], random_state=15)

# Custom encoder function to convert sentiment labels to binary values
def custom_encoder(df):
    df.replace(to_replace="surprise", value=1, inplace=True)
    df.replace(to_replace="love", value=1, inplace=True)
    df.replace(to_replace="joy", value=1, inplace=True)
    df.replace(to_replace="fear", value=0, inplace=True)
    df.replace(to_replace="anger", value=0, inplace=True)
    df.replace(to_replace="sadness", value=0, inplace=True)
    return df

df_sample['label'] = custom_encoder(df_sample['label'])
df_sample.dropna(subset=['label'], inplace=True)

# Data Preprocessing
lm = WordNetLemmatizer()

def text_transformation(df_col):
    corpus = []
    for text in df_col:
        # Skip NaN values
        if pd.isna(text):
            continue
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lm.lemmatize(word) for word in review if word not in set(get_stopwords('en'))]
        if review:  # Ensure that the review is not empty
            corpus.append(' '.join(review))
    return corpus

# Transform the text data
corpus = text_transformation(df_sample['text'])

# Vectorize the text data
cv = CountVectorizer(ngram_range=(1, 2))
X = cv.fit_transform(corpus)
y = df_sample['label']

# Define the parameters for GridSearchCV
parameters = {
    'max_features': ['auto', 'sqrt'],
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
grid_search.fit(X, y)
print("Best Parameters: ", grid_search.best_params_)

# Train the model with the best parameters
rfc = RandomForestClassifier(
    max_features=grid_search.best_params_['max_features'],
    max_depth=grid_search.best_params_['max_depth'],
    n_estimators=grid_search.best_params_['n_estimators'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    bootstrap=grid_search.best_params_['bootstrap']
)
rfc.fit(X, y)

# # Load the test dataset
# test_df = pd.read_csv('test.txt', delimiter=';', names=['text', 'label'])

# # Encode the labels in the test data
# test_df['label'] = custom_encoder(test_df['label'])
# y_test = test_df['label']

# Transform the test text data
# X_test = text_transformation(test_df['text'])
# testdata = cv.transform(X_test)

# Make predictions
# predictions = rfc.predict(testdata)

# Evaluate the model
# print(f'Accuracy Score: {accuracy_score(y_test, predictions)}')
# print(f'Precision Score: {precision_score(y_test, predictions)}')
# print(f'Recall Score: {recall_score(y_test, predictions)}')
# print("-" * 50)
# print(f'Classification Report:\n{classification_report(y_test, predictions)}')

# Define the sentiment prediction functions
def expression_check(prediction_input):
    if prediction_input == 0:
        print("Input statement has Negative Sentiment.")
    elif prediction_input == 1:
        print("Input statement has Positive Sentiment.")
    else:
        print("Invalid Statement.")

def sentiment_predictor(input_texts):
    if not isinstance(input_texts, list):
        input_texts = [input_texts]
    input_texts = text_transformation(input_texts)
    transformed_input = cv.transform(input_texts)
    predictions = rfc.predict(transformed_input)
    # for prediction in predictions:
    expression_check(predictions)

# Test the sentiment prediction functions
# input1 = ["this is a very wicked thing to do, you are stupid."]
# # input2 = ["You are a terrible person and it is bad."]

while True:
    input1 = input("Enter text: ")
    sentiment_predictor(input1)
    print("\n")


# sentiment_predictor(input1)
