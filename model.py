import pandas as pd
import re
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from stopwords import get_stopwords

# nltk.download('wordnet')
# nltk.data.path.append('/usr/local/share/nltk_data')

nltk.data.path.append('/usr/local/share/nltk_data')  # specify your local directory for nltk data
# nltk.download('wordnet', download_dir='./nltk_data')

# Load the datasets
df_train = pd.read_csv("train.txt", delimiter=';', names=['text', 'sentiment'])
df_val = pd.read_csv("val.txt", delimiter=';', names=['text', 'label'])

# Concatenate the dataframes and reset the index
df = pd.concat([df_train, df_val])

# Custom encoder function to convert sentiment labels to binary values
def custom_encoder(df):
    df.replace(to_replace="surprise", value=1, inplace=True)
    df.replace(to_replace="love", value=1, inplace=True)
    df.replace(to_replace="joy", value=1, inplace=True)
    df.replace(to_replace="fear", value=0, inplace=True)
    df.replace(to_replace="anger", value=0, inplace=True)
    df.replace(to_replace="sadness", value=0, inplace=True)
    return df

df['label'] = custom_encoder(df['label'])
df.dropna(subset=['label'], inplace=True)

# Data Preprocessing
lm = WordNetLemmatizer()

def text_transformation(df_col):
    corpus = []
    for text in df_col:
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
corpus = text_transformation(df['text'])

# Vectorize the text data
cv = CountVectorizer(ngram_range=(1, 2))
X = cv.fit_transform(corpus)
y = df['label']

# Train the model with the best parameters (pre-determined)
rfc = RandomForestClassifier(
    max_features='sqrt',  # changed from 'auto' to 'sqrt'
    max_depth=None,
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True
)
rfc.fit(X, y)

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
    expression_check(predictions[0])

# Loop for user input
if len(sys.argv) > 1:
    input_text = sys.argv[1]
    sentiment_predictor(input_text)
    print("\n")

else:
    print(" No statement provided.")
