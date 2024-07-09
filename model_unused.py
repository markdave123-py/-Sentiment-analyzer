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
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
# from scikitplot.metrics import plot_confusion_matrix
from stopwords import get_stopwords


# nltk.download('wordnet')

#Evaluate dataset and load them into dataframes



df_train = pd.read_csv("reduced.txt", delimiter=';', names=['text','sentiment'])
df_val = pd.read_csv("val.txt",delimiter=';',names=['text','label'])

#concatenate the dataframes and and an index  inplace dropping the old one

df = pd.concat([df_train,df_val])
df.reset_index(inplace=True,drop=True)
df.dropna(subset=['label'], inplace=True)



def custom_encoder(df):
    df.replace(to_replace ="surprise", value =1, inplace=True)
    df.replace(to_replace ="love", value =1, inplace=True)
    df.replace(to_replace ="joy", value =1, inplace=True)
    df.replace(to_replace ="fear", value =0, inplace=True)
    df.replace(to_replace ="anger", value =0, inplace=True)
    df.replace(to_replace ="sadness", value =0, inplace=True)

    return df


custom_encoder(df['label'])
df['label'].dropna(inplace=True)

# Check the class distribution in the dataset
# print(df['label'].value_counts())



#Data Preprocessing

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
        review = [lm.lemmatize(word) for word in review if not word in set(get_stopwords('en'))]
        if review:  # Ensure that the review is not empty
            corpus.append(' '.join(review))
    return corpus


corpus = text_transformation(df['text'])
# print(corpus)


cv = CountVectorizer(ngram_range=(1,2))

train_data = cv.fit_transform(corpus)

X = train_data

y = df['label']

parameters = {'max_features': ('auto','sqrt'),
             'n_estimators': [500, 1000, 1500],
             'max_depth': [5, 10, None],
             'min_samples_split': [5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10],
             'bootstrap': [True, False]}


grid_search = GridSearchCV(RandomForestClassifier(),parameters,cv=5,return_train_score=True,n_jobs=-1)
grid_search.fit(X,y)
grid_search.best_params_


# for i in range(432):
#     print('Parameters: ',grid_search.cv_results_['params'][i])
#     print('Mean Test Score: ',grid_search.cv_results_['mean_test_score'][i])
#     print('Rank: ',grid_search.cv_results_['rank_test_score'][i])


rfc = RandomForestClassifier(max_features=grid_search.best_params_['max_features'],max_depth=grid_search.best_params_['max_depth'],
                                  n_estimators=grid_search.best_params_['n_estimators'],min_samples_split=grid_search.best_params_['min_samples_split'],                                    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                    bootstrap=grid_search.best_params_['bootstrap'])
rfc.fit(X,y)


test_df = pd.read_csv('test.txt',delimiter=';',names=['text','label'])


X_test , y_test = test_df.text,test_df.label

test_df = custom_encoder(y_test)

test_corpus = text_transformation(X_test)

testdata = cv.transform(test_corpus)

predictions = rfc.predict(testdata)

# acc_score = accuracy_score(y_test,predictions)
# pre_score = precision_score(y_test,predictions)
# rec_score = recall_score(y_test,predictions)
# print('Accuracy_score: ',acc_score)
# print('Precision_score: ',pre_score)
# print('Recall_score: ',rec_score)
# print("-"*50)
# cr = classification_report(y_test,predictions)
# print(cr)




def expression_check(prediction_input):
    if prediction_input == 0:
        print("Input statement has Negative Sentiment.")
    elif prediction_input == 1:
        print("Input statement has Positive Sentiment.")
    else:
        print("Invalid Statement.")


def sentiment_predictor(input):
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = rfc.predict(transformed_input)
    expression_check(prediction)


input1 = ["I bought a new phone and it's so good."]
# input2 = ["I bought a new phone and it's so good."]

sentiment_predictor(input1)
# sentiment_predictor(input2)