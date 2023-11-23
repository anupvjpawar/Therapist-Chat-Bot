import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import cv2  # Import OpenCV
import base64

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')


# Load data and preprocess
img1 = cv2.imread('url_to_not_flagged_image.jpg')
img2 = cv2.imread('url_to_flagged_image.jpg')
bg1 = cv2.imread('backgroundimg.jpg')
bg1_height, bg1_width, _ = bg1.shape

# Resize img1 to 25% of the size of bg1
img1 = cv2.resize(img1, (int(0.25 * bg1_width), int(0.25 * bg1_height)))
        
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#def chatbot():
data = pd.read_csv(r"Sheet_1.csv",encoding= "latin1" )

data.drop(["Unnamed: 3","Unnamed: 4","Unnamed: 5",
           "Unnamed: 6","Unnamed: 7",], axis = 1, inplace =True)
data = pd.concat([data["class"],data["response_text"]], axis = 1)
data.dropna(axis=0, inplace =True)
#print(data)
data["class"] = [1 if each == "flagged" else 0 for each in data["class"]]
data.response_text[16]

first_text = data.response_text[16]
text = re.sub("[^a-zA-Z]"," ",first_text)
text = text.lower() 
#print (text)
text = nltk.word_tokenize(text)

text = [ word for word in text if not word in set(stopwords.words("english"))]
lemmatizer = WordNetLemmatizer()
text = [(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"),pos = "v"),pos="a")) for word in text]
#print(text)
description_list = []
for description in data.response_text:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower() 

    description = nltk.word_tokenize(description)
    description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemmatizer = WordNetLemmatizer()
    description = (lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"),pos = "v"),pos="a") for word in description)

    description = " ".join(description)
    description_list.append(description)

max_features = 100
count_vectorizer = CountVectorizer(max_features=max_features)
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
#print("Top {} Most Used Words: {}".format(max_features,count_vectorizer.get_feature_names()))

y = data.iloc[:,0].values
x = sparce_matrix

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)
print("Accuracy: {}".format(round(nb.score(y_pred.reshape(-1,1),y_test),2)))
#print(text)
    
# Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1("BlissBot!!!", style={'color': 'blue'}),
    dcc.Textarea(id='user-input', placeholder='Enter your text here...', style={'width': '100%', 'fontSize': 20}),
    html.Button('Submit', id='submit-button', n_clicks=0, style={'background-color': 'green', 'color': 'white'}),
    dcc.Loading(id="loading-output", type="default", children=[
        html.Div(id='output-container'),
        html.Div(id='image-container')
    ])
], style={'backgroundImage': 'url("data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', bg1)[1]).decode('utf-8') + '")'})

@app.callback(
    [Output('output-container', 'children'),
     Output('image-container', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    if not user_input or n_clicks == 0:
        return "", ""

    user_input = re.sub("[^a-zA-Z]", " ", user_input.lower())
    user_input = nltk.word_tokenize(user_input)
    user_input = [word for word in user_input if not word in set(stopwords.words("english"))]
    lemmatizer = WordNetLemmatizer()
    user_input = " ".join(
        lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"), pos="v"), pos="a")
        for word in user_input
    )
    user_input_vectorized = count_vectorizer.transform([user_input]).toarray()
    prediction = nb.predict(user_input_vectorized)[0]

    result = "Flagged: This text may contain concerning content." if prediction == 1 else "Not Flagged: This text seems fine."

    # Display OpenCV images
    img_result = html.Img(src=byte_to_base64(cv2.imencode('.png', img1)[1]),
                         style={'width': '50%', 'height': '50%'}) if prediction == 0 else html.Img(
        src=byte_to_base64(cv2.imencode('.png', img2)[1]), style={'width': '50%', 'height': '50%'})

    return html.Div([
        html.P('Input Text: {}'.format(user_input)),
        html.P('Prediction: {}'.format(result, style={'color': 'red' if prediction == 1 else 'green'}))
    ]), img_result

def byte_to_base64(byte_array):
    return 'data:image/png;base64,' + base64.b64encode(byte_array).decode('utf-8')

if __name__ == '__main__':
    app.run_server(debug=True, port=8053)