
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


### Mapping of word index to word
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('simple_rnn_imdb.h5')

### Helper function to decode review
def decode_review(review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in review])

### Function to preprocess review
def preprocess_review(review):
    review = review.lower().split()
    review = [word_index.get(i, 2) + 3 for i in review]
    review = sequence.pad_sequences([review], maxlen=500)
    return review

### Function to predict sentiment
def predict_sentiment(review):
    review = preprocess_review(review)
    prediction = model.predict(review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


### Streamlit code
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict sentiment')

### Input text box for user input
review = st.text_area('Enter a review')

if st.button('Predict'):
    if review:
        sentiment, confidence = predict_sentiment(review)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Confidence: {confidence:.4f}')
    else:
        st.write('Please enter a review')