from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle

# Load the model and vectorizer
model = pickle.load(open('sentimentmodel.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Assuming you saved the vectorizer during training
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        user_input = request.form['inputText']
        user_input = stemming(user_input)
        user_input_transformed = vectorizer.transform([user_input])  # Ensure input is in list format
        prediction_data = model.predict(user_input_transformed)
        prediction = "Positive" if prediction_data[0] == 1 else "Negative"
        return render_template('index.html', prediction_data=prediction)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
