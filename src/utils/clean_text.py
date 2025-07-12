import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english')) - {'why', 'how', 'what', 'when', 'where', 'who', 'do', 'does'}
lemmatizer = WordNetLemmatizer()

special_corrections = {
    "citriscidal": "citricidal",
    "mormens": "mormons",
    "blaphsemy": "blasphemy",
    "sholomister": "sholomister",
    "pterofobia": "pterophobia",
    "asparugus": "asparagus"
}


def clean_text(text, model_type='tfidf'):
    text = re.sub(r"http\S+|www\S+", "", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [special_corrections.get(word, word) for word in tokens]
    if model_type == 'tfidf':
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    else:
        tokens = [word for word in tokens if word.strip()]

    return ' '.join(tokens)
