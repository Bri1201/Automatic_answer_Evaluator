import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
import string


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN


def preprocess_tokens(tokens):
    tokens = [token.lower() for token in tokens]
    tagged_tokens = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
    return lemmatized_tokens


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stop_words.update(list(string.punctuation))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    preprocessed_tokens = preprocess_tokens(filtered_tokens)
    return ' '.join(preprocessed_tokens), preprocessed_tokens


def preprocess_sentences(text):
    sentences = sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence)[0] for sentence in sentences]
    return preprocessed_sentences
