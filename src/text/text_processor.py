import nltk as nl
from nltk.stem.porter import *
from nltk.corpus import stopwords
from string import punctuation

stemmer = PorterStemmer()


def remove_stop_words(words):
    """
    Will remove stop words from a list
    :param words: input list
    :return: output list of words without stop words.
    """
    custom_stop_words = set(stopwords.words('english') + list(punctuation))
    return [stemmer.stem(word) for word in words if word not in custom_stop_words]


def tokenize(text, stop_words):
    """
    Tokenizes a given text and also removes stop words.
    :param text:
    :param stop_words:
    :return:
    """
    words = nl.word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]


def text_cleaner(in_text):
    """

    :param in_text:
    :return:
    """
    cleaned_text = re.sub(r'([a-zA-Z])\\1{2,}', r'$1', in_text)
    cleaned_text = re.sub("\S*\d\S*", "", cleaned_text).strip()
    return re.sub(r'[^a-zA-Z0-9\s,.]', '', cleaned_text)


def get_cleaned_text(text):
    """

    :param text:
    :return:
    """

    in_text = text.lower()
    cleaned_text = text_cleaner(in_text)
    tokens = nl.word_tokenize(cleaned_text)

    # TODO : Use setemimng here.
    return remove_stop_words(tokens)
