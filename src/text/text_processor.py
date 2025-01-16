import nltk as nl

nl.download('punkt_tab')

from nltk.stem.porter import *
from nltk.corpus import stopwords
from string import punctuation

stemmer = PorterStemmer() #care reduce cuvintele la forma lor de bază


def remove_stop_words(words):
    """
    Will remove stop words from a list
    :param words: input list
    :return: output list of words without stop words.
    """
    custom_stop_words = set(stopwords.words('english') + list(punctuation))
    return [stemmer.stem(word) for word in words if word not in custom_stop_words] #elimină stop words din lista de cuvinte si aplica stemming pentru cuvintele ramase


def tokenize(text, stop_words):
    """
    Tokenizes a given text and also removes stop words.
    :param text:
    :param stop_words:
    :return:
    """
    words = nl.word_tokenize(text)
    words = [w.lower() for w in words] #transformare in litere mici
    return [w for w in words if w not in stop_words and not w.isdigit()] #elimină stop words și cifre.


def text_cleaner(in_text):
    """

    :param in_text:
    :return:
    """
    cleaned_text = re.sub(r'([a-zA-Z])\\1{2,}', r'$1', in_text) #elimină secvențele lungi de litere repetate
    cleaned_text = re.sub("\S*\d\S*", "", cleaned_text).strip() #șterge cuvintele care conțin cifre
    return re.sub(r'[^a-zA-Z0-9\s,.]', '', cleaned_text) #Curăță textul de caractere speciale


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
