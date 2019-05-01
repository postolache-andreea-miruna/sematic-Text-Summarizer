import math
from collections import Counter
from gensim.models import KeyedVectors


import nltk as nl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from text.text_processor import get_cleaned_text


def get_sen_length(sen):
    return len(nl.word_tokenize(sen))


def get_sen_pos(postion, total_sens):
    postion = 1 if postion == 0 else postion
    return (postion - 1) / total_sens


def get_proper_nouns(sentence):
    nnp = [word for word, pos in nl.pos_tag(nl.word_tokenize(str(sentence))) if pos == 'NNP']
    return len(nnp)


def extract_np(psent):
    for subtree in psent.subtrees():
        if subtree.label() == 'NP':
            yield ' '.join(word for word, tag in subtree.leaves())


def get_np_vps(sen):
    # Todo check grammar for verb phrases
    grammar = r"""
          NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
          {<NN>+}                 # chunk consecutive nouns
          VP: {<VB.*><NP|PP|CLAUSE>+$}
          """
    cp = nl.RegexpParser(grammar)

    tagged_sent = nl.pos_tag(sen.split())
    parsed_sent = cp.parse(tagged_sent)
    nps = []
    for npstr in extract_np(parsed_sent):
        nps.append(npstr)
    return nps


def vectorize_sent(sent, model):
    # print(pre_process(sent))
    return np.mean([model[w] for w in get_cleaned_text(sent) if w in model]
                   or [np.zeros(300)], axis=0)


def get_sen_vec_list(sens, model):
    index_vec_list = []
    for i, sen in enumerate(sens):
        vec = vectorize_sent(sen, model)
        index_vec_list.append(vec)
    return index_vec_list


def calculate_tf_idf_sum(sen, D, df_map):
    tokens = get_cleaned_text(sen)
    tf_map = dict(Counter(tokens))
    tf_idf_map = {k: (v * math.log(D / df_map.get(k, 1))) for k, v in tf_map.items()}
    return sum(tf_idf_map.values())


def get_sen_cohesiveness(sens, model):
    index_vec_list = get_sen_vec_list(sens, model)
    sen_score_map = {}
    # print(index_vec_list)
    c_s_array = (cosine_similarity(index_vec_list, index_vec_list))
    cohesiveness = (c_s_array.sum(axis=0)) - 1
    return {i: v for i, v in enumerate(cohesiveness)}


def get_sentences(text):
    return nl.sent_tokenize(text)


def get_val(val, min_val, max_val):
    min_val = 1 if min_val <= 0 else min_val
    return (val - min_val) / (max_val - min_val)


def load_model(model_path):
    return KeyedVectors.load_word2vec_format(model_path)