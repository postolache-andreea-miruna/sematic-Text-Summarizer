import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.cluster import KMeans

import nltk

nltk.download('averaged_perceptron_tagger_eng')

from text.text_processor import *
from util.util_functions import *

nltk.download('reuters')
nltk.download('stopwords')

from nltk.corpus import reuters, stopwords


def normalize_values(res_arr):
    # print([x['tf_idf'] for x in res_arr])
    min_max_map = {}
    for k in ['tf_idf', 'sen_length', 'proper_nouns', 'np_vps', 'c_score', 'sen_pos']:
        min_max_map.setdefault(k, {})['max'] = max([x[k] for x in res_arr]) #pentru fiecare caracteristica  se calculeaza valoarea maximului în intregul set de date res_arr; se asigura existenta unui sub dictionar pentru fiecare caracteristica
        min_max_map[k]['min'] = min([x[k] for x in res_arr]) #calcul minim pentru fiecare caracterstica si salvarea in sub dictionar
    result = []
    # print(min_max_map)
    for r in res_arr: #normalizare intre valorile de min si max obtinute mai sus
        tf_idf = get_val(r['tf_idf'], min_max_map['tf_idf']['min'], min_max_map['tf_idf']['max'])
        sen_length = get_val(r['sen_length'], min_max_map['sen_length']['min'], min_max_map['sen_length']['max'])
        proper_nouns = get_val(r['proper_nouns'], min_max_map['proper_nouns']['min'],
                               min_max_map['proper_nouns']['max'])
        np_vps = get_val(r['np_vps'], min_max_map['np_vps']['min'], min_max_map['np_vps']['max'])
        c_score = get_val(r['c_score'], min_max_map['c_score']['min'], min_max_map['c_score']['max'])
        sen_pos = r['sen_pos']
        total_score = 0.7 * tf_idf + 0.05 * sen_length + 0.025 * proper_nouns + \
                      0.025 * np_vps + 0.15 * c_score + 0.15 * sen_pos #cu cat ponderea e mai mare cu atat e mai importanta caracteristica
        obj = {'tf_idf': 0.7 * tf_idf, 'sen_length': 0.05 * sen_length, 'proper_nouns': 0.025 * proper_nouns,
               'np_vps': 0.025 * np_vps,
               'c_score': 0.15 * c_score, 'sen_pos': 0.15 * sen_pos, 'total_score': total_score, 'index': r['index']}
        result.append(obj)
    return result


def get_final_sen_ranks(text, D, df_map, model):
    sens = get_sentences(text)
    t = len(sens)
    coh_map = get_sen_cohesiveness(sens, model) #scor de coeziune pentru fiecare propoziție,  cât de bine se leagă propozițiile între ele.
    result_array = []

    for i, s in enumerate(sens):
        # print(s)
        tf_idf = calculate_tf_idf_sum(s, D, df_map) #calculeaza suma valorilor TF-IDF (pentru evaularea importantei unui cuvant in propozitie in raport cu D) pentru cuvintele propozitiei curente
        sen_length = get_sen_length(s) #lungimea propozitiei
        proper_nouns = get_proper_nouns(s) #Extrage substantive proprii
        np_vps = len(get_np_vps(s)) # extrage sintagmele nominale și verbale din propoziția s
        c_score = coh_map[i] # scorul de coeziune pentru propoziția curent
        pos = get_sen_pos(i, t) #poziția propoziției
        rank_obj = {'tf_idf': tf_idf, 'sen_length': sen_length, 'proper_nouns': proper_nouns,
                    'np_vps': np_vps, 'c_score': c_score, 'sen_pos': pos, 'index': i}
        # print("   ", rank_obj)
        result_array.append(rank_obj)
    result = normalize_values(result_array) #normalizare intre valori min max si intoarce lista  de scoruri totale pentru fiecare propozitie
    #wr = open("/tmp/score.json", "w") --inainte
    output_file_path = os.path.join(os.getcwd(), "score.json") #os.getcwd() returnează directorul curent, os.path.join creează calea completă către fișierul JSON
    with open(output_file_path, "w") as wr:
        for i, r in enumerate(result):
            r["sentence"] = sens[i] # Asociază propoziția curenta cu scorul corespunzător.
            wr.write(json.dumps(r) + "\n")
    return result, sens #intoarce scorurile si lista de propozitii


def get_sen_cluster_map(sens, model, n_clusters):
    X = get_sen_vec_list(sens, model) # listă de vectori numerici reprezentand semantic propozitiile
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X) #algoritmul KMeans - invatare nesupravegheata pentru gruparea propozitiilor in 3 clustere bazandu-se pe vectorii lor. Algoritmul împarte propozițiile în 3 grupuri, minimizând distanța între propozițiile din același grup și maximizând distanța între grupuri.
    labels = kmeans.labels_ #listă de etichete, unde fiecare etichetă reprezintă clusterul din care face parte propoziția respectivă
    return labels


def get_summary_indices(cluster_score_map, sens, sum_words):
    sorted_cl_map = {} #dicționar pentru scorurile propozițiilor, sortate descrescător.
    for key, score_map in cluster_score_map.items(): #iterare prin fiecare cluster și scorurile asociate.
        sorted_by_value = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)#Sortează propozițiile din fiecare cluster în funcție de scor (kv[1]), în ordine descrescătoare
        sorted_cl_map[key] = sorted_by_value

    # Get final sentence indices.
    final_sumamry_indices = [] #listă pentru indicii propozițiilor selectate pentru rezumat
    x = 0 #numărul de cuvinte selectate până acum
    i = 0 #indicele propoziției curente în fiecare cluster
    print(' summary words ', sum_words)
    while (x < sum_words):
        for key, score_map in sorted_cl_map.items(): #score_map = [(index,scor_propozitie),..]

            # The below if will ensure that  current cluster has ith senetence, it can not be always the case.
            if (len(score_map) > i):

                # This condition is for breaking during loop if num words are conaition is stastified.
                if x >= sum_words:#Se oprește selecția când se atinge numărul dorit de cuvinte.
                    break

                index = score_map[i][0] #indicele propoziției din clusterul curent, la poziția i
                token_len = len(nl.word_tokenize(sens[index])) #numărul de cuvinte din propoziția selectată
                if token_len > 15:
                    final_sumamry_indices.append(index)
                    # TODO : don't consider those sentences whose length is less than certain threshhold.
                    x += token_len

        i += 1
    return final_sumamry_indices


def get_summary_sentences(summary_indices, sens):
    return [text_cleaner(sens[x]) for x in summary_indices]


def calculate_df():
    stop_words = stopwords.words('english') + list(punctuation)

    df_map = {}
    print("going to read corpus for calculating document frequency")
    for file_id in reuters.fileids(): #reuters este corpus de stiri  cu articole organizate in categorii
        # TO Get unique occurrence of word
        words = set(tokenize(reuters.raw(file_id), stop_words)) #scoate cuvintele unice din textul documentul cu id curent eliminând stop words
        for w in words:
            df_map.setdefault(w, df_map.get(w, 0) + 1) #salveaza numarul de aparitii al cuvintelor in dictionar
    return df_map, len(reuters.fileids()) #dictionat si numar total documente


def get_summary(input_file_path, sum_percent, model_path):
    word_to_vec_model = load_model(model_path)
    df_map, D = calculate_df() #dicționar care arată de câte ori apare fiecare cuvânt în toate documentele din corpusul Reuters și numărul total de documente.

    with open(input_file_path, 'r') as my_file:
        data = my_file.read().replace('\n', '')
        data = data.replace(".", ". ")
    result_array, sens = get_final_sen_ranks(data, D, df_map, word_to_vec_model) #scorurile normalizarii si lista propozitii
    cluster_labels = get_sen_cluster_map(sens, word_to_vec_model, n_clusters=3) #labelurile conform alg KMeans
    total_words = sum([len(nl.word_tokenize(sen_tokens)) for sen_tokens in sens])
    sum_words = math.ceil((sum_percent / 100) * total_words) # Calculează numărul total de cuvinte care trebuie incluse în rezumat, pe baza procentului dorit
    final_cluster_score_map = {}
    for r in result_array:
        index = r['index']
        total_score = r['total_score']
        cluster = int(cluster_labels[index]) #clusterul pentru rezultatul propozitiei cu indexul curent
        final_cluster_score_map.setdefault(cluster, {})[index] = total_score #stocarea scorului total al propoziției curente în dicționarul final_cluster_score_map, organizat pe clustere.
    # print(final_cluster_score_map)
    summary_indices = get_summary_indices(final_cluster_score_map, sens, sum_words)# intoarce lista de index pt propozițiile care vor fi incluse în rezumat, bazându-se pe scorurile lor și pe numărul total de cuvinte dorit.
    summary_indices.sort()
    # print(summary_indices)
    return get_summary_sentences(summary_indices, sens)


