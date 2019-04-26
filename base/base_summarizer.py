import json

from text.text_processor import text_cleaner
from util.util_functions import *
from sklearn.cluster import KMeans
import sys


def normalize_values(res_arr):
    # print([x['tf_idf'] for x in res_arr])
    min_max_map = {}
    for k in ['tf_idf', 'sen_length', 'proper_nouns', 'np_vps', 'c_score', 'sen_pos']:
        min_max_map.setdefault(k, {})['max'] = max([x[k] for x in res_arr])
        min_max_map[k]['min'] = min([x[k] for x in res_arr])
    result = []
    # print(min_max_map)
    for r in res_arr:
        tf_idf = get_val(r['tf_idf'], min_max_map['tf_idf']['min'], min_max_map['tf_idf']['max'])
        sen_length = get_val(r['sen_length'], min_max_map['sen_length']['min'], min_max_map['sen_length']['max'])
        proper_nouns = get_val(r['proper_nouns'], min_max_map['proper_nouns']['min'],
                               min_max_map['proper_nouns']['max'])
        np_vps = get_val(r['np_vps'], min_max_map['np_vps']['min'], min_max_map['np_vps']['max'])
        c_score = get_val(r['c_score'], min_max_map['c_score']['min'], min_max_map['c_score']['max'])
        sen_pos = r['sen_pos']
        total_score = 0.7 * tf_idf + 0.05 * sen_length + 0.025 * proper_nouns + \
                      0.025 * np_vps + 0.15 * c_score + 0.15 * sen_pos
        obj = {'tf_idf': 0.7 * tf_idf, 'sen_length': 0.05 * sen_length, 'proper_nouns': 0.025 * proper_nouns,
               'np_vps': 0.025 * np_vps,
               'c_score': 0.15 * c_score, 'sen_pos': 0.15 * sen_pos, 'total_score': total_score, 'index': r['index']}
        result.append(obj)
    return result


def get_final_sen_ranks(text, D, df_map, model):
    sens = get_sentences(text)
    t = len(sens)
    coh_map = get_sen_cohesiveness(sens, model)
    result_array = []

    for i, s in enumerate(sens):
        # print(s)
        tf_idf = calculate_tf_idf_sum(s, D, df_map)
        sen_length = get_sen_length(s)
        proper_nouns = get_proper_nouns(s)
        np_vps = len(get_np_vps(s))
        c_score = coh_map[i]
        pos = get_sen_pos(i, t)
        rank_obj = {'tf_idf': tf_idf, 'sen_length': sen_length, 'proper_nouns': proper_nouns,
                    'np_vps': np_vps, 'c_score': c_score, 'sen_pos': pos, 'index': i}
        # print("   ", rank_obj)
        result_array.append(rank_obj)
    result = normalize_values(result_array)
    wr = open("/tmp/score.json", "w")
    for i, r in enumerate(result):
        r["sentence"] = sens[i]
        wr.write(json.dumps(r) + "\n")
    return result, sens


def get_sen_cluster_map(sens, model, n_clusters):
    X = get_sen_vec_list(sens, model)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels


def get_summary_indices(cluster_score_map, sens, sum_words):
    sorted_cl_map = {}
    for key, score_map in cluster_score_map.items():
        sorted_by_value = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        sorted_cl_map[key] = sorted_by_value
    # print(len(sorted_cl_map))

    # Get final sentence indices.
    final_sumamry_indices = []
    x = 0
    i = 0
    print(' summary words ', sum_words)
    while (x < sum_words):
        for key, score_map in sorted_cl_map.items():

            # The below if will ensure that  current cluster has ith senetence, it can not be always the case.
            if (len(score_map) > i):

                # This condition is for breaking during loop if num words are conaition is stastified.
                if x >= sum_words:
                    break

                index = score_map[i][0]
                # print(sens[index])
                token_len = len(nl.word_tokenize(sens[index]))
                if token_len > 15:
                    # print(" token len  ",token_len )
                    # print("\n\n\n")
                    # print(sens[index])
                    final_sumamry_indices.append(index)
                    # TODO : don't consider those sentences whose length is less than certain threshhold.
                    x += token_len
                # print("\n\n\n")
                # print("  x  ", x, "  score  ", score_map[i][1], "  cluster  ", key,  "  i ", i, "  index ", index )

        i += 1
    return final_sumamry_indices


def get_summary_sentences(summary_indices, sens):
    return [text_cleaner(sens[x]) for x in summary_indices]


def claculate_df(corpus_path):
    df_map = {}
    doc_tokens = [set(get_cleaned_text(json.loads(line)['text'])) for line in open(corpus_path)]
    D = len(doc_tokens)
    for token_set in doc_tokens:
        for token in token_set:
            df_map[token] = df_map.get(token, 0) + 1
    return df_map, D


def get_summary(input_file_path, sum_percent, word_to_vec_model, corpus_path):
    df_map, D = claculate_df(corpus_path)

    with open(input_file_path, 'r') as my_file:
        data = my_file.read().replace('\n', '')
        data = data.replace(".", ". ")
    result_array, sens = get_final_sen_ranks(data, D, df_map, word_to_vec_model)
    cluster_labels = get_sen_cluster_map(sens, word_to_vec_model, n_clusters=3)
    total_words = sum([len(nl.word_tokenize(sen_tokens)) for sen_tokens in sens])
    sum_words = math.ceil((sum_percent / 100) * total_words)
    final_cluster_score_map = {}
    for r in result_array:
        index = r['index']
        total_score = r['total_score']
        cluster = int(cluster_labels[index])
        final_cluster_score_map.setdefault(cluster, {})[index] = total_score
    # print(final_cluster_score_map)
    summary_indices = get_summary_indices(final_cluster_score_map, sens, sum_words)
    summary_indices.sort()
    # print(summary_indices)
    return get_summary_sentences(summary_indices, sens)


if __name__ == '__main__':
    # TODO : Need to discuss on which model it will work and how to get it in README
    model_path = sys.argv[0]
    print("Going to load word2Vec Model, it may take a minute or more")
    model = load_model(model_path)
    print("model loaded, now going to quickly generate summary")

    # TODO : Need to discuss the file schema and how to generate it in README
    corpus_file = sys.argv[1]

    summary_percent = 5
    input_file = sys.argv[2]
    summary_sens = get_summary(input_file, summary_percent, word_to_vec_model=model,
                               corpus_path=corpus_file)
    for s in summary_sens:
        print(s)
