import os
import dgl
import time
import pickle
import random
import numpy as np
import torch as th
from dgl.data.utils import load_graphs, save_graphs


def open_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        file_content = pickle.load(file)
        return file_content


def save_pkl_file(file_path, contents):
    with open(file_path, 'wb') as file:
        pickle.dump(contents, file)
    print("having saved pkl...")


def open_txt_file(file_path):
    with open(file_path, 'r') as file:
        contents = [line.rstrip("\n") for line in file.readlines()]
        return contents


def save_txt_file(file_path, contents):
    with open(file_path, 'w') as file:
        for paragraph in contents:
            file.write(paragraph + "\n")
    print("having saved txt...")


# 为所有节点提取元路径
def extract_all_node_metapath(data_dir, target_type, metapaths, relation, mid_types, num_node_type, num_category):

    glist, label_dict = load_graphs(data_dir + 'graph.bin')
    label_g = glist[0]

    # 防止数据泄露，节点分类时构建的语料库需要从reduced_graph中获取
    glist, label_dict = load_graphs(data_dir + "reduced_graph.bin")  
    g = glist[0]

    graph_node_name = open_pkl_file(data_dir + 'graph_node_name.pkl')
    metapath = metapaths[target_type]
    relation = relation[target_type]

    labels_range = [i for i in range(3)]
    labels = label_g.nodes[target_type].data['label'].tolist()
    labeled_node_id = []
    for i in range(len(labels)):
        if labels[i] in labels_range:
            labeled_node_id.append(i) 
    print(f"labeled_nodes num: {len(labeled_node_id)}")

    label_id2name = {0: "action", 1: "comedy", 2: "drama"}
    
    sampling_time = 10
    num_nodes = g.num_nodes(target_type)
        
    all_path_for_sampling_times = [[] for _ in range(num_nodes)]
    print("---------------------------------------")
    print(f"Sampling nodes of type {target_type}...")
    labels_output = []
    for p, path in enumerate(metapath):
        path_for_sampling_times = [[] for _ in range(num_nodes)]
        print(f"Sampling the {p}-th path...")
        for st in range(sampling_time):
            traces, types = dgl.sampling.random_walk(g=g, nodes=g.nodes(target_type), metapath=path)
            traces = traces.tolist()
            length = len(traces)
            print(f"Performing the {st}-th sampling...")
            for node in range(length):
                if node % 5000 == 0:
                    print(f"{node} nodes have been sampled...")
                path_i = []
                if traces[node][1] != -1:
                    path_i.append(target_type.replace('_', ' '))
                    path_i.append(graph_node_name[target_type][traces[node][0]].replace('_', ' '))
                    path_i.append(relation[p][0])
                    mid_type = mid_types[target_type][p][0]
                    path_i.append(mid_type.replace('_', ' '))
                    path_i.append(graph_node_name[mid_type][traces[node][1]].replace('_', ' '))
                    path_i.append(relation[p][1])
                    if len(traces[node]) <= 3:  # apa pa
                        path_i.append(target_type.replace('_', ' '))
                        path_i.append(graph_node_name[target_type][traces[node][2]].replace('_', ' '))
                    else:  # apcpa pcpa
                        mid_type = mid_types[target_type][p][1]
                        path_i.append(mid_type.replace('_', ' '))
                        path_i.append(graph_node_name[mid_type][traces[node][2]].replace('_', ' '))
                        path_i.append(relation[p][2])

                        mid_type = mid_types[target_type][p][2]
                        path_i.append(mid_type.replace('_', ' '))
                        path_i.append(graph_node_name[mid_type][traces[node][3]].replace('_', ' '))
                        path_i.append(relation[p][3])

                        mid_type = target_type
                        path_i.append(mid_type.replace('_', ' '))
                        path_i.append(graph_node_name[mid_type][traces[node][4]].replace('_', ' '))
                else:
                    path_i.append(target_type.replace('_', ' '))
                    path_i.append(graph_node_name[target_type][traces[node][0]].replace('_', ' '))
                
                path_i.append("</s>")
                path_i = " ".join(path_i)
                path_for_sampling_times[node].append(path_i)
        path_for_sampling_times = [list(set(item)) for item in path_for_sampling_times if item]
        k = 3
        path_for_sampling_times = [random.sample(item, min(k, len(item))) for item in path_for_sampling_times if item]
        for i, item in enumerate(path_for_sampling_times):
            path_for_sampling_times[i] = " ".join(item)
            all_path_for_sampling_times[i].append(path_for_sampling_times[i])
    all_path_for_sampling_times = [list(set(item)) for item in all_path_for_sampling_times if item]
    for i, item in enumerate(all_path_for_sampling_times):
        all_path_for_sampling_times[i] = " ".join(item)
        all_path_for_sampling_times[i].rstrip(' </s> ')
        if i in labeled_node_id:
            label = labels[i]
            append_ste_t = f"You can deduce the category of movie {graph_node_name[target_type][i].replace('_', ' ')} as <mask>."
            all_path_for_sampling_times[i] = f"For movie {graph_node_name[target_type][i].replace('_', ' ')}: [" + all_path_for_sampling_times[i] + "] " + append_ste_t
            labels_output.append(label_id2name[label].lower())
    # print(all_path_for_sampling_times[0])
    # print(f"length: {len(all_path_for_sampling_times)}")
    return all_path_for_sampling_times, labels_output


if __name__ == "__main__":
    data_dir = '../data/imdb/'  # your file path
    metapaths = {'movie': [['was acted by', 'acted in'], ['was directed by', 'directed']],  # mam mdm
                 'actor': [['acted in', 'was acted by'], ['acted in', 'was directed by', 'directed', 'was acted by']],  # ama amdma
                 'director': [['directed', 'was directed by'],['directed', 'was acted by', 'acted in', 'was directed by']]}  # dmd dmamd
    relation = {'movie': [['was acted by', 'acted in'], ['was directed by', 'directed']],
                'actor': [['acted in', 'was acted by'], ['acted in', 'was directed by', 'directed', 'was acted by']],  # ama amdma
                'director': [['directed', 'was directed by'],['directed', 'was acted by', 'acted in', 'was directed by']]}  # dmd dmamd
    mid_types = {'movie': [['actor'], ['director']],
                 'actor': [['movie'], ['movie', 'director', 'movie']], 
                 'director': [['movie'], ['movie', 'actor', 'movie']]}
    
    all_corpus = []
    num_node_type, num_category = 3, 3
    movie_corpus, labels = extract_all_node_metapath(data_dir, 'movie', metapaths, relation, mid_types, num_node_type, num_category)
    all_corpus.extend(movie_corpus)

    save_txt_file(data_dir + 'movie_metapath_nc.txt', all_corpus)
    save_txt_file(data_dir + 'movie_nc_labels.txt', labels)

    print(all_corpus[0])
    print(f"length: {len(all_corpus)}")  # 4278
    print(labels[0])
    print(f"length: {len(labels)}")  # 4278