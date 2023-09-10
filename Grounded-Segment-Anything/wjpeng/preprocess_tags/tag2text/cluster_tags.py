"""
description:
    This is a simple application for sentence embeddings: clustering
    Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.
tutorial:
    1. https://www.sbert.net/examples/applications/clustering/README.html
    2. https://www.sbert.net/docs/usage/semantic_textual_similarity.html
    3. https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/agglomerative.py
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--dest_file', type=str)
    parser.add_argument('--dis_threshold', type=float, default=1.5)
    args = parser.parse_args()

    # load corpus
    tag_list = []
    idx_list = []
    with open(args.src_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            idx = line.strip().split(',')[0]
            tag = line.strip().split(',')[1]
            idx_list.append(idx)
            tag_list.append(tag)

    # load embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # extract embeddings
    tag_embeddings = embedder.encode(tag_list)

    # Normalize the embeddings to unit length
    tag_embeddings = tag_embeddings / np.linalg.norm(tag_embeddings, axis=1, keepdims=True)

    # Perform k-mean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=args.dis_threshold)
    clustering_model.fit(tag_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_tags = {}
    for tag_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_tags:
            clustered_tags[cluster_id] = []

        clustered_tags[cluster_id].append(tag_list[tag_id])

    # save
    tag2cluster = {}
    for tag, cluster_id in zip(tag_list, cluster_assignment):
        tag2cluster[tag] = cluster_id
    ret = {
        'tag2cluster': tag2cluster,
        'clustered_tags': clustered_tags
    }

    np.save(args.dest_file, ret)

    # print
    for i, cluster in clustered_tags.items():
        print("Cluster ", i+1)
        print(cluster)
        print("")
