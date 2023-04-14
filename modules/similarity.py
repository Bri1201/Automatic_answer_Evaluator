import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize sentence transformer model
sent_transformer = SentenceTransformer('paraphrase-MiniLM-L12-v2')


def get_sentence_embeddings(sentences):
    embeddings = sent_transformer.encode(sentences)
    return embeddings


def get_cosine_similarity(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)


def get_sentence_similarity(sentences1, sentences2):
    embeddings1 = get_sentence_embeddings(sentences1)
    embeddings2 = get_sentence_embeddings(sentences2)
    similarity_matrix = get_cosine_similarity(embeddings1, embeddings2)
    max_similarities = np.max(similarity_matrix, axis=1)
    avg_similarity = np.mean(max_similarities)
    return avg_similarity
