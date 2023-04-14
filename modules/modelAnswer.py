import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def get_answer_similarity(student_answer, model_answer):
    # Split the model_answer and student_answer into segments of maximum length 512
    max_length = 512
    model_segments = [model_answer[i:i+max_length] for i in range(0, len(model_answer), max_length)]
    student_segments = [student_answer[i:i+max_length] for i in range(0, len(student_answer), max_length)]

    # Encode the model_segments and student_segments using RoBERTa
    model_embeddings = []
    student_embeddings = []
    for model_segment, student_segment in zip(model_segments, student_segments):
        encoded_model_segment = tokenizer.encode(model_segment, add_special_tokens=True, return_tensors='pt')
        encoded_student_segment = tokenizer.encode(student_segment, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(encoded_model_segment)
            student_output = model(encoded_student_segment)
        model_embedding = np.mean(model_output[0].numpy(), axis=1)
        student_embedding = np.mean(student_output[0].numpy(), axis=1)
        model_embeddings.append(model_embedding)
        student_embeddings.append(student_embedding)

    # Concatenate the embedding vectors for the encoded model_segments and student_segments
    model_embedding = np.concatenate(model_embeddings, axis=1)
    student_embedding = np.concatenate(student_embeddings, axis=1)

    # Calculate the cosine similarity between the embedding vectors
    similarity_score = cosine_similarity(model_embedding, student_embedding)[0][0]

    return similarity_score
