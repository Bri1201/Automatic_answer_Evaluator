from preprocess import preprocess_text, preprocess_sentences
from similarity import get_sentence_similarity

# Introduced it now
def length_penalty(student_len, model_len):
    if student_len == 0 or model_len == 0:
        return 0
    ratio = float(student_len) / model_len
    penalty = min(ratio, 1.0 / ratio)
    return penalty


def grade_answer(keywords, model_answer, student_answer):
    preprocessed_model_answer, model_tokens = preprocess_text(model_answer)
    preprocessed_student_answer, student_tokens = preprocess_text(student_answer)

    model_sentences = preprocess_sentences(model_answer)
    student_sentences = preprocess_sentences(student_answer)

    sentence_similarity = get_sentence_similarity(model_sentences)
