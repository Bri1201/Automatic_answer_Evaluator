#Importing libraries
import sklearn
import nltk
#For synonyms
from nltk.corpus import wordnet
#For count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import preprocess_keys, preprocess_text


def checkForKeywords(keywords, student_answer, model_answer):
    #Preprocess keywords and texts
    pp_keywords = preprocess_keys(keywords)
    pp_student = preprocess_text(student_answer)
    pp_model = preprocess_text(model_answer)
    
    # create CountVectorizer object and fit on document
    vectorizer = CountVectorizer()
    student_vec = vectorizer.fit_transform([pp_student])
    model_vec = vectorizer.transform([pp_model])
    
    # transform keywords into sparse matrix
    keyword_matrix = vectorizer.transform([''.join(pp_keywords)])
    
    # Compute two cosine similarities
    sim1 = sklearn.metrics.pairwise.cosine_similarity(student_vec, keyword_matrix)[0][0]
    sim2 = sklearn.metrics.pairwise.cosine_similarity(student_vec, model_vec)[0][0]
    
    return sim1, sim2


# Load the pre-trained model and tokenizer
model_name = "textattack/bert-base-uncased-rotten-tomatoes"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def semanticA(text1, text2, aspect):
    # Tokenize the texts and aspect
    tokens1 = tokenizer(aspect + " " + text1, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(aspect + " " + text2, return_tensors='pt', padding=True, truncation=True)

    # Perform semantic analysis on the texts and aspect
    with torch.no_grad():
        output1 = model(**tokens1)[0]
        semantic1 = torch.softmax(output1, dim=1)[0][1].item()

        output2 = model(**tokens2)[0]
        semantic2 = torch.softmax(output2, dim=1)[0][1].item()

    # Determine if the two texts express the same or opposite meaning
    if abs(semantic1 - semantic2) < 0.1:
        return 1
    else:
        return 0

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def getRelevantSentences(text, word):

    # get synonyms of the word
    synonyms = get_synonyms(word)

    # tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # list to store sentences containing the word or its synonym
    matched_sentences = []

    # loop through each sentence
    for sentence in sentences:
        # tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        # check if the word or any of its synonyms are present in the sentence
        if word in words:
            matched_sentences.append(sentence)
        else:
            for synonym in synonyms:
                if synonym in words:
                    # replace the synonym with the word
                    sentence = sentence.replace(synonym, word)
                    matched_sentences.append(sentence)
                    break

    # Return the matched sentences
    return matched_sentences

def compare_sentences(sentences1, sentences2, word):
    similarity_score = 0
    for s1 in sentences1:
        for s2 in sentences2:
            if word in s1 and word in s2:
                similarity_score += semanticA(s1, s2, word)
            else:
                synonyms = get_synonyms(word)
                for syn in synonyms:
                    if syn in s1 and syn in s2:
                        s1_new = s1.replace(syn, word)
                        s2_new = s2.replace(syn, word)
                        similarity_score += semanticA(s1_new, s2_new, word)
                        break
    return similarity_score

def compareTexts(text1, text2, keywords):
    pp_text1 = preprocess_text(text1)
    pp_text2 = preprocess_text(text2)
    pp_keywords = preprocess_keys(keywords)
    
    totalScore = 0
    
    for keyword in pp_keywords:
        list1 = getRelevantSentences(pp_text1, keyword)
        list2 = getRelevantSentences(pp_text2, keyword)
        
        score = compare_sentences(list1, list2, keyword)
        #print(keyword)
        #print(score)
        totalScore += score
        #print(totalScore)
    
    return totalScore/len(keywords)