from modelAnswer import get_answer_similarity
from keywords import compareTexts


#Import the function gradeAnswer from 

def grade_answer(keywords, model_answer, student_answer):
    sim1 = get_answer_similarity(student_answer, model_answer)
    print(sim1)
    sim2 = compareTexts(model_answer, student_answer, keywords)
    print(sim2)
    if(sim2 < 0.5 and sim1 < 0.99):
        return 0
    total_marks = sim1*0.6 + sim2*0.4
    #Returns percentage marks
    return total_marks*10
