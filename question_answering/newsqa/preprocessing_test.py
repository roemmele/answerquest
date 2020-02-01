import argparse
import pickle


class Question:
    def __init__(self, string):
        self.string = string
        self.answer = ""


class Document:
    def __init__(self, string, questions):
        self.string = string
        self.questions = questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", default=None, help="newsqa pickled dataset file")
    args = parser.parse_args()
    data = pickle.load(open(args.pickle_file, 'rb'))
    doc_list = []
    for index, document in enumerate(data):
        text = document['text']
        questions = document['questions']
        question_object_list = []
        for question in questions:
            consensus = question['consensus']
            question_obj = Question(question['q'])
            if 'badQuestion' in consensus or 'noAnswer' in consensus:
                question_obj.answer = ""
            else:
                question_obj.answer = text[consensus['s']:consensus['e']]
            question_object_list.append(question_obj)
        doc_list.append(Document(text, question_object_list))
    pickle.dump(doc_list, open('newsqa_test.pkl', 'wb'))
