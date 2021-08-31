import os
import argparse
import json
import subprocess


def get_answer_annotated_data(questions, answer_sents, answers, scores,
                              min_score=0.0):
    filtered_questions = []
    filtered_answer_sents = []

    for idx, (question,
              answer_sent,
              answer,
              score) in enumerate(zip(questions,
                                      answer_sents,
                                      answers,
                                      scores)):
        if score < min_score:
            continue
        answer_start_char = None
        while answer_start_char == None:
            try:
                answer_start_char = answer_sent.lower().index(answer.lower())
            except:
                if " " not in answer:  # Answer not found
                    break
                # Trim leading word from answer to see if subsegment can be found
                answer = answer[answer.index(" ") + 1:]
        if answer_start_char is not None:
            #         answer_start_chars.append(answer_start_char)
            answer_end_char = answer_start_char + len(answer)
            answer_sent = (answer_sent[:answer_start_char] + "<ANSWER> "
                           + answer_sent[answer_start_char:])
            answer_sent = (answer_sent[:answer_end_char + len("<ANSWER> ")]
                           + " </ANSWER>" +
                           answer_sent[answer_end_char + len(" <ANSWER>"):])
            filtered_answer_sents.append(answer_sent)
            filtered_questions.append(question)
    return {'answer_sent': filtered_answer_sents,
            'question': filtered_questions}


def run_proc(jar_dir, input_texts):
    '''Switch to directory that contains H&S code'''

    os.chdir(args.jar_dir)

    '''Run Java rule-based code via python subprocess. Ensure sense tagging and parsing servers are running on ports 
    specified in the multiple config/QuestionTransducer.properties files. Each of these config files specifies different
    ports for these servers. Code below will continuously cycle through each config so that it can distribute 
    the analyses across different ports, in order to speed things up.'''

    qg_output = {'question': [],
                 'answer_sent': [],
                 'answer': [],
                 'score': []}

    for text_idx, text in enumerate(input_texts):
        proc = subprocess.Popen(["java", "-Xmx1200m",
                                 "-cp", "question-generation.jar",
                                 "edu/cmu/ark/QuestionAsker",
                                 "--verbose", "--model", "models/linear-regression-ranker-reg500.ser.gz",
                                 "--just-wh", "--max-length", "30", "--downweight-pro",
                                 "--properties", args.config_file],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc_outputs = [proc_output.split("\t") for proc_output in
                        proc.communicate(text.encode())[0].decode('utf-8').strip().split("\n")]

        try:
            for question, answer_sent, answer, score in proc_outputs:
                #qg_output['text_id'].append(args.start_idx + text_idx)
                qg_output['question'].append(question)
                qg_output['answer_sent'].append(answer_sent)
                qg_output['answer'].append(answer)
                qg_output['score'].append(float(score))
        except:  # Possibly no questions were returned (e.g. error in parsing)
            continue
        print("PROCESSED TEXTS UP TO INDEX", text_idx, "\n\n")

    return (qg_output['question'],
            qg_output['answer_sent'],
            qg_output['answer'],
            qg_output['score'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-input_file",
                        help="Path to input texts (one text per line).",
                        type=str, required=True)
    parser.add_argument("--jar_dir", "-jar_dir",
                        help="Directory containing downloaded java program.",
                        type=str, required=True)
    # parser.add_argument("--start_idx", "-start_idx", type=int)
    # parser.add_argument("--end_idx", "-end_idx", type=int)
    parser.add_argument("--config_file", "-config_file",
                        help="Config file for java program",
                        type=str, required=True)
    parser.add_argument("--output_dir", "-output_dir",
                        help="Directory path of where to save Q&A output",
                        type=str, required=True)
    parser.add_argument("--min_score", "-min_score",
                        help="Filter questions below this quality score.\
                        By default, all generated questions will be saved.",
                        type=float, default=0.0)
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_texts = [text.strip() for text in f]

    orig_dir = os.path.dirname(os.path.abspath(__file__))

    questions, answer_sents, answers, scores = run_proc(args.jar_dir,
                                                        input_texts)

    os.chdir(orig_dir)

    annotated_qg_data = get_answer_annotated_data(questions, answer_sents, answers, scores,
                                                  min_score=args.min_score)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, 'input_texts.txt'), 'w') as f:
        f.write("\n".join(annotated_qg_data['answer_sent']))
        print("Saved answer-annotated sentences to",
              os.path.join(args.output_dir, 'input_texts.txt'))

    with open(os.path.join(args.output_dir, 'questions.txt'), 'w') as f:
        f.write("\n".join(annotated_qg_data['question']))
        print("Saved questions to",
              os.path.join(args.output_dir, 'questions.txt'))
