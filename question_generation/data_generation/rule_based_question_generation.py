import pandas
import os
import argparse
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", "-input_file", type=str)
parser.add_argument("--start_idx", "-start_idx", type=int)
parser.add_argument("--end_idx", "-end_idx", type=int)
parser.add_argument("--config_file", "-config_file", type=str)
parser.add_argument("--save_prefix", "-save_prefix", type=str)
args = parser.parse_args()

'''Switch to directory that contains H&S code'''

os.chdir("/home/mroemmele/question_generation/HS-QuestionGeneration/")

'''Load input texts'''

with open(args.input_file) as f:
    input_texts = [text.strip() for text in f][args.start_idx:args.end_idx]

'''Run Java rule-based code via python subprocess. Ensure sense tagging and parsing servers are running on ports 
specified in the multiple config/QuestionTransducer.properties files. Each of these config files specifies different
ports for these servers. Code below will continuously cycle through each config so that it can distribute 
the analyses across different ports, in order to speed things up.'''

qg_output = {'text_id': [], 'question': [],
             'answer_sent': [], 'answer': [], 'score': []}
# n_parallel_procs = 50
# input_chunks = [input_texts[idx:idx + n_parallel_procs]
#                 for idx in range(0, len(input_texts), n_parallel_procs)]

# for chunk_idx, chunk in enumerate(input_chunks):
#     procs = [subprocess.Popen(["java", "-Xmx1200m",
#                                "-cp", "question-generation.jar",
#                                "edu/cmu/ark/QuestionAsker",
#                                "--verbose", "--model", "models/linear-regression-ranker-reg500.ser.gz",
#                                "--just-wh", "--max-length", "30", "--downweight-pro",
#                                "--properties", args.config_file],
#                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#              for text in chunk]
#     print("Processing texts in chunk...")
#     proc_outputs = [[proc_output.split("\t") for proc_output in
#                      proc.communicate(text.encode())[0].decode('utf-8').strip().split("\n")]
#                     for proc, text in zip(procs, chunk)]
#     for output_idx, output in enumerate(proc_outputs):
#         # import pdb
#         # pdb.set_trace()
#         try:
#             for question, answer_sent, answer, score in output:
#                 qg_output['text_id'].append(args.start_idx + (chunk_idx * n_parallel_procs) + output_idx)
#                 qg_output['question'].append(question)
#                 qg_output['answer_sent'].append(answer_sent)
#                 qg_output['answer'].append(answer)
#                 qg_output['score'].append(float(score))
#         except:  # Possibly no questions were returned (e.g. error in parsing)
#             continue
#     print("PROCESSED TEXTS UP TO INDEX", args.start_idx + (chunk_idx + 1) * n_parallel_procs, "\n")

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
    # import pdb
    # pdb.set_trace()
    try:
        for question, answer_sent, answer, score in proc_outputs:
            qg_output['text_id'].append(args.start_idx + text_idx)
            qg_output['question'].append(question)
            qg_output['answer_sent'].append(answer_sent)
            qg_output['answer'].append(answer)
            qg_output['score'].append(float(score))
    except:  # Possibly no questions were returned (e.g. error in parsing)
        continue
    print("PROCESSED TEXTS UP TO INDEX", args.start_idx + text_idx, "\n\n")

qg_output = pandas.DataFrame(qg_output)
qg_output.to_csv(args.save_prefix + "_{}_to_{}.csv".format(args.start_idx, args.end_idx), index=False)
print("saved output to", args.save_prefix + "_{}_to_{}.csv".format(args.start_idx, args.end_idx))
