# answerquest

This repo contains the code for the answerquest library, which performs question generation and question answering from provided texts. In particular, it can be used to generate a list of Q&A pairs pertaining to the text, and also predict answers to given questions about the text.

## Installation

You can install the answerquest library directly from this repo using pip:

`pip install git+https://github.com/roemmele/answerquest.git`

This should install all the required dependencies (which includes pytorch, OpenNMT-py, transformers, and spacy).

To run the library, you'll also need the question generation (QG) and question answering (QA) model files (if using QA only, the QG models are not needed - see below under "Usage"). To download these files, clone the repo and run the download script in the top level of the repo:

`bash download_models.sh` 

This will download the folders "qg_augmented_model/" and "doc_qa_model/" in the directory where you ran the script. The folder "qg_augmented_model" contains the PyTorch question generation model "qg_augmented_model.pt" as well as the folder "gpt2_tokenizer_vocab/", which contains the tokenizer files needed for question generation preprocessing. The folder "doc_qa_model/" contains the folder "checkpoint-59499/", which in turn contains all the model files for question answering. You will need to specify these path names when loading the pipeline (see next).

## Usage

There are two main ways to interact with the library. You can use it to perform question-answer pair generation from given a text, by loading a pipeline with the QG and QA models. Alternatively, you can use the question answering component only to obtain answers to given questions, instead of additionally performing question generation.

### Q&A Pair Generation

In this repo, test.py shows examples of how to interact with the library. First load the pipeline by specifying the model files as arguments. For example, assuming the model files are in your current working directory:

```
In [1]: from answerquest import QnAPipeline

In [2]: qna_pipeline = QnAPipeline(
   ...:     qg_tokenizer_path="qg_augmented_model/gpt2_tokenizer_vocab/",
   ...:     qg_model_path="qg_augmented_model/qg_augmented_model.pt",
   ...:     qa_model_path="doc_qa_model/checkpoint-59499/"
   ...:     )
```
Then you can use the pipeline to generate question-answer items for a text. For example:
```
In [3]: input_text = "He ate the pie. The pie was served with ice cream."

In [4]: (sent_idxs,
   ...:  questions,
   ...:   answers) = qna_pipeline.generate_qna_items(
                            input_text,
   ...:                     filter_duplicate_answers=True,
   ...:                     filter_redundant=True,
   ...:                     sort_by_sent_order=True
                    )
PRED AVG SCORE: -0.7060, PRED PPL: 2.0259

In [5]: sent_idxs, questions, answers
Out[5]: 
((0, 1),
 ('What did he eat?', 'What was served with ice cream?'),
 ('pie', 'The pie'))
 ```

### Question Answering Only

If you don't need to generate questions and want to run the question answering component only, you can use the following code that loads the QA model only:

```
In [1]: from answerquest import QA

In [2]: qa_only = QA(model_path="doc_qa_model/checkpoint-59499/")

In [3]: pred_answer = qa_only.answer_question(input_text="He ate the pie. The pie was served with ice cream.",
   ...:                                       question="What was the pie served with?")

In [4]: pred_answer
Out[4]: 'ice cream.'
```

## Script

The script "generate_qna.py" provided here in this repo in scripts/ will produce question-answer items for each text in a file of newline-separated texts. See `python generate_qna.py -h` for details:

`usage: generate_qna.py [-h] --qg_tokenizer_path QG_TOKENIZER_PATH --qg_model_path QG_MODEL_PATH --qa_model_path QA_MODEL_PATH
                       --input_texts_path INPUT_TEXTS_PATH --qna_save_path QNA_SAVE_PATH [--qg_batch_size QG_BATCH_SIZE]
                       [--qa_batch_size QA_BATCH_SIZE] [--filter_duplicate_answers] [--filter_redundant] [--sort_by_sent_order]`

For example:

`python generate_qna.py -qg_tokenizer_path qg_augmented_model/gpt2_tokenizer_vocab -qg_model_path qg_augmented_model/qg_augmented_model.pt -qa_model_path doc_qa_model/checkpoint-59499 -input_texts_path [INPUT_TEXT_FILE] -qna_save_path [OUTPUT_JSON_FILE] -filter_duplicate_answers -filter_redundant -sort_by_sent_order`

## Service

### Q&A Pair Generation

The repo also contains an example of running Q&A generation as service. Run qna_server/server.py to start the service with the model files specified as parameters. See `python qna_server/server.py -h`:

`usage: server.py [-h] --qg_tokenizer_path QG_TOKENIZER_PATH --qg_model_path
                 QG_MODEL_PATH --qa_model_path QA_MODEL_PATH [--port PORT]`

For example:

`python qna_server/server.py -qg_tokenizer_path qg_augmented_model/gpt2_tokenizer_vocab -qg_model_path qg_augmented_model/qg_augmented_model.pt -qa_model_path doc_qa_model/checkpoint-59499`

By default the service will run on port 8082. It should automatically detect whether there is a GPU device to run on.

You can then make a POST question to the service to obtain question-answer pairs from a given input text. Here's an example of making this request in Python:
```
In [1]: import requests

In [2]: requests.post('http://0.0.0.0:8082/qg_with_qa', data={'input_text':"He ate the pie. The pie was served with ice cream."}).json()
                                                                             
Out[2]: 
{'answers': ['pie', 'The pie'],
 'questions': ['What did he eat?', 'What was served with ice cream?']}`
```

### Question Answering

There is also an example of running the question answering functionality only, without question generation, so that answers are predicted from an input text given a question. See `python qa_only_server/server.py -h:`

`usage: server.py [-h] --model_path MODEL_PATH`

For example:

`python qa_only_server/server.py -model_path doc_qa_model/checkpoint-59499/`

```
In [1]: requests.post('http://0.0.0.0:8080/documentsqa', json={'paragraph': "He ate the pie. The pie was served with ice cream.", "question": "What was the pie served with?"}).text
Out[2]: 'ice cream.'
```

## Training your own models

The QG and QA models are trained separately.

The [scripts/question_generation folder](scripts/question_generation) contains the files and instructions for training a QG model.  

The [scripts/question_answering folder](scripts/question_answering) contains the files for training a QA model (additional documentation still needed here).
