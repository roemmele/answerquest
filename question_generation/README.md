
# question_generation

This folder contains code that runs question-answer pair generation either via a command-line script or as a service.

## Dependencies

This code requires the packages numpy, pytorch, flask, spacy, nlkt, OpenNMT-py, rank-bm25, and transformers. You can create a conda environment with these dependencies via:

`conda create --name my_qg_env python=3 numpy pytorch flask spacy nltk`

`conda activate my_qg_env`

`pip install OpenNMT-py rank_bm25 transformers`

`python -m spacy download en` (This downloads the required english model file for spaCy text processing)

## Running the code

To run the code, you'll need the question generation (QG) and question answering (QA) model files. 

Download the QG model files [here](https://qna.sdl.com:8443/qg_augmented_model.zip). These files include both the generation model (qg_augmented_model.pt) as well as the tokenizer files needed for preprocessing (gpt2_tokenizer_vocab/).

Download the QA model files [here](https://qna.sdl.com:8443/doc_qa_model.zip). The contained folder checkpoint-59499/ is what you will need to specify when loading the service.

### Script

You can run "generate_qna.py" to produce question-answer items from a given input text. It should automatically detect whether there is a GPU device to run on. See "python generate_qna.py -h" for details.

`usage: generate_qna.py [-h] --qg_tokenizer_path QG_TOKENIZER_PATH
                       --qg_model_path QG_MODEL_PATH --qa_model_path
                       QA_MODEL_PATH --input_texts_path INPUT_TEXTS_PATH
                       --qna_save_path QNA_SAVE_PATH
                       [--qg_batch_size QG_BATCH_SIZE]
                       [--qa_batch_size QA_BATCH_SIZE]`

For example:

`python generate_qna.py -qg_tokenizer_path qg_augmented_model/gpt2_tokenizer_vocab -qg_model_path qg_augmented_model/qg_augmented_model.pt -qa_model_path doc_qa_model/checkpoint-59499 -input_texts_path [INPUT_TEXT_FILE] -qna_save_path [OUTPUT_JSON_FILE]`

### Service

The system can also be run as a Flask service. Run flask_app/server.py to start the service with the model files specified as parameters. See "`python flask_app/server.py -h`" for details:

`usage: server.py [-h] --qg_tokenizer_path QG_TOKENIZER_PATH --qg_model_path
                 QG_MODEL_PATH --qa_model_path QA_MODEL_PATH [--port PORT]`

For example:

`python flask_app/server.py -qg_tokenizer_path qg_augmented_model/gpt2_tokenizer_vocab -qg_model_path qg_augmented_model/qg_augmented_model.pt -qa_model_path doc_qa_model/checkpoint-59499`

By default the service will run on port 8082. It should automatically detect whether there is a GPU device to run on.

You can then make a POST question to the service to obtain question-answer pairs from a given input text. Here's an example of making this request in Python:

`In [1]: import requests`
                                                                                                        
`In [2]: requests.post('http://0.0.0.0:8082/qg_with_qa', data={'input_text':"He ate the pie. The pie was served with ice cream."}).json()`
                                                                                                          
`Out[2]: 
{'answers': ['pie', 'The pie'],
 'questions': ['What did he eat?', 'What was served with ice cream?']}`
