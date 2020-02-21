
# question_generation

This folder contains a Flask app that runs question-answer pair generation as a service. 

## Dependencies

This code requires the packages numpy, pytorch, flask, spacy, nlkt, OpenNMT-py, rank-bm25, and transformers. You can create a conda environment with these dependencies via:

`conda create --name my_qg_env python=3 numpy pytorch flask spacy nltk`
`conda activate my_qg_env`
`pip install OpenNMT-py rank_bm25 transformers`
`python -m spacy download en` (This downloads the required english model file for spaCy text processing)

## Running the code

To run this service, you'll need the question generation (QG) and question answering (QA) model files. 

Download the QG model files [here](https://qna.sdl.com:8443/qg_augmented_model.zip). These files include both the generation model (qg_augmented_model.pt) as well as the tokenizer files needed for preprocessing (gpt2_tokenizer_vocab/).

Download the QA model files [here](https://qna.sdl.com:8443/doc_qa_model.zip). The contained folder checkpoint-59499/ is what you will need to specify when loading the service.

Run flask_app/server.py to start the service with these model files specified as parameters. See "`python flask_app/server.py -h`" for details:

`usage: server.py [-h] --qg_tokenizer_path QG_TOKENIZER_PATH --qg_model_path
                 QG_MODEL_PATH --qa_model_path QA_MODEL_PATH [--gpu_id GPU_ID]
                 [--port PORT]`

For example:

`python flask_app/server.py -qg_tokenizer_path qg_augmented_model/gpt2_tokenizer_vocab -qg_model_path qg_augmented_model/qg_augmented_model.pt -qa_model_path doc_qa_model/checkpoint-59499`

By default the service will run on port 8082 and with gpu_id=-1 (CPU).

You can then make a POST question to the service to obtain question-answer pairs from a given input text. Here's an example of making this request in Python:

`In [1]: import requests`
                                                                                                        
`In [2]: requests.post('http://0.0.0.0:8082/qg_with_qa', data={'input_text':"He ate the pie. The pie was served with ice cream."}).json()`
                                                                                                          
`Out[2]: 
{'answers': ['pie', 'The pie'],
 'questions': ['What did he eat?', 'What was served with ice cream?']}`
