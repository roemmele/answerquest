# Creating a Question Generation (QG) Model

Make sure you've installed the library dependencies in the top-level setup.py.

## Data Pre-processing

In order to train a question generation model using our approach, you'll first need to tokenize a Q&A dataset where the answers to questions are annotated within the texts containing those answers. The QG model observes these answer-annotated texts during training. 

A tiny sample of SquAD/NewsQA items is provided in sample_data/ to show these steps, split into train/ and valid/ sets for a complete illustration. Note that these items have already been processed with the structure expected by the following scripts, so you'll want to emulate this design. Importantly, each line in questions.txt corresponds to a line in answer_sents.txt. The sentence in answer_sents.txt has annotations (i.e. \<ANSWER\>,\<\/ANSWER\>) that designate the start and end of the answer span pertaining to the corresponding question. 

First, we used the GPT-2 tokenizer provided by HuggingFace (link) to tokenize the texts prior to training. We adapted the tokenizer to handle the answer markers as special tokens so that they are not split into subtokens. To download and create the adapted tokenizer vocabulary, we ran:

```python create_tokenizer_vocab.py --save_path qg_tokenizer_vocab/```

Alternatively instead of running this command, you can just download the customized vocabulary via ``bash download_models.sh`` as instructed in the top level of the repo.

Then, to apply the tokenizer to the data in one of the partitions of sample_data/ (for example, train/), run:

```python tokenize_data.py --in_data_dir sample_data/train -out_data_dir tok_sample_data/train```

The -out_data_dir is a new directory that will be created containing the tokenized questions.txt and answer_sents.txt. Once you've done this for all training/validation data you want to use for the model, you're ready to train the model. 


## Training the model

The QG models are transformer sequence-to-sequence models implemented with [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), version 1.2.0. We use their standard scripts for data preprocessing and training. Below are the commands that were used, applied to the tok_sample_data/ we produced above as an example (these scripts are executed inside the OpenNMT-py repo where the scripts are located):

``python preprocess.py -train_src ~/masked_dataset/train/answer_sents.txt -train_tgt ~/masked_dataset/train/sent_questions.txt -valid_src ~/masked_dataset/valid/answer_sents.txt -valid_tgt ~/masked_dataset/valid/sent_questions.txt -save_data ~/masked_dataset/processed_sentence -src_seq_length 100 -dynamic_dict -share_vocab``

``python train.py -data ~/masked_dataset/processed_paragraph -save_model ~/qg_models/paragraph_copy_model_suggested_params -layers 4 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -param_init 0 -warmup_steps 8000 -decay_method noam -valid_steps 500 -world_size 1 -gpu_ranks 0 -batch_size 4096 -keep_checkpoint 2 -copy_attn -reuse_copy_attn -max_generator_batches 2 -normalization tokens -batch_type tokens -adam_beta2 0.998 -learning_rate 2 -param_init_glorot -label_smoothing 0.1 -accum_count 4 -save_checkpoint 500 -early_stopping 1``

When applied to the full dataset of SQuAD/NewsQA pairs processed in the format shown by sample_data/, this command yielded the model we refer to as ``Standard`` in the paper. For training the other model variations discussed in the paper, see below. 


## Model Variations

### RuleMimic

As explained in the paper, the RuleMimic model has the same architecture as the model trained above, but it is trained specifically on a dataset of Q&A pairs generated from a rule-based system. 

The system is that of [Hielman & Smith](http://www.cs.cmu.edu/~ark/mheilman/questions/). This sytem applied rule-based transformations to syntactic parse trees to derive questions from sentences, where each question has an answer contained in the transformed sentence. Additionally, the approach scores the generated questions based on a statistical model of quality. 

The result of the code here is a dataset with input sentence and question pairs (answers highlighted in the input sentence) along with their scores.The script rule_based_question_generation.py execute , 

You will first need to download the jar from the above site. The rule_based_question_generation.py script launches the jar as a subprocess. Before running the script, you should start the supersense tagging and parsing servers using the README instructions provided with the jar download.

To run the parsing server:
`java -Xmx1200m -cp question-generation.jar edu.cmu.ark.StanfordParserServer --grammar config/englishFactored.ser.gz --port 5556 --maxLength 40`

To run the supersense tagging server:
`java -Xmx500m -cp lib/supersense-tagger.jar edu.cmu.ark.SuperSenseTaggerServer  --port 5557 --model config/superSenseModelAllSemcor.ser.gz --properties config/QuestionTransducer.properties.1`

Start multiple servers by varying the port number. These port numbers need to be consistent with the ones specified in config/QuestionTransducer.properties.{X}. There is a properties file for each unique pair of supersenseServerPort and parserServerPort specifications.

Each properties file can be provided as input (via the -config_file parameter) to a different run of rule_based_question_generation.py so that each run will use the corresponding sense and parsing servers in that config. The script takes a file (-input_file) of newline-separated texts as inputs from which questions will be generated. Specify which chunk of texts each run should process using the start_idx and end_idx parameters, which correspond to line indices in the input file. The script will process texts will these indices specifically and then write the questions generated from these texts to a file {save_prefix}_{start_idx}_{end_idx}.csv. Here is an example:

`python rule_based_question_generation.py -input_file  ~/question_generation/newsqa_untok_data/train/unique_paragraphs.txt -config_file ~/question_generation/HS-QuestionGeneration/config/QuestionTransducer.properties.1 -start_idx 0 -end_idx 10 -save_prefix ~/question_generation/squad_data/hs_output_sample`



As described in the paper, we trained three models: one simply trained on the SQuAD/NewsQA Q&A pairs (the "Standard" model), one trained on the Q&A pairs derived from the rule-based method above ("RuleMimic"), and one initialized from the RuleMimic model and fine-tuned on the SQUaD/News QA pairs ("Augmented"). For training the Standard and RuleMimic models, we ran:


Training the Augmented model was exactly the same with just an additional parameter supplied to train.py specifying the location of the trained RuleMimic model:

``python train.py -data ~/fine_tuned_rule_based_squad/processed_data/processed -save_model ~/fine_tuned_rule_based_squad/trained_models/copy_model_suggested_params -layers 4 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -param_init 0 -warmup_steps 8000 -decay_method noam -valid_steps 500 -world_size 1 -gpu_ranks 0 -batch_size 4096 -keep_checkpoint 2 -copy_attn -reuse_copy_attn -max_generator_batches 2 -normalization tokens -batch_type tokens -adam_beta2 0.998 -learning_rate 2 -param_init_glorot -label_smoothing 0.1 -accum_count 4 -save_checkpoint 500 -early_stopping 1 -train_from ~/rule_based_entity_tagged_squad/trained_models/copy_model_suggested_params_step_3500.pt``

You can use OpenNMT-py's "translate.py" script to generate questions for any tokenized texts:

``python translate.py -model ~/qg_models/paragraph_copy_model_step_100000.pt -src ~/masked_dataset/valid/paragraphs.txt -verbose -gpu 0 -output ~/qg_outputs/paragraph_copy_model_outputs.txt``

Note that the output of this script will need to be detokenized, e.g. ``answerquest.utils.detokenize_fn(TOKENIZER_PATH, question)``.

