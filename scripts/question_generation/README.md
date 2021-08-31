# Creating the Question Generation (QG) Model

Make sure you've installed the library dependencies in the top-level setup.py.

## 1. Data Pre-Processing

In order to train a question generation model using our approach, you'll first need to tokenize a Q&A dataset where the answers to questions are annotated within the texts containing those answers. The QG model observes these answer-annotated texts during training. 

A tiny sample of SquAD/NewsQA items is provided in sample_data/ which is used here to illustrate the data processing steps. In particular, for the data in untokenized/ used as the initial input to the processing pipeline, each line in questions.txt corresponds to a line in input_texts.txt. Each text in input_texts.txt has annotations (i.e. \<ANSWER\>,\<\/ANSWER\>) that designate the start and end of the answer span pertaining to the corresponding question.

First, we used the GPT-2 tokenizer provided by the [HuggingFace transformers library](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer) to tokenize the texts prior to training. We adapted the tokenizer to handle the answer markers as special tokens in the input texts so that they are not split into subtokens. To download and create the adapted tokenizer vocabulary, we ran:

``python create_tokenizer_vocab.py -save_path qg_tokenizer_vocab/``

Alternatively instead of running this command, you can just download the customized vocabulary via ``bash download_models.sh`` as instructed in the top level of the repo.

Then, to apply the tokenizer to a data folder that contains input_texts.txt and questions.txt, designate a folder to save the tokenized data and run it on the folder containing the untokenized files. For example, for the files in sample_data/untokenized/train/):

``python tokenize_data.py -tokenizer_path qg_tokenizer_vocab/ -in_data_dir sample_data/untokenized/train/ -out_data_dir sample_data/tokenized/train``

The -out_data_dir is a new directory that will be created containing the tokenized questions.txt and input_texts.txt. Of course, repeat this command for any additional validation/test data folders. Once the tokenized data is created, you're ready to train the model.


## 2. Training the model

The QG models are transformer sequence-to-sequence models implemented with [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), version 1.2.0. Note that version 1.2.0 is important as this code is not compatible with newer versions of OpenNMT-py. We use their standard commands in this version for data preprocessing and training. Below are the commands that were used, applied to the sample_data/tokenized/ we produced above as an example:

``onmt_preprocess -train_src sample_data/tokenized/train/input_texts.txt -train_tgt sample_data/tokenized/train/questions.txt -valid_src sample_data/tokenized/valid/input_texts.txt -valid_tgt sample_data/tokenized/valid/questions.txt -save_data sample_data/onmt_processed_data -dynamic_dict -share_vocab -src_seq_length 200``

This creates data files prefixed by the -save_data path. Then, to train on this processed data:

``onmt_train -data sample_data/onmt_processed_data -save_model sample_model -layers 4 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -param_init 0 -warmup_steps 8000 -decay_method noam -valid_steps 200 -world_size 1 -gpu_ranks 0 -batch_size 4096 -keep_checkpoint 2 -copy_attn -reuse_copy_attn -max_generator_batches 2 -normalization tokens -batch_type tokens -adam_beta2 0.998 -learning_rate 2 -param_init_glorot -label_smoothing 0.1 -accum_count 4 -save_checkpoint 200 -early_stopping 1``

(Note: set -gpu_ranks based on your own GPU configuration or omit it to run on CPU). 

See the OpenNMT-py documentation for the explanation of all of these parameter settings. Some of these values were selected based on the recommendations [here](https://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model).

Once the model is trained you can use the onmt-translate command to generate questions for any given tokenized texts, e.g.:

``onmt_translate -model sample_model_step_[X].pt -src sample_data/tokenized/valid/input_texts.txt -gpu 0 -verbose -output sample_pred_questions.txt``

You'll need to additionally detokenize this output using the above tokenizer, e.g.

```from transformers import GPT2Tokenizer
from answerquest.utils import detokenize_fn
tokenizer = GPT2Tokenizer.from_pretrained('qg_tokenizer_vocab')
with open('sample_pred_questions.txt') as f:
    detok_questions = [detokenize_fn(tokenizer, question.strip().split(" ")) for question in f]
```

When applied to the full dataset of SQuAD/NewsQA pairs processed via these steps, this yielded the model we refer to as "Standard" in the paper. For training the other model variations discussed in the paper, see below. 


## 3. Model Variations

### RuleMimic Model

As explained in the paper, the RuleMimic model has the same architecture as the model trained above, but it is trained specifically on a dataset of Q&A pairs generated from a rule-based system. 

The system is that of [Heilman & Smith](http://www.cs.cmu.edu/~ark/mheilman/questions/). This system applied rule-based transformations to syntactic parse trees to derive questions from sentences, where each question has an answer contained in the transformed sentence. Additionally, the approach scores the generated questions based on a statistical model of quality. 

The script rule_based_question_generation.py takes any texts as input and produces a dataset with input sentence and question pairs in the same format as the sample data used above. Below are the steps needed to run this script.

First, download the full code directory from the above site (QuestionGeneration-10-01-12.tar.gz)  

This code includes a jar file, and rule_based_question_generation.py script launches the jar as a subprocess. Before running the script, you should separately start the supersense tagging and parsing servers using the README instructions provided with the downloaded code.

To run the parsing server, while inside the directory containing the jar, we did:
``java -Xmx1200m -cp question-generation.jar edu.cmu.ark.StanfordParserServer --grammar config/englishFactored.ser.gz --port 5556 --maxLength 40``

To run the supersense tagging server:
``java -Xmx500m -cp lib/supersense-tagger.jar edu.cmu.ark.SuperSenseTaggerServer  --port 5557 --model config/superSenseModelAllSemcor.ser.gz --properties config/QuestionTransducer.properties``

rule_based_question_generation.py executes the jar and saves its output with the same format and labels as the files in sample_data/. The system outputs questions aligned to single sentences, but because the program does its own sentence-segmentation, we provided full paragraphs as input, e.g. sample_data/raw_text/paragraphs.txt (Note that in contrast to the sample input_texts.txt which already contains answer annotations derived from a Q&A dataset, the text provided to the rule-based system naturally does not already have answer annotations - you can use any raw text).

You'll need to specify the path to the directory containing the jar as well as the config file it loads. The config we used is already contained in QuestionGeneration/config/QuestionTransducer.properties. Make sure the ports specified for the parsing and supersense tagging servers match where you are running them (i.e. what you specified in the above commands).

``python rule_based_question_generation.py -input_file sample_data/raw/paragraphs.txt -jar_dir QuestionGeneration/ -config_file QuestionGeneration/config/QuestionTransducer.properties -output_dir sample_data/rule_gen_output``

This will save the files input_texts.txt and questions.txt to sample_data/rule_gen_output. You can see that input_texts.txt has the same answer-annotation format as in sample_data/untokenized.

From there, you can process this data and train a model on it using the exact same steps outlined in Section 1 and 2. 

### Augmented Model

As detailed in the paper, our best-performing model (referred to as "Augmented") came from first training the RuleMimic model described above, then fine-tuning this model on the full SQuAD/NewsQA dataset. The idea behind this is that model first learns to mimic formal grammatical rules for deriving questions, then additionally simulates more informal, abstractive features of human-authored questions in these Q&A datasets.

We did this simply by fine-tuning the above RuleMimic model on the same SQuAD/NewsQA data described in Section 1. In OpenNMT-py, the -train_from parameter will initialize a new model with the weights of the existing model provided as this parameter. We used the exact same command from Section 2. For example, if you've already trained the model sample_rulemimic_model on the rule-generated data:

``onmt_train -data sample_data/onmt_processed_data -save_model sample_augmented_model -layers 4 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -param_init 0 -warmup_steps 8000 -decay_method noam -valid_steps 500 -world_size 1 -gpu_ranks 0 -batch_size 4096 -keep_checkpoint 2 -copy_attn -reuse_copy_attn -max_generator_batches 2 -normalization tokens -batch_type tokens -adam_beta2 0.998 -learning_rate 2 -param_init_glorot -label_smoothing 0.1 -accum_count 4 -save_checkpoint 500 -early_stopping 1 -train_from sample_rulemimic_model``

