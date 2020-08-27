import logging
import os
import sys
import argparse
import unittest
import torch
from answerquest import QnAPipeline, QA

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'INFO'))

args = None

input_text = '''Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'''

qna_output = {"sent_idxs": [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3],
              "questions": ["What is the name of the National Football League?",
                            "Who was an American football game?",
                            "What nationality is the game?",
                            "Who defeated the National Football Conference?",
                            "What did Denver earn?",
                            "What was the name of the American Football Conference?",
                            "What was defeated by Denver?",
                            "When was the game played?",
                            "Where was the game played?",
                            "Who emphasized the \"golden anniversary\"?",
                            "What is the Super Bowl?",
                            "What could be known as \"Super Bowl L\"?"],
              "answers": ["(NFL)",
                          "Super Bowl 50",
                          "American",
                          "Denver Broncos",
                          "third Super Bowl title",
                          "(AFC)",
                          "Carolina Panthers",
                          "February 7, 2016",
                          "Levi's Stadium",
                          "the league",
                          "the 50th Super Bowl",
                          "each Super Bowl game"]}


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.qna_pipeline = QnAPipeline(args.qg_tokenizer_path,
                                        args.qg_model_path,
                                        args.qa_model_path)
        self.qa_only = QA(args.qa_model_path)

    def test_qna_pipeline(self):
        (sent_idxs,
         questions,
         answers) = self.qna_pipeline.generate_qna_items(input_text,
                                                         filter_duplicate_answers=True,
                                                         filter_redundant=True,
                                                         sort_by_sent_order=True)
        self.assertIsNotNone(sent_idxs)
        self.assertIsNotNone(questions)
        self.assertIsNotNone(answers)
        self.assertEqual(len(sent_idxs), len(questions), len(answers))

        if args.analyze_output:
            self.assertListEqual(list(sent_idxs), qna_output['sent_idxs'])
            self.assertListEqual(list(questions), qna_output['questions'])
            self.assertListEqual(list(answers), qna_output['answers'])

    def test_qa_only(self):

        pred_answer = self.qa_only.answer_question(input_text=input_text,
                                                   question=qna_output['questions'][0])

        self.assertIsNotNone(pred_answer)
        self.assertNotEqual(pred_answer, "")

        if args.analyze_output:
            self.assertEqual(pred_answer, qna_output['answers'][0])

    def test_gpu_usage(self):
        if torch.cuda.is_available():
            self.assertEqual(self.qna_pipeline.qg_model._gpu, 0)
            self.assertEqual(self.qna_pipeline.qa_model.device.type, "cuda")
            self.assertEqual(self.qa_only.model.device.type, "cuda")
        else:
            self.assertEqual(self.qna_pipeline.qg_model._gpu, -1)
            self.assertEqual(self.qna_pipeline.qa_model.device.type, "cpu")
            self.assertEqual(self.qa_only.model.device.type, "cpu")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--qg_tokenizer_path", "-qg_tokenizer_path",
                        help="Specify path to HuggingFace GPT2Tokenizer config files.",
                        type=str, required=True)
    parser.add_argument("--qg_model_path", "-qg_model_path",
                        help="Specify path to PyTorch question generation model.",
                        type=str, required=True)
    parser.add_argument("--qa_model_path", "-qa_model_path",
                        help="Specify path to PyTorch question answering model.",
                        type=str, required=True)
    parser.add_argument("--analyze_output", "-analyze_output",
                        help="Specifically check if model output matches expected output from trained models.\
                        By default, only output format will be validated.",
                        action='store_true', default=False)
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    unittest.main()
