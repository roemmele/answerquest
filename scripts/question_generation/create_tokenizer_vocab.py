import argparse
from transformers import GPT2Tokenizer


def adapt_tokenizer(save_path):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add special tokens for designating answers in squad texts
    special_tokens = ["<ANSWER>", "</ANSWER>"]
    tokenizer.add_tokens(special_tokens)
    # Save tokenizer vocab with special tokens added
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt GPT-2 tokenizer for QG dataset by adding special tokens to vocabulary.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save_path", "-save_path",
                        help="Directory path of where to save tokenizer files.",
                        type=str, required=True)
    args = parser.parse_args()

    adapt_tokenizer(args.save_path)
    print("Saved tokenizer vocab to", args.save_path)
