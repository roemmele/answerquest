import setuptools

setuptools.setup(
    name="answerquest",
    version="1.0.0",
    author="SDL Research",
    packages=setuptools.find_packages(),
    setup_requires=['spacy', 'nltk'],
    install_requires=['numpy',
                      'torch<1.6',
                      'spacy',
                      'flask',
                      'transformers',
                      'rank-bm25',
                      'nltk',
                      'flask-cors',
                      'OpenNMT-py'],
    include_package_data=True
)

# Download the required spacy and nltk models
import spacy
spacy.cli.download('en')
import nltk
nltk.download('punkt')