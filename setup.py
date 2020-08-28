import setuptools

setuptools.setup(
    name="answerquest",
    version="1.0.0",
    author="SDL Research",
    packages=setuptools.find_packages(),
    install_requires=['numpy==1.19.1',
                      'torch==1.6.0',
                      'spacy==2.3.2',
                      'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz',
                      'transformers==3.0.2',
                      'rank-bm25==0.2.1',
                      'nltk==3.5',
                      'OpenNMT-py==1.2.0'],
    include_package_data=True
)
