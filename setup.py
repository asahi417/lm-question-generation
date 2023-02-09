from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
VERSION = '0.0.7'
NAME = 'lmqg'
LICENSE = 'MIT License'
setup(
    name=NAME,
    packages=find_packages(exclude=['tests', 'misc', 'asset']),
    version=VERSION,
    license=LICENSE,
    description='Language Model for Question Generation.',
    url='https://github.com/asahi417/lm-question-generation',
    download_url="https://github.com/asahi417/lm-question-generation/archive/v{}.tar.gz".format(VERSION),
    keywords=['language model', 'question-answering', 'question-generation'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        f'License :: OSI Approved :: {LICENSE}',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    extras_require={
        "api": [
            'uvicorn',
            'fastapi',
            'pydantic',
            'spacy_ke',
            'pydantic',
            'protobuf',
            'lmppl'
        ]
    },
    install_requires=[
        'psutil',
        'pytextrank',
        "torch",
        "tqdm",
        "requests",
        "pandas",
        "numpy",
        "transformers<=4.21.2",  # push-to-model is not working for latest version
        "huggingface-hub<=0.9.1",
        "sentencepiece",
        "datasets",
        "spacy",
        'sudachipy',
        'sudachidict_core',
        'bert-score',
        'pyemd',  # to compute moverscore
        'evaluate',
        "wandb",
        "ray",
        "ray[tune]"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'lmqg-train-search = lmqg.lmqg_cl.model_finetuning:main_training_search',
            'lmqg-eval = lmqg.lmqg_cl.model_evaluation:main',
            'lmqg-eval-qag = lmqg.lmqg_cl.model_evaluation_qag:main',
            'lmqg-eval-qa = lmqg.lmqg_cl.model_evaluation_qa:main',
            'lmqg-push-to-hf = lmqg.lmqg_cl.push_to_hf:main',
            'lmqg-generate-qa = lmqg.lmqg_cl.model_evaluation_qa_based_metric:main_generate_qa_pair',
            'lmqg-qae = lmqg.lmqg_cl.model_evaluation_qa_based_metric:main_qa_model_training'
        ]
    }
)

