from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
VERSION = '0.0.3'
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
            'pytextrank',
            'pydantic',
            'protobuf',
            'psutil'
        ]
    },
    install_requires=[
        "torch",
        "tqdm",
        "requests",
        "pandas",
        "numpy",
        "transformers",
        "sentencepiece",
        "datasets",
        "spacy",
        'sudachipy',
        'sudachidict_core',
        'bert-score',
        'pyemd'  # to compute moverscore
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'lmqg-train-search = lmqg.lmqg_cl.model_finetuning:main_training_search',
            'lmqg-eval = lmqg.lmqg_cl.model_evaluation:main',
            'lmqg-push-to-hf = lmqg.lmqg_cl.push_to_hf:main'
        ]
    }
)

