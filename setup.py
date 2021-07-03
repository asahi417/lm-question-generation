from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
VERSION = '0.0.0'
NAME = 'mt5sum'
LICENSE = 'MIT'
setup(
    name=NAME,
    packages=find_packages(exclude=['tests']),
    version=VERSION,
    license=LICENSE,
    description='mT5 finetuning on summarization task and crosslingual mt5sum evaluation.',
    url='https://github.com/asahi417/{}'.format(NAME),
    download_url="https://github.com/asahi417/{}/archive/v{}.tar.gz".format(NAME, VERSION),
    keywords=['nlp', 'language model', 'summarization', 'T5'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: {} License'.format(LICENSE),   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "torch",
        "tqdm",
        "requests",
        "pandas",
        "numpy",
        "transformers",
        "sentencepiece",
        "tensorboard",
        "datasets",
        "nltk",
        "rouge_score"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'mt5sum-train = mt5sum_cl.model_training:main'
            # 'mt5sum-eval = cli.evaluate:main'
        ]
    }
)

