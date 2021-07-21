""" Multilingual sentence splitter for answer extraction model training.
TODO: Cover all the language in TYDIQA dataset.
"""

import re
from typing import List
from langdetect import detect
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

__all__ = 'SentSplit'


class JASplitter:
    """ JA sentence splitter from https://github.com/himkt/konoha/blob/master/konoha/sentence_tokenizer.py """

    PERIOD = "。"
    PERIOD_SPECIAL = "__PERIOD__"
    PATTERNS = [re.compile(r"（.*?）"), re.compile(r"「.*?」")]

    @staticmethod
    def conv_period(item) -> str:
        return item.group(0).replace(JASplitter.PERIOD, JASplitter.PERIOD_SPECIAL)

    def __call__(self, document) -> List[str]:
        for pattern in JASplitter.PATTERNS:
            document = re.sub(pattern, self.conv_period, document)

        result = []
        for line in document.split("\n"):
            line = line.rstrip()
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            line = line.replace("。", "。\n")
            sentences = line.split("\n")

            for sentence in sentences:
                if not sentence:
                    continue

                period_special = JASplitter.PERIOD_SPECIAL
                period = JASplitter.PERIOD
                sentence = sentence.replace(period_special, period)
                result.append(sentence)

        return result


class Splitter:

    def __call__(self, document):
        out = sent_tokenize(document)
        return [re.sub(r'\s*\Z', ' ', i) for i in out[:-1]] + [out[-1]]


def setup_splitter(language):
    if language in ['ja', 'jp']:
        return JASplitter()
    else:
        return Splitter()


class SentSplit:

    def __init__(self, language: str = 'en'):
        self.language = language
        self.splitter = setup_splitter(self.language)

    def __call__(self, docs: str):
        la = detect(docs)
        if self.language != la:
            self.splitter = setup_splitter(la)
            self.language = la
        return self.splitter(docs)





