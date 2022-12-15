""" Spacy Pipeline
python -m spacy download en_core_web_sm  # English
python -m spacy download ja_core_news_sm  # Japanese
python -m spacy download zh_core_web_sm  # Chinese
python -m spacy download de_core_news_sm  # German
python -m spacy download es_core_news_sm  # Spanish
python -m spacy download it_core_news_sm  # Italy
python -m spacy download ko_core_news_sm  # Korean
python -m spacy download ru_core_news_sm  # Russian
python -m spacy download fr_core_news_sm  # French
"""
import spacy

__all__ = 'SpacyPipeline'

MODELS = {
    "en": "en_core_web_sm",
    "ja": "ja_core_news_sm",
    "zh": "zh_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "ko": "ko_core_news_sm",
    "ru": "ru_core_news_sm",
    "fr": "fr_core_news_sm"
}
VALID_METHODS = ['positionrank', 'textrank', 'biasedtextrank', 'positionrank', 'ner']


class SpacyPipeline:

    def __init__(self, language, algorithm: str = None):
        model = "en_core_web_sm" if language not in MODELS else MODELS[language]

        self.nlp = spacy.load(model)
        self.nlp.add_pipe("sentencizer")
        self.algorithm = algorithm
        self.library = None
        if self.algorithm is not None and self.algorithm != 'ner':
            assert algorithm in VALID_METHODS, f'invalid algorithm {algorithm}\n- valid list: {VALID_METHODS}'
            if self.algorithm == 'yake':
                import spacy_ke  # need to load yake
                self.nlp.add_pipe("yake")
                self.library = 'spacy_ke'
            elif self.algorithm in ['textrank', 'biasedtextrank', 'positionrank']:
                import pytextrank
                self.nlp.add_pipe(algorithm)
                self.library = 'pytextrank'
            else:
                raise ValueError(f'unknown algorithm: {self.algorithm}')

    def _get_keyword(self, output, original_document=None, n=None):
        if self.algorithm == 'ner':
            return [str(i) for i in output.ents]
        assert original_document is not None
        assert n is not None
        if self.library == 'spacy_ke':
            return [str(term) for term, score in output._.extract_keywords(n) if str(term) in original_document]
        return [str(i.text) for i in output._.phrases[:n] if str(i.text) in original_document]

    def sentence_keyword(self, string: str, n: int = 10):
        out = self.nlp(string)
        sentence = [str(i) for i in out.sents if len(i) > 0]
        keyword = self._get_keyword(out, string, n)
        return sentence, keyword

    def sentence(self, string: str):
        return [str(i) for i in self.nlp(string).sents if len(i) > 0]

    def token(self, string: str):
        return [str(i) for i in self.nlp.tokenizer(string)]

    def keyword(self, string: str, n: int = 10):
        return self._get_keyword(self.nlp(string), string, n)

    def ner(self, string: str, n: int = None):
        keywords = self.nlp(string).ents
        return keywords[:min(len(keywords), n)] if n is not None else keywords

    @property
    def language(self):
        return self.nlp.lang
