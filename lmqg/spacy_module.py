""" Spacy Pipeline
python -m spacy download en_core_web_sm  # English
python -m spacy download ja_core_news_sm  # Japanese
python -m spacy download zh_core_web_sm  # Chinese
"""
import spacy

__all__ = 'SpacyPipeline'

MODELS = {"en": "en_core_web_sm", "ja": "ja_core_news_sm", "zh": "zh_core_web_sm"}


class SpacyPipeline:

    def __init__(self, language, algorithm: str = 'textrank'):
        model = "en_core_web_sm" if language not in MODELS else MODELS[language]
        self.nlp = spacy.load(model)
        self.nlp.add_pipe("sentencizer")
        if algorithm == 'yake':
            import spacy_ke  # need to load yake
            self.nlp.add_pipe("yake")
            self.library = 'spacy_ke'
        elif algorithm in ['textrank', 'biasedtextrank', 'positionrank']:
            import pytextrank
            self.nlp.add_pipe(algorithm)
            self.library = 'pytextrank'
        else:
            raise ValueError('unknown algorithm: {}'.format(algorithm))

    def _get_keyword(self, output, original_document, n):
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

    @property
    def language(self):
        return self.nlp.lang
