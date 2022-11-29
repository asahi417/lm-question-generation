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


class SpacyPipeline:

    def __init__(self, language, algorithm: str = 'positionrank'):
        model = "en_core_web_sm" if language not in MODELS else MODELS[language]
        self.nlp = spacy.load(model)
        self.nlp.add_pipe("sentencizer")
        self.algorithm = algorithm
        if self.algorithm == 'yake':
            import spacy_ke  # need to load yake
            self.nlp.add_pipe("yake")
            self.library = 'spacy_ke'
        elif self.algorithm in ['textrank', 'biasedtextrank', 'positionrank']:
            import pytextrank
            self.nlp.add_pipe(algorithm)
            self.library = 'pytextrank'
        elif self.algorithm == 'ner':
            pass
        else:
            raise ValueError(f'unknown algorithm: {self.algorithm}')

    def _get_keyword(self, output, original_document=None, n=None):
        if self.algorithm == 'ner':
            return output.ents
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
        if self.algorithm == 'ner':
            return self.nlp(string).ents
        return self._get_keyword(self.nlp(string), string, n)

    @property
    def language(self):
        return self.nlp.lang


if __name__ == '__main__':
    tmp = "This page contains resources from the Cardiff NLP group at Cardiff University. More info about our organisation here: https://cardiffnlp.github.io"
    test = SpacyPipeline('en', 'ner')
    test.sentence_keyword(tmp)

    tmp = "このページには、カーディフ大学のカーディフ NLP グループのリソースが含まれています。私たちの組織の詳細については、https://cardiffnlp.github.io をご覧ください"
    test = SpacyPipeline('ja', 'ner')
    test.sentence_keyword(tmp)

    tmp = "Esta página contiene recursos del grupo Cardiff NLP en la Universidad de Cardiff. Más información sobre nuestra organización aquí: https://cardiffnlp.github.io"
    test = SpacyPipeline('es', 'ner')
    test.sentence_keyword(tmp)

    tmp = "Diese Seite enthält Ressourcen der Cardiff NLP-Gruppe an der Cardiff University. Weitere Informationen über unsere Organisation hier: https://cardiffnlp.github.io"
    test = SpacyPipeline('de', 'ner')
    test.sentence_keyword(tmp)

    tmp = "Cette page contient des ressources du groupe Cardiff NLP de l'Université de Cardiff. Plus d'informations sur notre organisation ici : https://cardiffnlp.github.io"
    test = SpacyPipeline('fr', 'ner')
    test.sentence_keyword(tmp)

    tmp = "이 페이지에는 카디프 대학의 카디프 그룹의 리소스가 포함되어 있습니다. 여기에서 우리 조직에 대한 자세한 정보"
    test = SpacyPipeline('ko')
    test.sentence_keyword(tmp)

    tmp = "Эта страница содержит ресурсы Кардиффской группы НЛП в Университете Кардиффа. Подробнее о нашей организации здесь: https://cardiffnlp.github.io"
    test = SpacyPipeline('ru')
    test.sentence_keyword(tmp)
