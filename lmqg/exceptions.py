""" Custom exceptions """


class ExceedMaxLengthError(Exception):
    """ Token exceed max length. """

    def __init__(self, max_length=None):
        self.message = f'Input sentence exceeds max length of {max_length}'
        super().__init__(self.message)


class HighlightNotFoundError(Exception):
    """ Highlight is not in the sentence. """

    def __init__(self, highlight: str, input_sentence: str):
        self.message = f'Highlight `{highlight}` not found in the input sentence `{input_sentence}`'
        super().__init__(self.message)


class AnswerNotFoundError(Exception):
    """ Answer cannot found in the context. """

    def __init__(self, context: str):
        self.message = f'Model cannot find any answer candidates in `{context}`'
        super().__init__(self.message)


class APIError(Exception):
    """ Error from huggingface inference API. """

    def __init__(self, context: str):
        self.message = f'Huggingface API Error:\n`{context}`'
        super().__init__(self.message)
