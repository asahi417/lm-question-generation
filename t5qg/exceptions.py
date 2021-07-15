""" Custom exceptions """


class ExceedMaxLengthError(Exception):
    """ Token exceed max length. """

    def __init__(self, max_length=None):
        self.message = 'Input sentence exceeds max length of {}'.format(max_length)
        super().__init__(self.message)


class HighlightNotFoundError(Exception):
    """ Highlight is not in the sentence. """

    def __init__(self, highlight: str, input_sentence: str):
        self.message = 'Highlight () not found in the input sentence ({})'.format(highlight, input_sentence)
        super().__init__(self.message)

