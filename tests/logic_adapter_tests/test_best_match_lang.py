from mock import MagicMock
from chatterbot.logic import BestMatchLang
from chatterbot.conversation import Statement, Response
from tests.base_case import ChatBotTestCase


class BestMatchLangTestCase(ChatBotTestCase):
    """
    Unit tests for the BestMatchLang logic adapter.
    """

    def setUp(self):
        super(BestMatchLangTestCase, self).setUp()
        from chatterbot.utils import nltk_download_corpus
        from chatterbot.comparisons import synset_distance

        nltk_download_corpus('stopwords')
        nltk_download_corpus('wordnet')
        nltk_download_corpus('punkt')

        self.adapter = BestMatchLang(
            statement_comparison_function=synset_distance,
            language='spanish',
            lang='spa'
        )

        # Add a mock storage adapter to the logic adapter
        self.adapter.set_chatbot(self.chatbot)

    def test_different_punctuation(self):
        possible_choices = [
            Statement('¿Quién eres?'),
            Statement('¿Eres bueno?'),
            Statement('Tú eres bueno')
        ]
        self.adapter.chatbot.storage.get_response_statements = MagicMock(
            return_value=possible_choices
        )

        statement = Statement('Eres bueno')
        match = self.adapter.get(statement)

        self.assertEqual('¿Eres bueno?', match)

    def test_no_known_responses(self):
        """
        In the case that a match is selected which has no known responses.
        In this case a random response will be returned, but the confidence
        should be zero because it is a random choice.
        """
        self.adapter.chatbot.storage.update = MagicMock()
        self.adapter.chatbot.storage.count = MagicMock(return_value=1)
        self.adapter.chatbot.storage.get_random = MagicMock(
            return_value=Statement('Random')
        )

        match = self.adapter.process(Statement('Blah'))

        self.assertEqual(match.confidence, 0)
        self.assertEqual(match.text, 'Random')

    def test_get_closest_statement2(self):
        """
        Note, the content of the in_response_to field for each of the
        test statements is only required because the logic adapter will
        filter out any statements that are not in response to a known statement.
        """
        possible_choices = [
            Statement('Es un pantano precioso.', in_response_to=[Response('Es un pantano precioso.')]),
            Statement('Es una ciénaga bonita.', in_response_to=[Response('Es una ciénaga bonita.')]),
            Statement('Esto huele como una ciénaga.', in_response_to=[Response('Esto huele como una ciénaga.')])
        ]

        self.adapter.chatbot.storage.filter = MagicMock(
            return_value=possible_choices
        )

        statement = Statement('Es una ciénaga preciosa.')
        match = self.adapter.get(statement)

        self.assertEqual('Es un pantano precioso.', match)
