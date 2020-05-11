""" Pattern and template based chatbot dialog engines """
import logging

# import pandas as pd

from ..etl import quizzes
from .. import spacy_language_model
from ..etl import knowledge_extraction as extract

log = logging.getLogger(__name__)
nlp = spacy_language_model.load('en_core_web_md')


def capitalizations(s):
    return (s, s.lower(), s.upper(), s.title())


class Bot:
    """ Bot that can ask questions and indicate whether user answered is correct

    >>> bot = Bot()
    >>> bot.prompt("What is your name?")
    'What is your name?'
    """

    def __init__(self):
        """ Load glossary from yaml file indicated by list of domain names """
        global nlp
        self.nlp = nlp
        self.qa = quizzes.load("qa-data-science.yml")
        self.vector = dict()
        self.vector['term'] = quizzes.term_vector_dict(self.glossary.keys())
        self.vector['definition'] = quizzes.term_vector_dict(self.glossary.values(), self.glossary.keys())

        self.synonyms = {term: term for term in self.glossary}
        

    def reply(self, statement):
        """ Suggest responses to a user statement string with [(score, reply_string)..]"""
        responses = []
        extracted_term = extract.whatis(statement) or extract.whatmeans(statement) or ''
        if extracted_term:
            for i, term in enumerate(capitalizations(extracted_term)):
                normalized_term = self.synonyms.get(term, term)
                if normalized_term in self.glossary:
                    responses.append((1 - .02 * i, self.glossary[normalized_term]['definition']))
        else:
            responses = [(0.05, "I don't understand. That doesn't sound like a question I can answer using my glossary.")]
        if not len(responses):
            responses.append((0.25, f"My glossaries and dictionaries don't seem to contain that term ('{extracted_term}')."))
        return responses
