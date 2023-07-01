import spacy 
import nltk
import string 
import numpy as np
from spacy.tokenizer import Tokenizer
from nltk.stem.porter import *
nlp = spacy.load('en_core_web_sm')
class NLP_helper:
    def sentence_tokenzation(self, sentence):
        # tokenizer=Tokenizer(nlp.vocab)
        # token = tokenizer(sentence)
        token =nltk.word_tokenize(sentence)
        return list(token)

    def word_stema(self, word):
        stemmer = PorterStemmer()
        stemma=stemmer.stem(word)
        return stemma
    def tokenezation_remove_stop_word(self,sentence):
        all_stopwords = nlp.Defaults.stop_words
        punc =set(string.punctuation )
        all_stopwords=all_stopwords.union(punc)
        text_tokens=self.sentence_tokenzation(sentence)
        tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
        return tokens_without_sw
    def sentence_vector(self, sentence, words):
        sentence=[self.word_stema(word) for word in sentence ]
        vect=np.zeros(len(words), dtype=np.float32)
        for index,value in enumerate(words):
            if value in sentence:
                vect[index]=1.0
        return vect
    

    

