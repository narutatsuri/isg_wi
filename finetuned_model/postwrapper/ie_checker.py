import re
import os
import nltk


class IEChecker():
    def __init__(self, 
                 data_path):
        """
        Loads text file containing database of idiomatic expressions (IEs).
        ARGUMENTS:
            data_path: path to text file containing all IEs
        """
        self.VERB_TYPES = ["VB","VBG","VBN","VBP","VBD"]
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        
        with open(os.path.join(os.path.dirname(__file__), data_path)) as handle:
            self.idioms = [idiom for idiom in [line.rstrip('\n') for line in handle] if idiom != '']

    def infer(self, 
              original, 
              paraphrase):
        """
        Check if paraphrase contains an IE that original does not.
        ARGUMENTS:
            original: input sentence string
            paraphrase: paraphrased sentence string
        """
        def normalize_phrase(sentence):
            tokens = nltk.pos_tag(nltk.word_tokenize(sentence))
            d = {}
            for token, type in tokens:
                if type in self.VERB_TYPES:
                    new_token = self.lemmatizer.lemmatize(token,'v')
                    if new_token != token: d[token] = new_token
            if not d:
                normalized_phrase = sentence
            res = sentence.split(" ")
            for i, word in enumerate(res):
                if word in d: res[i] = d[word]

            return " ".join(res)
            
        normalized_original = normalize_phrase(original)
        normalized_paraphrase = normalize_phrase(paraphrase)

        regex_result = False
        for regex_idiom in self.idioms:
            if not re.search(regex_idiom, normalized_original) and re.search(regex_idiom, normalized_paraphrase): 
                regex_result = True

        # Check if original and paraphrase are identical sentences:
        same_sentence = re.sub(r'[^A-Za-z]', '', paraphrase.lower()) != re.sub(r'[^A-Za-z]', '', original.lower())

        return regex_result and same_sentence