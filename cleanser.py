"""
This module implements a version of the noisy text normalization system
described in "Contextual Bearing on Linguistic Variation" by Gouws et al. (2011).

Author: Stephan Gouws
Contact: stephan@ml.sun.ac.za
"""

from decoder import Decoder
from generator import Generator


class TextCleanser(object):

    def __init__(self, decoder=None, generator=None):
        if generator is None:
            self.generator = Generator()
        else:
            self.generator = generator
        if decoder is None:
            self.decoder = Decoder()
        else:
            self.decoder = decoder

    def _get_replacements(self, cleantext, old_tokens):
        """return the token replacements that were made"""
        new_tokens = self.generator.tokenize(cleantext)
        # if new_tokens contain more tokens than old_tokens then alignment is
        # screwed
        if len(new_tokens) > len(old_tokens):
            replacements = -1
        else:
            replacements = []
            for i, new_tok in enumerate(new_tokens):
                if i >= len(old_tokens):
                    break
                old_tok = old_tokens[i]
                if new_tok != old_tok.lower():
                    replacements.append((old_tok, new_tok))

        return replacements

    def _cleanse(self, text, string_sim_func, gen_off_by_ones):
        replacements, old_tokens, candidates = self.generator.sent_generate_candidates(text, string_sim_func, gen_off_by_ones)
        cleantext, error = self.decoder.decode(candidates)
        replacements = self._get_replacements(cleantext, old_tokens)
        return cleantext, error, replacements

    def ibm_cleanse(self, text, gen_off_by_ones=False):
        """Accept noisy text, run through cleanser described in Gouws et al. 2011, and
        return the cleansed text."""
        return self._cleanse(text, self.generator.IBM_SIM, gen_off_by_ones)

    def phonetic_cleanse(self, text, gen_off_by_ones=False):
        return self._cleanse(text, self.generator.PHONETIC_ED_SIM, gen_off_by_ones)

    def ssk_cleanse(self, text, gen_off_by_ones=False):
        """Use subsequence overlap similarity function"""
        return self._cleanse(text, self.generator.SSK_SIM, gen_off_by_ones)


if __name__ == "__main__":
    tc = TextCleanser()

    testSents = ['test sentence one', 'test s3ntens tw0', 't0day iz awssam', 'i jus talk to her.she ridin wit us',
                 'Whts papppin tho happy new years to u an ya fam',
                 "Be sure 2 say HI to Wanda she's flying in from Toronto ;) 2 give a seminar on the art of correction, she'll b @ our booth",
                 "LOL i kno rite?", "Trying t fnd out if it does hav at as a word"]

    # test_confusion_set = [[(0.4, "w1"), (0.6, "w2")], [(0.3, "w3"),(0.3, "w4"), (0.4, "w5")]]

    for s in testSents:
        # c = gen.sent_generate_candidates(s)
        print("Sentence: {}".format(s))
        # print("Candidate list: {}".format(c))
        # word_lattice = gen.generate_word_lattice(c)
        # print("Word lattice: {}".format(word_lattice))
        cleantext, err, replacements = tc.ibm_cleanse(s)
        print("Decoding result: {}".format(cleantext))
