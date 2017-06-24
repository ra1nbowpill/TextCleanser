"""
    This module generates candidate words by comparing possibly noisy
    words against words in the lexicon and returning the words it considers
    most likely candidates.

    Author: Stephan Gouws
    Contact: stephan@ml.sun.ac.za
"""

import collections
import csv
import heapq
import os
import pickle as pickle
import re
import string
import sys
import time

from phonetic_algorithms import PhoneticAlgorithms
import string_functions as str_fun

TEXTCLEANSER_ROOT = os.path.split(os.path.realpath(__file__))[0] + os.sep

# constants
LEXICON_FILENAME = TEXTCLEANSER_ROOT + "data/combined_lex.pickl"
PHON_LEX_FILENAME = TEXTCLEANSER_ROOT + "data/phonetic_lex.pickl"
PHONE_LEX_KEYS_FILENAME = TEXTCLEANSER_ROOT + "data/phonetic_lex_keys.pickl"
SUB_LEX_FILENAME = TEXTCLEANSER_ROOT + "data/sub_lexicon.pickl"
STOPWORDS_FILENAME = TEXTCLEANSER_ROOT + "data/stopwords-set.pickl"
COMMON_ABBR_FILENAME = TEXTCLEANSER_ROOT + "data/common_abbrs.csv"

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
PUNC = string.punctuation
STOPWORDS = pickle.load(open(STOPWORDS_FILENAME, "rb"), encoding="UTF-8")

# placeholder for empty symbol, such that tokeniser doesn't split it
EMPTY_SYM = "EMPTYSYM"


class Generator:

    # CLASS CONSTANTS
    # Candidate word selection methods
    IBM_SIM = 0
    SSK_SIM = 1
    PHONETIC_ED_SIM = 2

    def __init__(self, lexicon=None):

        if lexicon is None:
            # load the lexicon from a pickled dump
            # we use the Top20K words in gigaword intersected with CMU
            # pronunciation dictionary, adapt this to your needs
            self.lexicon = pickle.load(open(LEXICON_FILENAME, 'rb'))
        else:
            self.lexicon = lexicon

        # load the phonetically indexed lexicon
        try:
            self.phon_lex = pickle.load(open(PHON_LEX_FILENAME, 'rb'))
            # print("Loaded phonetically indexed lexicon with {} partitions.".format(len(self.phon_lex.keys())))
        except IOError:
            # build phonetic lexicon
            self.phon_lex = self.build_phonetic_index()
            pickle.dump(self.phon_lex, open(PHON_LEX_FILENAME, 'wb'))
        # load the list of phonetic keys
        self.phonetic_keys = self.phon_lex.keys()

        # load lexicon sub-indexed by first consonant
        try:
            self.sub_lexicon = pickle.load(open(SUB_LEX_FILENAME, 'rb'))
            # print("Loaded sub-indexed lexicon with {} partitions.".format(len(self.sub_lexicon.keys())))
        except IOError:
            # build sub-lexicon
            # self.lexicon = cPickle.load(open(LEXICON_FILENAME, 'rb'))
            # for w in self.lexicon:      # remove one-letter entries
            #    if len(w) <= 1:
            #        del w
            self.sub_lexicon = self.build_sub_index()
            pickle.dump(self.sub_lexicon, open(SUB_LEX_FILENAME, 'wb'))

        # load list of common abbreviations, adapt common_abbrs.csv to your own
        # needs
        self.abbr_list = {}
        for row in csv.reader(open(COMMON_ABBR_FILENAME), delimiter=","):
            self.abbr_list[row[0]] = row[1]
        sys.stderr.write("Loaded list of {} common abbreviations.\n".format(len(self.abbr_list)))

        # print("Loaded lexicon of size: {}".format(len(self.lexicon)))

        # generate K possible noisy expansions per word.
        self.topK = 10

        # load phonetic similarity function
        self.phon_sim = PhoneticAlgorithms().double_metaphone
        # load subsequence-kernel similarity function
        ssk = str_fun.StringFunctions(lamb=0.8, p=2)
        self.ssk_sim = ssk.SSK

        # smiley regex
        self.smiley_regex = re.compile("[:;]-?[DP()\\\|bpoO0]{1,2}")  # recognise emoticons
        self.lt_gt_regex = re.compile("&[lr]t;[\d]?")
        # pnc = ''.join(l for l in [string.punctuation]) + " \t"  # remove double punctuation and spaces
        # regexes for detecting usernames, hashtags, rt's and urls at the token
        # level
        self.hashTags = re.compile("^#(.*)")
        self.username = re.compile("@\w+")
        self.rt = re.compile("^rt")
        # this regex from: http://snipplr.com/view/36992/improvement-of-url-interpretation-with-regex/
        # I made the http required (+) since otherwise it matches things like 'River.With'
        # where users did not put a space between the end of sentence and next
        # word
        self.urls = re.compile(
            "((https?://)+([-\w]+\.[-\w.]+)+\w(:\d+)?(/([-\w/_.]*(\?\S+)?)?)*)")
        # self.sim_cache = shelve.open('sim_cache')

    def build_phonetic_index(self):
        """Index the lexicon phonetically to reduce lookup time. However this failed
            epically and only reduced search space by half on average..."""
        print("Building phonetic index over {} words.".format(len(self.lexicon)))
        t1 = time.time()
        phon_lex = collections.defaultdict(list)
        dbl_metaphone = PhoneticAlgorithms().double_metaphone
        for w in self.lexicon:
            m = dbl_metaphone(w)
            phon_lex[m[0]].append(w)
            if m[1] is not None:
                phon_lex[m[1]].append(w)
        print("Finished building phonetic index in {} seconds.".format((time.time() - t1)))
        return phon_lex

    def build_sub_index(self):
        """Index the lexicon by the first consonant, or first letter if no
            consonant found, for faster lookup"""
        cons_lex = collections.defaultdict(list)
        for w in self.lexicon:
            first_letter = str_fun.get_first_cons(w)
            cons_lex[first_letter].append(w)
        return cons_lex

    def expand_abbrs(self, tok):
        """Look for and expand commonly occurring Internet and other abbreviations.
            Edit the list in data/common_abbrs.csv.
            Note that it is a deterministic replacement.
        """
        try:
            return self.abbr_list[tok]
        except KeyError:
            return None

    def rank_candidates(self, candidates, lexicon_list, sim_function, gen_off_by_ones):
        """Rank all tokens in candidates with tokens found in lexicon. To make the use
            of sub-lexicons possible, it uses lexicon_list[i] for i,token in enumerate(candidates),
            i.e. you pass a lexicon_list[i] for each token
            TODO: Might use lexicon and sub_lex parameters instead?
            """
        # Generate all 'off-by-one' variants
        if gen_off_by_ones:
            off_by_ones = []
            for candidate in candidates:
                # if w in lexicon]
                off_by_ones.extend([w for w in str_fun.word_edits(candidate)])
            candidates.extend(list(set(off_by_ones)))
            # print("{} after adding edits.".format(len(noisyWords)))

        # compute scores with words in lexicon
        K = self.topK
        top_k = []    # top K heap of candidate words

        cur_w = 0
        for i, cand_token in enumerate(candidates):
            # for w in self.lexicon:
            lexicon = lexicon_list[i]
            for w in lexicon:
                # if len(w)>=len(cand_token):
                sim = sim_function(cand_token, w)
                if sim == 0:      # don't add zero-prob words
                    continue
                if cur_w < K:        # first K, just insert into heap
                    heapq.heappush(top_k, (sim, w))
                else:           # next (N-K), insert if sim>smallest sim in heap (root)
                    try:
                        if sim > top_k[0][0]:
                            heapq.heapreplace(top_k, (sim, w))
                    except IndexError:
                        heapq.heappush(top_k, (sim, w))
                cur_w += 1

        conf_set = heapq.nlargest(K - 1, top_k, key=lambda x: x[0])
        return conf_set

    def hash_user_rt_url(self, tok):
        # if noisyWord in ['rt', 'hshtg', 'usr', 'url', EMPTY_SYM]:
        hash_tags = self.hashTags
        username = self.username
        rt = self.rt
        urls = self.urls
        if hash_tags.search(tok) or username.search(tok) or rt.search(tok) or urls.search(tok):
            return True
        else:
            return False

    def check_oov_but_valid(self, noisy_word):
        # check if it is a common abbreviation or contracted form
        # TODO: Is this the right place to do this?
        abbr = self.expand_abbrs(noisy_word.strip("'"))
        if abbr is not None:
            # TODO: this never replaces lol, etc with EMPTY_SYM after decoding
            #            return [(0.8, abbr), (0.2, noisyWord)]
            return [(1.0, abbr)]

        # check if it's a number, longer than 1 char
        # TODO: Do better detection to catch any token with no letters
        # e.g. "9/11" (dates), "4-4" (scores), etc. and ignore
        # Simple first idea, if it contains _no letters_, ignore

        if len([1 for c in noisy_word if c in "0123456789"]) > 4:
            return [(1.0, noisy_word)]

        if not re.search("[a-z]", noisy_word):
            return [(1.0, noisy_word)]

        # check for punctuation
        if noisy_word in PUNC:
            return [(1.0, noisy_word)]

        # check for USR or URL or EMPTY_SYM  special tokens
        # noisy=hashTags.sub('hshtg', username.sub('usr', rt.sub('rt', urls.sub('url', noisy))))
        if self.hash_user_rt_url(noisy_word) or noisy_word in [EMPTY_SYM]:
            return [(1.0, noisy_word)]

        # otherwise
        return None

    # @profile
    def word_generate_candidates(self, noisy_word, rank_method, off_by_ones=False):
        """Generate a confusion set of possible candidates for a word using some rank_method,
            currently supported methods include:
            Generator.IBM_SIM - implementation of the heuristic used in Contractor et al. 2010
            Generator.SSK_SIM - a 2-char string subsequence similarity function
            Generator.PHONETIC_ED_SIM - a phonetic edit distance"""

        oov_but_valid = self.check_oov_but_valid(noisy_word)
        if oov_but_valid:
            return oov_but_valid

        # TODO: This seems clumsy, lexicon should include STOPWORDS by default
        # O(1) for sets
        if noisy_word not in self.lexicon and noisy_word not in STOPWORDS:
            # expand noisy word
            noisy_words = str_fun.expand_word(noisy_word)

            # 1) IBM_SIMILARITY and SUBSEQUENCE-KERNEL SIMILARITY
            if rank_method in [Generator.IBM_SIM, Generator.SSK_SIM]:
                first_letters = [str_fun.get_first_cons(w) for w in noisy_words]
                lexicon = [self.sub_lexicon[first_letter]
                           for first_letter in first_letters]
                if rank_method == Generator.IBM_SIM:
                    sim_function = str_fun.contractor_sim
                else:
                    sim_function = self.ssk_sim
                candidates = noisy_words
                conf_set = self.rank_candidates(
                    candidates, lexicon, sim_function, off_by_ones)

            # 2) PHONETIC EDIT DISTANCE SIMILARITY
            elif rank_method == Generator.PHONETIC_ED_SIM:
                # candidates=set([self.phon_sim(w)[0] for w in noisy_words])
                # Include both primary and secondary codes!
                candidates = []
                for w in noisy_words:
                    phonetic_w = self.phon_sim(w)
                    if phonetic_w[0]:
                        candidates.append(phonetic_w[0])
                    if phonetic_w[1]:
                        candidates.append(phonetic_w[1])
                # sim_function=self.edit_dist
                lexicon = [self.phonetic_keys for _ in range(len(candidates))]
                sim_function = str_fun.phonetic_ed_sim
                phon_conf_set = self.rank_candidates(
                    candidates, lexicon, sim_function, off_by_ones)
                # print("noisy_words: {}".format(noisy_words))
                # print("phonetic codes: {}".format(candidates))
                # print("phonetic confusion set: {}".format(phon_conf_set))
                # expand phonetic codes into their likely candidate words
                conf_set = []
                for sim, phonetic_code in phon_conf_set:
                    conf_set.extend([(sim, w)
                                     for w in self.phon_lex[phonetic_code]])
                # retain the top 20
                conf_set = conf_set[:10]

            else:
                raise NotImplementedError(
                    "Unknown rank_method supplied as argument.")

            # heuristic: add original word with same prob as lowest prob word
            # in case it's a valid OOV word!
            try:
                weight = conf_set[-1][0]
            except IndexError:
                weight = 1.0        # there's nothing in the confusion set
            if weight == 0:
                weight = 0.2        # add original word with some low probability
            conf_set.append((weight, noisy_word))

            conf_set = [tok for tok in conf_set if tok[0] > 0 and tok[1] != '']
            if len(conf_set) == 0:
                conf_set = [(1.0, '*E*')]
            return conf_set

        else:
            # this is a valid word in the lexicon
            # NOTE: This would miss (i) (special case) accidental homophilous misspellings, such as 'rite' and 'right'
            # also (ii) (the general case) spelling errors where a spelling error leads to a valid word in the lexicon
            # 'it' -> 'is', 'are' -> 'art', 'party' -> 'part', etc.
            # only way to get (ii) is to use a spelling error approach and include all 'off-by-one' errors in the
            # confusion set, e.g. as computed by Norvig's spelling corrector.
            # For (i) need to use phonetic lookup.
            return [(1.0, noisy_word)]

    def get_oov(self, noisy_word):
        if self.check_oov_but_valid(noisy_word) or noisy_word in self.lexicon or noisy_word in STOPWORDS:
            return []
        else:
            return str_fun.expand_word(noisy_word)

    def sent_preprocess(self, sent):
        # collapse punctuation marks occurring > 1 into one
        # rem_dbl_punc = re.compile(r"([{}])\1+".format(pnc))
        # remove punctuation chars repeated > once
        pass

    def fix_bad_tokenisation(self, tokens):
        """Fix some general tokenisation issues"""

        # Split run-on tokens where we have two words separated by a punctuation mark,
        # or one word with leading or trailing punctuation
        punc = [p for p in PUNC if p not in ("'", "@", "#")]
        split_runon_tokens = re.compile("(\w*)([{}]*)(\w*)".format(punc))
        out_tokens = []
        for tok in tokens:
            if self.hash_user_rt_url(tok):
                # keep as is so we don't split '@user'->['@', 'user'], or urls
                # into god knows what
                out_tokens.append(tok)
            else:
                out_tokens.extend([tok for tok in split_runon_tokens.sub(
                    r'\1 \2 \3', tok).split(' ') if tok != ''])
        return out_tokens

    def sent_generate_candidates(self, sent, rank_method, off_by_ones=False, log_oov=False):
        """Generate 'confusion set' from a sentence.
            Return (r,w,c) replacements made (smileys), words after tokenisation and confusion set."""
        # perform some simple pre-processing
        # 1) remove emoticons
        # for bookkeeping, keep track of all replacements
        replacements = []

        # For the evaluation task, do not remove smileys
        """for change in self.smiley_regex.findall(sent):
            replacements.append((change, ''))      # (x,y) for x->y
        # TODO: Handle 'left-handed' emoticons..
        sent = self.smiley_regex.sub('', sent)
        """

        # replace all spurious &lt; and &gt; html artifacts..
        sent = self.lt_gt_regex.sub('', sent)

        # split into words
        # TODO: Get a better tokeniser...
        words = sent.split()
        # fix bad tokenisation issues
        words = self.fix_bad_tokenisation(words)

        if log_oov:
            confusion_set = [self.get_oov(nw.lower()) for nw in words]
        else:
            # get candidates for each word
            confusion_set = [self.word_generate_candidates(nw.lower(), rank_method, off_by_ones)
                             for nw in words]

        return replacements, words, confusion_set


"""
    TODO ideas list:
    1) pre-process and normalise letters occurring more than 2 consecutive times..
    2) Implement a recaser, using the original sentence as guide for recasing clean sentence
    3) Find suitable heuristic to avoid 'normalising' words which are correct e.g.
        neologisms, names, etc. This is tougher and might require training a classifier to
        detect OOV words, as in (Han and Baldwin, 2011).
"""

if __name__ == "__main__":

    testWords = ['t0day', 'today', 't0d4y', 'w0t', 'wh4t']
    testSents = ['test sentence one', 'test s3ntens tw0', 't0day iz awssam']
    test_confusion_set = [[(0.4, "w1"), (0.6, "w2")],
                          [(0.3, "w3"), (0.3, "w4"), (0.4, "w5")]]

    gen = Generator()

    for word in testWords:
        # print(gen.expand_word(word))
        print(gen.word_generate_candidates(word, Generator.PHONETIC_ED_SIM))

    for sent in testSents:
        _, _, c = gen.sent_generate_candidates(sent, Generator.IBM_SIM)
        print("Sentence: {}".format(sent))
        print("Candidate list: {}".format(c))
