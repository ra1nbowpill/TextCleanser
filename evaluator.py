import re

from cleanser import TextCleanser
from generator import Generator
import sys
import os

TEXTCLEANSER_ROOT = os.path.split(os.path.realpath(__file__))[0] + os.sep


class Evaluator(object):
    """
    This module computes evaluation statistics on text normalisation performance
    """

    def __init__(self):
        self.cleanser = TextCleanser()
        gen = self.cleanser.generator
        cln = self.cleanser
        self.cleanse_methods = {gen.IBM_SIM: cln.ibm_cleanse,
                                gen.SSK_SIM: cln.ssk_cleanse,
                                gen.PHONETIC_ED_SIM: cln.phonetic_cleanse}
        self.gold_sent_clean = []
        self.gold_word_pairs = []
        self.gold_sent_pairs = []

    def load_gold_standard(self, file, data_accessor=None):
        """ load a parallel noisy-clean gold standard dataset into (noisyword, cleaned_word) tuples"""

        if data_accessor is None:  # default
            # load Bo Han's dataset
            sent_tok_pairs = []

            for line in file:
                # read nr of tokens
                if re.search("^[0-9]+?\n$", line):
                    # this shows how many consecutive tokens are printed next,
                    # we just discard this
                    if len(sent_tok_pairs) > 0:
                        self.gold_sent_pairs.append(sent_tok_pairs)
                        sent_tok_pairs = []
                    continue
                noisy, clean = [w.strip(' \n') for w in line.split('\t')]

                sent_tok_pairs.append((noisy, clean))
                self.gold_word_pairs.append((noisy, clean))

            # self.gold_sent_pairs = [self.gold_sent_pairs[73]]

            print("Loaded {} gold word pairs.".format(len(self.gold_word_pairs)))
            print("Loaded {} gold sentence pairs.".format(len(self.gold_sent_pairs)))

        else:
            # TODO: include ability to handle input-output sentences, segment,
            # align, load as pairs
            raise NotImplementedError("No other accessors implemented.")

    def log_oov_from_gold_pairs(self, oov_file):
        """ Log all OOV tokens in gold standard to log file"""
        for (noisy, clean) in self.gold_word_pairs:
            result = self.gen.get_oov(noisy)
            if not result:
                # write all expanded variants to file
                oov_file.write(','.join(result) + '\n')

    def log_repl(self, repl_log_f, tweet_id, repl):
        """Log all replacements made to disk"""
        if type(repl) == "list":
            for r in repl:
                repl_log_f.write("{}\t{}\t{}\n".format(tweet_id, r[0], r[1]))

    def get_cleanser_output(self, rankmethod, input=None, log_replacements=False, range_=None):
        """Expects input as [[(token 1, gold token 1), ..], ..] sentence pairs"""
        # run the cleanser using the specified $rankmethod over all sentence
        # pairs
        if rankmethod in self.cleanse_methods.keys():
            selectormethod = self.cleanse_methods[rankmethod]
        else:
            selectormethod = self.cleanse_methods[Generator.SSK_SIM]

        if log_replacements:
            repl_log = open('replacements.log', 'w')

        if not input:
            input = self.gold_sent_pairs

        for sent_num, sent_in_tok in enumerate(input):
            if range_ is not None and sent_num not in range_:
                continue

            print("Processing sentence {}".format(sent_num))

            # reconstruct sentence string, assuming it's in the form [(token 1,
            # gold token 1), ..]
            sent_str = ' '.join([tok_pair[0] for tok_pair in sent_in_tok])

            sent_clean, error, replacements = selectormethod(
                sent_str, gen_off_by_ones=False)
            if log_replacements:
                self.log_repl(repl_log, sent_num, replacements)

            if error:
                print("Error: {}".format(error))
                continue
            print("In: {}\nOut:{}".format(sent_str, sent_clean))
            # pack as (original token, cleaned token, gold token)
            in_out_gold = [(s[0], c, s[1]) for s, c in zip(sent_in_tok, sent_clean.split())]
            self.gold_sent_clean.append(in_out_gold)

        print("self.gold_sent_clean has {} items".format(len(self.gold_sent_clean)))

    def clean_word(self, w):
        """
            compare output-gold pairs only based on the alphanumeric characters they contain.
        """
        return re.sub("[^a-z0-9]", "", w)

    def log_oracle_pairs(self, oracle_log_f):
        oracle_log_f.write("====New oracle log\n")
        clean = self.clean_word
        oracle_in_out_gold = [(clean(w[0]), clean(w[1]), clean(
            w[2])) for sent in self.gold_sent_clean for w in sent if len(w) == 3 and w[0] != w[2]]
        for p in oracle_in_out_gold:
            oracle_log_f.write('\t'.join(p) + '\n')

        oracle_log_f.close()

    def compute_wer(self, sent_output_gold_pairs=None):
        """flatten out the [[(word,gold), ...], ..] list, compute global word error rate"""
        if not sent_output_gold_pairs:
            # take cleaned out + gold output from gold standard
            # i.e. run get_cleanser_output first
            print("Taking output,gold pairs.")
            # sent_output_gold_pairs = [(w[1],w[2]) for sent in self.gold_sent_clean for w in sent if len(w)==3]
            # ORACLE EXPERIMENT: Compare only output when we know INPUT is incorrect when input!=gold
            # w == (in, out, gold), therefore check if w[0]==w[2]
            clean = self.clean_word
            sent_output_gold_pairs = [(clean(w[1]), clean(w[2]))
                                      for sent in self.gold_sent_clean
                                      for w in sent if clean(w[0]) != clean(w[2])]

        self.log_oracle_pairs(open('oracle_pairs.log', 'a'))
        # print(sent_output_gold_pairs)
        # print(self.gold_sent_clean)

        num_incorrect = 0
        # this computes N as ONLY the number of ORACLE pairs (known to be different)
        # N=len(sent_output_gold_pairs)
        # this computes N over ALL in-out tokens
        N = sum([len(s) for s in self.gold_sent_clean])
        for out_gold_pair in sent_output_gold_pairs:
            # or out_gold_pair[0]=="*E*": # out_gold_pair == (output, gold)
            if out_gold_pair[0] == out_gold_pair[1]:
                continue
            else:
                num_incorrect += 1
        return float(num_incorrect) / float(N)

    def compute_p_r_f(self, token_norm=None):
        """
            Compute precision, recall and f-score following
            Han et al. "Automatically Constructing a Normalisation Dictionary for Microblogs"
        """

        normalized = 0
        true_positiv = 0
        true_negativ = 0
        false_positiv = 0
        false_negativ = 0
        require_normalization = 0

        for t_i, t_o, t_c in [t for sent in self.gold_sent_clean for t in sent]:
            if token_norm is not None:
                t_i = token_norm(t_i)
                t_c = token_norm(t_c)
                t_o = token_norm(t_o)

            must_be_normalized = t_i != t_o
            if must_be_normalized:
                require_normalization += 1

            was_normalized = t_c != t_i
            if was_normalized:
                normalized += 1

            right_normalization = t_c == t_o

            if not must_be_normalized:
                if was_normalized:
                    false_positiv += 1
                else:
                    true_negativ += 1
            if must_be_normalized:
                if not was_normalized:
                    false_negativ += 1
                elif right_normalization:
                    true_positiv += 1
                elif not right_normalization:
                    false_negativ += 1

        if normalized != 0:
            precision = true_positiv / normalized
            # false_alarm = false_negativ / normalized
        else:
            precision = 0
            # false_alarm = 0

        if require_normalization != 0:
            recall = true_positiv / require_normalization
        else:
            recall = 0

        if precision + recall != 0:
            f_score = (2 * precision * recall) / (precision + recall)
        else:
            f_score = 0

        return precision, recall, f_score

    def compute_bleu(self, sentence_pair):
        """ Compute the bleu score"""
        pass


if __name__ == "__main__":

    gold_file = open(TEXTCLEANSER_ROOT + 'data/han_dataset/corpus.tweet2')

    if len(sys.argv) >= 1:
        if sys.argv[1] == '-':
            gold_file = sys.stdin
        elif os.path.exists(sys.argv[1]):
            gold_file = open(sys.argv[1])

    ev = Evaluator()
    ev.load_gold_standard(gold_file)

    ev.get_cleanser_output(rankmethod=Generator.SSK_SIM)

    prf = ev.compute_p_r_f(token_norm=lambda t: t.lower())
    print("PRF = {}".format(prf))
