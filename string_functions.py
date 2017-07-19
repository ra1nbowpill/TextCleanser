"""
Created on 14 Apr 2011

@author: stephan
"""
import shelve
import itertools
import math

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
TRANSLIT = {
    '0': ['0', 'o'],
    '1': ['1', 'l', 'one'],
    '2': ['2', 'to', 'two'],
    '3': ['3', 'e', 'three'],
    '4': ['4', 'a', 'four', 'for'],
    '5': ['5', 's', 'five'],
    '6': ['6', 'six', 'b'],
    '7': ['7', 't', 'seven'],
    '8': ['8', 'eight', 'ate'],
    '9': ['9', 'nine', 'g'],
    '@': ['@', 'at'],
    '&': ['&', 'and']
}


def get_first_cons(word):
    """Get first consonant or just first letter if no consonant found"""
    for l in word:
        if l in CONSONANTS:
            return l
    return word[0]


def lcs_len1(xs, ys):
    """
    from: http://wordaligned.org/articles/longest-common-subsequence
    Return the length of the LCS of xs and ys.

    Example:
    >> lcs_length("HUMAN", "CHIMPANZEE")
    4
    """
    ny = len(ys)
    curr = list(itertools.repeat(0, 1 + ny))
    for x in xs:
        prev = list(curr)
        for i, y in enumerate(ys):
            if x == y:
                curr[i + 1] = prev[i] + 1
            else:
                curr[i + 1] = max(curr[i], prev[i + 1])
    return curr[ny]


def lcs_len2(str1, str2):
    """
        http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Longest_common_subsequence#Computing_the_length_of_the_LCS
    """
    m = len(str1)
    n = len(str2)

    # An (m+1) times (n+1) matrix
    matrix = [[0] * (n + 1) for _ in range(m + 1)]
    i = 1
    j = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1] + 1
            else:
                matrix[i][j] = max(matrix[i][j - 1], matrix[i - 1][j])
    return matrix[i][j]


def cs(s):
    """
        generates the consonant skeleton of a word, 'shop' -> 'shp'
    """
    return ''.join([l for l in s.lower() if l not in 'aeiou'])


def lcs_ratio(word1, word2):
    """
        Return the LCS / max(len(w1),len(w2))
    """
    return float(lcs_len1(word1, word2)) / max(len(word1), len(word2))


def edit_dist(s1, s2):
    if len(s1) < len(s2):
        return edit_dist(s2, s1)
    if not s1:
        return len(s2)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one
            # character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_edits(word):
    """Norvig's spelling corrector code for generating off-by-one candidates"""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in ALPHABET if b]
    inserts = [a + c + b for a, b in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def contractor_sim(s1, s2):
    """
        Implementation of Contractor et al.'s similarity function
    """
    lcs_val = lcs_ratio(s1, s2)
    sim_val = float(lcs_val / (edit_dist(cs(s1), cs(s2)) + 1))
    return sim_val


def phonetic_ed_sim(s1, s2):
    """Computes the phonetic edit distance SIMILARITY between two strings"""
    return math.exp(-(edit_dist(s1, s2)))


def expand_word(noisy_word):

    def _need_expand(noisy_word):
        for l in noisy_word:
            if l in TRANSLIT:
                return True
        return False

    def _choose(lst, res):
        if not lst:
            return res
        if not res:
            return _choose(lst[1:], lst[0])
        else:
            return _choose(lst[1:], [p + elt for p in res for elt in lst[0]])

    if not _need_expand(noisy_word):
        return [noisy_word]

    conf_net = [TRANSLIT[l] if l in TRANSLIT else [l] for l in noisy_word]
    candidates = _choose(conf_net, [])
    return candidates


class StringFunctions:

    def __init__(self, lamb, p):
        self.ssk_cache = shelve.open('ssk_lamb{}_p{}_cache'.format(round(lamb, 2), p))
        self.lamb = lamb
        self.p = p

    def SSK(self, xi, xj):
        """Return subsequence kernel"""
        cache = self.ssk_cache
        lamb = self.lamb
        p = self.p
        if len(xi) < p:
            p = len(xi)
        if len(xj) < p:
            p = len(xj)

        def SSKernel(xi, xj, lamb, p):
            mykey = str((xi, xj)) if xi > xj else str((xj, xi))
            if mykey not in cache:
                dps = []
                for i in range(len(xi)):
                    dps.append([lamb**2 if xi[i] == xj[j]
                                else 0 for j in range(len(xj))])
                dp = []
                for i in range(len(xi) + 1):
                    dp.append([0] * (len(xj) + 1))
                k = [0] * (p + 1)
                for l in range(2, p + 1):
                    for i in range(len(xi)):
                        for j in range(len(xj)):
                            dp[i + 1][j + 1] = dps[i][j] + lamb * dp[i][j + 1] + lamb * dp[i + 1][j] - lamb**2 * dp[i][j]
                            if xi[i] == xj[j]:
                                dps[i][j] = lamb**2 * dp[i][j]
                                k[l] = k[l] + dps[i][j]
                cache[mykey] = k[p]
            return cache[mykey]
        # return lambda xi, xj: SSKernel(xi,xj,lamb,p)/(SSKernel(xi,xi,lamb,p)
        # * SSKernel(xj,xj,lamb,p))**0.5
        num = SSKernel(xi, xj, lamb, p)
        den = (SSKernel(xi, xi, lamb, p) * SSKernel(xj, xj, lamb, p))**0.5

        if den == 0:
            # print("SSK ERROR: den==0!! xi={}, xj={}".format(xi, xj))
            # special case
            if len(xi) == 1 or len(xj) == 1:
                # print('entering loop1')
                s = xi
                t = xj
                if len(xi) > len(xj):
                    s = xj
                    t = xi
                for i, c in enumerate(t):
                    if c == s[0]:
                        # print('in loop 2')
                        return lamb**i / float(len(t))
            return 0.01  # crude override, return low similarity

        return num / den


if __name__ == "__main__":
    print("String kernel module. Test case output:")
    kern = StringFunctions(lamb=0.9, p=2)
    test_list = [['today', 'today'], ['today', '2day'], ['today', 'tdy'], ['today', 'tomorrow'],
                 ['with', 'wth'], ['today', 'yesterday'], ['with', 'wif']]
    test_list2 = [('today', 'tdy'), ('today', 't'), ('today', 'd')]
    for test_case in test_list2:
        w1 = test_case[0]
        w2 = test_case[1]
        kern_sim = kern.SSK(w1, w2)
        # print(kern_sim)
        print("<{}, {}> = {}".format(w1, w2, round(kern_sim, 4)))
