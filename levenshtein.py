import os
import numpy as np

import string
import re

dataDir = '/u/cs401/A3/data/'
# dataDir = "/mnt/c/Users/j9108c/BitTorrent Sync/school/UofT/CSC401/a3/test/cs401/data/"

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    O(nm) time and space complexity.

    Parameters
    ----------
    r : reference. list of strings
    h : hypothesis. list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively.

    Examples
    --------
    >>> Levenshtein("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> Levenshtein("who is there".split(), "".split())
    1.0 0 0 3
    >>> Levenshtein("".split(), "who is there".split())
    Inf 0 3 0
    """
    n = len(r)
    m = len(h)
    R = np.empty((n+2, m+2))

    r.insert(0, "<s>")
    r.append("</s>")
    h.insert(0, "<s>")
    h.append("</s>")

    R[0, :] = np.arange(m+2) # first row
    R[:, 0] = np.arange(n+2) # first column

    B = np.empty((n+2, m+2))
    B[0, :] = np.arange(m+2)
    B[:, 0] = np.arange(n+2)

    temp = {}
    for i in range(1, n+1):
        for j in range(1, m+1):
            temp["mat"] = R[i-1, j-1] # match
            temp["sub"] = R[i-1, j-1]+1 # substitution
            temp["ins"] = R[i, j-1]+1 # insertion
            temp["del"] = R[i-1, j]+1 # deletion
            ascending = sorted(temp.items(), key=lambda x: x[1]) # sort dict by values, ascending
            R[i, j] = ascending[0][1] # min value out of the four
            op = ascending[0][0] # operation
            if (op == "mat"):
                B[i, j] = 0
            elif (op == "sub"):
                B[i, j] = 1
            elif (op == "ins"):
                B[i, j] = 2
            elif (op == "del"):
                B[i, j] = 3
                
    wer = R[n, m] / n

    unique, counts = np.unique(B, return_counts=True)
    ops_counts = dict(zip(unique, counts))
    subs = ops_counts[1]
    ins = ops_counts[2]
    dels = ops_counts[3]

    return wer, subs, ins, dels

def preproc(line):
    punc_to_remove = string.punctuation
    punc_to_remove = punc_to_remove.replace("[", "")
    punc_to_remove = punc_to_remove.replace("]", "")

    line = re.sub(r"[{}]".format(punc_to_remove), "", line) # remove all punctuation except "[" and "]"
    line = re.sub(r" +", " ", line) # remove duplicate spaces
    line = line.strip()
    line = line.lower()

    return line

if __name__ == "__main__":
    wers_google = []
    wers_kaldi = []

    for subdir,dirs,files in os.walk(dataDir):
        # print(subdir)
        # print(dirs)
        # print(files)
        # print("")
        if (subdir.split("/")[-1].startswith("S")): # speaker dir
            speaker = subdir.split("/")[-1]
            
            r = open(f"{subdir}/transcripts.txt").read().splitlines()
            for index,line in enumerate(r):
                r[index] = preproc(line)
            num_lines_r = sum([1 for line in r])

            h_google = open(f"{subdir}/transcripts.Google.txt").read().splitlines()
            for index,line in enumerate(h_google):
                h_google[index] = preproc(line)
            # num_lines_h_google = sum([1 for line in h_google])

            h_kaldi = open(f"{subdir}/transcripts.Kaldi.txt").read().splitlines()
            for index,line in enumerate(h_kaldi):
                h_kaldi[index] = preproc(line)
            # num_lines_h_kaldi = sum([1 for line in h_kaldi])

            if (num_lines_r > 0): # https://piazza.com/class/kjixofiil3j2q5?cid=719
                for i in range(num_lines_r):
                    wer, subs, ins, dels = Levenshtein(r[i].split(" "), h_google[i].split(" "))
                    wers_google.append(wer)
                    print(f"{speaker} Google {i} {wer} S:{subs}, I:{ins}, D:{dels}")

                    wer, subs, ins, dels = Levenshtein(r[i].split(" "), h_kaldi[i].split(" "))
                    wers_kaldi.append(wer)
                    print(f"{speaker} Kaldi {i} {wer} S:{subs}, I:{ins}, D:{dels}")

                print("")

    print(f"google WER avg: {np.mean(wers_google)}")
    print(f"kaldi WER avg: {np.mean(wers_kaldi)}")
    print(f"google WER sd: {np.std(wers_google)}")
    print(f"kaldi WER sd: {np.std(wers_kaldi)}")
