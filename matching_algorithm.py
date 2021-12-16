import numpy as np
from itertools import groupby

def accuracy_fn(ref, pred, index=None):
  if len(ref) == 0 and len(pred) != 0:
    return 0
  if len(ref) == 0 and len(pred) == 0:
    return 1.0
  if index:
    ref = ref[: min(len(ref), index)]
    pred = pred[: min(len(pred), index)]
  if len(ref) > len(pred):
    pred = pred + [-1] * (len(ref) - len(pred))
  if len(pred) > len(ref):
    pred = pred[: len(pred)]
  return sum(1 for x,y in zip(ref, pred) if x == y) / len(ref)   

# Needleman Wunsch algorithm inspired by: https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5
def nw(x, y, match = 1, mismatch = 1, gap = 1):
    nx = len(x)
    ny = len(y)

    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx, nx + 1)
    F[0,:] = np.linspace(0, -ny, ny + 1)

    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4

    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i][0] == y[j][0]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1
    # Reverse the strings.
    # rx = ''.join(rx)[::-1]
    # ry = ''.join(ry)[::-1]
    # print(rx)
    # print(ry)
    # return rx, ry
    return rx[::-1], ry[::-1]

def phoneme_match(ref, pred, words, wp_mapping):
  indexified_ref = [(elt, idx) for idx, elt in enumerate(ref)]
  indexified_pred = [(elt, idx) for idx, elt in enumerate(pred)]
  
  rx, ry = nw(indexified_ref, indexified_pred)

  mispronounced_word_idxs = []
  prev_idx = 0

  # print(rx)
  # print(ry)
  for i in range(len(rx)):
    if rx[i] == '-':
      mispronounced_word_idxs.append(wp_mapping[prev_idx])
      continue
    if rx[i][0] != ry[i][0]:
      mispronounced_word_idxs.append(wp_mapping[rx[i][1]])
    prev_idx = rx[i][1]
  
  mispronounced_word_idxs = [word_idx[0] for word_idx in groupby(mispronounced_word_idxs)]
  return mispronounced_word_idxs

def evaluate_test_case(ref, pred, words, wp_mapping, correct_mispronounced_idxs):
  predicted_misprounced_idxs = phoneme_match(ref, pred, words, wp_mapping)
  acc =  accuracy_fn(correct_mispronounced_idxs, predicted_misprounced_idxs)
  if acc != 1.0:
    print(f'CORRECT: {correct_mispronounced_idxs}\nPREDICT: {predicted_misprounced_idxs}')
  return acc


# x = ["G","A","T","A","C", "A"]
# y = ["G","A","T","G","C","U", "Z"]
# words = ["GOAT", "ACAI"]
# true_mappings = [0, 0, 0, 1, 1, 1]
# mispredict_idxs = [1]


# print(evaluate_test_case(x, y, words, true_mappings, mispredict_idxs))

# TEST CASE 1 - PERFECT MATCH
words = ['we', "don't", 'know', 'this', 'guy'] 
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = []
pred_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 1: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 2 - PERFECT MISMATCH
words = ['we', "don't", 'know', 'this', 'guy'] 
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [0, 1, 2, 3, 4]
pred_phonemes = ['a', 'iz', 'dl', 'h', 'w', 'na', 'owe', 'dha', 'ihe', 'sa', 'gdcl', 'ag', 'agy']
print('TEST CASE 2: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 3 - DIFF LENGTH MISMATCH
words = ['we', "don't", 'know', 'this', 'guy'] 
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh',]
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [0, 1, 2, 3]
pred_phonemes = ['a', 'iz', 'dl', 'h', 'w', 'na', 'owe', 'dha', 'ihe', 'sa', 'gdcl', 'ag', 'agy']
print('TEST CASE 3: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))


# TEST CASE 4 - WRONG PHONEME IN A WORD
words = ['we', "don't", 'know', 'this', 'guy'] 
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [3]
pred_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'th', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 4: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))


# TEST CASE 5 - SWAPPED WORDS
words = ['we', "don't", 'know', 'this', 'guy'] 
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [1, 2]
pred_phonemes = ['w', 'iy', 'dcl', 'n', 'ow', 'd', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 5: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 6 - SKIPPED WORD
words = ['we', "don't", 'know', 'this', 'guy'] 
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [1]
pred_phonemes = ['w', 'iy', 'dcl', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 6: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 7 - SKIPPED 2 WORDS
words = ['we', "don't", 'know', 'this', 'guy'] 
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [1, 2]
pred_phonemes = ['w', 'iy', 'dcl', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 7: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 8 - SKIPPED WORD BUT NEXT WORD STARTS WITH SAME PHONEME
words = ['we', "don't", 'know', 'this', 'guy'] 
# replaced 'n' with 'dcl' for this test case to avoid needing a new utterance
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'dcl', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay'] 
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [1]
pred_phonemes = ['w', 'iy', 'dcl', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 8: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 9 - REPEATED WORD BUT MISPRONOUNCED THE FIRST TIME
words = ['we', "know", 'know', 'this', 'guy'] 
# replaced 'n' with 'dcl' for this test case to avoid needing a new utterance
ref_phonemes = ['w', 'iy', 'n', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [1]
ref_phonemes = ['w', 'iy', 'kn', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 9: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 10 - EXTRA SOUNDS WITHIN WORD
words = ['we', "know", 'know', 'this', 'guy'] 
# replaced 'n' with 'dcl' for this test case to avoid needing a new utterance
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [1]
pred_phonemes = ['w', 'iy', 'dcl', 'a', 'a', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 10: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

# TEST CASE 11 - EXTRA SOUNDS BETWEEN WORDS
words = ['we', "know", 'know', 'this', 'guy'] 
# replaced 'n' with 'dcl' for this test case to avoid needing a new utterance
ref_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
mapping = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 
mispredict_idxs = [1]
pred_phonemes = ['w', 'iy', 'dcl', 'd', 'ow', 'a', 'n', 'ow', 'dh', 'ih', 's', 'gcl', 'g', 'ay']
print('TEST CASE 11: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))


# TEST CASE 12 - LONG SENTENCE CORRECT
words = ['such', 'legislation', 'was', 'clarified', 'and', 'extended', 'from', 'time', 'to', 'time', 'thereafter']
# replaced 'n' with 'dcl' for this test case to avoid needing a new utterance
ref_phonemes = ['s', 'ah', 'sh', 'l', 'eh', 'dcl', 'jh', 'ax', 's', 'l', 'ey', 'sh', 'en', 'w', 'ax', 'z', 'kcl', 'k', 'l', 'eh', 'axr', 'f', 'ay', 'dcl', 'en', 'ix', 'kcl', 'k', 's', 'tcl', 't', 'eh', 'n', 'd', 'ix', 'dcl', 'f', 'em', 'tcl', 't', 'ay', 'm', 'tcl', 't', 'ax', 'tcl', 't', 'ay', 'm', 'dh', 'eh', 'r', 'ae', 'f', 'tcl', 't', 'axr']
mapping = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10]
mispredict_idxs = []
pred_phonemes = ['s', 'ah', 'sh', 'l', 'eh', 'dcl', 'jh', 'ax', 's', 'l', 'ey', 'sh', 'en', 'w', 'ax', 'z', 'kcl', 'k', 'l', 'eh', 'axr', 'f', 'ay', 'dcl', 'en', 'ix', 'kcl', 'k', 's', 'tcl', 't', 'eh', 'n', 'd', 'ix', 'dcl', 'f', 'em', 'tcl', 't', 'ay', 'm', 'tcl', 't', 'ax', 'tcl', 't', 'ay', 'm', 'dh', 'eh', 'r', 'ae', 'f', 'tcl', 't', 'axr']
print('TEST CASE 12: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))


# TEST CASE 13 - MULTIPLE MISTAKES
words = ['such', 'legislation', 'was', 'clarified', 'and', 'extended', 'from', 'time', 'to', 'time', 'thereafter']
# replaced 'n' with 'dcl' for this test case to avoid needing a new utterance
ref_phonemes = ['s', 'ah', 'sh', 'l', 'eh', 'dcl', 'jh', 'ax', 's', 'l', 'ey', 'sh', 'en', 'w', 'ax', 'z', 'kcl', 'k', 'l', 'eh', 'axr', 'f', 'ay', 'dcl', 'en', 'ix', 'kcl', 'k', 's', 'tcl', 't', 'eh', 'n', 'd', 'ix', 'dcl', 'f', 'em', 'tcl', 't', 'ay', 'm', 'tcl', 't', 'ax', 'tcl', 't', 'ay', 'm', 'dh', 'eh', 'r', 'ae', 'f', 'tcl', 't', 'axr']
mapping =      [0,     0,    0,   1,   1,      1,    1,    1,   1,   1,    1,    1,    1,   2,    2,   2,    3,    3,   3,    3,    3,    3,    3,    3,     4,     5,   5,    5,    5,    5,   5,    5,   5,    5,   5,    5,    6,    6,    7,    7,    7,   7,    8,    8,    8,    9,    9,    9,   9,   10,   10,   10,  10,   10,  10,    10,   10]
mispredict_idxs = [1, 10]
pred_phonemes = ['s', 'ah', 'sh', 'l', 'eh', 'ax', 's', 'l', 'ey', 'sh', 'en', 'w', 'ax', 'z', 'kcl', 'k', 'l', 'eh', 'axr', 'f', 'ay', 'dcl', 'en', 'ix', 'kcl', 'k', 's', 'tcl', 't', 'eh', 'n', 'd', 'ix', 'dcl', 'f', 'em', 'tcl', 't', 'ay', 'm', 'tcl', 't', 'ax', 'tcl', 't', 'ay', 'm', 'dh', 'eh', 'r', 'ae',  'eh', 'r', 'ae', 'tcl', 't', 'axr']
print('TEST CASE 13: ', evaluate_test_case(ref_phonemes, pred_phonemes, words, mapping, mispredict_idxs))

