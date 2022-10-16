import sys

from amrlib.evaluate.smatch_enhanced import compute_scores
GOLD=sys.argv[1]
PRED=sys.argv[2]
compute_scores(PRED, GOLD)
