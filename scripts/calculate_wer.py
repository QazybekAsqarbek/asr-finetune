import json
import numpy as np

def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    reference=reference.strip().lower()
    hypothesis=hypothesis.strip().lower()
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer

def word_errors(reference, hypothesis, ignore_case=True, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower().strip().replace('  ',' ').replace('  ',' ')
        hypothesis = hypothesis.lower().strip().strip().replace('  ',' ').replace('  ',' ')

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


# --- SETTINGS ---
# PREDICTIONS_FILE = "askarbek_predictions.txt"
# PREDICTIONS_FILE = "baseline_predictions.txt"
# PREDICTIONS_FILE = "finetune_predictions.txt"
PREDICTIONS_FILE = "finetune_predictions_151225.txt"
MANIFEST_FILE = "f_test_25nov.json"

def main():
    print("📊 Loading data...")
    
    # 1. Load Ground Truth (Reference)
    references = []
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            references.append(data['text'])

    # 2. Load Predictions (Hypothesis)
    hypotheses = []
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            hypotheses.append(line.strip())

    # 3. Validation
    if len(references) != len(hypotheses):
        print(f"❌ Error: Mismatch in line counts!")
        print(f"References (Manifest): {len(references)}")
        print(f"Hypotheses (Predictions): {len(hypotheses)}")
        return

    print(f"🔄 Calculating metrics for {len(references)} samples...")

    wer_scores = []
    total_edits = 0
    total_words = 0
    
    worst_samples = []

    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        # Calculate specific WER for this sentence
        score = wer(ref, hyp, ignore_case=True)
        wer_scores.append(score)
        
        # Calculate edits for Global WER
        edits, length = word_errors(ref, hyp, ignore_case=True)
        total_edits += edits
        total_words += length

        # Keep track of bad results (e.g., WER > 50%) for debugging
        if score > 0.5:
            worst_samples.append((score, ref, hyp))

    # 4. Statistics
    avg_wer = np.mean(wer_scores)
    min_wer = np.min(wer_scores)
    max_wer = np.max(wer_scores)
    
    # Global WER is the industry standard (Sum of Errors / Sum of Words)
    # It handles short/long sentence bias better than simple Average
    global_wer = total_edits / total_words if total_words > 0 else 0

    print("\n" + "="*30)
    print("       RESULTS       ")
    print("="*30)
    print(f"Global WER:  {global_wer:.2%}")
    print(f"Average WER: {avg_wer:.2%}")
    print(f"Min WER:     {min_wer:.2%}")
    print(f"Max WER:     {max_wer:.2%}")
    print("="*30)

    # Optional: Print Top 3 Worst Errors
    worst_samples.sort(key=lambda x: x[0], reverse=True)
    print("\nExample Worst Predictions:")
    for score, r, h in worst_samples[:3]:
        print(f"WER: {score:.2%}")
        print(f"Ref: {r}")
        print(f"Hyp: {h}")
        print("-" * 20)

if __name__ == "__main__":
    main()