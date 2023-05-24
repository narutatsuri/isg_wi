import difflib


def ChangeChecker(original, 
                  paraphrase):
    """
    Check if the replacement follows the ISG-WI task criteria.
    ARGUMENTS: 
        original: input sentence string
        paraphrase: paraphrased sentence string
    """
    original = original.split(); paraphrase = paraphrase.split()
    diffs = []
    for i,s in enumerate(difflib.ndiff(original, paraphrase)):
        if s[0]==" ": continue
        else:
            diffs.append(i)

    good = True
    for index, i in enumerate(diffs[:-1]):
        if i + 1 != diffs[index + 1]:
            good = False

    return good