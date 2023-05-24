from .ie_checker import IEChecker
from .semantic_checker import SemanticChecker
from .change_checker import ChangeChecker
from .grammar_fixer import GrammarFixer
from .fact_fixer import FactFixer


class PostWrapper():
    def __init__(self,
                 fact_fixer_model,
                 semantic_checker,
                 change_checker_path):
        """
        Loads all wrapper models.
        ARGUMENTS: 
            fact_fixer_model: path to NER model
            semantic_checker: path to NLI model
            change_checker_path: path to idiomatic expressions database
        """
        self.grammar_fixer = GrammarFixer()
        self.fact_fixer = FactFixer(fact_fixer_model)
        self.semantic_checker = SemanticChecker(semantic_checker)
        self.ie_checker = IEChecker(change_checker_path)

    def infer(self, 
              original, 
              paraphrase):
        """
        Applies all wrapper modules and fixes/checks quality of paraphrase.
        ARGUMENTS:
            original: input sentence string
            paraphrase: paraphrased sentence string
        """
        paraphrase = self.grammar_fixer.infer(paraphrase)
        paraphrase = self.fact_fixer.infer(original, paraphrase)

        adequacy = self.semantic_checker.infer(original, paraphrase)
        richness = self.ie_checker.infer(original, paraphrase)
        correctness = ChangeChecker(original, paraphrase)

        if not (adequacy and richness and correctness):
            return -1, adequacy, richness, correctness
        return paraphrase, adequacy, richness, correctness