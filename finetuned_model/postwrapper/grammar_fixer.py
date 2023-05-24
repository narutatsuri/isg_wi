import language_tool_python


class GrammarFixer():
    def __init__(self):
        """
        Loads Grammar Fixer model by calling LanguageTool's API. 
        """
        self.grammar_model = language_tool_python.LanguageTool("en-US")

    def infer(self, 
              sentence):
        """
        Correct grammatical mistakes in passed sentence.
        ARGUMENTS:
            sentence: input sentence string 
        """
        return self.grammar_model.correct(sentence)