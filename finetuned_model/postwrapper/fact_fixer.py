from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification


class FactFixer():
    def __init__(self,
                 model_path):
        """
        Loads NER model at model_path. 
        ARGUMENTS:
            model_path: string to path of NER model. 
        """
        self.model = pipeline("ner", 
                              model=AutoModelForTokenClassification.from_pretrained(model_path), 
                              tokenizer=AutoTokenizer.from_pretrained(model_path))

    def infer(self, 
              original, 
              paraphrase):
        """
        Given input and paraphrased sentence, check if proper nouns are incorrectly modified 
        and correct any erroneous misplacements.
        ARGUMENTS:
            original: input sentence string
            paraphrase: paraphrased sentence string
        """
        original_ne = [item["word"] for item in self.model(original)]
        paraphrase_ne = [item["word"] for item in self.model(paraphrase)]

        for index, ne in enumerate(original_ne):
            if paraphrase_ne[index].lower() != ne.lower():
                paraphrase = paraphrase.replace(paraphrase[index], ne)
        
        return paraphrase