from .backbone import BackboneModel
from .postwrapper import PostWrapper


class finetuned_model():
    def __init__(self, 
                 args):
        """
        Initialize fine-tuned model. 
        ARGUMENTS: 
            args: 
                backbone_tokenizer: path to tokenizer
                backbone_dir: path to fine-tuned backbone model
        """
        self.backbone = BackboneModel(args["backbone_tokenizer"], args["backbone_dir"]) 
        self.postprocessing = PostWrapper(fact_fixer_model=args["fact_fixer"], 
                                          semantic_checker=args["semantic_checker"],
                                          change_checker_path=args["richness_data_path"])

    def infer(self, 
              original, 
              use_wrapper):
        """
        Generates paraphrase suggestion. 
        ARGUMENTS:
            original: input sentence string
            use_wrapper: boolean for whether to use wrapper or not
        """
        paraphrase = self.backbone.infer(original)
        
        if use_wrapper:
            paraphrase, adequacy, richness, correctness = self.postprocessing.infer(original, paraphrase)
            return paraphrase, adequacy, richness, correctness
        else:
            return paraphrase, None, None, None