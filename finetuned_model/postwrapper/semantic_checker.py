from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SemanticChecker():
    def __init__(self,
                 model_path,
                 max_length=256):
        """
        Loads NLI model for inference.
        ARGUMENTS:
            model_path: string to path of NLI model. 
            max_length: max token length to model. Default is 256. 
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def infer(self, 
              original, 
              paraphrase):
        """
        Check if the semantic content is preserved in the paraphrase.
        ARGUMENTS:
            original: input sentence string
            paraphrase: paraphrased sentence string
        """
        tokenized_input_seq_pair = self.tokenizer.encode_plus(original, 
                                                              paraphrase,
                                                              max_length=self.max_length,
                                                              return_token_type_ids=True, 
                                                              truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             labels=None)
        
        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

        return predicted_probability[0] > predicted_probability[1]