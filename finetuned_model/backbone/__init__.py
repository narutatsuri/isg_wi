from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset


class BackboneModel():
    def __init__(self, 
                 backbone_tokenizer,
                 backbone_dir):
        """
        Loads backbone model.
        ARGUMENTS:
            backbone_tokenizer: path to tokenizer
            backbone_dir: path to fine-tuned backbone model
        """
        # Default values for Backbone model
        self.max_input_length = 256
        self.max_output_length = 256
        self.batch_size = 4
        self.weight_decay = 0.1
        self.learning_rate = 4e-5
        self.epochs = 3
        self.num_beams = 8

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone_dir)

    def tokenizer_fn(self, 
                     batch):
        """
        Tokenize batch of inputs.
        ARGUMENTS:
            batch: batch of string inputs
        """
        inputs = self.tokenizer(batch["input"], padding="max_length", truncation=True, max_length=self.max_input_length)
        outputs = self.tokenizer(batch["output"], padding="max_length", truncation=True, max_length=self.max_output_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask

        return batch

    def train(self,
              data_dir, 
              output_dir):
        """
        Train backbone model on data.
        ARGUMENTS:
            data_dir: path to training data
            output_dir: path to save trained model
        """
        dataset = load_dataset("csv", data_files={'train': data_dir, 'test': data_dir})
        tokenized_datasets = dataset.map(self.tokenizer_fn, batched=True)
        training_args = TrainingArguments(output_dir=output_dir, evaluation_strategy="epoch")

        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=tokenized_datasets['train'],
                          eval_dataset=tokenized_datasets['test'])

        trainer.train()
        
    def infer(self, 
              sentence):
        """
        Generate an annotation for the input sentence.
        ARGUMENTS:
            sentence: input sentence to model
        """
        inputs = self.tokenizer([sentence], 
                                max_length=self.max_input_length, 
                                truncation=True, 
                                return_tensors="pt")        
        output = self.model.generate(**inputs, 
                                     num_beams=self.num_beams, 
                                     do_sample=True, 
                                     max_length=self.max_output_length)
        decoded_output = self.tokenizer.batch_decode(output, 
                                                     skip_special_tokens=True)[0]

        return decoded_output