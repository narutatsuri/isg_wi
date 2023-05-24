"""
Annotates a testing set using the fine-tuned model. 
NOTE: CSV file passed as dataset_dir must contain one column with name "input". 
"""
from finetuned_model.wrapper import finetuned_model
import pandas as pd
from tqdm import tqdm
import argparse


# Construct argument parser
parser = argparse.ArgumentParser()
# Arguments for dataset
parser.add_argument("--dataset_dir", type=str)
parser.add_argument("--save_dir", type=str)
# Arguments for model
parser.add_argument("--backbone_tokenizer", type=str, default="t5-base")
parser.add_argument("--backbone_dir", type=str, default="parrot-phrec")
parser.add_argument("--fact_fixer", type=str, default="dslim/bert-base-NER")
parser.add_argument("--semantic_checker", type=str, default="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
parser.add_argument("--richness_data_path", type=str, default="PIE_dict_regex.txt")
parser.add_argument("--use_wrapper", action="store_true")
args = vars(parser.parse_args())

# Load fine-tuned model
model = finetuned_model(args)
# Load testing dataset
dataset = pd.read_csv(args["dataset_dir"])
# Add columns
dataset["output"] = [None] * len(dataset)
dataset["adequacy"] = [None] * len(dataset)
dataset["richness"] = [None] * len(dataset)
dataset["correctness"] = [None] * len(dataset)

for row_index, row in tqdm(dataset.iterrows(), total=len(dataset)):
    original = row["input"]
    paraphrase, adequacy, richness, correctness = model.infer(original, args["use_wrapper"])

    if paraphrase != -1:
        dataset.iloc[row_index, 1] = paraphrase
        dataset.iloc[row_index, 2] = adequacy
        dataset.iloc[row_index, 3] = richness
        dataset.iloc[row_index, 4] = correctness

dataset.to_csv(args["save_dir"], index=False)