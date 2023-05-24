import argparse
from finetuned_model.backbone import BackboneModel
import pandas as pd


# Construct argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--dataset_dir", type=str)
parser.add_argument("--model_save_dir", type=str)
args = vars(parser.parse_args())

# Load dataset (Assumes CSV format)
dataset = pd.read_csv(args["dataset_dir"])[["input", "output"]]

model = BackboneModel(args)
model.train(args["dataset_dir"], args["model_save_dir"])