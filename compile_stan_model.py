#!/usr/bin/env python
import os.path
import pickle
import argparse
from pystan import StanModel

def main():
    args = get_args()
    model_path = args.model
    output_path = args.output

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model = StanModel(file=model_path, model_name=model_name)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)
        print(f"Saved compiled Stan model to {output_path}")

def get_args():
    arg_parser = argparse.ArgumentParser(
        description="Compile a Stan model and save as a pickle object.")
    arg_parser.add_argument("--model", type=str, required=True)
    arg_parser.add_argument("--output", type=str, required=True)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
