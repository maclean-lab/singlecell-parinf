#!/usr/bin/env python
# get cell lists such that some cells are already sampled but some are not
import os.path
from argparse import ArgumentParser
import numpy as np
import pandas as pd

def main():
    args = get_args()
    input_list_path = args.input_list
    input_list = pd.read_csv(input_list_path, index_col=None, sep='\t')

    num_sampled = args.num_sampled
    sampled_cells = input_list.loc[1:num_sampled + 1, 'Cell']
    unsampled_cells = input_list.loc[num_sampled + 1:, 'Cell']
    unsampled_cells = unsampled_cells.sample(num_sampled)
    mixed_cells = pd.concat((sampled_cells, unsampled_cells),
                            ignore_index=True)
    rng = np.random.default_rng()
    rng.shuffle(mixed_cells)

    num_output_lists = args.num_output_lists
    output_list_length = args.output_list_length
    output_list_prefix = os.path.splitext(input_list_path)[0]
    root_cell = args.root_cell

    for i in range(num_output_lists):
        output_list_path = f'{output_list_prefix}_mixed_{i}.txt'
        output_list = pd.Series([root_cell])
        output_list = output_list.append(
            mixed_cells[ i * output_list_length:(i + 1) * output_list_length],
            ignore_index=True)
        output_list.to_csv(output_list_path, sep='\t', index=False,
                           header=['Cell'])

def get_args():
    arg_parser = ArgumentParser(
        description='Get cell lists such that some cells are already sampled '
        'but some are not')
    arg_parser.add_argument('--input_list', type=str)
    arg_parser.add_argument('--root_cell', type=int)
    arg_parser.add_argument('--num_sampled', type=int)
    arg_parser.add_argument('--num_unsampled', type=int)
    arg_parser.add_argument('--num_output_lists', type=int)
    arg_parser.add_argument('--output_list_length', type=int)

    return arg_parser.parse_args()

if __name__ == '__main__':
    main()
