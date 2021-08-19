#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import moving_average, pdf_multi_plot

def main():
    # get filter option from command line
    arg_parser = argparse.ArgumentParser(description='Plot all trajectories.')
    arg_parser.add_argument('--filter_type', dest='filter_type', type=str,
                            default=None)
    arg_parser.add_argument('--cell_list', dest='cell_list', type=str,
                            default=None)
    arg_parser.add_argument('--output', dest='output', type=str,
                            default='trajecotries.pdf')
    args = arg_parser.parse_args()

    # load trajectories
    print('Loading trajectories')
    y_raw = np.loadtxt('canorm_tracjectories.csv', delimiter=',')

    # apply filter
    if args.filter_type == 'moving_average':
        y = moving_average(y_raw)
    elif args.filter_type == 'savitzky_golay':
        from scipy.signal import savgol_filter
        y = savgol_filter(y_raw, 51, 2)
    else:
        print('No filter specified or unknown type of filter. Using raw '
              + 'trajectories')

        y = y_raw

    # reorder cells if given a list of cells
    if args.cell_list:
        cell_list = pd.read_csv(args.cell_list, delimiter='\t', index_col=False)
        y = y[cell_list['Cell'], ]
        cell_names = [f'cell {cell_id}' for cell_id in cell_list['Cell']]
    else:
        cell_names = [f'cell {cell_id}' for cell_id in range(y.shape[0])]

    # make the trajectory plot
    pdf_multi_plot(plt.plot, y, args.output, titles=cell_names,
                   show_progress=True)
    print('Trajectory plot saved')

if __name__ == '__main__':
    main()
