#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import load_trajectories, pdf_multi_plot

def main():
    # get filter option from command line
    arg_parser = argparse.ArgumentParser(description='Plot all trajectories.')
    arg_parser.add_argument('--filter_type', dest='filter_type', type=str,
                            default=None)
    arg_parser.add_argument('--cell_list', dest='cell_list', type=str,
                            default=None)
    arg_parser.add_argument('--output', dest='output', type=str,
                            default='trajecotries.pdf')
    arg_parser.add_argument('--page_width', dest='page_width', type=float,
                            default=8.5)
    arg_parser.add_argument('--page_height', dest='page_height', type=float,
                            default=11.0)
    arg_parser.add_argument('--num_rows', dest='num_rows', type=int,
                            default=4)
    arg_parser.add_argument('--num_cols', dest='num_cols', type=int,
                            default=2)
    arg_parser.add_argument('--plot_fmt', dest='plot_fmt', type=str,
                            default='')
    args = arg_parser.parse_args()

    # load trajectories
    print('Loading trajectories')
    y, _, ts = load_trajectories(200, filter_type=args.filter_type,
                                downsample_offset=300)

    # reorder cells if given a list of cells
    if args.cell_list:
        cell_list = pd.read_csv(args.cell_list, delimiter='\t', index_col=False)
        y = y[cell_list['Cell'], ]
        cell_names = [f'cell {cell_id}' for cell_id in cell_list['Cell']]
    else:
        cell_names = [f'cell {cell_id}' for cell_id in range(y.shape[0])]

    traj_data = [(ts, y[i, :]) for i in range(y.shape[0])]

    # make the trajectory plot
    plot_kwargs = {}
    if 'o' in args.plot_fmt:
        # when plotting as dicrete circles, do not fill inside the circles
        plot_kwargs['fillstyle'] = 'none'

    pdf_multi_plot(plt.plot, traj_data, args.output, args.plot_fmt,
                   num_rows=args.num_rows, num_cols=args.num_cols,
                   page_size=(args.page_width, args.page_height),
                   titles=cell_names, show_progress=True, **plot_kwargs)
    print('Trajectory plot saved')

if __name__ == '__main__':
    main()
