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
    arg_parser.add_argument('--first_cell_order', dest='first_cell_order',
                            type=int, default=0)
    arg_parser.add_argument('--last_cell_order', dest='last_cell_order',
                            type=int, default=-1)
    arg_parser.add_argument('--t0', dest='t0', type=int, default=200)
    arg_parser.add_argument('--output', dest='output', type=str,
                            default='trajectories.pdf')
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
    arg_parser.add_argument('--multi_plot', dest='multi_plot',
                            action='store_true')
    args = arg_parser.parse_args()

    # load trajectories
    print('Loading trajectories')
    y, _, ts = load_trajectories(args.t0, filter_type=args.filter_type,
                                 downsample_offset=300)
    ts += args.t0

    # reorder cells if given a list of cells
    if args.cell_list:
        cell_list = pd.read_csv(args.cell_list, delimiter='\t', index_col=False)
        y = y[cell_list['Cell'], ]
        cell_names = [f'cell {cell_id}' for cell_id in cell_list['Cell']]
    else:
        cell_names = [f'cell {cell_id}' for cell_id in range(y.shape[0])]

    if args.last_cell_order == -1:
        y = y[args.first_cell_order:, :]
    else:
        y = y[args.first_cell_order:args.last_cell_order + 1, :]

    # make the trajectory plot
    if args.multi_plot:
        # plot trajectories on separate subplots
        plot_kwargs = {}
        if 'o' in args.plot_fmt:
            # when plotting as discrete circles, do not fill inside the circles
            plot_kwargs['fillstyle'] = 'none'

        traj_data = [(ts, y[i, :]) for i in range(y.shape[0])]
        pdf_multi_plot(plt.plot, traj_data, args.output, args.plot_fmt,
                    num_rows=args.num_rows, num_cols=args.num_cols,
                    page_size=(args.page_width, args.page_height),
                    titles=cell_names, show_progress=True, **plot_kwargs)
    else:
        plt.figure(figsize=(args.page_width, args.page_height))
        plt.plot(ts, y.T, args.plot_fmt, alpha=0.3)
        plt.xlabel('Time (seconds)')
        plt.ylabel(r'Ca${}^{2+}$ response (AU)')
        plt.tight_layout()
        plt.savefig(args.output)
        plt.close()

    print('Trajectory plot saved')

if __name__ == '__main__':
    main()
