#!/usr/bin/env python
# Find two cell chains of same length from a cell list file, such that the
# first n cells are different but all cells after are identical.
import argparse
import pandas as pd

def main():
    # get command-line arguments
    args = parse_args()
    cell_list_path = args.cell_list
    head_length = args.head_length
    tail_length = args.tail_length
    output_prefix = args.output_prefix

    # load cell list
    cell_list = pd.read_csv(cell_list_path, sep="\t", index_col=False)

    # find a cell with multiple children such that we can traverse down from
    # either child with head_length levels. this cell is the first cell in the
    # tail part
    child_counts = cell_list["Parent"].value_counts()
    child_counts = child_counts[child_counts > 1]
    print_blank_line = False  # indicator for printing a blank line
    for cell, _ in child_counts.iteritems():
        # try traveling up from current cell for tail_length levels
        tail = travel_up(cell_list, cell, tail_length)

        # try traveling down from both children for head_length levels
        children = cell_list["Cell"][cell_list["Parent"] == cell]
        heads = []
        for child in children:
            head = travel_down(cell_list, child, head_length)
            if head:
                heads.append(head)

        # concatenate heads and tail and save to file
        if len(heads) > 2 and tail:
            heads = sorted(heads, key=lambda h: len(h), reverse=True)

            # print a blank line if needed
            if print_blank_line:
                print()
            else:
                print_blank_line = True

            print(f"Cell {cell} has multiple branches under it")
            print(f"Longest branch has length {len(heads[0])}")
            print(f"Second longest branch has length {len(heads[1])}")

            save_cell_list(heads[0], tail, f"{output_prefix}_{cell}_1.txt")
            save_cell_list(heads[1], tail, f"{output_prefix}_{cell}_2.txt")

def travel_up(cell_list, cell, n):
    """Find the path between from a given cell and its (n-1)-th ancestor,
    inclusive
    """
    up_list = [cell]
    for _ in range(n - 1):
        parent = cell_list.loc[cell_list["Cell"] == cell, "Parent"].item()
        if parent == -1:  # root node reached
            break

        up_list.append(parent)
        cell = parent

    return up_list

def travel_down(cell_list, cell, n):
    """Find a path between from a given cell and one of its (n-1)-th descendant,
    inclusive
    Implemented as recursive DFS
    """
    down_list = [cell]
    if n == 1:  # base case 1, no need to go down further
        return down_list

    max_level = 1  # maximum levels from DFS
    children = cell_list.loc[cell_list["Parent"] == cell, "Cell"]
    if children.empty:  # base case 2, leaf reached
        return down_list

    for child in children:
        child_list = travel_down(cell_list, child, n - 1)
        if len(child_list) >= max_level:
            down_list = [cell] + child_list
            max_level = len(down_list)

    return down_list

def save_cell_list(head, tail, cell_list_path):
    cells = list(reversed(head)) + tail
    parents = [-1] + cells[:-1]
    cell_list = pd.DataFrame({"Cell": cells, "Parent": parents})

    cell_list.to_csv(cell_list_path, sep="\t", index=False)

def parse_args():
    """Parse command-line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--cell_list", type=str, required=True)
    arg_parser.add_argument("--head_length", type=int, required=True,
                            help="Number of different cells at the beginning")
    arg_parser.add_argument("--tail_length", type=int, required=True,
                            help="Number of identical cells at the end")
    arg_parser.add_argument("--output_prefix", type=str, required=True)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
