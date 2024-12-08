from pathlib import Path
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scores", default="Meilenstein3/stylegan3_fid.txt", type=Path)
    return parser.parse_args()

def load_data_old(scores):
    """
    Loads data from a given file containing scores.

    Args:
        scores (str): The path to the file containing the scores.

    Returns:
        tuple: Two lists, x and y, where x contains integer values from the first column 
               and y contains float values from the second column of the file.
    """
    x, y = [], []
    with open(scores, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = [p.strip() for p in line.strip().split()]
            x.append(int(parts[0]))
            y.append(float(parts[1]))
    return x, y

def load_data(scores):
    """
    Load data from a file containing scores.

    Args:
        scores (str): The file path to the scores file. The file should have a header line followed by lines
                      containing an integer and a float separated by whitespace.

    Returns:
        dict: A dictionary where the keys are integers (from the first column of the file) and the values are
              floats (from the second column of the file).
    """
    data = {}
    with open(scores, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = [p.strip() for p in line.strip().split()]
            data[int(parts[0])] = float(parts[1])
    return data

def linear_interpolation(x, x1, y1, x2, y2):
    """
    Perform linear interpolation to find the value of y at a given x.

    Parameters:
    x (float): The x-value at which to interpolate.
    x1 (float): The x-value of the first known point.
    y1 (float): The y-value of the first known point.
    x2 (float): The x-value of the second known point.
    y2 (float): The y-value of the second known point.

    Returns:
    float: The interpolated y-value at the given x.
    """
    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return y

def closest_divisible_by_50(n):
    """
    Calculate the closest lower and higher numbers that are divisible by 50.

    Args:
        n (int): The input number.

    Returns:
        tuple: A tuple containing two integers:
            - The closest lower number divisible by 50.
            - The closest higher number divisible by 50.
    """
    lower = (n // 50) * 50
    higher = ((n // 50) + 1) * 50 
    return lower, higher

def get_marker_y(data, marker_x):
    """
    Calculate the y-values for given x-values using linear interpolation.

    Args:
        data (dict): A dictionary where keys are x-values and values are y-values.
        marker_x (list): A list of x-values for which y-values need to be calculated.

    Returns:
        list: A list of y-values corresponding to the given x-values.
    """
    marker_y = []
    for x in marker_x:
        lower, higher = closest_divisible_by_50(x)
        y = linear_interpolation(x, lower, data[lower], higher, data[higher])
        marker_y.append(y)
    return marker_y

def visualize(data):
    """
    Visualizes the FID score over time for StyleGAN3 training.

    Parameters:
    data (dict): A dictionary where keys are the training steps (int) and values are the corresponding FID scores (float).

    The function creates a line plot of the FID scores over the training steps, adds markers at every 70 steps, and saves the plot as 'stylegan3_fid_over_time.png'.
    """
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    lineplot = sns.lineplot(x=data.keys(), y=data.values())
    line_color = lineplot.lines[0].get_color()
    plt.title('StyleGAN3 FID Score Over Time')
    plt.xlabel('Step')
    plt.ylabel('FID Score')
    plt.ylim(0, None)

    marker_x = range(70, max(data.keys()), 70)
    marker_y = get_marker_y(data, marker_x)
    for x, y in zip(marker_x, marker_y):
        plt.plot(x, y, 'o', markersize=4, color=line_color)

    plt.scatter(marker_x, marker_y, color=line_color, label='Epochs', zorder=5)
    plt.legend()
    plt.savefig("stylegan3_fid_over_time.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.scores)
    x, y = load_data_old(args.scores)
    visualize(data)