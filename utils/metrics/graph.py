#!/usr/bin/env python3
import matplotlib.pyplot as plt

# GRAPHS
def graph(history, title, xlabel, ylabel, history_name, history_val_name, save_path):
    plt.figure(figsize=(8.0,8.0))# (default: [6.4, 4.8])
    plt.title(title, fontsize='x-large')
    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel(ylabel, fontsize='large')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    # assign different colors
    colors = {'blue': 'b', 'green': 'g', 'red': 'r', 'cyan': 'c',
              'magenta': 'm', 'yellow': 'y', 'black': 'k', 'white': 'w'}
    plt.plot(history[history_name], colors['cyan'])
    plt.plot(history[history_val_name], colors['magenta'])
    plt.legend(['Train', 'Validation'], loc='upper left', fontsize='large')
    plt.savefig(save_path)  # da lasciare prima di show()
    plt.show()
    plt.close()

