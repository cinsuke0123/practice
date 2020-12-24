#!-*-conding:utf-8-*-
import matplotlib.pyplot as plt
__all__ = ["lineplot", "scatter"]


class lineplot:

    def __init__(self, xlim, ylim, filename):
        self.xlim = xlim
        self.ylim = ylim
        self.filename = filename

    def plot(self, x, y):
        plt.grid()
        plt.plot(x, y)
        plt.xlim(self.xlim), plt.ylim(self.ylim)
        plt.savefig(self.filename)

class scatter:

    def __init__(self, xlim, ylim, filename):
        self.xlim = xlim
        self.ylim = ylim
        self.filename = filename

    def plot(self, x, y):
        plt.grid()
        plt.scatter(x, y)
        plt.xlim(self.xlim), plt.ylim(self.ylim)
        plt.savefig(self.filename)
