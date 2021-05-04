import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Weather MO experiment.")
parser.add_argument("--name", default="train_plot.pdf", type=str)
args = parser.parse_args()

its = []
fs = []
for line in sys.stdin:
    if line.startswith("it:"):
        its.append(float(line.split(":")[1].strip("F").strip(" ")))
        fs.append(float(line.split(":")[2].strip(" ")))
its = np.array(its)
fs = np.array(fs)

N = 20
ma = np.convolve(fs, np.ones(N) / N, mode="valid")
fig = plt.figure(figsize=(8, 2))
plt.plot(its[: -N + 1], ma, c="red", ls=":")
plt.scatter(its, fs, alpha=0.1)
plt.yscale("log")
# plt.xscale("log")
plt.savefig(args.name)
plt.show()
