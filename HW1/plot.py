import matplotlib.pyplot as plt
import sys

# Read the times 
fptimes = []
with open("fptree_times.txt", "r") as f:
    for line in f:
        tt =  0
        if line.find('h')!=-1:
            fptimes.append(float(line[:line.index('h')]) * 3600 + float(line[line.index('h'):line.index('m')]) * 60 + float(line[line.index('m')+1: line.index('s')]))
        else:
            fptimes.append(float(line[:line.index('m')]) * 60 + float(line[line.index('m')+1: line.index('s')]))

aptimes = []
with open("apriori_times.txt", "r") as f:
    for line in f:
        if line.find('h')!=-1:
            aptimes.append(float(line[:line.index('h')]) * 3600 + float(line[line.index('h'):line.index('m')]) * 60 + float(line[line.index('m')+1: line.index('s')]))
        else:
            aptimes.append(float(line[:line.index('m')]) * 60 + float(line[line.index('m')+1: line.index('s')]))

xs = [0.05, 0.1, 0.25, 0.5, 0.9]

aptimes = aptimes[::-1]
fptimes = fptimes[::-1]




plt.plot(xs[len(xs)-len(fptimes):], fptimes, label="FPTree")
plt.plot(xs[len(xs)-len(aptimes):], aptimes, label="AP")
plt.xlabel("Support")
plt.ylabel("Time (s)")
plt.legend()
plt.title("FP-Tree vs. AP | Time vs. Support")
plt.savefig(sys.argv[1])
