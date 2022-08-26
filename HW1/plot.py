import matplotlib.pyplot as plt

# Read the times 
fptimes = []
with open("fptree_times.txt", "r") as f:
    for line in f:
        fptimes.append(float(line[:line.index('m')]) * 60 + float(line[line.index('m')+1: line.index('s')]))

aptimes = []
with open("apriori_times.txt", "r") as f:
    for line in f:
        aptimes.append(float(line[:line.index('m')]) * 60 + float(line[line.index('m')+1: line.index('s')]))

xs = [0.05, 0.1, 0.25, 0.5, 0.9]
plt.plot(xs, fptimes, label="FPTree")
plt.plot(xs, aptimes, label="AP")
plt.xlabel("Support")
plt.ylabel("Time (s)")
plt.legend()
plt.title("FP-Tree vs. AP | Time vs. Support")
plt.savefig('out.png')
