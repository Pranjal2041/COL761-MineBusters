# { time ./build/apps/program fptree $1 0.9 temp.dat; } 2>&1 | awk '/real/{print $2}' > fptree_times.txt
# { time ./build/apps/program fptree $1 0.5 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
# { time ./build/apps/program fptree $1 0.25 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt

# { time ./build/apps/program apriori $1 0.9 temp.dat; } 2>&1 | awk '/real/{print $2}' > apriori_times.txt
# { time ./build/apps/program apriori $1 0.5 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt
# { time ./build/apps/program apriori $1 0.25 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt

# { time ./build/apps/program fptree $1 0.1 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
# { time ./build/apps/program fptree $1 0.05 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
# { time ./build/apps/program apriori $1 0.1 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt
# { time ./build/apps/program apriori $1 0.05 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt

python plot.py $2
