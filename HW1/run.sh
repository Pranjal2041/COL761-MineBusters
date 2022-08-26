{ time ./build/apps/program fptree testcases/3/test3.dat 0.05 temp.dat; } 2>&1 | awk '/real/{print $2}' > fptree_times.txt
{ time ./build/apps/program fptree testcases/3/test3.dat 0.1 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
{ time ./build/apps/program fptree testcases/3/test3.dat 0.25 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
{ time ./build/apps/program fptree testcases/3/test3.dat 0.5 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
{ time ./build/apps/program fptree testcases/3/test3.dat 0.9 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt

{ time ./build/apps/program apriori testcases/3/test3.dat 0.05 temp.dat; } 2>&1 | awk '/real/{print $2}' > apriori_times.txt
{ time ./build/apps/program apriori testcases/3/test3.dat 0.1 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt
{ time ./build/apps/program apriori testcases/3/test3.dat 0.25 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt
{ time ./build/apps/program apriori testcases/3/test3.dat 0.5 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt
{ time ./build/apps/program apriori testcases/3/test3.dat 0.9 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt

python plot.py