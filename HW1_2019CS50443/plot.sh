{ time ./build/apps/program fptree $1 90 temp.dat; } 2>&1 | awk '/real/{print $2}' > fptree_times.txt
{ time ./build/apps/program fptree $1 50 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
{ time ./build/apps/program fptree $1 25 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt

{ time ./build/apps/program apriori $1 90 temp.dat; } 2>&1 | awk '/real/{print $2}' > apriori_times.txt
{ time ./build/apps/program apriori $1 50 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt
{ time ./build/apps/program apriori $1 25 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt

{ time ./build/apps/program fptree $1 10 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
{ time ./build/apps/program fptree $1 5 temp.dat; } 2>&1 | awk '/real/{print $2}' >> fptree_times.txt
{ time ./build/apps/program apriori $1 10 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt
{ time ./build/apps/program apriori $1 5 temp.dat; } 2>&1 | awk '/real/{print $2}' >> apriori_times.txt

python plot.py $2
