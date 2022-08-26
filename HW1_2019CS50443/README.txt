Github Repo -
https://github.com/Pranjal2041/COL761-MineBusters


Files - 
├── 2019CS50443.sh
├── compile.sh
├── Makefile
├── plot.py
├── README.txt
├── src
│   ├── apriori.cpp
│   ├── apriori.hpp
│   ├── fptree.cpp
│   ├── fptree.hpp
│   ├── main.cpp
│   ├── util.cpp
│   └── util.hpp
└── plot.sh



Team Members -
1. Harsh Agrawal (2019CS10431) [45%]: Worked on the FPTree algorithm. 
2. Pranjal Aggarwal (2019CS50443) [45%]: Worked on the Apriori algorithm and constructed plots and submission scripts. 
3. Akash Suryawanshi (2019CS50416) [10%]: Started with Apriori algorithm.

Explanation - 

1.) From the graph, we observe that FPTree algorithm takes lesser time than apriori for larger support values(>0.5). This is because of the minimum overhead time, in reading the database and specially Constructing the header tables, and conditional trees.
2.) However, for smaller values of support, FPTree is more efficient than Apriori. This is because FPTree requires a fixed(2) number of all database accesses. On the other hand, Apriori requires O(k) accesses to the dataset, and infact proportional to the number of candidates as well. Therefore, as the support decreases, the number of candidates increases, which substantially increase the database accesses required by the Apriori algorithm.
3.) Also, for smaller support values, both FPTree and Apriori algorithm takes a lot of time, and may not run in 1 hour for large datasets. This is because, of lesser ability to prune, and handling large number of candidate sets. However, FPTree keeps becoming more efficient than Apriori as we keep decreasing the support. For example, in 'webdocs.dat' case, FPTree was able to run on support of 10% in less than 40 minutes, while Apriori didn't complete within even a few hours!      
