Files bundled:

q1.py -> Format inter-onversion and plotting script for various graph mining algorithms
q2.py -> Index creation and querying script for graph databases.
q3.py -> Elbow plot generation script for K-Means clustering. 
CS1190431.pdf -> report
*.sh -> scripts as mentioned in the assignment specifications

Contributors:

1. Harsh Agrawal (2019CS10431) [33%]
2. Pranjal Aggarwal (2019CS50443) [33%]
3. Akash Suryawanshi (2019CS50416) [33%]

Usage:

Q1 ->
 `./time.sh <path to graph database file> <path to output plot>`

This will load the database of graphs convert them to appropriate format and and benchmark the running time of the various algorithms. Finally the plot will be generated in the path specified as the second argument.
Q2 -> 
`sh index.sh <graph dataset>`
`sh query.sh`

Specifications as mentioned in the assignment doc.

Q3 ->
 `./elbow_plot.sh <dataset> <dimension> q3_<dimension>_<RollNo>.png`
Works as specified in the assignment doc.