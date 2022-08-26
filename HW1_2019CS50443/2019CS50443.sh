if ($1 == "-apriori"){
    ./build/apps/program apriori $2 $3 $4
}elif ($1 == "-fptree"){
    ./build/apps/program fptree $2 $3 $4
}elif ($1 == "-plot"){
    bash plot.sh $2 $3
}