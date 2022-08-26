// #pragma once
#include <bits/stdc++.h>

#include "util.hpp"
#include<iostream> 
#include<vector> // for vector 
#include<algorithm> // for copy() and assign() 
#include<iterator> // for back_inserter 

void solve_apriori(string dataset_name, float support, vector<vector<string>> &output);
vector<vector<string>> generateCandidateSet(vector<vector<string>>& frequent_itemsets);