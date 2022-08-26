#pragma once
#include <bits/stdc++.h>

#include "util.hpp"
using namespace std;

class Apriori {
public:
    float support;
    void solve_apriori(string dataset_name, float support, vector<vector<string>>& output);
    
}