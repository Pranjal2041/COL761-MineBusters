// #pragma once
#include <bits/stdc++.h>

#include <algorithm>  // for copy() and assign()
#include <iostream>
#include <iterator>  // for back_inserter
#include <vector>    // for vector

#include "util.hpp"
class AprioriSolver {
 public:
  static void solve(string dataset_name, float support, string output_filename);

 private:
  static void generateCandidateSet(vector<vector<string>>& frequent_itemsets,
                                   vector<vector<string>>& candidate_set);
  static void _sighandler(int signum);
  static void make_output();
  static string output_filename;
  static vector<vector<string>> output;
};