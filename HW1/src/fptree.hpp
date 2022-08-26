#pragma once
#include <bits/stdc++.h>

#include "util.hpp"

using namespace std;
struct FPNode {
  FPNode* parent;
  FPNode* next;
  unordered_map<string, unique_ptr<FPNode>> children;
  string item;
  int count;
};

class FPTree {
 public:
  FPTree();

  void add_transaction(vector<string>& transaction, int transaction_count = 1);
  void finalize_transactions();
  void make_conditional(string item, unique_ptr<FPTree>& tree);
  int get_count(string item);
  void make_frequent_itemsets(vector<unordered_set<string>>& frqnt_itemset,
                              int support);
  const static string ROOT_ITEM;

 private:
  unique_ptr<FPNode> root;
  unordered_map<string, FPNode*> header_table;
  unordered_map<string, FPNode*> tail_table;

  // Stores the header entries in the order of their increasing frequency.
  vector<string> header_order;
  unordered_map<string, int> item_counts;
  bool is_finalized = false;

  void validate_item(string item);
};

class FPTreeSolver {
 public:
  static void solve(string dataset_name, float support, string output_filename);

 private:
  static void _sighandler(int signum);
  static void make_output();
  static vector<unordered_set<string>> frqnt_itemsets;
  static string output_filename;
  static vector<vector<string>> output;
};
