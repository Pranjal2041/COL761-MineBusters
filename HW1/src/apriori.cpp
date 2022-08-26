// #pragma once
#include "apriori.hpp"
string AprioriSolver::output_filename;
vector<vector<string>> AprioriSolver::output;
void AprioriSolver::generateCandidateSet(
    vector<vector<string>> &frequent_itemsets, vector<vector<string>> &ck) {
  ck = vector<vector<string>>();
  // for (auto &f1 : frequent_itemsets){
  for (int ii = 0; ii < frequent_itemsets.size(); ii++) {
    auto f1 = frequent_itemsets[ii];
    for (int jj = ii + 1; jj < frequent_itemsets.size(); jj++) {
      auto f2 = frequent_itemsets[jj];
      auto k = f1.size();
      bool flag = true;
      for (int i = 0; i < k - 1; i++) {
        if (f1[i] != f2[i]) {
          flag = false;
          break;
        }
      }
      if (!flag) {
        continue;  // In some situations we can break instead?
      }
      // Assert: First k - 1 elements are same
      if (f1[k - 1] >= f2[k - 1]) {
        continue;  // We can further optimize it. Just don't use advanced for
                   // loop :)
      }
      vector<string> c;
      copy(f1.begin(), f1.end(), back_inserter(c));
      c.push_back(f2[k - 1]);
      ck.push_back(c);
    }
  }
}

void AprioriSolver::_sighandler(int signum) {
  if (signum == SIGALRM) {
    cout << "timeout" << endl;
    AprioriSolver::make_output();
    cout << "Output dumped!" << endl;
  }
}

void AprioriSolver::make_output() {
  sort(AprioriSolver::output.begin(), AprioriSolver::output.end(),
       compare_vec_lexico);
  write_output(AprioriSolver::output, AprioriSolver::output_filename);
}
void AprioriSolver::solve(string dataset_name, float support,
                          string output_filename) {
  AprioriSolver::output_filename = output_filename;
  signal(SIGALRM, AprioriSolver::_sighandler);
  alarm(TIMEOUT);
  auto dataset = FileIterator(dataset_name);
  unordered_map<string, int> item_count;

  vector<vector<string>> frequent_itemsets;
  vector<int> frequent_counts;

  int total_transactions = 0;
  vector<string> transaction;
  dataset.next(transaction);
  while (transaction.size() != 0) {
    for (auto &t : transaction) {
      if (item_count.find(t) == item_count.end()) {
        item_count[t] = 1;
      } else {
        item_count[t]++;
      }
    }
    total_transactions++;
    if (total_transactions % 1000 == 0) {
      cout << total_transactions << endl;
    }
    dataset.next(transaction);
  }
  cout << "First 1 Line of Algo Complete" << endl;

  int num_support = ceil(total_transactions * support);

  for (auto c : item_count) {
    if (c.second >= num_support) {
      vector<string> temp;
      temp.push_back(c.first);
      frequent_itemsets.push_back(temp);
      frequent_counts.push_back(c.second);
    }
  }
  vector<int> indices;
  for (int i = 0; i < frequent_itemsets.size(); i++) {
    indices.push_back(i);
  }

  sort(indices.begin(), indices.end(), [&](int a, int b) {
    return frequent_itemsets[a][0] < frequent_itemsets[b][0];
  });
  vector<vector<string>> f_new;
  vector<int> f_counts;
  for (int idx = 0; idx < indices.size(); idx++) {
    f_new.push_back(frequent_itemsets[indices[idx]]);
    f_counts.push_back(frequent_counts[indices[idx]]);
  }

  cout << "First 2 Lines of Algo Complete" << endl;

  for (int k = 2;; k++) {
    cout << k << endl;
    for (auto &f : f_new) {
      AprioriSolver::output.push_back(f);
    }
    vector<vector<string>> ck;
    AprioriSolver::generateCandidateSet(f_new, ck);
    if (ck.size() == 0) {
      break;
    }

    auto dataset = FileIterator(dataset_name);
    unordered_set<string> transaction;
    dataset.next(transaction);

    vector<int> freq_counts = vector<int>(ck.size(), 0);
    int tt = 0;
    while (transaction.size() != 0) {
      tt++;
      for (int i = 0; i < ck.size(); i++) {
        auto c = ck[i];
        vector<string> outp(c.size());
        bool flag = true;
        for (auto cc : c) {
          if (transaction.find(cc) == transaction.end()) {
            flag = false;
            break;
          }
        }
        if (flag) {
          freq_counts[i]++;
        }
      }
      dataset.next(transaction);
    }

    f_new.clear();
    f_counts.clear();
    for (int i = 0; i < ck.size(); i++) {
      auto f = ck[i];
      auto cnts = freq_counts[i];
      if (cnts >= num_support) {
        f_new.push_back(f);
        f_counts.push_back(cnts);
      }
    }
  }
  AprioriSolver::make_output();
}