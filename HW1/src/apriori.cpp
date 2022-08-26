#include "apriori.hpp"

class Apriori {
    float support;
}

unordered_map<vector<string>, int> findNextFrequentSet(unordered_map<vector<string>, int> C, float support) {
  unordered_map<vector<string>, int> F_next;
  for (auto candidate: C) {
    if (candidate.second >= support) {
      F_next[candidate.first] = candidate.second;
    }
  }
  unordered_map<vector<string>, int> F_sorted;
  for (auto f: F_next) {
    vector<string> itemset_vec(f.begin(), f.end());
    sort(itemset_vec.begin(), itemset_vec.end());
    F_sorted[itemset_vec] = f.second;
  }
  return F_sorted;
}

unordered_map<vector<string>, int> findNextCandidateSet(unordered_map<vector<string>, int> Fi, float support, string dataset_name){
  unordered_map<vector<string>, int> C_next;
  auto dataset = FileIterator(dataset_name);
  dataset.next(transaction);
  k = Fi.size()
  // vector<tuple<int, int>>
  for (auto x: Fi) {
    for (auto y: Fi) {
      if (x != y) {
        int i = 0;
        vector<string> tmp;
        while (x[i] == y[i]) {
          tmp.push_back(x[i]);
          i++;
          if (i == k-1) {
            if (x[i-1] < y[i-1]) {
              tmp.push_back(x[i-1])
              tmp.push_back(y[i-1])
            }
            else {
              tmp.push_back(y[i-1])
              tmp.push_back(x[i-1])
            }
            if !isPrunable(tmp) {
              C_next[tmp] = 0
              while (transaction.size() != 0) {
                for (auto& t : transaction) {
                  if checkItemset(t, tmp) {
                    C_next[tmp]++;
                  }
                }
                dataset.next(transaction);
              }
            }
          }
        }
      }
    }
  }
  return C_next;
}



bool isPrunable(vector<string> c, unordered_map<vector<string>, int> F) {
  for (auto xi: c) {
    vector<string> newvec;
    newvec = c;
    remove(newvec.begin(),newvec.end(),xi);
    if !checkSubsetinF(newvec, F) {
      return true;
    }
  }
  return false;
}

bool checkSubsetinF(vector<string> c, unordered_map<vector<string>, int> F) {
  set<string> s(c.begin(),c.end());
  for (auto f: F) {
    set<int> fx(f.first.begin(),f.first.end());
    if (s == fx) {
      return true;
    }
  }
  return false;
}

bool checkItemset(auto t, vector<string> c) {
  allPresent = true;
  for (auto i: c) {
    if (t.find(i) == t.end()) {
        allPresent = false;
        break;
      }
  }
  return allPresent;
}

void solve_apriori(string dataset_name, float support, vector<vector<string>>& output) {
  auto dataset = FileIterator(dataset_name);
  unordered_map<string, int> item_count;
  unordered_map<vector<string>, int> F1;
  int total_transactions = 0;
  vector<string> transaction;
  dataset.next(transaction);
  while (transaction.size() != 0) {
    for (auto& t : transaction) {
      if (item_count.find(t) == item_count.end()) {
        item_count[t] = 1;
      } else {
        item_count[t]++;
      }
    }
    total_transactions++;
    dataset.next(transaction);
    }
  for (auto c: item_count) {
    if (c.second >= support) {
      vector<string> temp;
      temp.push_back(c.first)
      F1[temp] = c.second;
    }
  }
  printf("First pass done. Found %d total transactions\n", total_transactions);
  unordered_map<vector<string>, int> F_prev = F1
  while (true) {
    C_next = findNextCandidateSet(F_prev, support, dataset_name);
    if (!C_next.empty()) {
      F_new = findNextFrequentSet(C_next, support);
      if (F_new.empty()) {
        return F_prev;
      }
    }
    else {
      return F_prev;
    }
    F_prev = F_new;
  }
}