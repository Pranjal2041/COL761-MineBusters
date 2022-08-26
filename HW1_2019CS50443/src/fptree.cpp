#include "fptree.hpp"
const string FPTree::ROOT_ITEM = "root";
FPTree::FPTree() {
  root = make_unique<FPNode>();
  root->parent = nullptr;
  root->item = FPTree::ROOT_ITEM;
  root->count = 0;
}

void FPTree::add_transaction(vector<string>& transaction,
                             int transaction_count) {
  auto curr = this->root.get();
  curr->count += transaction_count;
  for (auto& t : transaction) {
    if (curr->children.find(t) == curr->children.end()) {
      curr->children[t] = make_unique<FPNode>();
      curr->children[t]->parent = curr;
      curr->children[t]->item = t;
      curr->children[t]->count = transaction_count;

      if (header_table.find(t) == header_table.end()) {
        header_table[t] = curr->children[t].get();
        tail_table[t] = curr->children[t].get();
      } else {
        tail_table[t]->next = curr->children[t].get();
        tail_table[t] = tail_table[t]->next;
      }
    } else {
      curr->children[t]->count += transaction_count;
    }
    curr = curr->children[t].get();
  }
}

int FPTree::get_count(string item) {
  if (item_counts.find(item) != item_counts.end()) {
    return item_counts[item];
  }
  validate_item(item);

  auto curr = header_table[item];
  int count = 0;
  while (curr != nullptr) {
    count += curr->count;
    curr = curr->next;
  }
  return count;
}

void FPTree::finalize_transactions() {
  this->header_order = vector<string>();
  this->item_counts = unordered_map<string, int>();

  for (auto& kv : header_table) {
    header_order.push_back(kv.first);
  }

  for (auto& hd_item : header_order) {
    auto count = get_count(hd_item);
    item_counts[hd_item] = count;
  }

  sort(header_order.begin(), header_order.end(), [&](string a, string b) {
    if (item_counts[a] != item_counts[b]) {
      return item_counts[a] < item_counts[b];
    }
    return a < b;
  });
  is_finalized = true;
}

void FPTree::validate_item(string item) {
  assert(header_table.find(item) != header_table.end());
}

void FPTree::make_conditional(string item, unique_ptr<FPTree>& tree) {
  this->validate_item(item);

  tree = make_unique<FPTree>();

  auto leaf = this->header_table[item];
  while (leaf != nullptr) {
    vector<string> tx;
    auto branch_tail = leaf->parent;
    while (branch_tail->item != FPTree::ROOT_ITEM) {
      tx.push_back(branch_tail->item);
      branch_tail = branch_tail->parent;
    }
    reverse(tx.begin(), tx.end());
    tree->add_transaction(tx, leaf->count);
    leaf = leaf->next;
  }
  tree->finalize_transactions();
}
void FPTree::make_frequent_itemsets(
    vector<unordered_set<string>>& frqnt_itemset, int support) {
  assert(this->is_finalized);
  for (auto& item : this->header_order) {
    if (this->get_count(item) < support) {
      continue;
    }
    unique_ptr<FPTree> cnd_tree;
    vector<unordered_set<string>> cnd_frq_itemsets;

    cnd_frq_itemsets.push_back(unordered_set<string>({item}));

    this->make_conditional(item, cnd_tree);
    cnd_tree->make_frequent_itemsets(cnd_frq_itemsets, support);
    for (auto& cnd_frq_itemset : cnd_frq_itemsets) {
      cnd_frq_itemset.insert(item);
      frqnt_itemset.push_back(cnd_frq_itemset);
    }
  }
}
vector<unordered_set<string>> FPTreeSolver::frqnt_itemsets;
string FPTreeSolver::output_filename;
vector<vector<string>> FPTreeSolver::output;

void FPTreeSolver::_sighandler(int signum) {
  if (signum == SIGALRM) {
    cout << "timeout" << endl;
    make_output();
    cout << "Output dumped!" << endl;
  }
}

void FPTreeSolver::make_output() {
  for (auto& itemset : FPTreeSolver::frqnt_itemsets) {
    vector<string> itemset_vec(itemset.begin(), itemset.end());
    sort(itemset_vec.begin(), itemset_vec.end());
    FPTreeSolver::output.push_back(itemset_vec);
  }
  sort(FPTreeSolver::output.begin(), FPTreeSolver::output.end(),
       compare_vec_lexico);
  write_output(output, FPTreeSolver::output_filename);
}

void FPTreeSolver::solve(string dataset_name, float support,
                         string output_filename) {
  FPTreeSolver::output_filename = output_filename;
  signal(SIGALRM, FPTreeSolver::_sighandler);
  alarm(TIMEOUT);
  auto dataset = FileIterator(dataset_name);
  unordered_map<string, int> item_count;
  int total_transactions = 0;

  auto start = chrono::high_resolution_clock::now();

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
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::seconds>(end - start).count();
  printf("First pass done. Found %d total transactions\n", total_transactions);
  printf("Time taken: %d seconds\n", duration);

  int num_support = ceil(total_transactions * support);

  auto tree = make_unique<FPTree>();
  dataset = FileIterator(dataset_name);

  transaction.clear();
  start = chrono::high_resolution_clock::now();

  dataset.next(transaction);
  int trans_done = 0;
  while (transaction.size() != 0) {
    vector<string> tx;
    for (auto& t : transaction) {
      if (item_count[t] >= num_support) {
        tx.push_back(t);
      }
    }
    sort(tx.begin(), tx.end(), [&](string a, string b) {
      if (item_count[a] != item_count[b]) {
        return item_count[a] > item_count[b];
      }
      return a > b;
    });
    tree->add_transaction(tx);
    dataset.next(transaction);
  }
  end = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<chrono::seconds>(end - start).count();
  cout << "Second pass done. Tree constructed" << endl;
  printf("Time taken: %d seconds\n", duration);

  tree->finalize_transactions();

  cout << "Constructing frequent itemsets with support " << num_support << endl;
  start = chrono::high_resolution_clock::now();
  tree->make_frequent_itemsets(FPTreeSolver::frqnt_itemsets, num_support);
  end = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<chrono::seconds>(end - start).count();
  cout << "Frequent itemsets constructed in " << duration << " seconds" << endl;

  FPTreeSolver::make_output();
}