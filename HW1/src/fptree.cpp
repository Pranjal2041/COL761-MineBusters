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

void solve_fptree(string dataset_name, float support,
                  vector<vector<string>>& output) {
  auto dataset = FileIterator(dataset_name);
  unordered_map<string, int> item_count;
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

  auto tree = make_unique<FPTree>();
  dataset = FileIterator(dataset_name);

  transaction.clear();
  dataset.next(transaction);
  while (transaction.size() != 0) {
    sort(transaction.begin(), transaction.end(), [&](string a, string b) {
      if (item_count[a] != item_count[b]) {
        return item_count[a] > item_count[b];
      }
      return a > b;
    });
    tree->add_transaction(transaction);
    dataset.next(transaction);
  }

  tree->finalize_transactions();

  vector<unordered_set<string>> frqnt_itemsets;

  int num_support = ceil(total_transactions * support);
  tree->make_frequent_itemsets(frqnt_itemsets, num_support);

  output = vector<vector<string>>();
  auto compare_vec_lexico = [](vector<string>& a, vector<string>& b) {
    auto sz = min(a.size(), b.size());
    for (int i = 0; i < sz; i++) {
      if (a[i] != b[i]) {
        return a[i] < b[i];
      }
    }
    return a.size() < b.size();
  };

  for (auto& itemset : frqnt_itemsets) {
    vector<string> itemset_vec(itemset.begin(), itemset.end());
    sort(itemset_vec.begin(), itemset_vec.end());
    output.push_back(itemset_vec);
  }
  sort(output.begin(), output.end(), compare_vec_lexico);
}