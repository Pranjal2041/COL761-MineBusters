#include "util.hpp"
// Create an iterator for reading dataset file

FileIterator::FileIterator(string filename) {
  // Open file
  file.open(filename);
  // Check if file is open
  if (!file.is_open()) {
    cout << "Error: File not found" << endl;
    exit(1);
  }
}

// Read next line from file
void FileIterator::next(vector<string>& transaction) {
  string line;
  getline(file, line);
  transaction = vector<string>();
  stringstream ss(line);
  string item;
  while (getline(ss, item, ' ')) {
    transaction.push_back(item);
  }
}

void FileIterator::next(unordered_set<string>& transaction) {
  string line;
  getline(file, line);
  transaction = unordered_set<string>();
  stringstream ss(line);
  string item;
  while (getline(ss, item, ' ')) {
    transaction.insert(item);
  }
}

void write_output(vector<vector<string>> data, string filename) {
  ofstream output_file(filename);
  for (auto i = 0ul; i < data.size(); i++) {
    for (auto j = 0ul; j < data[i].size(); j++) {
      output_file << data[i][j] << " ";
    }
    output_file << endl;
  }
}

void print_v(vector<string>& v) {
  for (auto i = 0ul; i < v.size(); i++) {
    cout << v[i] << " ";
  }
  cout << endl;
}

bool compare_vec_lexico(vector<string>& a, vector<string>& b) {
  auto sz = min(a.size(), b.size());
  for (int i = 0; i < sz; i++) {
    if (a[i] != b[i]) {
      return a[i] < b[i];
    }
  }
  return a.size() < b.size();
};
