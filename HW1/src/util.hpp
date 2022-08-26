#pragma once
#include <bits/stdc++.h>
using namespace std;

// Create an iterator for reading dataset file
class FileIterator {
 public:
  FileIterator(string filename);

  // Read next line from file
  void next(vector<string>& transaction);
  void next(unordered_set<string>& transaction);

 private:
  ifstream file;
};

void write_output(vector<vector<string>> data, string filename);
void print_v(vector<string>& v);