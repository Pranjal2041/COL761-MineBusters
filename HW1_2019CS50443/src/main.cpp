#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "apriori.hpp"
#include "fptree.hpp"
#include "util.hpp"

using namespace std;

// main function with argc and argv
int main(int argc, char *argv[]) {
  if (argc != 5) {
    cout << "Usage: " << argv[0]
         << " <algorithm> <dataset_name> <support> <output_filename>" << endl;
    return 1;
  }
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string algorithm = argv[1];
  string dataset_name = argv[2];
  float support = atof(argv[3]) / 100.0f;
  string output_filename = argv[4];

  if (algorithm == "apriori") {
    cout << "Running apriori algorithm" << endl;
    cout << "Dataset: " << dataset_name << endl;
    cout << "Support: " << support << endl;
    cout << "Output: " << output_filename << endl;
  } else if (algorithm == "fptree") {
    cout << "Running fptree algorithm" << endl;
    cout << "Dataset: " << dataset_name << endl;
    cout << "Support: " << support << endl;
    cout << "Output: " << output_filename << endl;
  } else {
    cout << "Unknown algorithm: " << algorithm << endl;
    return 1;
  }

  vector<vector<string>> output;
  if (algorithm == "apriori") {
    AprioriSolver::solve(dataset_name, support, output_filename);

  } else if (algorithm == "fptree") {
    FPTreeSolver::solve(dataset_name, support, output_filename);
  }
  return 0;
}
