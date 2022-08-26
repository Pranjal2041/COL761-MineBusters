#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "fptree.hpp"
#include "util.hpp"

using namespace std;

// vector<vector<string>> solve_apriori(string dataset_name, float support) {
//   cout << "Solving using Apriori Algorithm" << endl;
//   return vector<vector<string>>();
// }

// vector<vector<string>> solve_fptree(string dataset_name, float support) {
//   cout << "Solving using FPTree Algorithm" << endl;
//   return vector<vector<string>>();
// }

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
  float support = atof(argv[3]);
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

  // Sample Usage of FileIterator to read the dataset file
  // ------------------------------------------------------------
  // FileIterator file_iterator(dataset_name);
  // vector<string> line = file_iterator.next();
  // while (line.size() != 0) {
  //   for (int i = 0; i < line.size(); i++) {
  //     cout << line[i] << " ";
  //   }
  //   cout << endl;
  //   line = file_iterator.next();
  // }
  // ------------------------------------------------------------

  vector<vector<string>> output;
  if (algorithm == "apriori") {
    // solve_apriori(dataset_name, support);
  } else if (algorithm == "fptree") {
    solve_fptree(dataset_name, support, output);
  }

  // Write output to output_file
  write_output(output, output_filename);

  return 0;
}
