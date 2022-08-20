#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>


using namespace std;

// Create an iterator for reading dataset file
class FileIterator {
public:
    FileIterator(string filename) {
        // Open file
        file.open(filename);
        // Check if file is open
        if (!file.is_open()) {
            cout << "Error: File not found" << endl;
            exit(1);
        }
    }

    // Read next line from file
    vector<string> next() {
        string line;
        getline(file, line);
        vector<string> transaction;
        stringstream ss(line);
        string item;
        while (getline(ss, item, ' ')) {
            transaction.push_back(item);
        }
        return transaction;
    }
private:
    ifstream file;
};

void write_output(vector<vector<string>> data, string filename){
    ofstream output_file(filename);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            output_file << data[i][j] << " ";
        }
        output_file << endl;
    }
}

vector<vector<string>> solve_apriori(string dataset_name, float support){
    cout << "Solving using Apriori Algorithm" << endl;
    return vector<vector<string>>();
}

vector<vector<string>> solve_fptree(string dataset_name, float support){
    cout << "Solving using FPTree Algorithm" << endl;
    return vector<vector<string>>();
}

// main function with argc and argv
int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " <algorithm> <dataset_name> <support> <output_filename>" << endl;
        return 1;
    }
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
    }
    else {
        cout << "Unknown algorithm: " << algorithm << endl;
        return 1;
    }    

    // Sample Usage of FileIterator to read the dataset file
    // ------------------------------------------------------------
    FileIterator file_iterator(dataset_name);
    vector<string> line = file_iterator.next();    
    while (line.size() != 0) {
        for (int i = 0; i < line.size(); i++) {
            cout << line[i] << " ";
        }
        cout << endl;
        line = file_iterator.next();
    }
    // ------------------------------------------------------------

    vector<vector<string>> output; 
    if (algorithm == "apriori") {
        output = solve_apriori(dataset_name, support);
    } else if (algorithm == "fptree"){
        output = solve_fptree(dataset_name, support);
    }

    // Write output to output_file
    write_output(output, output_filename);


    return 0;
}
