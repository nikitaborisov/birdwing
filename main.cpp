// main.cpp
#include "include/gpu_ntt.h"
#include "include/multiply.h"
#include "include/config.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>

using namespace std;

// this function will read integers from a binary file into two vectors A and B.
// binary file must contain 32-bit limbs for both numbers, back to back
bool read_limb_file(const std::string& filename,
                    std::vector<TestDataTypeUint>& A,
                    std::vector<TestDataTypeUint>& B) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Get file size
    fin.seekg(0, std::ios::end);
    std::streamsize size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    if (size % (sizeof(TestDataTypeUint) * 2) != 0) {
        std::cerr << "File size not divisible by 2 * sizeof(limb)\n";
        return false;
    }

    size_t num_limbs = size / (sizeof(TestDataTypeUint) * 2);
    A.resize(num_limbs);
    B.resize(num_limbs);

    // Read first number
    fin.read(reinterpret_cast<char*>(A.data()), num_limbs * sizeof(TestDataTypeUint));
    // Read second number
    fin.read(reinterpret_cast<char*>(B.data()), num_limbs * sizeof(TestDataTypeUint));

    fin.close();
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " [merge|4step|benchmark] [optional: file input]\n";
        return 1;
    }

    string option = argv[1];
    vector<TestDataTypeUint> A = {1, 2, 3, 4};
    vector<TestDataTypeUint> B = {5, 6, 7, 8};
    vector<TestDataTypeUint> C;

    // if file input is provided, read from file
    if (argc >= 3) {
        string filename = argv[2];
        if (!read_limb_file(filename, A, B)) {
            cerr << "Failed to read input file: " << filename << "\n";
            return 1;
        }
        cout << "[Host] Read " << A.size() << " limbs per number from " << filename << "\n";
        // print the limbs
        // cout << "A: ";
        // for (auto x : A)
        //     cout << x << " ";
        // cout << "\nB: ";
        // for (auto x : B)
        //     cout << x << " ";
        // cout << endl;
    } else {
        // Use default small example if no file
        A.assign({1, 2, 3, 4});
        B.assign({5, 6, 7, 8});
    }

    if (option == "merge") {
        cout << "[Host] Multiplying small polynomials using merge method...\n";
        // host_multiply_merge(A, B, C);
        cout << "Result (merge): ";
        for (auto x : C)
            cout << x << " ";
        cout << endl;
    } else if (option == "4step") {
        cout << "[Host] Multiplying small polynomials using 4-step method...\n";
        // host_multiply_4step(A, B, C);
        cout << "Result (4step): ";
        for (auto x : C)
            cout << x << " ";
        cout << endl;
    } else if (option == "benchmark") {
        cout << "[Benchmark] Comparing merge vs 4step performance...\n";
        vector<TestDataTypeUint> C_merge, C_4step;

        auto start_merge = chrono::high_resolution_clock::now();
        // host_multiply_merge(A, B, C_merge);
        auto end_merge = chrono::high_resolution_clock::now();
        double time_merge = chrono::duration<double, milli>(end_merge - start_merge).count();

        auto start_4step = chrono::high_resolution_clock::now();
        // host_multiply_4step(A, B, C_4step);
        auto end_4step = chrono::high_resolution_clock::now();
        double time_4step = chrono::duration<double, milli>(end_4step - start_4step).count();

        cout << "Merge time:  " << time_merge << " ms\n";
        cout << "4Step time:  " << time_4step << " ms\n";

        cout << "Result (merge): ";
        for (auto x : C_merge)
            cout << x << " ";
        cout << "\nResult (4step): ";
        for (auto x : C_4step)
            cout << x << " ";
        cout << endl;

        if (C_merge == C_4step)
            cout << "[OK] Results match!\n";
        else
            cout << "[WARNING] Results differ!\n";
    } else {
        cerr << "Unknown option: " << option << "\n";
        cerr << "Usage: " << argv[0] << " [merge|4step|benchmark]\n";
        return 1;
    }

    cout << "Done.\n";
    return 0;
}
