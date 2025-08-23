#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>

int main() {
    int size = 200;                 // reservoir size
    double targetRadius = 0.95;     // desired spectral radius

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(size, size);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Fill sparse random weights
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            W(i, j) = dist(rng);

    // Compute spectral radius
    Eigen::EigenSolver<Eigen::MatrixXd> solver(W);
    auto eigenvalues = solver.eigenvalues();
    double rho = 0.0;
    for (int i = 0; i < eigenvalues.size(); i++)
        if (std::abs(eigenvalues[i]) > rho) rho = std::abs(eigenvalues[i]);

    std::cout << "Original spectral radius: " << rho << "\n";

    // Rescale to target
    if (rho > 0.0) W *= (targetRadius / rho);

    std::cout << "Rescaled spectral radius: " << targetRadius << "\n";

    // Save binary
    std::ofstream out("reservoir.bin", std::ios::binary);
    int32_t rows = W.rows();
    int32_t cols = W.cols();
    out.write(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    out.write(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    out.write(reinterpret_cast<char*>(W.data()), rows * cols * sizeof(double));
    out.close();

    std::cout << "Binary reservoir saved as reservoir.bin\n";
}
