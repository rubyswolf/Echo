#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <stdexcept>
#include <sndfile.h>
#include <string>
#include <Eigen/Dense>

// ========== Matrix IO ==========
std::vector<std::vector<double>> loadMatrix(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    int32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));

    std::vector<std::vector<double>> mat(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++)
        file.read(reinterpret_cast<char*>(mat[i].data()), cols * sizeof(double));

    return mat;
}


// ========== Reservoir ==========
class Reservoir {
public:
    Reservoir(const std::vector<std::vector<double>>& W)
        : W(W), state(W.size(), 0.0) {
    }

    void reset() {
        std::fill(state.begin(), state.end(), 0.0);
    }

    void step(double input) {
        // Simple recurrent update: state = ReLU(W*state + input)
        std::vector<double> new_state(state.size(), 0.0);
        for (size_t i = 0; i < W.size(); i++) {
            double sum = input; // feed input directly to each neuron
            for (size_t j = 0; j < W[i].size(); j++) {
                sum += W[i][j] * state[j];
            }
            new_state[i] = (1-alpha) * state[i] + alpha * std::tanh(sum); // Activation
        }
        state = std::move(new_state);
    }

    const std::vector<double>& getState() const {
        return state;
    }

    size_t size() const {
        return state.size();
    }

private:
    std::vector<std::vector<double>> W;
    std::vector<double> state;
	double alpha = 0.1; // Leaky integrator coefficient
 };

// ========== Readout ==========
class Readout {
public:
    // --- Constructor for live/realtime mode ---
    Readout(const Reservoir* res)
        : reservoir(res)
    {
        if (!reservoir) {
            throw std::runtime_error("Reservoir pointer is null in live Readout");
        }

        // Initialize random weights
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        weights.resize(reservoir->size());
        for (auto& w : weights) w = dist(rng);
    }

    // --- Constructor for offline (trained) mode ---
    Readout(std::vector<double> trainedWeights)
        : reservoir(nullptr), weights(std::move(trainedWeights))
    {
        // no reservoir needed in offline processing
    }

    // --- Live processing ---
    double processSample() const {
        if (!reservoir) {
            throw std::runtime_error("processSample called in offline mode");
        }

        double sum = 0.0;
        const auto& st = reservoir->getState();
        for (size_t i = 0; i < st.size(); i++) {
            sum += weights[i] * st[i];
        }
        return sum;
    }

    // --- Offline processing ---
    std::vector<double> processEcho(const Eigen::MatrixXd& Echo) const {
        std::vector<double> output(Echo.rows());
        for (int t = 0; t < Echo.rows(); t++) {
            output[t] = 0.0;
            for (int j = 0; j < Echo.cols(); j++) {
                output[t] += weights[j] * Echo(t, j);
            }
        }
        return output;
    }

private:
    const Reservoir* reservoir;   // only valid in live mode
    std::vector<double> weights;  // trained or random init
};

// ===== WAV loader helper =====
std::vector<double> loadMonoWav(const std::string& filename, SF_INFO& sfinfo) {
    SNDFILE* infile = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!infile) throw std::runtime_error("Error opening " + filename);
    if (sfinfo.channels != 1) throw std::runtime_error("Input must be mono");
    std::vector<double> data(sfinfo.frames);
    sf_readf_double(infile, data.data(), sfinfo.frames);
    sf_close(infile);
    return data;
}

// ===== WAV writer helper =====
void writeMonoWav(const std::string& filename, const std::vector<double>& data, SF_INFO sfinfo) {
    SF_INFO outinfo = sfinfo;
    outinfo.channels = 1;
    SNDFILE* outfile = sf_open(filename.c_str(), SFM_WRITE, &outinfo);
    if (!outfile) throw std::runtime_error("Error opening " + filename);
    sf_writef_double(outfile, data.data(), sfinfo.frames);
    sf_close(outfile);
}

// ========== Main ==========
int main() {
    try {
        std::cout << "Loading reservoir\n";
        auto W = loadMatrix("reservoir0.95.bin");
        Reservoir reservoir(W);

        // ===== Train on input1/output1 =====
        std::cout << "Loading training input/output\n";
        SF_INFO sfinfo1;
        auto input1 = loadMonoWav("input1.wav", sfinfo1);
        auto output1 = loadMonoWav("output1.wav", sfinfo1); // target

        int T1 = sfinfo1.frames;
        int N = reservoir.size();
        Eigen::MatrixXd Echo1(T1, N);
        Eigen::VectorXd Target1(T1);

        std::cout << "Collecting reservoir states\n";
        reservoir.reset();
        for (int t = 0; t < T1; t++) {
            reservoir.step(input1[t]);
            for (size_t j = 0; j < N; j++)
                Echo1(t, j) = reservoir.getState()[j];
            Target1(t) = output1[t];
        }

        std::cout << "Training readout\n";
        double lambda = 1e-6;
        Eigen::MatrixXd XtX = Echo1.transpose() * Echo1;
        Eigen::VectorXd Xty = Echo1.transpose() * Target1;
        Eigen::VectorXd w = (XtX + lambda * Eigen::MatrixXd::Identity(N, N)).ldlt().solve(Xty);

        Readout readout(std::vector<double>(w.data(), w.data() + w.size()));

        // ===== Apply to input2 =====
        std::cout << "Loading new input\n";
        SF_INFO sfinfo2;
        auto input2 = loadMonoWav("input1.wav", sfinfo2);
        std::vector<double> output2(input2.size());

        std::cout << "Processing new input\n";
        reservoir.reset();
        for (size_t t = 0; t < input2.size(); t++) {
            reservoir.step(input2[t]);
            // Live readout
            double y = 0.0;
            const auto& st = reservoir.getState();
            for (size_t j = 0; j < N; j++)
                y += w[j] * st[j];
            output2[t] = y;
        }

        std::cout << "Writing echo_output2.wav\n";
        writeMonoWav("echo_output2.wav", output2, sfinfo2);

        std::cout << "Done!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}