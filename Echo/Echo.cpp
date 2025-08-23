#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <stdexcept>
#include <sndfile.h>
#include <string>

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
    Readout(const Reservoir* res) : reservoir(res) {
        // Randomize weights for readout
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        weights.resize(reservoir->size());
        for (auto& w : weights) w = dist(rng);
    }

    double process() const {
        double sum = 0.0;
        const auto& st = reservoir->getState();
        for (size_t i = 0; i < st.size(); i++) {
            sum += weights[i] * st[i];
        }
        return sum;
    }

private:
    const Reservoir* reservoir;
    std::vector<double> weights;
};

// ========== Main ==========
int main() {
    // 1. Load reservoir weight matrix
    auto W = loadMatrix("reservoir.bin");

    // 2. Initialize reservoir
    Reservoir reservoir(W);

    // 3. Two randomized readouts
    Readout readoutL(&reservoir);
    Readout readoutR(&reservoir);

    // 4. Load mono WAV
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open("rickroll.wav", SFM_READ, &sfinfo);
    if (!infile) {
        std::cerr << "Error opening input.wav\n";
        return 1;
    }
    if (sfinfo.channels != 1) {
        std::cerr << "Input must be mono\n";
        return 1;
    }
    std::vector<double> input(sfinfo.frames);
    sf_readf_double(infile, input.data(), sfinfo.frames);
    sf_close(infile);

    // 5. Prepare stereo output
    std::vector<double> output(sfinfo.frames * 2);

    // 6. Process samples
    for (sf_count_t n = 0; n < sfinfo.frames; n++) {
        double sample = input[n];
        reservoir.step(sample);
        double left = readoutL.process();
        double right = readoutR.process();
        output[2 * n] = left;
        output[2 * n + 1] = right;
    }

    // 7. Write stereo WAV
    SF_INFO outinfo = sfinfo;
    outinfo.channels = 2;
    SNDFILE* outfile = sf_open("output.wav", SFM_WRITE, &outinfo);
    if (!outfile) {
        std::cerr << "Error opening output.wav\n";
        return 1;
    }
    sf_writef_double(outfile, output.data(), sfinfo.frames);
    sf_close(outfile);

    std::cout << "Processing complete -> output.wav\n";
    return 0;
}
