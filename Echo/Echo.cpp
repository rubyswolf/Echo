#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <stdexcept>
#include <string>
#include <deque>
#include <algorithm>
#include <cmath>
#include <conio.h>
#include <sndfile.h>
#include <Eigen/Dense>

// ================= Matrix IO =================
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

// ================= Reservoir =================
class Reservoir {
public:
    Reservoir(const std::vector<std::vector<double>>& W)
        : W(W), state(W.size(), 0.0) {
    }

    void reset() { std::fill(state.begin(), state.end(), 0.0); }

    void step(double input) {
        std::vector<double> new_state(state.size(), 0.0);
        for (size_t i = 0; i < W.size(); i++) {
            double sum = input; // input injected identically
            for (size_t j = 0; j < W[i].size(); j++) sum += W[i][j] * state[j];
            new_state[i] = (1 - alpha) * state[i] + alpha * std::tanh(sum);
        }
        state = std::move(new_state);
    }

    const std::vector<double>& getState() const { return state; }
    size_t size() const { return state.size(); }

private:
    std::vector<std::vector<double>> W;
    std::vector<double> state;
    double alpha = 0.8; // leaky coefficient
};

// ========= Helpers for WAV =========
std::vector<double> loadMonoWav(const std::string& filename, SF_INFO& sfinfo) {
    SNDFILE* infile = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!infile) throw std::runtime_error("Error opening " + filename);
    if (sfinfo.channels != 1) throw std::runtime_error("Input must be mono");
    std::vector<double> data(sfinfo.frames);
    sf_readf_double(infile, data.data(), sfinfo.frames);
    sf_close(infile);
    return data;
}

void writeMonoWav(const std::string& filename, const std::vector<double>& data, SF_INFO sfinfo) {
    SF_INFO outinfo = sfinfo;
    outinfo.channels = 1;
    outinfo.frames = (sf_count_t)data.size();   // set frames correctly
    SNDFILE* outfile = sf_open(filename.c_str(), SFM_WRITE, &outinfo);
    if (!outfile) throw std::runtime_error("Error opening " + filename);
    sf_count_t written = sf_writef_double(outfile, data.data(), (sf_count_t)data.size());
    if (written != (sf_count_t)data.size()) {
        std::cerr << "Warning: wrote " << written << " frames, expected " << data.size() << "\n";
    }
    sf_close(outfile);
}

// ================= Delayed Readout (Method 2 / Block-Coord Training) =================
// We learn a continuous nonnegative delay d_j per feature j, with linear interpolation.
// Training alternates: (a) closed-form ridge for w given d; (b) gradient step on d given w.
class DelayedReadout {
public:
    DelayedReadout(size_t features, int Dmax, double lambda = 1e-6, double gamma = 0.0, double lr = 1e-3, int ls = 1)
        : N(features), Dmax(Dmax), lambda(lambda), gamma(gamma), lr(lr), ls(ls) {
        w.assign(N, 0.0);
        d.assign(N, 0.0);  // init at zero delay
    }

    // Train using Echo (T x N) and target (T)
    void train(const Eigen::MatrixXd& Echo, const Eigen::VectorXd& Target,
        int iters = 200, bool verbose = true) {
        const int T = (int)Echo.rows();
        Eigen::MatrixXd Xs(T, (int)N); // shifted design
        Eigen::MatrixXd Delta(T, (int)N); // x(t-k-1) - x(t-k) per (t,j)

        bool stopRequested = false;
        for (int it = 0; it < iters; ++it) {

            // ---- (1) Multiple gradient descent steps for delays before ridge ----
            for (int step = 0; step < ls; ++step) {
                // Build shifted Echo and Delta
                buildShifted(Echo, d, Xs, &Delta);

                // Residuals using current weights
                Eigen::VectorXd w_vec((int)N);
                for (size_t j = 0; j < N; ++j) w_vec[(int)j] = w[j];
                Eigen::VectorXd yhat = Xs * w_vec;
                // 1) Basic mse & max-abs
                double mse_cpp = (yhat - Target).squaredNorm() / (double)T;
                double maxabs = (yhat - Target).cwiseAbs().maxCoeff();

                // 2) normalized MSE (relative to target variance)
                double targ_mean = Target.mean();
                double var = (Target.array() - targ_mean).square().sum() / std::max(1, T - 1);
                double nmse = var > 0 ? mse_cpp / var : mse_cpp;

                // 3) cross-correlation lag search (small window)
                int maxLagCheck = std::min((int)Dmax * 2, T / 2); // search range
                int bestLag = 0; double bestScore = -1e308;
                for (int lag = -maxLagCheck; lag <= maxLagCheck; ++lag) {
                    int tstart = std::max(0, -lag);
                    int tend = std::min(T, T - lag);
                    double acc = 0.0;
                    for (int t = tstart; t < tend; ++t) acc += yhat[t] * Target[t + lag];
                    if (acc > bestScore) { bestScore = acc; bestLag = lag; }
                }

                // 4) weighted mean delay (abs weights) & top-k
                double wnum = 0, wden = 0;
                for (size_t j = 0;j < N;++j) { wnum += std::abs(w[j]) * d[j]; wden += std::abs(w[j]); }
                double wmean = (wden > 0) ? (wnum / wden) : 0.0;
                std::vector<std::pair<double, int>> ord;
                ord.reserve(N);
                for (size_t j = 0;j < N;++j) ord.emplace_back(std::abs(w[j]), (int)j);
                std::sort(ord.begin(), ord.end(), std::greater<>());
                int K = std::min(10, (int)ord.size());
                std::cout << "Iter " << it << ": mse=" << mse_cpp << " nmse=" << nmse
                    << " maxabs=" << maxabs << " bestLag=" << bestLag
                    << " mean_d=" << meanDelay() << " wmean_d=" << wmean << "\n";
                std::cout << " Top " << K << ": ";
                for (int i = 0;i < K;++i) {
                    int j = ord[i].second;
                    std::cout << "(" << ord[i].first << "," << d[j] << ") ";
                }
                std::cout << "\n";
                Eigen::VectorXd e = yhat - Target;

                // Gradient for delays
                std::vector<double> grad_d(N, 0.0);
                for (size_t j = 0; j < N; ++j) {
                    double acc = 0.0;
                    for (int t = 0; t < T; ++t) acc += e[t] * Delta(t, (int)j);
                    grad_d[j] = 2.0 * w[j] * acc + gamma * d[j];
                }

                // Update and clamp delays
                for (size_t j = 0; j < N; ++j) {
                    d[j] -= lr * grad_d[j];
                    if (d[j] < 0.0) d[j] = 0.0;
                    if (d[j] > (double)Dmax) d[j] = (double)Dmax;
                }
            }

            // ---- (2) Closed-form ridge for weights ----
            buildShifted(Echo, d, Xs, nullptr); // rebuild after final delay step
            Eigen::MatrixXd XtX = Xs.transpose() * Xs;
            Eigen::VectorXd Xty = Xs.transpose() * Target;
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity((int)N, (int)N);
            Eigen::VectorXd w_vec = (XtX + lambda * I).ldlt().solve(Xty);
            for (size_t j = 0; j < N; ++j) w[j] = w_vec[(int)j];

            // ---- (3) Verbose output ----
            if (verbose) {
                Eigen::VectorXd yhat = Xs * w_vec;
                Eigen::VectorXd e = yhat - Target;
                double mse = e.squaredNorm() / (double)T;
                std::cout << "Iter " << it << ": MSE=" << mse
                    << ", mean d=" << meanDelay() << "\n";
            }

            // ---- (4) Stop request (ESC key) ----
            if (_kbhit()) {
                int ch = _getch();
                if (ch == 27) { // ESC key
                    std::cout << "\nEscape pressed -> stopping after iteration " << it << "\n";
                    stopRequested = true;
                    break;
                }
            }
        }

        std::cout << "Training finished (iters=" << (stopRequested ? "early" : "all") << ")\n";

        // Cache k and a for fast inference
        computeKA();
    }


    // Offline apply to a pre-collected Echo (same Echo used to build live reservoir states)
    std::vector<double> processEcho(const Eigen::MatrixXd& Echo) const {
        Eigen::MatrixXd Xs;
        buildShifted(Echo, d, Xs, nullptr);
        Eigen::VectorXd w_vec((int)N);
        for (size_t j = 0; j < N; ++j) w_vec[(int)j] = w[j];
        Eigen::VectorXd yhat = Xs * w_vec;
        return std::vector<double>(yhat.data(), yhat.data() + yhat.size());
    }

    // Live streaming apply: call step(state) per sample, returns y
    double processStateLive(const std::vector<double>& state) {
        // push state to buffer
        if ((int)ring.size() < Dmax + 2) {
            ring.push_back(state);
        }
        else {
            ring.pop_front();
            ring.push_back(state);
        }
        // compute y = sum_j w_j * ((1-a_j) x[t-k_j, j] + a_j x[t-k_j-1, j])
        double y = 0.0;
        const int T = (int)ring.size();
        for (size_t j = 0; j < N; ++j) {
            int k = k_cached[j];
            double a = a_cached[j];
            int idx0 = std::max(0, T - 1 - k);
            int idx1 = std::max(0, idx0 - 1);
            double x0 = ring[idx0][j];
            double x1 = ring[idx1][j];
            double xs = (1.0 - a) * x0 + a * x1;
            y += w[j] * xs;
        }
        return y;
    }

    const std::vector<double>& weights() const { return w; }
    const std::vector<double>& delays() const { return d; }
    int maxDelay() const { return Dmax; }

private:
    size_t N;
    int Dmax, ls;
    double lambda, gamma, lr;
    std::vector<double> w;   // readout weights
    std::vector<double> d;   // per-feature delay (samples)

    // cache for live
    std::vector<int> k_cached;   // floor(d)
    std::vector<double> a_cached; // frac(d)
    std::deque<std::vector<double>> ring; // recent reservoir states

    double meanDelay() const {
        double s = 0; for (double v : d) s += v; return s / std::max<size_t>(1, N);
    }

    void computeKA() {
        k_cached.resize(N);
        a_cached.resize(N);
        for (size_t j = 0; j < N; ++j) {
            int k = (int)std::floor(d[j]);
            double a = d[j] - (double)k;
            if (k < 0) { k = 0; a = 0.0; }
            if (k > Dmax) { k = Dmax; a = 0.0; }
            k_cached[j] = k; a_cached[j] = a;
        }
    }

    // Build shifted matrix Xs and optionally Delta = x(t-k-1)-x(t-k)
    static void buildShifted(const Eigen::MatrixXd& Echo,
        const std::vector<double>& d,
        Eigen::MatrixXd& Xs,
        Eigen::MatrixXd* Delta) {
        const int T = (int)Echo.rows();
        const int N = (int)Echo.cols();
        Xs.resize(T, N);
        if (Delta) Delta->resize(T, N);
        for (int j = 0; j < N; ++j) {
            double dj = d[(size_t)j];
            if (dj < 0) dj = 0; // safety
            int k = (int)std::floor(dj);
            double a = dj - (double)k;
            for (int t = 0; t < T; ++t) {
                int idx0 = t - k; if (idx0 < 0) idx0 = 0; if (idx0 >= T) idx0 = T - 1;
                int idx1 = idx0 - 1; if (idx1 < 0) idx1 = 0;
                double x0 = Echo(idx0, j);
                double x1 = Echo(idx1, j);
                Xs(t, j) = (1.0 - a) * x0 + a * x1;
                if (Delta) (*Delta)(t, j) = (x1 - x0); // derivative wrt d in this k-region
            }
        }
    }
};

// ================= Main =================
int main() {
    try {
        std::cout << "Loading reservoir\n";
        auto W = loadMatrix("reservoir0.95.bin");
        Reservoir reservoir(W);

        // ---- Load training audio ----
        std::cout << "Loading training input/output\n";
        SF_INFO sfinfo1{};
        auto input1 = loadMonoWav("input1.wav", sfinfo1);
        auto output1 = loadMonoWav("output1.wav", sfinfo1);
        int T1 = (int)sfinfo1.frames;
        int N = (int)reservoir.size();

        // ---- Collect Echo states ----
        std::cout << "Collecting reservoir states\n";
        Eigen::MatrixXd Echo1(T1, N);
        Eigen::VectorXd Target1(T1);
        reservoir.reset();
        for (int t = 0; t < T1; t++) {
            reservoir.step(input1[(size_t)t]);
            const auto& st = reservoir.getState();
            for (int j = 0; j < N; j++) Echo1(t, j) = st[(size_t)j];
            Target1[t] = output1[(size_t)t];
        }

        // ---- Train delayed readout ----
        std::cout << "Training delayed readout (method 2)\n";
        int Dmax = 64;          // max delay in samples (tune)
        double lambda = 1e-6;   // ridge for weights
        double gamma = 1e-6;    // small L2 on delays; set 0 to disable
        double lr = 0.1;       // delay learning rate
        int ls = 10;        // delay learning steps
        int iters = 200;        // outer iterations

        DelayedReadout readout((size_t)N, Dmax, lambda, gamma, lr, ls);
        readout.train(Echo1, Target1, iters, /*verbose=*/true);

        // ---- Offline: apply to training echo (to inspect fit) ----
        std::cout << "Processing training input (offline)\n";
        auto y1 = readout.processEcho(Echo1);
        writeMonoWav("echo_output1.wav", y1, sfinfo1);
        auto file_back = loadMonoWav("echo_output1.wav", sfinfo1);
        if (file_back.size() == y1.size()) {
            double mse_rw = 0.0;
            for (size_t t = 0;t < y1.size();++t) { double e = file_back[t] - y1[t]; mse_rw += e * e; }
            mse_rw /= (double)y1.size();
            std::cout << "Written file vs in-memory y1 MSE = " << mse_rw << "\n";
        }
        else {
            std::cout << "Written file length mismatch: file " << file_back.size() << " vs y1 " << y1.size() << "\n";
        }

        // ---- Live: apply to new input ----
        std::cout << "Loading new input\n";
        SF_INFO sfinfo2{};
        auto input2 = loadMonoWav("input2.wav", sfinfo2);
        std::vector<double> output2(input2.size());

        std::cout << "Processing new input (live)\n";
        reservoir.reset();
        for (size_t t = 0; t < input2.size(); t++) {
            reservoir.step(input2[t]);
            const auto& st = reservoir.getState();
            output2[t] = readout.processStateLive(st);
        }
        writeMonoWav("echo_output2.wav", output2, sfinfo2);

        // ---- Print a small summary ----
        const auto& d = readout.delays();
        double dmean = 0; for (double v : d) dmean += v; dmean /= std::max<size_t>(1, d.size());
        std::cout << "Done! Mean learned delay = " << dmean << " samples\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
