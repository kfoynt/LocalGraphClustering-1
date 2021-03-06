/**
 * Randomized proximal coordinate descent for l1 regularized pagerand vector
 * INPUT:
 *     alpha     - teleportation parameter between 0 and 1
 *     rho       - l1-reg. parameter
 *     v         - seed node
 *     ai,aj,a   - Compressed sparse row representation of A
 *     d         - vector of node strengths
 *     epsilon   - accuracy for termination criterion
 *     n         - size of A
 *     ds        - the square root of d
 *     dsinv     - 1/ds
 *     offset    - offset for zero based arrays (matlab) or one based arrays (julia)
 *
 * OUTPUT:
 *     p              - PageRank vector as a row vector
 *     not_converged  - flag indicating that maxiter has been reached
 *     grad           - last gradient
 *
 */


// start include write data to files
#include <iostream>
#include <fstream>
// end
#include <vector>
#include <cmath>
#include "include/routines.hpp"
#include "include/proxl1PRrand_c_interface.h"

using namespace std;

namespace proxl1PRrand 
{
template<typename vtype, typename itype>
void updateGrad(vtype node, double& rho, double& alpha, double* q, double* grad, double* ds, double* dsinv, itype* ai, vtype* aj, vector<bool>& visited, vector<vtype>& candidates, vector<double>& norms) {
    double c = (1-alpha)/2;
    double ra = rho*alpha;
    double rads = rho*alpha*ds[node];
    double dq = -grad[node]-rads;
    double cdqdsinv = c*dq*dsinv[node];
    q[node] += dq;
    grad[node] = -rads-c*dq;

    vtype neighbor;
    for (itype j = ai[node]; j < ai[node + 1]; ++j) {
        neighbor = aj[j];
        grad[neighbor] -= cdqdsinv*dsinv[neighbor];
        norms[neighbor] = abs(grad[neighbor]*dsinv[neighbor]);
        if (!visited[neighbor] && q[neighbor] - grad[neighbor] >= ra*ds[neighbor]) {
            visited[neighbor] = true;
            candidates.push_back(neighbor);
        }
    }
}

template<typename vtype, typename itype>
vector<double> getQdiag(vtype numNodes, itype* ai, vtype* aj, double* d, double alpha) {
    vector<double> Qdiag(numNodes);
    vector<bool> Adiag(numNodes, false);

    for (vtype node = 0; node < numNodes; ++node) {
        itype idx = ai[node];
        while (idx < ai[node + 1] && aj[idx] < node) ++idx;
        Adiag[node] = idx < ai[node + 1] && aj[idx] == node;
    }

    double c = (1 - alpha) / 2;
    for (vtype node = 0; node < numNodes; ++node) {
        Qdiag[node] = 1.0 - c - c * Adiag[node] / d[node];
    }
    return Qdiag;
}

template<typename vtype>
void writeLog(vtype numNodes, const string& fname, double* q) {
    ofstream file(fname, fstream::app);
    if (file.is_open()) {
        for (vtype i = 0; i < numNodes; ++i) {
            file << q[i] << ',';
        }
        file << '\n';
        file.close();
    }
}

template<typename vtype, typename itype>
void writeGraph(vtype numNodes, const string& fname, itype* ai, vtype* aj) {
    ofstream file(fname);
    if (file.is_open()) {
        itype idx = 0;
        for (vtype node = 0; node < numNodes; ++node) {
            for (vtype neighbor = 0; neighbor < numNodes; ++neighbor) {
                if (idx < ai[node + 1] && aj[idx] == neighbor) {
                    ++idx;
                    file << 1 << ',';
                } else {
                    file << 0 << ',';
                }
            }
            file << '\n';
        }
        file.close();
    }
}
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRrand(vtype numNodes, double epsilon, double alpha, double rho, double* q, double* d, double* ds, double* dsinv, double* grad)
{
	vtype not_converged = 0;
    vtype seed = 3; // randomly choose
    double maxNorm = 2;
    grad[seed] = -alpha*dsinv[seed];
    maxNorm = abs(grad[seed]*dsinv[seed]);

    vector<vtype> candidates(1, seed);
    vector<bool> visited(numNodes, false);
    vector<double> norms(numNodes, 0);
    norms[seed] = maxNorm;
    visited[seed] = true;
    // exp start
    proxl1PRrand::writeGraph(numNodes, "../../../../../logs/graph.txt", ai, aj);
    vector<double> Qdiag = proxl1PRrand::getQdiag(numNodes, ai, aj, d, alpha);
    cout << "Q diagnal: ";
    for (double v : Qdiag)
        cout << v << ", ";
    cout << endl;
    // exp end
    double threshold = (1+epsilon)*rho*alpha;
    while (maxNorm > threshold) {
        vtype r = rand() % candidates.size();  // TODO rand() type?
        proxl1PRrand::updateGrad(candidates[r], rho, alpha, q, grad, ds, dsinv, ai, aj, visited, candidates, norms);
        // write q for analysis
        proxl1PRrand::writeLog(numNodes, "../../../../../logs/test.txt", q);
        // end
        maxNorm = 0;
        for (vtype i = 0; i < numNodes; ++i) {
            maxNorm = max(maxNorm, norms[i]);
        }
    }
    for (vtype i = 0; i < numNodes; ++i) {
        q[i] = abs(q[i])*ds[i];
    }
    return not_converged;
}

uint32_t proxl1PRrand32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double alpha,
                         double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                         double* dsinv, double epsilon, double* grad, double* p, double* y,
                         uint32_t maxiter, uint32_t offset, double max_time)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    return g.proxl1PRrand(n, epsilon, alpha, rho, p, d, ds, dsinv, grad);
}

int64_t proxl1PRrand64(int64_t n, int64_t* ai, int64_t* aj, double* a, double alpha,
                        double rho, int64_t* v, int64_t v_nums, double* d, double* ds,
                        double* dsinv,double epsilon, double* grad, double* p, double* y,
                        int64_t maxiter, int64_t offset, double max_time)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    return g.proxl1PRrand(n, epsilon, alpha, rho, p, d, ds, dsinv, grad);
}

uint32_t proxl1PRrand32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, double alpha,
                            double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                            double* dsinv, double epsilon, double* grad, double* p, double* y,
                            uint32_t maxiter, uint32_t offset, double max_time)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    return g.proxl1PRrand(n, epsilon, alpha, rho, p, d, ds, dsinv, grad);
}