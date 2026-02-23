#pragma GCC optimize("O3")
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <functional>

using namespace std;
void matrixMuliplyer(int n,
    const vector<vector<short>> &A,
    const vector<vector<short>> &B,
    vector<vector<long long>> &C,
    int l, int r) {
    for (int i=l; i<r;i++) {
        for (int j=0; j<n; j++) {
            long long c = 0;
            for (int k=0; k<n; k++){
                c += 1LL * A[i][k] * B[j][k];
            }
            C[i][j] = c;
        }
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<short>> A(n, vector<short>(n, 0));
    vector<vector<short>> B(n, vector<short>(n, 0));
    vector<vector<long long>> C(n, vector<long long>(n, 0));
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cin >> A[i][j];
        }
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cin >> B[j][i];
        }
    }
    int T = min(n, 16);
    int base = n/T;
    int rem = n%T;

    vector<thread> threads;
    threads.reserve(T);

    for (int k=0; k < T;k++) {
        int l = base*k + min(k, rem);
        int r = l + base + (k < rem ? 1 : 0);
        threads.emplace_back(matrixMuliplyer,
                                n,
                                    cref(A),
                                    cref(B),
                                    ref(C),
                                    l, r);
    }
    for (int i=0; i<T; i++) {
        threads[i].join();
    }

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << C[i][j] << " ";
        }
        cout << "\n";
    }
    return 0;
}