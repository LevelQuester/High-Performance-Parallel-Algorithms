#pragma GCC optimize("O3")
#include <iostream>
#include <vector>
#include <thread>
#include <functional>

using namespace std;
int dist[2001][2001];

void floyd(int n, int k, int l, int r) {
    for (int i=l; i<r;i++) {
        for (int j=0; j<n; j++) {
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
        }
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> dist[i][j];
        }
    }
    int T = min(n, 16);
    int base = n/T;
    int rem = n%T;
    for (int k=0; k<n; k++) {
        vector<thread> threads;
        threads.reserve(T);
        for (int i=0; i<T;i++) {
            int l = base*i + min(i, rem);
            int r = l + base + (i < rem ? 1 : 0);
            threads.emplace_back(floyd,
                n, k, l, r);
        }
        for (int i=0; i<T; i++) {
            threads[i].join();
        }
    }

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << dist[i][j] << ' ';
        }
        cout << endl;
    }
    return 0;
}