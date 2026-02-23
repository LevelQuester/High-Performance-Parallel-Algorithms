#pragma GCC optimize("O3")
#include <iostream>
#include <vector>

using namespace std;
int dist[2001][2001];

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

    for (int k=0; k<n;k++)
#pragma omp parallel for num_threads(min(n, 16))
        for (int i=0; i<n;i++) {
            for (int j=0; j<n; j++) {
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
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