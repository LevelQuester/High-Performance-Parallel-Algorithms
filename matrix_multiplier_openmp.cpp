#include <iostream>
#include <omp.h>

using namespace std;
short A[4001][4001];
short B[4001][4001];
int C[4001][4001];
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;

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

#pragma omp parallel for collapse(2)
    for (int i=0; i<n;i++) {
        for (int j=0; j<n; j++) {
            int c = 0;
            for (int k=0; k<n; k++){
                c += 1 * A[i][k] * B[j][k];
            }
            C[i][j] = c;
        }
    }

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << C[i][j] << " ";
        }
        cout << "\n";
    }
    return 0;
}