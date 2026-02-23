#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static constexpr int maxBlockLength = 100000;

struct Hash {
  int h[4];
  __host__ __device__ Hash() {}
  __host__ __device__ Hash(int v) { h[0] = h[1] = h[2] = h[3] = v; }
  __host__ __device__ Hash(int v1, int v2, int v3, int v4) {
    h[0] = v1;
    h[1] = v2;
    h[2] = v3;
    h[3] = v4;
  }
  __host__ __device__ int &operator[](int idx) { return h[idx]; }
  __host__ __device__ int operator[](int idx) const { return h[idx]; }
  __host__ void print() {
    uint8_t *data = reinterpret_cast<uint8_t *>(h);
    for (int i = 0; i < 16; ++i)
      printf("%02x", data[i]);
    printf("\n");
  }
};

__host__ __device__ inline uint32_t rotl(uint32_t value, uint32_t amount) {
  return (value << amount) | (value >> (32 - amount));
}

int md5preprocess(char *line, size_t size) {
  size_t realSize = size;
  line[realSize] = 0x80;
  ++realSize;
  while ((realSize + 8) % 64) {
    line[realSize] = 0;
    ++realSize;
  }
  *reinterpret_cast<uint64_t *>(line + realSize) = uint64_t(size * 8);
  return (int)((realSize + 8) / 64);
}

__constant__ uint32_t d_k[64];
__constant__ uint32_t d_s[64];

static constexpr uint32_t k_host[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

static constexpr uint32_t s_host[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

__host__ Hash md5_host(char *data, int numBlocks) {
  Hash h(0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476);

  for (int block = 0; block < numBlocks; ++block) {
    uint32_t *M = reinterpret_cast<uint32_t *>(data + block * 64);
    Hash c(h[0], h[1], h[2], h[3]);

    for (uint32_t i = 0; i < 64; ++i) {
      uint32_t F, g;
      if (i < 16) {
        F = (c[1] & c[2]) | (~c[1] & c[3]);
        g = i;
      } else if (i < 32) {
        F = (c[3] & c[1]) | (~c[3] & c[2]);
        g = (5 * i + 1) & 0xf;
      } else if (i < 48) {
        F = c[1] ^ c[2] ^ c[3];
        g = (3 * i + 5) & 0xf;
      } else {
        F = c[2] ^ (c[1] | ~c[3]);
        g = (7 * i) & 0xf;
      }

      F = F + c[0] + k_host[i] + M[g];
      c = Hash(c[3], c[1] + rotl(F, s_host[i]), c[1], c[2]);
    }

    h[0] += c[0];
    h[1] += c[1];
    h[2] += c[2];
    h[3] += c[3];
  }
  return h;
}

__device__ inline int count_leading_zeros(const Hash &h) {
  int z0 = __clz(__byte_perm(h.h[0], 0, 0x0123));
  if (z0 < 32)
    return z0;
  int z1 = __clz(__byte_perm(h.h[1], 0, 0x0123));
  if (z1 < 32)
    return 32 + z1;
  int z2 = __clz(__byte_perm(h.h[2], 0, 0x0123));
  if (z2 < 32)
    return 64 + z2;
  int z3 = __clz(__byte_perm(h.h[3], 0, 0x0123));
  return 96 + z3;
}

__global__ void find_seed(const uint32_t *global_data, int numBlocks, int t,
                          int *success_flag, uint32_t *result_seed,
                          uint64_t start_nonce) {
  if (*success_flag)
    return;

  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t my_seed[4];
  my_seed[0] = (uint32_t)(tid & 0xFFFFFFFF);
  my_seed[1] = (uint32_t)((tid >> 32) ^ start_nonce);
  my_seed[2] = 0;
  my_seed[3] = 0;

  Hash h(0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476);

  for (int block = 0; block < numBlocks; ++block) {
    uint32_t M[16];
    for (int i = 0; i < 16; ++i) {
      M[i] = global_data[block * 16 + i];
    }
    if (block == 0) {
      M[4] = my_seed[0];
      M[5] = my_seed[1];
      M[6] = my_seed[2];
      M[7] = my_seed[3];
    }

    Hash c(h[0], h[1], h[2], h[3]);
    for (uint32_t i = 0; i < 64; ++i) {
      uint32_t F, g;
      if (i < 16) {
        F = (c[1] & c[2]) | (~c[1] & c[3]);
        g = i;
      } else if (i < 32) {
        F = (c[3] & c[1]) | (~c[3] & c[2]);
        g = (5 * i + 1) & 0xf;
      } else if (i < 48) {
        F = c[1] ^ c[2] ^ c[3];
        g = (3 * i + 5) & 0xf;
      } else {
        F = c[2] ^ (c[1] | ~c[3]);
        g = (7 * i) & 0xf;
      }

      F = F + c[0] + d_k[i] + M[g];
      c = Hash(c[3], c[1] + rotl(F, d_s[i]), c[1], c[2]);
    }

    h[0] += c[0];
    h[1] += c[1];
    h[2] += c[2];
    h[3] += c[3];
  }

  if (count_leading_zeros(h) >= t) {
    if (atomicCAS(success_flag, 0, 1) == 0) {
      result_seed[0] = my_seed[0];
      result_seed[1] = my_seed[1];
      result_seed[2] = my_seed[2];
      result_seed[3] = my_seed[3];
    }
  }
}

void print_seed(uint32_t *seed) {
  uint8_t *data = reinterpret_cast<uint8_t *>(seed);
  for (int i = 0; i < 16; ++i)
    printf("%02x", data[i]);
  printf("\n");
}

int main() {

  cudaMemcpyToSymbol(d_k, k_host, 64 * sizeof(uint32_t));
  cudaMemcpyToSymbol(d_s, s_host, 64 * sizeof(uint32_t));

  int z, t;
  if (!(std::cin >> z >> t))
    return 0;

  std::string dummy;
  std::getline(std::cin, dummy);

  char *block_host = new char[maxBlockLength + 200];
  uint32_t *d_data;
  cudaMalloc(&d_data, maxBlockLength + 200);

  int *d_success_flag;
  cudaMalloc(&d_success_flag, sizeof(int));

  uint32_t *d_result_seed;
  cudaMalloc(&d_result_seed, 16);

  uint32_t prev_hash[4] = {0, 0, 0, 0};

  int threads_per_block = 256;
  int blocks_per_grid = 65535;

  for (int i = 0; i < z; ++i) {
    std::string msg;
    std::getline(std::cin, msg);
    int msg_len = msg.length();
    if (!msg.empty() && msg.back() == '\r') {
      msg.pop_back();
      msg_len = msg.length();
    }

    uint32_t *host_M = reinterpret_cast<uint32_t *>(block_host);
    host_M[0] = prev_hash[0];
    host_M[1] = prev_hash[1];
    host_M[2] = prev_hash[2];
    host_M[3] = prev_hash[3];
    host_M[4] = 0;
    host_M[5] = 0;
    host_M[6] = 0;
    host_M[7] = 0;
    std::memcpy(block_host + 32, msg.c_str(), msg_len);

    int size = 32 + msg_len;
    int numBlocks = md5preprocess(block_host, size);

    cudaMemcpy(d_data, block_host, numBlocks * 64, cudaMemcpyHostToDevice);

    int success = 0;
    cudaMemcpy(d_success_flag, &success, sizeof(int), cudaMemcpyHostToDevice);

    uint64_t nonce = 1;
    while (!success) {
      find_seed<<<blocks_per_grid, threads_per_block>>>(
          d_data, numBlocks, t, d_success_flag, d_result_seed, nonce);
      cudaDeviceSynchronize();
      cudaMemcpy(&success, d_success_flag, sizeof(int), cudaMemcpyDeviceToHost);
      nonce += 1337;
    }

    uint32_t found_seed[4];
    cudaMemcpy(found_seed, d_result_seed, 16, cudaMemcpyDeviceToHost);

    print_seed(found_seed);

    host_M[4] = found_seed[0];
    host_M[5] = found_seed[1];
    host_M[6] = found_seed[2];
    host_M[7] = found_seed[3];

    Hash next_hash = md5_host(block_host, numBlocks);
    prev_hash[0] = next_hash[0];
    prev_hash[1] = next_hash[1];
    prev_hash[2] = next_hash[2];
    prev_hash[3] = next_hash[3];
  }

  cudaFree(d_data);
  cudaFree(d_success_flag);
  cudaFree(d_result_seed);
  delete[] block_host;

  return 0;
}
