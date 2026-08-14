#include <cstdio>
#include <cstdint>
__global__ void t(const unsigned* bits, float* o_s, float* o_fs, float* o_c) {
    int i = threadIdx.x;
    float x = __uint_as_float(bits[i]);
    o_s[i] = sinf(x);
    float s, c; __sincosf(x, &s, &c);
    o_fs[i] = s;
    o_c[i] = cosf(x);
}
int main() {
    unsigned h[4] = {0x007fffffu, 0x00000001u, 0x00400000u, 0x34000000u};
    unsigned *db; float *ds, *df, *dc;
    cudaMalloc(&db, 16); cudaMalloc(&ds, 16); cudaMalloc(&df, 16); cudaMalloc(&dc, 16);
    cudaMemcpy(db, h, 16, cudaMemcpyHostToDevice);
    t<<<1,4>>>(db, ds, df, dc);
    float hs[4], hf[4], hc[4];
    cudaMemcpy(hs, ds, 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(hf, df, 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(hc, dc, 16, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; i++)
        printf("bits=%08x x=%.6e sinf=%.6e (bits %08x) __sinf=%.6e cosf=%.9f\n",
               h[i], *(float*)&h[i], hs[i], *(unsigned*)&hs[i], hf[i], hc[i]);
    return 0;
}
