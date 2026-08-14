/* Batch wrapper for CORE-MATH cr_sinf / cr_cosf.
   Build: gcc -O3 -shared -static -o crtrig.dll crtrig_wrapper.c sinf.c cosf.c */
#include <stdint.h>

float cr_sinf(float);
float cr_cosf(float);

__declspec(dllexport) void cr_sinf_array(const uint32_t *in, uint32_t *out, int64_t n)
{
  for (int64_t i = 0; i < n; i++) {
    union { float f; uint32_t u; } a, b;
    a.u = in[i];
    b.f = cr_sinf(a.f);
    out[i] = b.u;
  }
}

__declspec(dllexport) void cr_cosf_array(const uint32_t *in, uint32_t *out, int64_t n)
{
  for (int64_t i = 0; i < n; i++) {
    union { float f; uint32_t u; } a, b;
    a.u = in[i];
    b.f = cr_cosf(a.f);
    out[i] = b.u;
  }
}

/* Contiguous bit-pattern range [start, start+n): avoids building input arrays. */
__declspec(dllexport) void cr_sinf_range(uint32_t start, uint32_t *out, int64_t n)
{
  for (int64_t i = 0; i < n; i++) {
    union { float f; uint32_t u; } a, b;
    a.u = start + (uint32_t)i;
    b.f = cr_sinf(a.f);
    out[i] = b.u;
  }
}

__declspec(dllexport) void cr_cosf_range(uint32_t start, uint32_t *out, int64_t n)
{
  for (int64_t i = 0; i < n; i++) {
    union { float f; uint32_t u; } a, b;
    a.u = start + (uint32_t)i;
    b.f = cr_cosf(a.f);
    out[i] = b.u;
  }
}

__declspec(dllexport) float cr_sinf1(float x) { return cr_sinf(x); }
__declspec(dllexport) float cr_cosf1(float x) { return cr_cosf(x); }
