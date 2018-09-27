#pragma once
#include <cstddef>
#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))
template <typename T, std::size_t N>
constexpr std::size_t count_of(T const (&)[N]) noexcept { return N; }

#ifdef _DEBUG
extern double *buf_max;
#define _SET_BUF_MAX(BM) buf_max = BM;
#else
#define _SET_BUF_MAX(BM)
#endif

double randm();
