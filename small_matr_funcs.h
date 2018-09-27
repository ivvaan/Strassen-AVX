#pragma once
#include "utils.h"

typedef void(*FSimpleMatrProd)(double *, unsigned, double *, unsigned, double *, unsigned);

struct fix_size_prod_funcs {
    FSimpleMatrProd multiply, plus_multiply;
};

extern fix_size_prod_funcs prod_funcs_t[17];

constexpr unsigned get_funcs_numb()
{
    return count_of(prod_funcs_t);
}

inline FSimpleMatrProd get_mul_func(unsigned sz)
{
    return prod_funcs_t[sz].multiply;

}

inline FSimpleMatrProd get_plus_mul_func(unsigned sz)
{
    return  prod_funcs_t[sz].plus_multiply;

}
/*unsigned get_funcs_numb();
FSimpleMatrProd get_mul_func(unsigned sz);
FSimpleMatrProd get_plus_mul_func(unsigned sz);*/

bool small_matr_mul(unsigned SZ_, double * C_, double * A_, double * B_);
