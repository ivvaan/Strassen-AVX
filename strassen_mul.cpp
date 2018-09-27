#include "strassen_mul.h"
#include "small_matr_funcs.h"
#include "base_operations.h"
#include "utils.h"
#include <crtdbg.h>
#include <memory.h>
#include <boost/smart_ptr/scoped_array.hpp>
#include <immintrin.h>
#include <iostream>
#include <fstream>




void get_num_calc(unsigned SZ, double(&num_calc)[12])
{
    const unsigned small_matr_size = 130;
    for (unsigned i = 0; i < count_of(num_calc); i++)
        num_calc[i] = 0;
    double &small_matr_strassen = num_calc[8], &small_matr_padding = num_calc[9];
    double &big_matr_strassen = num_calc[10], &big_matr_padding = num_calc[11];
    double n = 1;
    while (SZ > small_matr_size) {
        if (SZ % 2) {
            big_matr_padding += n*SZ*SZ;
            --SZ;
        }
        big_matr_strassen += n*SZ*SZ;
        n *= 7;
        SZ /= 2;
    }
    while (SZ >= 16) {
        if (SZ % 2) {
            small_matr_padding += n*SZ*SZ;
            --SZ;
        }
        small_matr_strassen += n*SZ*SZ;
        n *= 7;
        SZ /= 2;
    }
    num_calc[SZ - 8] = n;
};

double get_weight(unsigned SZ, unsigned enlarge)
{
    //time in ns on my computer
    /* const double op_wgts[] = { 1.088E+02,2.012E+02,2.789E+02,3.879E+02,      //to multiply 8x8,9x9,10x10,11x11 matrix
    3.418E+02,5.112E+02,6.702E+02,8.304E+02,//to multiply 12x12,13x13,14x14,15x15 matrix
    5.573E-01,1.238E+00,4.462E+00,1.361E+00 }; //to perform (per element) strassen additive small, padding small, strassen additive big, padding big
    const double enlarge_per_el = 6.354;*/
    const double op_wgts[] = { 1.088E+02,2.012E+02,2.789E+02,3.879E+02,      //to multiply 8x8,9x9,10x10,11x11 matrix
        3.418E+02,5.112E+02,6.702E+02,8.304E+02,//to multiply 12x12,13x13,14x14,15x15 matrix
        1.09E+00,1.33E+00,1.61E+00,1.97E+00 }; //to perform (per element) strassen additive small, padding small, strassen additive big, padding big
    const double enlarge_per_el = 3.76;
    SZ += enlarge;
    double num_calc[12];
    get_num_calc(SZ, num_calc);
    double wgt = num_calc[0] * op_wgts[0];
    for (unsigned i = 1; i < count_of(num_calc); i++)
        wgt += num_calc[i] * op_wgts[i];
    return wgt + (enlarge ? enlarge_per_el*SZ*SZ : 0);
};

unsigned get_best_enl(unsigned SZ, unsigned range)
{
    double min_v = get_weight(SZ, 0);
    unsigned min_e = 0;
    for (unsigned i = 1; i < range; ++i) {
        double cur_v = get_weight(SZ, i);
        if (cur_v < min_v)
        {
            min_v = cur_v; min_e = i;
        }
    }
    return min_e;
};


void strassen_mul_suffix(unsigned sz, unsigned CS, double *C00, double *C01, double *C10, double *C11, double *S00, double *S01, double *S10)
{
    auto C_delta = CS - sz;
    auto C_last_row = C00 + CS*sz;
    auto sz4 = sz % 4;
    while (C00 < C_last_row) {
        for (auto C_last_col = C00 + (sz - sz4); C00 < C_last_col; C00 += 4, C01 += 4, C10 += 4, C11 += 4, S00 += 4, S01 += 4, S10 += 4)
        {
            __m256d ymm00 = _mm256_loadu_pd(C00);
            __m256d ymm01 = _mm256_loadu_pd(C01);
            __m256d ymm10 = _mm256_loadu_pd(C10);
            __m256d ymm11 = _mm256_loadu_pd(C11);
            __m256d zmm00 = _mm256_loadu_pd(S00);
            __m256d zmm01 = _mm256_loadu_pd(S01);
            __m256d zmm10 = _mm256_loadu_pd(S10);
            ymm01 = _mm256_add_pd(ymm01, zmm00);
            zmm10 = _mm256_add_pd(zmm10, ymm01);
            ymm00 = _mm256_add_pd(zmm00, ymm00);
            _mm256_storeu_pd(C00, ymm00);
            ymm00 = _mm256_add_pd(ymm01, ymm11);
            ymm00 = _mm256_sub_pd(ymm00, zmm01);
            _mm256_storeu_pd(C01, ymm00);
            ymm00 = _mm256_sub_pd(zmm10, ymm10);
            _mm256_storeu_pd(C10, ymm00);
            ymm00 = _mm256_add_pd(ymm11, zmm10);
            _mm256_storeu_pd(C11, ymm00);
        }
        for (auto C_last_col = C00 + sz4; C00 < C_last_col; ++C00, ++C01, ++C10, ++C11, ++S00, ++S01, ++S10)
        {
            *C01 += *S00;
            *S10 += *C01;
            *C00 += *S00;
            *C01 += *C11 - *S01;
            *C10 = *S10 - *C10;
            *C11 += *S10;
        }
        C00 += C_delta;
        C01 += C_delta;
        C10 += C_delta;
        C11 += C_delta;
    }
};


void strassen_padding_calc(unsigned SZ, double *buf, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
    auto SZ_minus_one = SZ - 1;// LAST
                               //coping last column of B  (hereafter referred to as B[][LAST]) to buf
    for (auto B_ = buf, B_end = buf + SZ_minus_one, B_cur_last = B + SZ_minus_one; B_ < B_end; ++B_, B_cur_last += BS)
        *B_ = *B_cur_last;   //buf[]=B[][LAST]

    for (auto B_ = B + SZ_minus_one*BS, C_end = C + CS*SZ_minus_one; C < C_end; C += CS, A += AS) {
        C_add_a_mul_B(SZ_minus_one, C, A[SZ_minus_one], buf);   //C[I][] += A[I][LAST]*B[][LAST] 
        C[SZ_minus_one] = vectA_mul_vectB(SZ, A, B_);       // C[I][LAST]=A[i][]*B[LAST][], B[LAST][] is last row of B 
    }
    for (auto C_end = C + SZ; C < C_end; ++C, B += BS)*C = vectA_mul_vectB(SZ, A, B);// C[LAST][I]=A[LAST][]*B[i][]

};




/*
Strassen - Winograd algorithm has 15 additive (+-) matrix operation (original Strassen has 18)

C11  C12     A11  A12     B11  B12
=            ×
C21  C22     A21  A22     B21  B22


S1 ← A21 + A22
S2 ← S1 − A11
S3 ← A11 − A21
S4 ← A12 − S2
T1 ← B12 − B11
T2 ← B22 − T1
T3 ← B22 − B12
T4 ← T2 − B21

P1 ← A11 × B11
P2 ← A12 × B21
P3 ← S4 × B22
P4 ← A22 × T4
P5 ← S1 × T1
P6 ← S2 × T2
P7 ← S3 × T3

U1 ← P1 + P6
U2 ← U1 + P7
U3 ← U1 + P5
C11 ← P1 + P2
C12 ← U3 + P3
C21 ← U2 − P4
C22 ← U2 + P5
*/






void  strassen_recur_mul_by_transposed(unsigned SZ, double *buf, double *C00, unsigned CS, double *A00, unsigned AS, double *B00, unsigned BS)
{

    if (SZ < get_funcs_numb()) {
        get_mul_func(SZ)(C00, CS, A00, AS, B00, BS);
    }
    else {

        auto sz = SZ / 2;

        auto A01 = A00 + sz;
        auto A10 = A00 + AS*sz;
        auto A11 = A10 + sz;

        auto C01 = C00 + sz;
        auto C10 = C00 + CS*sz;
        auto C11 = C10 + sz;

        // B - transposed matrix!!
        auto B01 = B00 + BS*sz;
        auto B10 = B00 + sz;
        auto B11 = B01 + sz;

        // storage for temporary matrices
        auto subm_size = sz*sz;
        auto *S00 = buf; buf += subm_size;
        auto *S01 = buf; buf += subm_size;
        auto *S10 = buf; buf += subm_size;
        auto *T00 = buf; buf += subm_size;
        _ASSERT(buf<buf_max);


        matrix_sum(sz, S00, sz, A10, AS, A11, AS);
        matrix_sub(sz, T00, sz, B01, BS, B00, BS);
        strassen_recur_mul_by_transposed(sz, buf, C11, CS, S00, sz, T00, sz);
        matrix_sub(sz, T00, sz, B11, BS, T00, sz);
        matrix_sub(sz, S00, sz, S00, sz, A00, AS);
        strassen_recur_mul_by_transposed(sz, buf, C01, CS, S00, sz, T00, sz);
        matrix_sub(sz, T00, sz, T00, sz, B10, BS);
        matrix_sub(sz, S00, sz, S00, sz, A01, AS);

        strassen_recur_mul_by_transposed(sz, buf, S01, sz, S00, sz, B11, BS);
        strassen_recur_mul_by_transposed(sz, buf, C10, CS, A11, AS, T00, sz);
        matrix_sub(sz, T00, sz, B11, BS, B01, BS);
        matrix_sub(sz, S00, sz, A00, AS, A10, AS);

        strassen_recur_mul_by_transposed(sz, buf, S10, sz, S00, sz, T00, sz);
        strassen_recur_mul_by_transposed(sz, buf, S00, sz, A00, AS, B00, BS);
        strassen_recur_mul_by_transposed(sz, buf, C00, CS, A01, AS, B10, BS);
        strassen_mul_suffix(sz, CS, C00, C01, C10, C11, S00, S01, S10);
        if (SZ % 2)
            strassen_padding_calc(SZ, S00, C00, CS, A00, AS, B00, BS);
    }
}

void strassen_transp_and_mul(unsigned SZ, double *buf, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)  //
{
    auto *BT = buf; buf += SZ*SZ;
    _ASSERT(buf<buf_max);
    transp(SZ, BT, B, BS);
    strassen_recur_mul_by_transposed(SZ, buf, C, CS, A, AS, BT, SZ);
};

double *enlarge_matrix(unsigned SZ, unsigned SZ_new, boost::scoped_array<double> &M_, double *M)
{
    M_.reset(new double[SZ_new*SZ_new]);
    auto M_new = M_.get();
    auto M_end = M + SZ*SZ;
    auto nbytescopy = SZ * sizeof(M[0]);
    auto nbyteszero = (SZ_new - SZ) * sizeof(M[0]);
    auto nbytesall = nbytescopy + nbyteszero;

    for (; M < M_end; M += SZ, M_new += SZ_new) {
        memcpy(M_new, M, nbytescopy);
        memset(M_new + SZ, 0, nbyteszero);
    }
    for (unsigned i = 0; i < SZ_new - SZ; i++, M_new += SZ_new) {
        memset(M_new, 0, nbytesall);
        M_new[SZ + i] = 1;
    }


    return M_.get();

};

void copy_on(double *M, unsigned SZ, double *M_src, unsigned SZ_src)
{
    auto nbytescopy = SZ * sizeof(M[0]);
    for (auto M_end = M + SZ*SZ; M < M_end; M += SZ, M_src += SZ_src)  memcpy(M, M_src, nbytescopy);
};




void  strassen_mul(unsigned SZ_, double *C_, double *A_, double *B_, int enl = 0)
{
    if (small_matr_mul(SZ_, C_, A_, B_))return;
    //if positive enl is passed to the function - use it.
    if (enl == 0) enl = get_best_enl(SZ_, SZ_ / 3);  // find optimal matrix enlarge by default 
    if (enl <0)enl = 0;     // if enl is negative don't change matrix size.
    
    double *C, *A, *B;
    auto SZ = SZ_;
    boost::scoped_array<double> c, a, b;
    if (enl) {
        SZ = SZ_ + enl;
        c.reset(new double[SZ*SZ]);
        C = c.get();
        A = enlarge_matrix(SZ_, SZ, a, A_);
        B = enlarge_matrix(SZ_, SZ, b, B_);
    }
    else
    {
        C = C_;
        A = A_;
        B = B_;
    }
    inplace_transpose(SZ, B);
    unsigned buf_size = (4 * SZ*SZ) / 3 + 1;
    boost::scoped_array<double> buf(new double[buf_size]);
    _SET_BUF_MAX(buf.get() + buf_size);
    strassen_recur_mul_by_transposed(SZ, buf.get(), C, SZ, A, SZ, B, SZ);
    if (SZ == SZ_)inplace_transpose(SZ, B);
    else copy_on(C_, SZ_, C, SZ);
};



void block_mul(unsigned SZ_, double *C_, double *A_, double *B_)
{
    if (small_matr_mul(SZ_, C_, A_, B_))return;
    double *C, *A, *B;
    auto SZ = SZ_;
    boost::scoped_array<double> c, a, b;
    const unsigned bs = 12;
    if (SZ % bs) {
        SZ = SZ_ - SZ_ % bs + bs;
        c.reset(new double[SZ*SZ]);
        C = c.get();
        A = enlarge_matrix(SZ_, SZ, a, A_);
        B = enlarge_matrix(SZ_, SZ, b, B_);
    }
    else
    {
        C = C_;
        A = A_;
        B = B_;
    }
    inplace_transpose(SZ, B);

    FSimpleMatrProd f_t = get_mul_func(bs);
    FSimpleMatrProd f_t_p = get_plus_mul_func(bs);

    for (unsigned i = 0; i<SZ; i += bs)
        for (unsigned j = 0; j < SZ; j += bs)
        {
#define IJ_M_PTR(M,I,J) &M[I*SZ+J]
            f_t(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, 0), SZ, IJ_M_PTR(B, j, 0), SZ);
            for (unsigned k = bs; k<SZ; k += bs)
                f_t_p(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, k), SZ, IJ_M_PTR(B, j, k), SZ);
        }
    if (SZ == SZ_)inplace_transpose(SZ, B);
    else copy_on(C_, SZ_, C, SZ);


}

