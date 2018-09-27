#include "base_operations.h"
#include <immintrin.h>
#include <math.h>
//SUM ------------------------------------------------------------------
void  matrix_sum(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
    auto A_delta = AS - sz;
    auto B_delta = BS - sz;
    auto C_delta = CS - sz;
    auto sz4 = sz % 4;
    auto *C_last_row = C + CS*sz;
    while (C < C_last_row) {
        auto C_last_col = C + (sz - sz4);
        for (; C < C_last_col; A += 4, B += 4, C += 4) {
            __m256d ymm0 = _mm256_loadu_pd(A);
            __m256d ymm1 = _mm256_loadu_pd(B);
            ymm0 = _mm256_add_pd(ymm0, ymm1);
            _mm256_storeu_pd(C, ymm0);
        }
        for (C_last_col += sz4; C < C_last_col; ++A, ++B, ++C)
            *C = *A + *B;
        A += A_delta;
        B += B_delta;
        C += C_delta;
    }
};

/*void  matrix_sum(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
double *C_last_col;
auto A_delta = AS - sz;
auto B_delta = BS - sz;
auto C_delta = CS - sz;
double *C_last_row = C + CS*sz;
while (C < C_last_row){
C_last_col = C + sz;
for (; C < C_last_col; ++A, ++B, ++C)
*C = *A + *B;
A += A_delta;
B += B_delta;
C += C_delta;
}
};*/


//SUB ------------------------------------------------------------------

void  matrix_sub(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
    auto A_delta = AS - sz;
    auto B_delta = BS - sz;
    auto C_delta = CS - sz;
    auto sz4 = sz % 4;
    auto *C_last_row = C + CS*sz;
    while (C < C_last_row) {
        auto C_last_col = C + (sz - sz4);
        for (; C < C_last_col; A += 4, B += 4, C += 4) {
            __m256d ymm0 = _mm256_loadu_pd(A);
            __m256d ymm1 = _mm256_loadu_pd(B);
            ymm0 = _mm256_sub_pd(ymm0, ymm1);
            _mm256_storeu_pd(C, ymm0);
        }
        for (C_last_col += sz4; C < C_last_col; ++A, ++B, ++C)
            *C = *A - *B;
        A += A_delta;
        B += B_delta;
        C += C_delta;
    }
};


/*void  matrix_sub(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
double *C_last_col;
auto A_delta = AS - sz;
auto B_delta = BS - sz;
auto C_delta = CS - sz;
double *C_last_row = C + CS*sz;
while (C < C_last_row){
C_last_col = C + sz;
for (; C < C_last_col; ++A, ++B, ++C)
*C = *A - *B;
A += A_delta;
B += B_delta;
C += C_delta;
}

};

*/

void C_add_a_mul_B(unsigned sz, double *C, double a, double *B)
{
    //for (auto B_end = B + sz; B < B_end; ++B, ++C) *C += a*(*B);
    double a_[] = { a,a };
    __m256d amm = _mm256_broadcastsd_pd(_mm_loadu_pd(a_));
    auto B_end = B + sz - sz % 4;
    while (B < B_end) {
        __m256d bmm = _mm256_mul_pd(_mm256_loadu_pd(B), amm);
        __m256d cmm = _mm256_add_pd(_mm256_loadu_pd(C), bmm);
        _mm256_storeu_pd(C, cmm);
        B += 4; C += 4;
    }
    B_end += sz % 4;
    while (B < B_end) *C++ += a*(*B++);
};

double vectA_mul_vectB(unsigned sz, double *A, double *B)
{
    /*	auto res = (*A)*(*B);
    auto B_end = B + sz;
    for (++A, ++B; B < B_end; ++A, ++B) res += (*A)*(*B);
    return res;
    */
    auto A_end = A + sz % 4;
    auto B_end = B + sz;
    double res = 0;
    while (A<A_end)   res += (*A++)*(*B++);
    if (B >= B_end)return res;
    __m256d smm = _mm256_mul_pd(_mm256_loadu_pd(A), _mm256_loadu_pd(B));
    __m256d cmm;
    A += 4; B += 4;
    while (B < B_end) {
        cmm = _mm256_mul_pd(_mm256_loadu_pd(A), _mm256_loadu_pd(B));
        smm = _mm256_hadd_pd(smm, cmm);
        A += 4; B += 4;
    }
    smm = _mm256_hadd_pd(smm, smm);
    return res + ((double*)&smm)[0] + ((double*)&smm)[2];
};

void matrA_mul_vectB(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B)
{
    for (auto C_end = C + sz*CS; C < C_end; A += AS, C += CS)
        *C = vectA_mul_vectB(sz, A, B);
}

void transp(unsigned SZ, double *BT, double *B, unsigned BS)
{
    for (unsigned i = 0; i<SZ; i++)
        for (unsigned j = 0; j<SZ; j++)
            BT[j*SZ + i] = B[i*BS + j];
}

void transp(unsigned SZ, double *BT, unsigned BTS, double *B, unsigned BS)
{
    for (unsigned i = 0; i<SZ; i++)
        for (unsigned j = 0; j<SZ; j++)
            BT[j*BTS + i] = B[i*BS + j];
}

void change_sign(unsigned SZ, double *M, unsigned MS)
{
    auto M_delta = MS - SZ;
    for (auto M_last_row = M + SZ*MS; M < M_last_row; M += M_delta)
        for (auto M_last_col = M + SZ; M < M_last_col; M++)
            *M = -*M;

}

void inplace_transpose(unsigned SZ, double *B)
{
    for (unsigned i = 0; i<SZ; i++)
        for (unsigned j = i + 1; j<SZ; j++)
        {
            auto b = B[i*SZ + j];
            B[i*SZ + j] = B[j*SZ + i];
            B[j*SZ + i] = b;
        }

}

double matr_dif(unsigned SZ, double *A, double *B)
{
    double avr = 0;
    double dif = fabs(A[0] - B[0]);
    for (unsigned i = 0; i<SZ; ++i)
        for (unsigned j = 0; j < SZ; ++j)
        {
            avr += fabs(A[i*SZ + j]) + fabs(B[i*SZ + j]);
            double d = fabs(A[i*SZ + j] - B[i*SZ + j]);
            if (d>dif)dif = d;
        }
    //if (avr == 0)return 0;
    return 2 * (dif*SZ)*SZ / avr;
};
double matr_dif2(unsigned SZ, double *A, double *B)
{
    double avr = 0;
    double dif = fabs(A[0] - B[0]);
    double mx = fabs(A[0]) + fabs(B[0]);
    for (unsigned i = 0; i<SZ; ++i)
        for (unsigned j = 0; j < SZ; ++j)
        {
            //avr += fabs(A[i*SZ + j]) + fabs(B[i*SZ + j]);
            double m = fabs(A[i*SZ + j]) + fabs(B[i*SZ + j]);
            if (m > mx) mx = m;
            double d = fabs(A[i*SZ + j] - B[i*SZ + j]);
            if (d>dif) dif = d;
        }
    //if (avr == 0)return 0;
    return 2 * dif / mx;
};
