// Strassen.cpp : Defines the entry pounsigned for the console application.
//

//#include "stdafx.h"
#include <stdio.h>
#include <tchar.h>
#include <time.h>

#include <iostream>
#include <fstream>



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

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/arithmetic/mod.hpp>
#include <boost/preprocessor/arithmetic/div.hpp>
#include <boost/preprocessor/control/expr_if.hpp>

typedef void (*FSimpleMatrProd)(double *, unsigned, double *, unsigned, double *, unsigned);

#define IJ_MATR_ELEM(M,I,J) M[(I)*M##S+(J)]
#define KTH_T_TERM(z, k, J_IDX) BOOST_PP_EXPR_IF(k,+) A[k]*B_[k]
#define IJK_T_CYCLE_K(z,J_IDX,n) *C=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);C++;B_+=BS;
#define IJK_T_CYCLE_K_P(z,J_IDX,n) *C+=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);C++;B_+=BS;

#define IJK_T_CYCLE_J(z,I_IDX,n) {auto B_=B; BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K,n); C+=CS-n; A+=AS;}
#define IJK_T_CYCLE_J_P(z,I_IDX,n) {auto B_=B; BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K_P,n); C+=CS-n; A+=AS;}

#define MATR_T_PROD_FUNC_IJK(z,n,P) [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS)\
{ BOOST_PP_REPEAT_1ST(n,IJK_T_CYCLE_J##P,n)}



#include <immintrin.h>

#define BMATR_AVECT_PROD44(B_, A_reg, res_reg) 	 ymm0 = _mm256_loadu_pd(B_); ymm1 = _mm256_loadu_pd(B_ + BS); \
 ymm2 = _mm256_loadu_pd(B_ + 2 * BS); ymm3 = _mm256_loadu_pd(B_ + 3 * BS); \
 ymm0 = _mm256_mul_pd(ymm0, A_reg); ymm1 = _mm256_mul_pd(ymm1, A_reg); \
 ymm2 = _mm256_mul_pd(ymm2, A_reg); ymm3 = _mm256_mul_pd(ymm3, A_reg); \
 ymm0 = _mm256_hadd_pd(ymm0, ymm1); ymm1 = _mm256_hadd_pd(ymm2, ymm3); \
 ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21); ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc); \
 res_reg = _mm256_add_pd(ymm2, ymm3);

#define BM_AV_PROD4_(z,k,unused) BMATR_AVECT_PROD44(B_, Amm##k, ymm0);res = _mm256_add_pd(ymm0, res); B_ += 4;
#define BM_AV_PROD4n(z,pref,n) BOOST_PP_IF(pref,B_ = B + pref *4* BS; C_ += 4;,auto *B_ = B;auto *C_ = C;); BMATR_AVECT_PROD44(B_, AmmF, res); B_ += 4; BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(n,4),1), BM_AV_PROD4_,unused); _mm256_storeu_pd(C_, res);
#define BM_AV_PROD4n_P(z,pref,n) BOOST_PP_IF(pref,B_ = B + pref *4* BS; C_ += 4;,auto *B_ = B;auto *C_ = C;); BMATR_AVECT_PROD44(B_, AmmF, res); B_ += 4; BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(n,4),1), BM_AV_PROD4_,unused);\
ymm1 = _mm256_loadu_pd(C_);res = _mm256_add_pd(res, ymm1); _mm256_storeu_pd(C_, res);


#define KTH_TERM(z, k, unused) +A[k]*B_[k]
#define SUF_SUM(z,DIM,unused) BOOST_PP_REPEAT_FROM_TO_3RD(BOOST_PP_SUB(DIM,BOOST_PP_MOD(DIM, 4)),DIM,KTH_TERM,unused)
#define INC_C_(z,k,DIM) B_ = B+k*4*BS;C_[0]+=0 SUF_SUM(z,DIM,unused);B_+=BS;\
C_[1]+=0 SUF_SUM(z,DIM,unused);B_+=BS;\
C_[2]+=0 SUF_SUM(z,DIM,unused);B_+=BS;\
C_[3]+=0 SUF_SUM(z,DIM,unused);

#define INC_C(z,k,DIM) BOOST_PP_EXPR_IF(BOOST_PP_MOD(DIM, 4),INC_C_(z,k,DIM))

#define SUF_SECTION(z,k,unused) ymm1 = _mm256_loadu_pd(B_ + 4*(1+k)); ymm1 = _mm256_mul_pd(ymm1, Amm##k); ymm0 = _mm256_hadd_pd(ymm0, ymm1);
//ymm1 = _mm256_loadu_pd(B_ + 4); ymm1 = _mm256_mul_pd(ymm1, Amm0); ymm0 = _mm256_hadd_pd(ymm0, ymm1);
//ymm1 = _mm256_loadu_pd(B_ + 8); ymm1 = _mm256_mul_pd(ymm1, Amm1); ymm0 = _mm256_hadd_pd(ymm0, ymm1);
//BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(DIM, 4),1), SUF_SECTION,unused)

//ymm1 = _mm256_loadu_pd(B_ + 4); ymm1 = _mm256_mul_pd(ymm1, Amm0); ymm0 = _mm256_hadd_pd(ymm0, ymm1); \

#define CALC_C_SUF_TERM(z,k,DIM) B_ = B +(DIM-DIM%4+k) * BS;\
ymm0 = _mm256_loadu_pd(B_); ymm0 = _mm256_mul_pd(ymm0, AmmF); \
BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(DIM, 4),1), SUF_SECTION,unused)\
ymm0 = _mm256_hadd_pd(ymm0, ymm0);C_[k] = ((double*)&ymm0)[0] + ((double*)&ymm0)[2] SUF_SUM(z,DIM,unused);

#define CALC_C_SUF(z,k,DIM) C_ += 4; BOOST_PP_REPEAT_2ND(BOOST_PP_MOD(DIM, 4),CALC_C_SUF_TERM,DIM);



#define CALC_C_SUF_TERM_P(z,k,DIM) B_ = B +(DIM-DIM%4+k) * BS;\
ymm0 = _mm256_loadu_pd(B_); ymm1 = _mm256_loadu_pd(B_ + 4);\
ymm0 = _mm256_mul_pd(ymm0, AmmF); ymm1 = _mm256_mul_pd(ymm1, Amm0);\
ymm0 = _mm256_hadd_pd(ymm0, ymm1);ymm0 = _mm256_hadd_pd(ymm0, ymm0);\
C_[k] += ((double*)&ymm0)[0] + ((double*)&ymm0)[2] SUF_SUM(z,DIM,unused);

#define CALC_C_SUF_P(z,k,DIM) C_ += 4; BOOST_PP_REPEAT_2ND(BOOST_PP_MOD(DIM, 4),CALC_C_SUF_TERM_P,DIM);

#define CALC_C_SUFFIX(z,k,DIM) BOOST_PP_EXPR_IF(BOOST_PP_MOD(DIM, 4),CALC_C_SUF(z,k,DIM))
#define CALC_C_SUFFIX_P(z,k,DIM) BOOST_PP_EXPR_IF(BOOST_PP_MOD(DIM, 4),CALC_C_SUF_P(z,k,DIM))

#define BM_AV_PROD_INC_C(z,n,DIM) BM_AV_PROD4n(z, n, DIM); INC_C(z, n, DIM);
#define BM_AV_PROD_INC_C_P(z,n,DIM) BM_AV_PROD4n_P(z, n, DIM); INC_C(z, n, DIM);

#define AMM_DECL(z,k,unused) __m256d Amm##k = _mm256_loadu_pd(A + 4*(1+k));

#define GEN_MUL_BODY(DIM,S)	double *C_last_row = C + CS*DIM;while (C < C_last_row) {\
__m256d ymm0, ymm1, ymm2, ymm3, res; __m256d AmmF = _mm256_loadu_pd(A);\
BOOST_PP_REPEAT_1ST(BOOST_PP_SUB(BOOST_PP_DIV(DIM, 4),1),AMM_DECL,unused)\
BOOST_PP_REPEAT_1ST(BOOST_PP_DIV(DIM, 4),BM_AV_PROD_INC_C##S,DIM)\
CALC_C_SUFFIX##S(z, unused, DIM);A += AS;C += CS;}


//#define MATR_T_PROD_FUNC(z, n, P) MATR_T_PROD_FUNC_IJK(z, n, P)
#define MATR_T_PROD_FUNC(z, n, P) [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS){GEN_MUL_BODY(n,P)}


FSimpleMatrProd prod_funcs_t[] = { //prod_funcs_t[i] i=8..15 is function C=A*transp(B) for matrices ixi
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    MATR_T_PROD_FUNC(z, 8,),
    MATR_T_PROD_FUNC(z, 9,),
    MATR_T_PROD_FUNC(z, 10,),
    MATR_T_PROD_FUNC(z, 11,),
    MATR_T_PROD_FUNC(z, 12,),
    MATR_T_PROD_FUNC(z, 13,),
    MATR_T_PROD_FUNC(z, 14,),
    MATR_T_PROD_FUNC(z, 15,)
    //MATR_T_PROD_FUNC(z, 16,), 
};


FSimpleMatrProd prod_funcs_p_t[] = {  //prod_funcs_p[i] i=8..15 is function C+=A*transp(B) for matrices ixi
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    MATR_T_PROD_FUNC(z, 8, _P),
    MATR_T_PROD_FUNC(z, 9, _P),
    MATR_T_PROD_FUNC(z, 10, _P),
    MATR_T_PROD_FUNC(z, 11, _P),
    MATR_T_PROD_FUNC(z, 12, _P),
    MATR_T_PROD_FUNC(z, 13, _P),
    MATR_T_PROD_FUNC(z, 14, _P),
    MATR_T_PROD_FUNC(z, 15, _P)
    //MATR_T_PROD_FUNC(z, 16,_P), 
};

void mul9(double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
    double *C_last_row = C + CS * 9;
    while (C < C_last_row) {
        __m256d ymm0, ymm1, ymm2, ymm3, res;
        __m256d AmmF = _mm256_loadu_pd(A);
        __m256d Amm0 = _mm256_loadu_pd(A + 4 * (1 + 0));
        auto *B_ = B;
        auto *C_ = C;

        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, AmmF);
        ymm1 = _mm256_mul_pd(ymm1, AmmF);
        ymm2 = _mm256_mul_pd(ymm2, AmmF);
        ymm3 = _mm256_mul_pd(ymm3, AmmF);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        res = _mm256_add_pd(ymm2, ymm3);

        B_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm0);
        ymm1 = _mm256_mul_pd(ymm1, Amm0);
        ymm2 = _mm256_mul_pd(ymm2, Amm0);
        ymm3 = _mm256_mul_pd(ymm3, Amm0);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;

        _mm256_storeu_pd(C_, res);

        B_ = B + 0 * 4 * BS;
        C_[0] += 0 + A[8] * B_[8];
        B_ += BS;
        C_[1] += 0 + A[8] * B_[8];
        B_ += BS;
        C_[2] += 0 + A[8] * B_[8];
        B_ += BS;
        C_[3] += 0 + A[8] * B_[8];

        B_ = B + 1 * 4 * BS;
        C_ += 4;

        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, AmmF);
        ymm1 = _mm256_mul_pd(ymm1, AmmF);
        ymm2 = _mm256_mul_pd(ymm2, AmmF);
        ymm3 = _mm256_mul_pd(ymm3, AmmF);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        res = _mm256_add_pd(ymm2, ymm3);

        B_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm0);
        ymm1 = _mm256_mul_pd(ymm1, Amm0);
        ymm2 = _mm256_mul_pd(ymm2, Amm0);
        ymm3 = _mm256_mul_pd(ymm3, Amm0);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;

        _mm256_storeu_pd(C_, res);

        B_ = B + 1 * 4 * BS;
        C_[0] += 0 + A[8] * B_[8];
        B_ += BS;
        C_[1] += 0 + A[8] * B_[8];
        B_ += BS;
        C_[2] += 0 + A[8] * B_[8];
        B_ += BS;
        C_[3] += 0 + A[8] * B_[8];

        C_ += 4;
        B_ = B + (9 - 9 % 4 + 0) * BS;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + 4);
        ymm0 = _mm256_mul_pd(ymm0, AmmF);
        ymm1 = _mm256_mul_pd(ymm1, Amm0);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm0 = _mm256_hadd_pd(ymm0, ymm0);
        C_[0] = ((double*)&ymm0)[0] + ((double*)&ymm0)[2] + A[8] * B_[8];

        ;
        A += AS;
        C += CS;
    }
}

void mul13(double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
    double *C_last_row = C + CS * 13;
    while (C < C_last_row) {
        __m256d ymm0, ymm1, ymm2, ymm3, res;
       
       /* __m256d Amm0 = _mm256_loadu_pd(A + 4 * (1 + 0));
        __m256d Amm1 = _mm256_loadu_pd(A + 4 * (1 + 1)); */
        auto *B_ = B;
        auto *C_ = C;
        auto *A_ = A;
        __m256d Amm = _mm256_loadu_pd(A_);
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm);
        ymm1 = _mm256_mul_pd(ymm1, Amm);
        ymm2 = _mm256_mul_pd(ymm2, Amm);
        ymm3 = _mm256_mul_pd(ymm3, Amm);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        res = _mm256_add_pd(ymm2, ymm3);

        B_ += 4; A_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm0);
        ymm1 = _mm256_mul_pd(ymm1, Amm0);
        ymm2 = _mm256_mul_pd(ymm2, Amm0);
        ymm3 = _mm256_mul_pd(ymm3, Amm0);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm1);
        ymm1 = _mm256_mul_pd(ymm1, Amm1);
        ymm2 = _mm256_mul_pd(ymm2, Amm1);
        ymm3 = _mm256_mul_pd(ymm3, Amm1);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;

        _mm256_storeu_pd(C_, res);

        B_ = B + 0 * 4 * BS;
        C_[0] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[1] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[2] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[3] += 0 + A[12] * B_[12];

        B_ = B + 1 * 4 * BS;
        C_ += 4;

        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, AmmF);
        ymm1 = _mm256_mul_pd(ymm1, AmmF);
        ymm2 = _mm256_mul_pd(ymm2, AmmF);
        ymm3 = _mm256_mul_pd(ymm3, AmmF);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        res = _mm256_add_pd(ymm2, ymm3);

        B_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm0);
        ymm1 = _mm256_mul_pd(ymm1, Amm0);
        ymm2 = _mm256_mul_pd(ymm2, Amm0);
        ymm3 = _mm256_mul_pd(ymm3, Amm0);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm1);
        ymm1 = _mm256_mul_pd(ymm1, Amm1);
        ymm2 = _mm256_mul_pd(ymm2, Amm1);
        ymm3 = _mm256_mul_pd(ymm3, Amm1);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;

        _mm256_storeu_pd(C_, res);

        B_ = B + 1 * 4 * BS;
        C_[0] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[1] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[2] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[3] += 0 + A[12] * B_[12];

        B_ = B + 2 * 4 * BS;
        C_ += 4;

        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, AmmF);
        ymm1 = _mm256_mul_pd(ymm1, AmmF);
        ymm2 = _mm256_mul_pd(ymm2, AmmF);
        ymm3 = _mm256_mul_pd(ymm3, AmmF);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        res = _mm256_add_pd(ymm2, ymm3);

        B_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm0);
        ymm1 = _mm256_mul_pd(ymm1, Amm0);
        ymm2 = _mm256_mul_pd(ymm2, Amm0);
        ymm3 = _mm256_mul_pd(ymm3, Amm0);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;
        ymm0 = _mm256_loadu_pd(B_);
        ymm1 = _mm256_loadu_pd(B_ + BS);
        ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
        ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
        ymm0 = _mm256_mul_pd(ymm0, Amm1);
        ymm1 = _mm256_mul_pd(ymm1, Amm1);
        ymm2 = _mm256_mul_pd(ymm2, Amm1);
        ymm3 = _mm256_mul_pd(ymm3, Amm1);
        ymm0 = _mm256_hadd_pd(ymm0, ymm1);
        ymm1 = _mm256_hadd_pd(ymm2, ymm3);
        ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
        ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);
        ymm0 = _mm256_add_pd(ymm2, ymm3);

        res = _mm256_add_pd(ymm0, res);
        B_ += 4;

        _mm256_storeu_pd(C_, res);

        B_ = B + 2 * 4 * BS;
        C_[0] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[1] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[2] += 0 + A[12] * B_[12];
        B_ += BS;
        C_[3] += 0 + A[12] * B_[12];

        C_ += 4;
        B_ = B + (13 - 13 % 4 + 0) * BS;
        ymm0 = _mm256_loadu_pd(B_);
        ymm0 = _mm256_mul_pd(ymm0, AmmF);

        ymm1 = _mm256_loadu_pd(B_ + 4); ymm1 = _mm256_mul_pd(ymm1, Amm0);ymm0 = _mm256_hadd_pd(ymm0, ymm1);

        ymm1 = _mm256_loadu_pd(B_ + 8);ymm1 = _mm256_mul_pd(ymm1, Amm1); ymm0 = _mm256_hadd_pd(ymm0, ymm1);

        ymm0 = _mm256_hadd_pd(ymm0, ymm0);
        C_[0] = ((double*)&ymm0)[0] + ((double*)&ymm0)[2] + A[12] * B_[12];

        ;
        A += AS;
        C += CS;
    }
}


//SUM ------------------------------------------------------------------

void  STRASSEN_SIMPLE_SUM(unsigned sz, double *C, unsigned CS,  double *A, unsigned AS, double *B, unsigned BS)
{
	auto A_delta = AS - sz;
	auto B_delta = BS - sz;
	auto C_delta = CS - sz;
	auto sz4 = sz % 4;
	auto *C_last_row = C + CS*sz;
	while (C < C_last_row){
    auto C_last_col = C + (sz - sz4);
		for (; C < C_last_col; A += 4, B += 4, C += 4){
			__m256d ymm0 = _mm256_loadu_pd(A);
			__m256d ymm1 = _mm256_loadu_pd(B);
			ymm0=_mm256_add_pd(ymm0, ymm1);
			_mm256_storeu_pd(C, ymm0);
		}
		for (C_last_col += sz4; C < C_last_col; ++A, ++B, ++C)
			*C = *A + *B;
		A += A_delta;
		B += B_delta;
		C += C_delta;
	}
}; 

/*void  STRASSEN_SIMPLE_SUM(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
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

void  STRASSEN_SIMPLE_SUB(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
	auto A_delta = AS - sz;
	auto B_delta = BS - sz;
	auto C_delta = CS - sz;
	auto sz4 = sz % 4;
	auto *C_last_row = C + CS*sz;
	while (C < C_last_row){ 
    auto C_last_col = C + (sz - sz4);
		for (; C < C_last_col; A += 4, B += 4, C += 4){
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


/*void  STRASSEN_SIMPLE_SUB(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
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


void STRASSEN_MUL_SUFFIX(unsigned sz, unsigned CS, double *C00, double *C01, double *C10, double *C11, double *S00, double *S01, double *S10)
{
	auto C_delta = CS - sz;
	auto C_last_row = C00 + CS*sz;
	auto sz4 = sz % 4;
	while (C00 < C_last_row) {
		for (auto C_last_col = C00 + (sz - sz4); C00 < C_last_col;C00+=4,C01+=4,C10+=4,C11+=4,S00+=4,S01+=4,S10+=4)
		{  
			__m256d ymm00 = _mm256_loadu_pd(C00);
			__m256d ymm01 = _mm256_loadu_pd(C01);
			__m256d ymm10 = _mm256_loadu_pd(C10);
			__m256d ymm11 = _mm256_loadu_pd(C11);
			__m256d zmm00 = _mm256_loadu_pd(S00);
			__m256d zmm01 = _mm256_loadu_pd(S01);
			__m256d zmm10 = _mm256_loadu_pd(S10);
			ymm01 = _mm256_add_pd(ymm01,zmm00);
			zmm10 = _mm256_add_pd(zmm10,ymm01);
			ymm00 = _mm256_add_pd(zmm00,ymm00);
			_mm256_storeu_pd(C00,ymm00);
			ymm00 = _mm256_add_pd(ymm01,ymm11);
			ymm00 = _mm256_sub_pd(ymm00,zmm01);
			_mm256_storeu_pd(C01,ymm00);
			ymm00 = _mm256_sub_pd(zmm10,ymm10);
			_mm256_storeu_pd(C10,ymm00);
			ymm00 = _mm256_add_pd(ymm11,zmm10);
			_mm256_storeu_pd(C11,ymm00);
		}
	  for (auto C_last_col = C00 + sz4; C00 < C_last_col; ++C00, ++C01, ++C10, ++C11, ++S00, ++S01, ++S10)
	  {
		  *C01 += *S00;
		  *S10 += *C01;
		  *C00 += *S00;
		  *C01 += *C11 - *S01;
		  *C10 = *S10 - *C10;
		  *C11 += *S10;
/*		  auto ymm00 = *C00;
		  auto ymm01 = *C01;
		  auto ymm10 = *C10;
		  auto ymm11 = *C11;
		  auto zmm00 = *S00;
		  auto zmm01 = *S01;
		  auto zmm10 = *S10;
		  ymm01 = ymm01 + zmm00;
		  zmm10 = zmm10 + ymm01;
		  *C00 = zmm00 + ymm00;
		  *C01 = ymm01 + ymm11 - zmm01;
		  *C10 = zmm10 - ymm10;
		  *C11 = ymm11 + zmm10;*/
	  }
	  C00 += C_delta;
		C01 += C_delta; 
		C10 += C_delta;
		C11 += C_delta;
	}
};

void C_add_a_mul_B(unsigned sz, double *C, double a, double *B_)
{
	for (auto B_end = B_ + sz; B_ < B_end; ++B_, ++C) *C += a*(*B_);
};

double A_mul_B(unsigned sz, double *A_,  double *B_)
{
	auto res = (*A_)*(*B_);
	auto B_end = B_ + sz;
	for (++A_, ++B_; B_ < B_end; ++A_, ++B_) res += (*A_)*(*B_);
	return res;
};

void strassen_padding_calc(unsigned SZ, double *buf, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
	auto SZ_minus_one = SZ - 1;// LAST
	//coping last elements of B to buf (hereafter referred to as B[][LAST])
	for (auto B_ = buf, B_end = buf + SZ_minus_one, B_cur_last = B + SZ_minus_one; B_ < B_end; ++B_, B_cur_last += BS)
		*B_ = *B_cur_last;
	for (auto B_ = B + SZ_minus_one*BS, C_end = C + CS*SZ_minus_one; C < C_end; C += CS, A += AS){
		//C[I][] += A[I][LAST]*B[][LAST]
		C_add_a_mul_B(SZ_minus_one, C, A[SZ_minus_one], buf);
		// C[I][LAST]=A[i][]*B[LAST][]
		C[SZ_minus_one] = A_mul_B(SZ, A, B_);
	}
  for (auto  C_end = C + SZ; C < C_end; ++C, B += BS)*C = A_mul_B(SZ, A, B);// C[LAST][I]=A[LAST][]*B[i][]
};

void  STRASSEN_RECUR_MUL(unsigned SZ, double *buf, double *C00, unsigned CS, double *A00, unsigned AS, double *B00, unsigned BS)
{
	
	if (SZ < 16){ 
      prod_funcs_t[SZ](C00, CS, A00, AS, B00, BS); 
  }
	else{
		
		auto sz = SZ / 2;

		auto a10 = AS*sz;
		auto b10 = BS*sz;
		auto c10 = CS*sz;

		auto A01 = A00 + sz;
		auto A10 = A00 + AS*sz;
		auto A11 = A10 + sz;

		// B - transposed matrix!!
		auto B01 = B00 + BS*sz;
		auto B10 = B00 + sz;
		auto B11 = B01 + sz;

		auto C01 = C00 + sz;
		auto C10 = C00 + CS*sz;
		auto C11 = C10 + sz;

		auto subm_size = sz*sz;

		
		auto *S00 = buf; buf += subm_size;
		auto *S01 = buf; buf += subm_size;
		auto *S10 = buf; buf += subm_size;
		auto *T00 = buf; buf += subm_size;
		STRASSEN_SIMPLE_SUM(sz, S00, sz, A10, AS, A11, AS);
		STRASSEN_SIMPLE_SUB(sz, T00, sz, B01, BS, B00, BS);
		STRASSEN_RECUR_MUL(sz, buf, C11, CS, S00, sz, T00, sz);
		STRASSEN_SIMPLE_SUB(sz, T00, sz, B11, BS, T00, sz);
		STRASSEN_SIMPLE_SUB(sz, S00, sz, S00, sz, A00, AS);
		STRASSEN_RECUR_MUL(sz, buf, C01, CS, S00, sz, T00, sz);
		STRASSEN_SIMPLE_SUB(sz, T00, sz, T00, sz, B10, BS);
		STRASSEN_SIMPLE_SUB(sz, S00, sz, S00, sz, A01, AS);
		
		STRASSEN_RECUR_MUL(sz, buf, S01, sz, S00, sz, B11, BS);
		STRASSEN_RECUR_MUL(sz, buf, C10, CS, A11, AS, T00, sz);
		STRASSEN_SIMPLE_SUB(sz, T00, sz, B11, BS, B01, BS);
		STRASSEN_SIMPLE_SUB(sz, S00, sz, A00, AS, A10, AS);
		
		STRASSEN_RECUR_MUL(sz, buf, S10, sz, S00, sz, T00, sz);
		STRASSEN_RECUR_MUL(sz, buf, S00, sz, A00, AS, B00, BS);
		STRASSEN_RECUR_MUL(sz, buf, C00, CS, A01, AS, B10, BS);
		STRASSEN_MUL_SUFFIX(sz, CS, C00, C01, C10, C11, S00, S01, S10);
		if (SZ % 2)
			strassen_padding_calc(SZ, buf, C00, CS, A00, AS, B00, BS);
	}
}

void transp(unsigned SZ, double(*B))
{
    for (unsigned i = 0; i<SZ; i++)
        for (unsigned j = i + 1; j<SZ; j++)
        {
            auto b = B[i*SZ + j];
            B[i*SZ + j] = B[j*SZ + i];
            B[j*SZ + i] = b;
        }

}


void  strassen_mul(unsigned SZ, double(*C), double(*A), double(*B))
{
	transp(SZ,B);
	boost::scoped_array<double> buf(new double[(8 * SZ*SZ) / 3 +1]);
	STRASSEN_RECUR_MUL(SZ, buf.get(), C, SZ, A, SZ,  B, SZ);
	transp(SZ,B);
};


void simple_mul(unsigned SZ, double(*C), double (*A), double (*B))
{
    for(unsigned i = 0;i<SZ;++i)
        for (unsigned j = 0; j < SZ; ++j)
        {
            C[i*SZ+j] = A[i*SZ] * B[j];
                for(unsigned k=1;k<SZ;++k)
                   C[i*SZ + j]+= A[i*SZ + k] * B[k*SZ + j];
        }
   
};

double *enlage_matrix(unsigned SZ, unsigned SZ_new, boost::scoped_array<double> &M_, double *M)
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

void copy_on(unsigned SZ, unsigned SZ_new, double *M, double *M_new)
{
    auto nbytescopy = SZ * sizeof(M[0]);
    for (auto M_end = M + SZ*SZ; M < M_end; M += SZ, M_new += SZ_new)  memcpy(M, M_new, nbytescopy);
};

void block_mul_t(unsigned SZ_,double(*C_), double(*A_), double(*B_))
{

    double *C, *A, *B;
    auto SZ = SZ_;
    boost::scoped_array<double> c, a, b;
    const unsigned bs = 8;
    if (SZ % bs) {
        SZ = SZ_ - SZ_ % bs + bs;
        c.reset(new double[SZ*SZ]);
        C = c.get();
        A = enlage_matrix(SZ_, SZ, a, A_);
        B = enlage_matrix(SZ_, SZ, b, B_);
    }
    else
    {
        C = C_;
        A = A_;
        B = B_;
    }
    transp(SZ,B);
 
	FSimpleMatrProd f_t_8 = prod_funcs_t[bs];
	FSimpleMatrProd f_t_8_p = prod_funcs_p_t[bs];

    for (unsigned i = 0; i<SZ; i += bs)
        for (unsigned j = 0; j < SZ; j += bs)
        {
#define IJ_M_PTR(M,I,J) &M[I*SZ+J]
            f_t_8(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, 0), SZ, IJ_M_PTR(B, j, 0), SZ);
            for (unsigned k = bs; k<SZ; k += bs)
                f_t_8_p(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, k), SZ, IJ_M_PTR(B, j, k), SZ);
        }
    if(SZ==SZ_)transp(SZ,B);
    else copy_on(SZ_, SZ, C_, C);


}

template <unsigned SZ>
double matr_dif(double(*A)[SZ], double(*B)[SZ])
{
	double avr = 0;
	double dif = fabs(A[0][0] - B[0][0]);
	for (unsigned i = 0; i<SZ; ++i)
		for (unsigned j = 0; j < SZ; ++j)
		{
		  avr += fabs(A[i][j]) + fabs(B[i][j]);
		  double d = fabs(A[i][j] - B[i][j]);
		  if (d>dif)dif = d;
		}
	//if (avr == 0)return 0;
	return 2*(dif*SZ)*SZ / avr;
};

const unsigned matr_size = 13;
typedef double row_t[matr_size];

template <unsigned SZ>
void print_matrix(char *s, double(*M)[SZ])
{
    std::cout << s;
    std::cout << "\n";
    for (unsigned i = 0; i<matr_size; i++) {
        for (unsigned j = 0; j < matr_size; j++)
            std::cout << M[i][j] << " ";
        std::cout << "\n";
    };
    std::cout << "\n";
}
unsigned main()
{
#define ALLOC_MATR(M)	boost::scoped_array<row_t> M##_(new row_t[matr_size]); row_t *M = M##_.get()
	ALLOC_MATR(a);
	ALLOC_MATR(b);
	ALLOC_MATR(S);
	ALLOC_MATR(R);
  ALLOC_MATR(ST);
  //ALLOC_MATR(B);
  ALLOC_MATR(BT);
  srand((unsigned)time(NULL));
    for(unsigned i=0;i<matr_size;i++)
        for (unsigned j = 0; j < matr_size; j++){
		a[i][j] = rand()/1000;
		b[i][j] = rand() / 1000;
        };
	std::cout << "press enter to start\n";
	getchar();
	std::cout << "wait...\n";
  int n_repeats=1;
#define _CUB(a) ((a)*(a)*(a)+1)
#define _REPEAT for(n_repeats=0;n_repeats<_CUB(2.5*1024.0/matr_size);n_repeats++)
#define _REPEAT
/*  prod_funcs_t[8] = mul8;
  prod_funcs_p_t[8] = mul8p;
  prod_funcs_t[9] = mul9;
  prod_funcs_p_t[9] = mul9p;
  prod_funcs_t[10] = mul10;*/
//prod_funcs_t[13] = mul13; 
  auto t_ = clock();
  simple_mul(matr_size, reinterpret_cast<double *>(S), reinterpret_cast<double *>(a), reinterpret_cast<double *>(b));
  double d_simple = (double)(clock() - t_) / CLOCKS_PER_SEC;
  t_ = clock();
  _REPEAT strassen_mul(matr_size, reinterpret_cast<double *>(ST), reinterpret_cast<double *>(a), reinterpret_cast<double *>(b));
  double d_strassen = (double)(clock() - t_) / CLOCKS_PER_SEC;
  t_ = clock();
  //_REPEAT recur_mul(R, a, b);
  double d_recur = (double)(clock() - t_) / CLOCKS_PER_SEC;
  /*t_ = clock();
  //block_mul(B, a, b);
  double d_block = (double)(clock() - t_) / CLOCKS_PER_SEC;*/
  
  t_ = clock();
  _REPEAT block_mul_t(matr_size, reinterpret_cast<double *>(BT), reinterpret_cast<double *>(a), reinterpret_cast<double *>(b));
  double d_block_t = (double)(clock() - t_) / CLOCKS_PER_SEC;
  double calc_dif =2000.0/matr_size;
  calc_dif = calc_dif*calc_dif*calc_dif / n_repeats;

  std::cout << "size " << matr_size << "\n";
  std::cout << "n repeats " << n_repeats << "\n";

  //std::cout << "simple " << d_simple << "\n";
  //std::cout << "recur " << d_recur << "\n";
  //std::cout << "block " << d_block << "\n";
  std::cout << "block t " << d_block_t/n_repeats << "\n";
  std::cout << "strassen t " << d_strassen / n_repeats << "\n";
  std::cout << "block d " << d_block_t*calc_dif << "\n";
  std::cout << "strassen d " << d_strassen*calc_dif << "\n";
    std::cout << "diff simple strassen " << matr_dif(S, ST) << "\n";

  //std::cout << "diff block block transp " << matr_dif(B, BT) << "\n";
  std::cout << "diff block transp strassen " << matr_dif(BT, ST) << "\n";
  //std::cout << "diff block recur " << matr_dif(B, R) << "\n";
  //std::cout << "diff block strassen " << matr_dif(B, ST) << "\n";
  //std::cout << "ratio simple/strassen  " << d_simple / d_strassen << "\n";
  std::cout << "ratio block transp/strassen  " << d_block_t / d_strassen << "\n";
  //std::cout << "ratio recur/strassen  " << d_recur / d_strassen << "\n";
     std::cout << "\n";
  print_matrix("A", a);
  print_matrix("B", b);

  std::cout << "\n";
  //print_matrix("block", B);
  print_matrix("simple", S);
  print_matrix("block", BT);
  print_matrix("strassen", ST);
/* */
    getchar();
    return 0;
}

