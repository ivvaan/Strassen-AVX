#include "small_matr_funcs.h"
#include "base_operations.h"
#include "utils.h"

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/arithmetic/mod.hpp>
#include <boost/preprocessor/arithmetic/div.hpp>
#include <boost/preprocessor/control/expr_if.hpp>
#include <immintrin.h>







#define RESTRICTED_PTR __restrict
//#define RESTRICTED_PTR 

#define IJ_MATR_ELEM(M,I,J) M[(I)*M##S+(J)]
#define KTH_T_TERM(z, k, J_IDX) BOOST_PP_EXPR_IF(k,+) A[k]*B_[k]
#define IJK_T_CYCLE_K(z,J_IDX,n) BOOST_PP_EXPR_IF(J_IDX,C++;B_+=BS;) *C=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);
#define IJK_T_CYCLE_K_P(z,J_IDX,n) BOOST_PP_EXPR_IF(J_IDX,C++;B_+=BS;) *C+=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);

#define IJK_T_CYCLE_J(z,I_IDX,n) BOOST_PP_EXPR_IF(I_IDX,C+=CS-n+1;A+=AS;) {auto *RESTRICTED_PTR B_=B;  BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K,n); }
#define IJK_T_CYCLE_J_P(z,I_IDX,n) BOOST_PP_EXPR_IF(I_IDX,C+=CS-n+1;A+=AS;) {auto *RESTRICTED_PTR B_=B;  BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K_P,n); }

#define MATR_T_PROD_FUNC_IJK(n,P) [](double * RESTRICTED_PTR C,unsigned CS,double  * RESTRICTED_PTR A,unsigned AS,double * RESTRICTED_PTR B,unsigned BS)\
{ BOOST_PP_REPEAT_1ST(n,IJK_T_CYCLE_J##P,n)}

#define MATR_T_PROD_SRUCT_IJK(n) {MATR_T_PROD_FUNC_IJK(n,),MATR_T_PROD_FUNC_IJK(n,_P)}




#define BMATR_AVECT_PROD44(B_, A_reg, res_reg) 	 ymm0 = _mm256_loadu_pd(B_); ymm1 = _mm256_loadu_pd(B_ + BS); \
 ymm2 = _mm256_loadu_pd(B_ + 2 * BS); ymm3 = _mm256_loadu_pd(B_ + 3 * BS); \
 ymm0 = _mm256_mul_pd(ymm0, A_reg); ymm1 = _mm256_mul_pd(ymm1, A_reg); \
 ymm2 = _mm256_mul_pd(ymm2, A_reg); ymm3 = _mm256_mul_pd(ymm3, A_reg); \
 ymm0 = _mm256_hadd_pd(ymm0, ymm1); ymm1 = _mm256_hadd_pd(ymm2, ymm3); \
 ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21); ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc); \
 res_reg = _mm256_add_pd(ymm2, ymm3);

#define BM_AV_PROD4_(z,k,unused) BMATR_AVECT_PROD44(B_, Amm##k, ymm0);res = _mm256_add_pd(ymm0, res); B_ += 4;
#define BM_AV_PROD4n(z,pref,n) BOOST_PP_IF(pref,B_ = B + pref *4* BS; C_ += 4;,auto *RESTRICTED_PTR B_ = B;auto *RESTRICTED_PTR C_ = C;); BMATR_AVECT_PROD44(B_, AmmF, res); B_ += 4; BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(n,4),1), BM_AV_PROD4_,unused); _mm256_storeu_pd(C_, res);
#define BM_AV_PROD4n_P(z,pref,n) BOOST_PP_IF(pref,B_ = B + pref *4* BS; C_ += 4;,auto *RESTRICTED_PTR B_ = B;auto *RESTRICTED_PTR C_ = C;); BMATR_AVECT_PROD44(B_, AmmF, res); B_ += 4; BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(n,4),1), BM_AV_PROD4_,unused);\
ymm1 = _mm256_loadu_pd(C_);res = _mm256_add_pd(res, ymm1); _mm256_storeu_pd(C_, res);


#define KTH_TERM(z, k, unused) +A[k]*B_[k]
#define SUF_SUM(z,DIM,unused) BOOST_PP_REPEAT_FROM_TO_3RD(BOOST_PP_SUB(DIM,BOOST_PP_MOD(DIM, 4)),DIM,KTH_TERM,unused)
#define INC_C_(z,k,DIM) B_ = B+k*4*BS;C_[0]+=0 SUF_SUM(z,DIM,unused);B_+=BS;\
C_[1]+=0 SUF_SUM(z,DIM,unused);B_+=BS;\
C_[2]+=0 SUF_SUM(z,DIM,unused);B_+=BS;\
C_[3]+=0 SUF_SUM(z,DIM,unused);

#define INC_C(z,k,DIM) BOOST_PP_EXPR_IF(BOOST_PP_MOD(DIM, 4),INC_C_(z,k,DIM))

#define SUF_SECTION(z,k,unused) ymm1 = _mm256_loadu_pd(B_ + 4*(1+k)); ymm1 = _mm256_mul_pd(ymm1, Amm##k); ymm0 = _mm256_hadd_pd(ymm0, ymm1);

#define CALC_C_SUF_TERM(z,k,DIM) B_ = B +(DIM-DIM%4+k) * BS;\
ymm0 = _mm256_loadu_pd(B_); ymm0 = _mm256_mul_pd(ymm0, AmmF); \
BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(DIM, 4),1), SUF_SECTION,unused)\
ymm0 = _mm256_hadd_pd(ymm0, ymm0);C_[k] = ((double*)&ymm0)[0] + ((double*)&ymm0)[2] SUF_SUM(z,DIM,unused);

#define CALC_C_SUF(z,k,DIM) C_ += 4; BOOST_PP_REPEAT_2ND(BOOST_PP_MOD(DIM, 4),CALC_C_SUF_TERM,DIM);



#define CALC_C_SUF_TERM_P(z,k,DIM) B_ = B +(DIM-DIM%4+k) * BS;\
ymm0 = _mm256_loadu_pd(B_); ymm0 = _mm256_mul_pd(ymm0, AmmF); \
BOOST_PP_REPEAT_3RD(BOOST_PP_SUB(BOOST_PP_DIV(DIM, 4),1), SUF_SECTION,unused)\
ymm0 = _mm256_hadd_pd(ymm0, ymm0);C_[k] += ((double*)&ymm0)[0] + ((double*)&ymm0)[2] SUF_SUM(z,DIM,unused);

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

#define MATR_T_PROD_FUNC(n, P) [](double * RESTRICTED_PTR C,unsigned CS,double * RESTRICTED_PTR A,unsigned AS,double * RESTRICTED_PTR B,unsigned BS){GEN_MUL_BODY(n,P)}

#define SMALL_MATR_T_PROD_STRUCT(n) MATR_T_PROD_SRUCT_IJK(n)
#define MATR_T_PROD_STRUCT(n) {MATR_T_PROD_FUNC(n,),MATR_T_PROD_FUNC(n,_P)}

fix_size_prod_funcs prod_funcs_t[] = {
    //prod_funcs_t[i].multiply is function C=A*transp(B) for matrices ixi
    //prod_funcs_t[i].plus_multiply is function C+=A*transp(B) for matrices ixi
    { nullptr, nullptr },
    { [](double *C,unsigned,double *A,unsigned,double *B,unsigned) { *C = A[0] * B[0]; },[](double *C,unsigned,double *A,unsigned,double *B,unsigned) { *C += A[0] * B[0]; } },
    SMALL_MATR_T_PROD_STRUCT(2),
    SMALL_MATR_T_PROD_STRUCT(3),
    SMALL_MATR_T_PROD_STRUCT(4),
    SMALL_MATR_T_PROD_STRUCT(5),
    SMALL_MATR_T_PROD_STRUCT(6),
    SMALL_MATR_T_PROD_STRUCT(7),
    MATR_T_PROD_STRUCT(8),
    MATR_T_PROD_STRUCT(9),
    MATR_T_PROD_STRUCT(10),
    MATR_T_PROD_STRUCT(11),
    MATR_T_PROD_STRUCT(12),
    MATR_T_PROD_STRUCT(13),
    MATR_T_PROD_STRUCT(14),
    MATR_T_PROD_STRUCT(15),
    MATR_T_PROD_STRUCT(16)
};

bool small_matr_mul(unsigned SZ_, double *C_, double *A_, double *B_)
{
    if (SZ_ < count_of(prod_funcs_t)) {
        if (SZ_ > 0) {
            inplace_transpose(SZ_, B_);
            prod_funcs_t[SZ_].multiply(C_, SZ_, A_, SZ_, B_, SZ_);
            inplace_transpose(SZ_, B_);
        }
        return true;
    }
    return false;
}


