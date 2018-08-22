// Strassen.cpp : Defines the entry pounsigned for the console application.
//

//#include "stdafx.h"
#include <stdio.h>
#include <tchar.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>




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
#define IJK_T_CYCLE_K(z,J_IDX,n) BOOST_PP_EXPR_IF(J_IDX,C++;B_+=BS;) *C=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);
#define IJK_T_CYCLE_K_P(z,J_IDX,n) BOOST_PP_EXPR_IF(J_IDX,C++;B_+=BS;) *C+=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);

#define IJK_T_CYCLE_J(z,I_IDX,n) BOOST_PP_EXPR_IF(I_IDX,C+=CS-n+1;A+=AS;) {auto B_=B;  BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K,n); }
#define IJK_T_CYCLE_J_P(z,I_IDX,n) BOOST_PP_EXPR_IF(I_IDX,C+=CS-n+1;A+=AS;) {auto B_=B;  BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K_P,n); }

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


#define SMALL_MATR_T_PROD_FUNC(z, n, P) MATR_T_PROD_FUNC_IJK(z, n, P)
//#define MATR_T_PROD_FUNC(z, n, P) MATR_T_PROD_FUNC_IJK(z, n, P)
#define MATR_T_PROD_FUNC(z, n, P) [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS){GEN_MUL_BODY(n,P)}


FSimpleMatrProd prod_funcs_t[] = { //prod_funcs_t[i] i=8..15 is function C=A*transp(B) for matrices ixi
    nullptr, 
    [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS) { *C = A[0] * B[0]; },
    SMALL_MATR_T_PROD_FUNC(z, 2,),
    SMALL_MATR_T_PROD_FUNC(z, 3,),
    SMALL_MATR_T_PROD_FUNC(z, 4,),
    SMALL_MATR_T_PROD_FUNC(z, 5,),
    SMALL_MATR_T_PROD_FUNC(z, 6,),
    SMALL_MATR_T_PROD_FUNC(z, 7,),
    MATR_T_PROD_FUNC(z, 8,),
    MATR_T_PROD_FUNC(z, 9,),
    MATR_T_PROD_FUNC(z, 10,),
    MATR_T_PROD_FUNC(z, 11,),
    MATR_T_PROD_FUNC(z, 12,),
    MATR_T_PROD_FUNC(z, 13,),
    MATR_T_PROD_FUNC(z, 14,),
    MATR_T_PROD_FUNC(z, 15,),
    MATR_T_PROD_FUNC(z, 16,)
};


FSimpleMatrProd prod_funcs_p_t[] = {  //prod_funcs_p[i] i=8..15 is function C+=A*transp(B) for matrices ixi
    nullptr,
    [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS) { *C += A[0] * B[0]; },
    SMALL_MATR_T_PROD_FUNC(z, 2,_P),
    SMALL_MATR_T_PROD_FUNC(z, 3,_P),
    SMALL_MATR_T_PROD_FUNC(z, 4,_P),
    SMALL_MATR_T_PROD_FUNC(z, 5,_P),
    SMALL_MATR_T_PROD_FUNC(z, 6,_P),
    SMALL_MATR_T_PROD_FUNC(z, 7,_P),
    MATR_T_PROD_FUNC(z, 8, _P),
    MATR_T_PROD_FUNC(z, 9, _P),
    MATR_T_PROD_FUNC(z, 10, _P),
    MATR_T_PROD_FUNC(z, 11, _P),
    MATR_T_PROD_FUNC(z, 12, _P),
    MATR_T_PROD_FUNC(z, 13, _P),
    MATR_T_PROD_FUNC(z, 14, _P),
    MATR_T_PROD_FUNC(z, 15, _P),
    MATR_T_PROD_FUNC(z, 16,_P) 
};

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))
template <typename T, std::size_t N>
constexpr std::size_t count_of(T const (&)[N]) noexcept
{
    return N;
}



void get_num_calc(unsigned SZ, double (&num_calc)[12])
{
    const unsigned small_matr_size = 130;
    for (unsigned i = 0; i < count_of(num_calc); i++)
        num_calc[i]=0;
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
  return wgt + (enlarge? enlarge_per_el*SZ*SZ : 0);
};

unsigned get_best_enl(unsigned SZ, unsigned range)
{
	double min_v = get_weight(SZ, 0);
	unsigned min_e = 0;
	for (unsigned i = 1; i < range; ++i){
		double cur_v = get_weight(SZ, i);
		if (cur_v < min_v)
			{ min_v = cur_v; min_e = i; }
	}
	return min_e;
};

//SUM ------------------------------------------------------------------

void  matrix_sum(unsigned sz, double *C, unsigned CS,  double *A, unsigned AS, double *B, unsigned BS)
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


void strassen_mul_suffix(unsigned sz, unsigned CS, double *C00, double *C01, double *C10, double *C11, double *S00, double *S01, double *S10)
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
	  }
	  C00 += C_delta;
		C01 += C_delta; 
		C10 += C_delta;
		C11 += C_delta;
	}
};

void C_add_a_mul_B(unsigned sz, double *C, double a, double *B)
{
	//for (auto B_end = B + sz; B < B_end; ++B, ++C) *C += a*(*B);
    double a_[] = { a,a };
    __m256d amm = _mm256_broadcastsd_pd(_mm_loadu_pd(a_));
    auto B_end = B + sz-sz%4;
    while (B < B_end) {   
        __m256d bmm = _mm256_mul_pd(_mm256_loadu_pd(B),amm);
        __m256d cmm = _mm256_add_pd(_mm256_loadu_pd(C), bmm);
        _mm256_storeu_pd(C, cmm);
        B+=4; C+=4;
    }
    B_end += sz % 4;
    while (B < B_end) *C++ += a*(*B++);
};

double vectA_mul_vectB(unsigned sz, double *A,  double *B)
{
/*	auto res = (*A)*(*B);
	auto B_end = B + sz;
	for (++A, ++B; B < B_end; ++A, ++B) res += (*A)*(*B);
	return res;
 */ 
  auto A_end = A + sz%4;
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
  return res+ ((double*)&smm)[0] + ((double*)&smm)[2];
};

void matrA_mul_vectB(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B)
{
    for (auto C_end=C+sz*CS; C < C_end; A += AS,C += CS)
        *C = vectA_mul_vectB(sz, A, B);
}

void strassen_padding_calc(unsigned SZ, double *buf, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
	auto SZ_minus_one = SZ - 1;// LAST
	//coping last column of B  (hereafter referred to as B[][LAST]) to buf
	for (auto B_ = buf, B_end = buf + SZ_minus_one, B_cur_last = B + SZ_minus_one; B_ < B_end; ++B_, B_cur_last += BS)
		*B_ = *B_cur_last;   //buf[]=B[][LAST]

	for (auto B_ = B + SZ_minus_one*BS, C_end = C + CS*SZ_minus_one; C < C_end; C += CS, A += AS){
    	C_add_a_mul_B(SZ_minus_one, C, A[SZ_minus_one], buf);   //C[I][] += A[I][LAST]*B[][LAST] 
	    C[SZ_minus_one] = vectA_mul_vectB(SZ, A, B_);       // C[I][LAST]=A[i][]*B[LAST][], B[LAST][] is last row of B 
	}
	for (auto C_end =C + SZ; C < C_end; ++C, B += BS)*C = vectA_mul_vectB(SZ, A, B);// C[LAST][I]=A[LAST][]*B[i][]
	
};

#ifdef _DEBUG
double *buf_max;
#endif

void  strassen_recur_mul_by_transposed(unsigned SZ, double *buf, double *C00, unsigned CS, double *A00, unsigned AS, double *B00, unsigned BS)
{
	
	if (SZ < 16){ 
      prod_funcs_t[SZ](C00, CS, A00, AS, B00, BS); 
  }
	else{
		
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

typedef void(*FSimpleInvFunc)(double *, unsigned, double *, unsigned);

FSimpleInvFunc inv_funcs[] = {
    nullptr,
    [](double *I,unsigned IS,double *A,unsigned AS) {I[0] = 1 / A[0]; },
    [](double *I,unsigned IS,double *A,unsigned AS) {
        auto det = A[0] * A[AS + 1] - A[1] * A[AS]; 
        I[0] = A[AS + 1] / det; I[1] = -A[1] / det; 
        I[IS] = -A[AS] / det; I[IS + 1] = A[0] / det;
    },
    [](double *I,unsigned IS,double *M,unsigned MS) {
        auto M1 = M + MS;
        auto M2 = M1 + MS;
        auto A = M1[1] * M2[2] - M1[2] * M2[1];
        auto B = M1[2] * M2[0] - M1[0] * M2[2];
        auto C = M1[0] * M2[1] - M1[1] * M2[0];
        auto det_rev = 1 / (M[0] * A + M[1] * B + M[2] * C);
        I[0] = det_rev*A;
        I[1] = det_rev*(M[2] * M2[1] - M[1] * M2[2]);
        I[2] = det_rev*(M[1] * M1[2] - M[2] * M1[1]);
        I += IS;
        I[0] = det_rev*B;
        I[1] = det_rev*(M[0] * M2[2] - M[2] * M2[0]);
        I[2] = det_rev*(M[2] * M1[0] - M[0] * M1[2]);
        I += IS;
        I[0] = det_rev*C;
        I[1] = det_rev*(M[1] * M2[0] - M[0] * M2[1]);
        I[2] = det_rev*(M[0] * M1[1] - M[1] * M1[0]);
    }
};

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
void transp_and_set_last_col_zero(unsigned SZ, double *BT, double *B, unsigned BS)
{
    for (unsigned i = 0; i<SZ - 1; i++)
        for (unsigned j = 0; j<SZ; j++)
            BT[j*SZ + i] = B[i*BS + j];
    for (unsigned j = 1; j<SZ + 1; j++)
        BT[j*SZ - 1] = 0;
}



void strassen_transp_and_mul(unsigned SZ, double *buf, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)  //
{
    auto *BT = buf; buf += SZ*SZ;
    _ASSERT(buf<buf_max);
    transp(SZ, BT, B, BS);
    strassen_recur_mul_by_transposed(SZ, buf, C, CS, A, AS, BT, SZ);
};


void change_sign(unsigned SZ, double *M, unsigned MS)
{
    auto M_delta = MS - SZ;
    for (auto M_last_row = M + SZ*MS; M < M_last_row; M+=M_delta) 
        for (auto M_last_col = M + SZ; M < M_last_col; M++)
            *M = -*M;
    
}

void  strassen_recur_inv(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS);

void  strassen_recur_inv_even(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
{
        auto sz = SZ / 2;
        auto A = A_;
        auto B = A + sz;
        auto C = A + AS*sz;
        auto D = C + sz;

        auto I00 = I;
        auto I01 = I00 + sz;
        auto I10 = I00 + IS*sz;
        auto I11 = I10 + sz;

        /*


        | I00  I01 |   | A   B | -1
        |          | = |       |
        | I10  I11 |   | C   D |
        see https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion for formulas
        */

        auto subm_size = sz*sz;

        auto AI = buf; buf += subm_size;
        strassen_recur_inv(sz, buf, AI, sz, A, AS); // AI - inverse A 
        auto T1 = buf; buf += subm_size;
        strassen_transp_and_mul(sz, buf, T1, sz, AI, sz, B, AS);
        auto T2 = buf; buf += subm_size;
        strassen_transp_and_mul(sz, buf, T2, sz, C, AS, T1, sz);
        matrix_sub(sz, T2, sz,T2, sz, D, AS ); //T2=C*AI*B-D (T2=-Z where Z=D-C*AI*B)
        strassen_recur_inv(sz, buf, I11, IS, T2, sz); //-ZI calculated and stored in I11
        
        strassen_transp_and_mul(sz, buf, I01, IS, T1, sz, I11, IS);  //I01=-AI*B*ZI
        
        transp(sz, T2, AI, sz); // T2=AIT
        strassen_recur_mul_by_transposed(sz, buf, T1, sz, T2, sz, C, AS);  //T1=AIT*CT
        
        strassen_recur_mul_by_transposed(sz, buf, T2, sz, I01, IS, T1, sz); //T2=-AI*B*ZI*C*AI

        matrix_sub(sz, I00, IS, AI, sz, T2, sz);  // I00=AI+AI*B*ZI*C*AI
        strassen_recur_mul_by_transposed(sz, buf, I10, IS, I11, IS, T1, sz); // I10=-ZI*C*AI
        change_sign(sz, I11, IS);  //I11 = ZI
 
}

void copy_on(unsigned row_len, unsigned row_numb, double *T, unsigned TS, double *F, unsigned FS)  // copy submatrix row_len*row_numb from F to T
{
    auto nbytescopy = row_len * sizeof(T[0]);
    for (auto T_end = T + TS*row_numb; T < T_end; T += TS, F += FS)  memcpy(T, F, nbytescopy);
};

void copy_and_change_sign(unsigned SZ, double *T, unsigned TS, double *F, unsigned FS)  // copy submatrix SZ*SZ from F to T and change sign
{
    auto T_delta = TS - SZ;
    auto F_delta = FS - SZ;
    for (auto T_last_row = T + SZ*TS; T < T_last_row; T += T_delta,F+=F_delta)
        for (auto T_last_col = T + SZ; T < T_last_col; T++,F++)
            *T = -*F;
};

void print_matrix(char *s, unsigned SZ, double *M, unsigned MS = 0);
void print_matrix(char *s, int l, unsigned SZ, double *M, unsigned MS = 0);
    
//#define _PRA(M)    if(SZ==5){print_matrix(#M,__LINE__,sz_A,M); getchar();}

void  strassen_recur_inv_odd(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
{
    auto sz_D = SZ / 2;
    auto sz_A = SZ - sz_D; 
    _ASSERT(sz_A==sz_D+1);

    auto A = A_;
    auto B_enlarged = A + sz_D; //enlarged B has additional first column from A and its  size is sz_A*sz_A instead sz_D*sz_A
    auto C_enlarged = A + AS*sz_D;  //enlarged C has additional first row from A and its  size is sz_A*sz_A instead sz_A*sz_D
    auto D_actual = C_enlarged +AS+ sz_A;   //  size sz_D*sz_D

    auto I00 = I;
    auto I01_enlarged = I00 + sz_D;  //enlarged I01 has additional first column from I00 and its  size is sz_A*sz_A 
    auto I10_enlarged = I00 + IS*sz_D; //enlarged I10 has additional first row from I00 and its  size is sz_A*sz_A
    auto I10_actual = I10_enlarged + IS;  //  size sz_A*sz_D
    auto I11_enlarged = I10_enlarged + sz_D; //enlarged I11 has additional first row and first column from I10 and I01 and its  size is sz_A*sz_A
    auto I11_actual = I10_actual + sz_A;  //  size sz_D*sz_D

 

//    | I00  I01 |   | A   B_enlarged | -1
//    |          | = |       |
//    | I10  I11 |   | C_enlarged   D |
//    see https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion for formulas

    auto subm_size_A = sz_A*sz_A;
    auto subm_size_D = sz_D*sz_D;
    auto sz_A_bytes = sz_A * sizeof(A[0]);

    auto AI = buf; buf += subm_size_A;
    strassen_recur_inv(sz_A, buf, AI, sz_A, A, AS); // AI - inverse A 
    //_PRA(AI);
    auto AImulB = buf; buf += subm_size_A;   
    
    auto BT = buf; buf += subm_size_A;
    transp(sz_A, BT, B_enlarged, AS);
    //B_enlarged has  size sz_A*sz_A and  BT has actual size sz_A*sz_D, 
    //so we need to make zero first BT row 
    memset(BT, 0, sz_A_bytes);
    //_PRA(BT);
    strassen_recur_mul_by_transposed(sz_A, buf, AImulB, sz_A, AI, sz_A, BT, sz_A);

    auto CmulAImulB = BT;

    auto C_stored = buf; buf += sz_A;
    // store first row of C_enlarged (last row of A) and make it zero
    memcpy(C_stored,C_enlarged, sz_A_bytes);
    memset(C_enlarged, 0, sz_A_bytes);
    strassen_transp_and_mul(sz_A, buf, CmulAImulB, sz_A, C_enlarged, AS, AImulB, sz_A);
    //_PRA(CmulAImulB);
    auto sz_A_plus_one = sz_A + 1;
    auto Z = CmulAImulB+ sz_A_plus_one;
    
    matrix_sub(sz_D, Z, sz_A, Z, sz_A, D_actual, AS); //Z=C_enlarged*AI*B_enlarged-D 

    strassen_recur_inv(sz_D, buf, I11_actual, IS, Z, sz_A); //ZI calculated  and placed to I11

    auto ZIT = CmulAImulB;
    transp(sz_D, Z, sz_A, I11_actual, IS);   // ZIT - enlarged transposed ZI
    //_PRA(ZIT);
    strassen_recur_mul_by_transposed(sz_A, buf, I01_enlarged, IS, AImulB, sz_A, ZIT, sz_A);  //I01_enlarged=AI*B_enlarged*ZI_enlarged
    
    auto AIT = ZIT;
    transp(sz_A, AIT, AI, sz_A);
    auto T1 = AImulB;
    strassen_recur_mul_by_transposed(sz_A, buf, T1, sz_A, AIT, sz_A, C_enlarged, AS);//T1=AIT*CT_enlarged
    memcpy(C_enlarged, C_stored, sz_A_bytes); //restoring last row of A matrix (first row of C_enlarged)
    _ASSERT(buf<buf_max);
    buf -= sz_A;
    //_PRA(T1);
    auto T2 = AIT;
    strassen_recur_mul_by_transposed(sz_A, buf, T2, sz_A, I01_enlarged, IS, T1, sz_A); //T2=-AI*B_enlarged*ZI*C_enlarged*AI
    strassen_recur_mul_by_transposed(sz_D, buf, I10_actual+1, IS, I11_actual, IS, T1+ sz_A_plus_one, sz_A); // I10=ZI*C*AI
    matrA_mul_vectB(sz_D, I10_actual, IS, I11_actual, IS, T1 + 1);  // I10=ZI*C*AI

    change_sign(sz_D, I11_actual, IS);  //I11 = -ZI

    matrix_sub(sz_A, I00, IS, AI, sz_A, T2, sz_A);  // I00=AI-AI*B_enlarged*ZI_enlarged*C_enlarged*AI
}

void  strassen_recur_inv(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
{
    if (SZ < count_of(inv_funcs)) {
        inv_funcs[SZ](I, IS, A_, AS);
    }
    else {
        if (SZ % 2) 
            strassen_recur_inv_odd(SZ, buf, I, IS, A_, AS);
        else strassen_recur_inv_even(SZ, buf, I, IS, A_, AS);
    }
};



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

void copy_on( double *M, unsigned SZ, double *M_src,unsigned SZ_src)
{
	auto nbytescopy = SZ * sizeof(M[0]);
	for (auto M_end = M + SZ*SZ; M < M_end; M += SZ, M_src += SZ_src)  memcpy(M, M_src, nbytescopy);
};


bool small_matr_mul(unsigned SZ_, double *C_, double *A_, double *B_)
{
    if (SZ_ < count_of(prod_funcs_t)) {
        if (SZ_ > 0) {
            inplace_transpose(SZ_, B_);
            prod_funcs_t[SZ_](C_, SZ_, A_, SZ_, B_, SZ_);
            inplace_transpose(SZ_, B_);
        }
        return true;
    }
    return false;
}

#ifdef _DEBUG
#define _SET_BUF_MAX buf_max = buf.get() + buf_size;
#else
#define _SET_BUF_MAX
#endif

void  strassen_mul(unsigned SZ_, double *C_, double *A_, double *B_, int enl=0)
{
  if (small_matr_mul(SZ_, C_, A_, B_))return;
  //if positive enl is passed to the function - use it.
  if(enl==0) enl = get_best_enl(SZ_, SZ_ / 3);  // find optimal matrix enlarge by default 
  if (enl <0)enl = 0;     // if enl is negative don't change matrix size.
	double *C, *A, *B;
	auto SZ = SZ_;
	boost::scoped_array<double> c, a, b;
	if (enl) {
		SZ = SZ_+enl;
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
  _SET_BUF_MAX;
	strassen_recur_mul_by_transposed(SZ, buf.get(), C, SZ, A, SZ,  B, SZ);
	if (SZ == SZ_)inplace_transpose(SZ, B);
	else copy_on(C_,SZ_, C,SZ);
};

void  strassen_inv(unsigned SZ_, double *I_, double *A_, int enl = 0)
{
    auto SZ = SZ_;
    auto *I=I_;
    auto *A = A_;
    unsigned buf_size = (4 * SZ*SZ) / 3 + 8 * SZ + 1;
    boost::scoped_array<double> buf(new double[buf_size]);
    _SET_BUF_MAX;
    strassen_recur_inv(SZ, buf.get(), I, SZ, A, SZ);
};

void simple_mul(unsigned SZ, double *C, double *A, double *B)
{
    for(unsigned i = 0;i<SZ;++i)
        for (unsigned j = 0; j < SZ; ++j)
        {
            C[i*SZ+j] = A[i*SZ] * B[j];
                for(unsigned k=1;k<SZ;++k)
                   C[i*SZ + j]+= A[i*SZ + k] * B[k*SZ + j];
        }
   
};

void block_mul(unsigned SZ_,double *C_, double *A_, double *B_)
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
    inplace_transpose(SZ,B);
 
	FSimpleMatrProd f_t = prod_funcs_t[bs];
	FSimpleMatrProd f_t_p = prod_funcs_p_t[bs];

    for (unsigned i = 0; i<SZ; i += bs)
        for (unsigned j = 0; j < SZ; j += bs)
        {
#define IJ_M_PTR(M,I,J) &M[I*SZ+J]
            f_t(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, 0), SZ, IJ_M_PTR(B, j, 0), SZ);
            for (unsigned k = bs; k<SZ; k += bs)
                f_t_p(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, k), SZ, IJ_M_PTR(B, j, k), SZ);
        }
    if(SZ==SZ_)inplace_transpose(SZ,B);
    else copy_on(C_,SZ_,C,SZ);


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
            double m= fabs(A[i*SZ + j]) + fabs(B[i*SZ + j]);
            if (m > mx) mx = m;
            double d = fabs(A[i*SZ + j] - B[i*SZ + j]);
            if (d>dif) dif = d;
        }
    //if (avr == 0)return 0;
    return 2 * dif / mx;
};


void print_matrix(char *s, unsigned SZ, double *M, unsigned MS)
{
    if (MS == 0)MS = SZ;
    std::cout << s << "\n";
    for (unsigned i = 0; i<MS*MS; i += MS) {
        auto M_ = M + i;
        for (unsigned j = 0; j<SZ; j++)
            std::cout << M_[j] << " ";
        std::cout << "\n";
    };
    std::cout << "\n";
}
void print_matrix(char *s, int l, unsigned SZ, double *M, unsigned MS)
{
    if (MS == 0)MS = SZ;
    std::cout << s <<" "<<l<< "\n";
    for (unsigned i = 0; i<MS*MS; i += MS) {
        auto M_ = M + i;
        for (unsigned j = 0; j<SZ; j++)
            std::cout << M_[j] << " ";
        std::cout << "\n";
    };
    std::cout << "\n";
}



double randm()
{
    static   std::random_device generator;
    //static   std::default_random_engine generator;
    static   std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
};

void make_random(unsigned matr_size, double *M, unsigned MS = 0)
{
    if (MS)
        for (unsigned i = 0; i<matr_size; i++)
            for (unsigned j = 0; j < matr_size; j++)
                M[i*MS + j] = randm();
    else
        for (unsigned i = 0; i < matr_size*matr_size; i++)
            M[i] = randm();
};
void make_unit(unsigned matr_size, double *M, unsigned MS = 0)
{
    if (MS == 0) MS = matr_size;
        for (unsigned i = 0; i<matr_size; i++)
            for (unsigned j = 0; j < matr_size; j++)
                M[i*MS + j] = 0;
        for (unsigned i = 0; i<matr_size; i++)M[i*MS + i] = 1;
 
};

void copy_matrix(unsigned matr_size, double *dest, double *src)
{
    memcpy(dest, src, matr_size*matr_size*sizeof(dest[0]));
};

#define ALLOC_MATR(M)	boost::scoped_array<double> M##__(new double[matr_size*matr_size]); double *M = M##__.get()
#define ALLOC_RANDOM_MATR(M)	boost::scoped_array<double> M##__(new double[matr_size*matr_size]); double *M = M##__.get(); make_random(matr_size,M)

#define _CUB(a) ((a)*(a)*(a)+1)
#define _REPEAT for(n_repeats=0;n_repeats<_CUB(2300.0/matr_size);n_repeats++)
//#define _REPEAT

void compare_strassen_block(unsigned matr_size)
{
    ALLOC_RANDOM_MATR(A);
    ALLOC_RANDOM_MATR(B);
    ALLOC_MATR(ST);
    ALLOC_MATR(BL);
    int n_repeats = 1;
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    unsigned enl = get_best_enl(matr_size, matr_size / 2);
    _REPEAT strassen_mul(matr_size, ST, A, B,enl);
    double d_strassen=static_cast<duration<double>>(high_resolution_clock::now() - start).count();

    start = high_resolution_clock::now();
    _REPEAT block_mul(matr_size, BL, A, B);
    double d_block = static_cast<duration<double>>(high_resolution_clock::now() - start).count();
    std::cout << "size=" << matr_size <<" size Str="<< matr_size + enl<< " repeats=" << n_repeats << " block t=" << d_block / n_repeats;
    std::cout << " strassen t=" << d_strassen / n_repeats << " strassen p=" << get_weight(matr_size, enl)*1e-9;
    std::cout << " diff=" << matr_dif(matr_size, BL, ST) << " ratio=" << d_block / d_strassen << "\n";

};

void print_stat(unsigned SZ, double t, unsigned enl)
{
    double num_calc[12];
    get_num_calc(SZ, num_calc);
    std::cout << SZ-enl << "\t" << SZ << "\t" << t << "\t";
    for (unsigned i = 0; i < count_of(num_calc); i++)
        std::cout << num_calc[i] << "\t";
    std::cout << (enl ? SZ*SZ : 0) << "\t" << (enl ? 1 : 0) << "\n";
};

void strassen_stat(unsigned m_s, unsigned enl)
{
    unsigned matr_size = m_s - enl; 
    ALLOC_RANDOM_MATR(A);
    ALLOC_RANDOM_MATR(B);
    ALLOC_MATR(ST);
    int n_repeats = 1;

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    _REPEAT strassen_mul(matr_size, ST, A, B, enl);
    double d = static_cast<duration<double>>(high_resolution_clock::now() - start).count()/n_repeats;
    print_stat(m_s, d, enl);

};

void test_matrix_mul(unsigned bs)
{
    if (bs < 8 || bs>15) { std::cout << "wrong input"; return;}
    const unsigned repeats = 20000;
    unsigned matr_size = 350;
    ALLOC_RANDOM_MATR(A);
    ALLOC_RANDOM_MATR(B);
    ALLOC_MATR(C);
    int n_repeats = 0;

    using namespace std::chrono;
    FSimpleMatrProd mul = prod_funcs_t[bs];
    auto start = high_resolution_clock::now();
    for (unsigned k = 0; k<repeats; ++k)
        for (unsigned i = 0; i<matr_size-bs; i += bs)
            for (unsigned j = 0; j<matr_size-bs; j += bs)
            {
                mul(&C[i*matr_size+j], matr_size, &A[i*matr_size + j], matr_size, &B[i*matr_size + j], matr_size);
                ++n_repeats;
            }
    double d = static_cast<duration<double>>(high_resolution_clock::now() - start).count() / n_repeats;
    std::cout <<  bs << "\t" << d << "\n";

};

#ifdef DIAGN_PRINT
#define _PRN(M) print_matrix(#M,matr_size,M);
#else
#define _PRN(M)
#endif

int main(int argc, char* argv[])
{
  /*  auto matr_size = 7;
    ALLOC_RANDOM_MATR(A);
    ALLOC_MATR(AI);
    ALLOC_MATR(P);
    ALLOC_MATR(E);
    //for (int i = 0; i < matr_size*matr_size; i++)A[i] = (i%5)+1;    
    _PRN(A);

    make_unit(matr_size, E);
    strassen_inv(matr_size,AI,A);


    strassen_mul(matr_size, P, AI, A,-1);
    std::cout << matr_size<<"\n";
    std::cout << "diff E P " << matr_dif2(matr_size, E, P) << "\n";

    _PRN(AI);
    _PRN( P);

    getchar();
    return 0; */

    /*    auto matr_size = 1024;
    ALLOC_RANDOM_MATR(A);
    ALLOC_RANDOM_MATR(B);
    ALLOC_MATR(ST);
    ALLOC_MATR(BL);

     strassen_mul(matr_size, ST, A, B, matr_size);
     block_mul(matr_size, BL, A, B);
    std::cout << "\n";
    std::cout << "diff block strassen " << matr_dif(matr_size, BL, ST) << "\n";

//    print_matrix("simple", matr_size, S);
//    print_matrix("block", matr_size, BL);
//    print_matrix("strassen", matr_size, ST);
   
        getchar();*/

    const unsigned from_default = 200;
    const unsigned to_default = 4100;
    unsigned from(from_default), to(to_default);
    bool wait = false;
    bool stat = false;
    if (argc == 1) { 
        std::cout<<"usage: Strassen -fF -tT -w\n";
    }
    else {
        for (int i = 1; i<argc; i++)
            if (argv[i][0] == '-')
            {
                switch (argv[i][1])
                {
                case 'f':
                {
                    from = atoi(argv[i] + 2);
                    if ((from<16) || (from>16 * 1024))
                    {
                        from = from_default;
                        std::cout << "Some error in -f param " << from_default << " used instead.\n";
                    }
                }
                break;
                case 't':
                {
                    to = atoi(argv[i] + 2);
                    if ((to<16) || (to>16 * 1024))
                    {
                        to = to_default;
                        std::cout << "Some error in -t param " << to_default << " used instead.\n";
                    }
                }
                break;
                case 'w':
                {
                    wait = true;
                }
                break;
                case 's':
                {
                    stat = true;
                }
                break;
                }
            }

    }
    if (from >= to) {
        std::cout << "Somehow -f>=-t, to fix it -t is increased to -f+1\n";
        to = from + 1;
    }
  std::cout << "Finally command is\n Strassen -f"<<from<<" -t"<<to;
  if (wait) std::cout << " -w\n"; else std::cout << "\n";
  if (wait)
  {
      std::cout << "press enter to start\n";
      getchar();
      std::cout << "wait...\n";
  }
  for (unsigned i = from; i < to; i++)  if (stat) strassen_stat(i,randm()*7);else compare_strassen_block(i);
  //for (unsigned i = 8; i < 16; i++) test_matrix_mul(i);
  if (wait) getchar();
    return 0; 
}

