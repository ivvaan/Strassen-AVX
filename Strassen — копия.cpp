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

typedef void (*FSimpleMatrProd)(double *, unsigned, double *, unsigned, double *, unsigned);

#define IJ_MATR_ELEM(M,I,J) M[(I)*M##S+(J)]
#define KTH_T_TERM(z, k, J_IDX) BOOST_PP_IF(k,+,) A[k]*B_[k]
#define IJK_T_CYCLE_K(z,J_IDX,n) *C=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);C++;B_+=BS;
#define IJK_T_CYCLE_K_P(z,J_IDX,n) *C+=BOOST_PP_REPEAT_3RD(n,KTH_T_TERM,J_IDX);C++;B_+=BS;

#define IJK_T_CYCLE_J(z,I_IDX,n) {auto B_=B; BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K,n); C+=CS-n; A+=AS;}
#define IJK_T_CYCLE_J_P(z,I_IDX,n) {auto B_=B; BOOST_PP_REPEAT_2ND(n,IJK_T_CYCLE_K_P,n); C+=CS-n; A+=AS;}

#define MATR_T_PROD_FUNC_IJK(z,n,P) [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS)\
{ BOOST_PP_REPEAT_1ST(n,IJK_T_CYCLE_J##P,n)}

#define MATR_T_PROD_FUNC(z, n, P) MATR_T_PROD_FUNC_IJK(z, n, P)


FSimpleMatrProd prod_funcs_t[] = { //prod_funcs_t[i] i=8..15 is function C=A*transp(B) for matrices ixi
	nullptr, nullptr, nullptr,
	MATR_T_PROD_FUNC(z, 3, ),  // just to see how macros expanded
	nullptr, nullptr, nullptr, nullptr,
	MATR_T_PROD_FUNC(z, 8, ),
	MATR_T_PROD_FUNC(z, 9, ),
	MATR_T_PROD_FUNC(z, 10, ),
	MATR_T_PROD_FUNC(z, 11, ),
	MATR_T_PROD_FUNC(z, 12, ),
	MATR_T_PROD_FUNC(z, 13, ),
	MATR_T_PROD_FUNC(z, 14, ),
	MATR_T_PROD_FUNC(z, 15, )
	//MATR_T_PROD_FUNC(z, 16,), 
};


FSimpleMatrProd prod_funcs_p_t[] = {  //prod_funcs_p[i] i=8..15 is function C+=A*transp(B) for matrices ixi
	nullptr, nullptr, nullptr,
	MATR_T_PROD_FUNC(z, 3, _P),  // just to see how macros expanded
	nullptr, nullptr, nullptr, nullptr,
	MATR_T_PROD_FUNC(z, 8, _P),
	MATR_T_PROD_FUNC(z, 9, _P),
	MATR_T_PROD_FUNC(z, 10, _P),
	MATR_T_PROD_FUNC(z, 11, _P),
	MATR_T_PROD_FUNC(z, 12, _P),
	MATR_T_PROD_FUNC(z, 13, _P),
	MATR_T_PROD_FUNC(z, 14, _P),
	MATR_T_PROD_FUNC(z, 15, _P)
	//MATR_T_PROD_FUNC_IJK(z, 16,_P), 
};



#include <immintrin.h>


void mul8(double * C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
	double *C_last_row = C + CS*8;
	while (C < C_last_row){
		__m256d ymm0, ymm1, ymm2, ymm3, res;
		__m256d Amm0 = _mm256_loadu_pd(A);
    __m256d Amm1 = _mm256_loadu_pd(A + 4);
      auto *B_ = B;
		  ymm0 = _mm256_loadu_pd(B_);
		  ymm1 = _mm256_loadu_pd(B_ + BS);
			ymm2 = _mm256_loadu_pd(B_ + 2*BS);
			ymm3 = _mm256_loadu_pd(B_ + 3*BS);
			ymm0 = _mm256_mul_pd(ymm0, Amm0);
			ymm1 = _mm256_mul_pd(ymm1, Amm0);
			ymm2 = _mm256_mul_pd(ymm2, Amm0);
			ymm3 = _mm256_mul_pd(ymm3, Amm0);

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

			_mm256_storeu_pd(C, res);

			B_ = B+4*BS;


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

			res = _mm256_add_pd(ymm2, ymm3);

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

			_mm256_storeu_pd(C+4, res);
		
		A += AS;
		C += CS;
	}

};

void mul8p(double * C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
	double *C_last_row = C + CS * 8;
	while (C < C_last_row){
		__m256d ymm0, ymm1, ymm2, ymm3, ymm4;
		__m256d ymm5 = _mm256_loadu_pd(A);
		__m256d ymm6 = _mm256_loadu_pd(A + 4);
		auto *B_ = B;
		ymm0 = _mm256_loadu_pd(B_);
		ymm1 = _mm256_loadu_pd(B_ + BS);
		ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
		ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
		ymm0 = _mm256_mul_pd(ymm0, ymm5);
		ymm1 = _mm256_mul_pd(ymm1, ymm5);
		ymm2 = _mm256_mul_pd(ymm2, ymm5);
		ymm3 = _mm256_mul_pd(ymm3, ymm5);

		ymm0 = _mm256_hadd_pd(ymm0, ymm1);
		ymm1 = _mm256_hadd_pd(ymm2, ymm3);
		ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
		ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);

		ymm4 = _mm256_add_pd(ymm2, ymm3);

		B_ += 4;
		ymm0 = _mm256_loadu_pd(B_);
		ymm1 = _mm256_loadu_pd(B_ + BS);
		ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
		ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
		ymm0 = _mm256_mul_pd(ymm0, ymm6);
		ymm1 = _mm256_mul_pd(ymm1, ymm6);
		ymm2 = _mm256_mul_pd(ymm2, ymm6);
		ymm3 = _mm256_mul_pd(ymm3, ymm6);

		ymm0 = _mm256_hadd_pd(ymm0, ymm1);
		ymm1 = _mm256_hadd_pd(ymm2, ymm3);
		ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
		ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);

		ymm0 = _mm256_add_pd(ymm2, ymm3);

		ymm0 = _mm256_add_pd(ymm0, ymm4);

		ymm1 = _mm256_loadu_pd(C);
		ymm0 = _mm256_add_pd(ymm0, ymm1);
		_mm256_storeu_pd(C, ymm0);

		B_ = B + 4 * BS;
		ymm0 = _mm256_loadu_pd(B_);
		ymm1 = _mm256_loadu_pd(B_ + BS);
		ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
		ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
		ymm0 = _mm256_mul_pd(ymm0, ymm5);
		ymm1 = _mm256_mul_pd(ymm1, ymm5);
		ymm2 = _mm256_mul_pd(ymm2, ymm5);
		ymm3 = _mm256_mul_pd(ymm3, ymm5);

		ymm0 = _mm256_hadd_pd(ymm0, ymm1);
		ymm1 = _mm256_hadd_pd(ymm2, ymm3);
		ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
		ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);

		ymm4 = _mm256_add_pd(ymm2, ymm3);

		B_ += 4;
		ymm0 = _mm256_loadu_pd(B_);
		ymm1 = _mm256_loadu_pd(B_ + BS);
		ymm2 = _mm256_loadu_pd(B_ + 2 * BS);
		ymm3 = _mm256_loadu_pd(B_ + 3 * BS);
		ymm0 = _mm256_mul_pd(ymm0, ymm6);
		ymm1 = _mm256_mul_pd(ymm1, ymm6);
		ymm2 = _mm256_mul_pd(ymm2, ymm6);
		ymm3 = _mm256_mul_pd(ymm3, ymm6);

		ymm0 = _mm256_hadd_pd(ymm0, ymm1);
		ymm1 = _mm256_hadd_pd(ymm2, ymm3);
		ymm2 = _mm256_permute2f128_pd(ymm0, ymm1, 0x21);
		ymm3 = _mm256_blend_pd(ymm0, ymm1, 0xc);

		ymm0 = _mm256_add_pd(ymm2, ymm3);

		ymm0 = _mm256_add_pd(ymm0, ymm4);

		ymm1 = _mm256_loadu_pd(C+4);
		ymm0 = _mm256_add_pd(ymm0, ymm1);
		_mm256_storeu_pd(C+4, ymm0);

		A += AS;
		C += CS;
	}

};


/*typedef void(*FAdditiveOp)(double *, double *, double *);
#define ADDITIVE_OP(z, k, OP) ymm0 = _mm256_loadu_pd(A);A+=4; ymm1 = _mm256_loadu_pd(B);B+=4;\
ymm0=_mm256_##OP##_pd(ymm0, ymm1); _mm256_storeu_pd(C, ymm0); C+=4;

#define UNROLL_FUNC(z,n,OP) [](double *C,double *A,double *B) {__m256d ymm0; __m256d ymm1; BOOST_PP_REPEAT_1ST(n,ADDITIVE_OP,OP)}
#define UNROLL_ELEM(z,n,OP) ,UNROLL_FUNC(z,n,OP)
#define DEFINE_UNROLL_FUNCS(name,n,OP) FAdditiveOp name []={nullptr BOOST_PP_REPEAT_FROM_TO_2ND(1,n,UNROLL_ELEM,OP)};

DEFINE_UNROLL_FUNCS(unroll_sum, 17, add)
DEFINE_UNROLL_FUNCS(unroll_sub, 17, sub)*/

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
void  STRASSEN_RECUR_MUL(unsigned SZ, double *buf, double *C00, unsigned CS, double *A00, unsigned AS, double *B00, unsigned BS)
{
	
	if (SZ < 16){ prod_funcs_t[SZ](C00, CS, A00, AS, B00, BS); }
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
	}
}


template <unsigned SZ>
void  strassen_mul(double(*C)[SZ], double(*A)[SZ], double(*B)[SZ])
{
	transp(B);
	boost::scoped_array<double> buf(new double[(8 * SZ*SZ) / 3 +1]);
	STRASSEN_RECUR_MUL(SZ, buf.get(), reinterpret_cast<double *>(C), SZ, reinterpret_cast<double *>(A), SZ,  reinterpret_cast<double *>(B), SZ);
	transp(B);
};

template <unsigned SZ>
void simple_mul(double(*C)[SZ], double (*A)[SZ], double (*B)[SZ])
{
    for(unsigned i = 0;i<SZ;++i)
        for (unsigned j = 0; j < SZ; ++j)
        {
            C[i][j] = A[i][0] * B[0][j];
                for(unsigned k=1;k<SZ;++k)
                   C[i][j]+= A[i][k] * B[k][j];
        }
   
};


template <unsigned SZ>
void transp(double(*B)[SZ])
{
    for(unsigned i=0;i<SZ;i++)
        for (unsigned j = i+1; j<SZ; j++)
        {
            auto b = B[i][j];
            B[i][j] = B[j][i];
            B[j][i] = b;
        }

}

template <unsigned SZ>
void block_mul_t(double(*C)[SZ], double(*A)[SZ], double(*B)[SZ])
{
    transp(B);
	FSimpleMatrProd f_t_8 = prod_funcs_t[8];
	FSimpleMatrProd f_t_8_p = prod_funcs_p_t[8];

    for (unsigned i = 0; i<SZ; i += 8)
        for (unsigned j = 0; j < SZ; j += 8)
        {
#define IJ_M_PTR(M,I,J) reinterpret_cast<double *>(&M[I][J])
            f_t_8(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, 0), SZ, IJ_M_PTR(B, j, 0), SZ);
            for (unsigned k = 8; k<SZ; k += 8)
                f_t_8_p(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, k), SZ, IJ_M_PTR(B, j, k), SZ);
        }
    transp(B);
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

const unsigned matr_size = 1024*2;
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
#define _CUB(a) (a)*(a)*(a)+1
#define _REPEAT for(int i=0;i<_CUB(3*1024/matr_size);i++)
//#define _REPEAT
	prod_funcs_t[8] = mul8;
	prod_funcs_p_t[8] = mul8p;
  auto t_ = clock();
 //simple_mul(S, a, b);
  double d_simple = (double)(clock() - t_) / CLOCKS_PER_SEC;
  t_ = clock();
  _REPEAT strassen_mul(ST, a, b);
  double d_strassen = (double)(clock() - t_) / CLOCKS_PER_SEC;
  t_ = clock();
  //_REPEAT recur_mul(R, a, b);
  double d_recur = (double)(clock() - t_) / CLOCKS_PER_SEC;
  /*t_ = clock();
  //block_mul(B, a, b);
  double d_block = (double)(clock() - t_) / CLOCKS_PER_SEC;*/
  
  t_ = clock();
  _REPEAT block_mul_t(BT, a, b);
  double d_block_t = (double)(clock() - t_) / CLOCKS_PER_SEC;

	std::cout << "size " << matr_size << "\n";

  //std::cout << "simple " << d_simple << "\n";
  //std::cout << "recur " << d_recur << "\n";
  //std::cout << "block " << d_block << "\n";
  std::cout << "block transp " << d_block_t << "\n";
  std::cout << "strassen " << d_strassen << "\n";
//  std::cout << "diff simple strassen " << matr_dif(S, ST) << "\n";

  //std::cout << "diff block block transp " << matr_dif(B, BT) << "\n";
  std::cout << "diff block transp strassen " << matr_dif(BT, ST) << "\n";
  //std::cout << "diff block recur " << matr_dif(B, R) << "\n";
  //std::cout << "diff block strassen " << matr_dif(B, ST) << "\n";
  //std::cout << "ratio simple/strassen  " << d_simple / d_strassen << "\n";
  std::cout << "ratio block transp/strassen  " << d_block_t / d_strassen << "\n";
  //std::cout << "ratio recur/strassen  " << d_recur / d_strassen << "\n";
  /*  std::cout << "\n";
  print_matrix("A", a);
  print_matrix("B", b);
  transp(b);
  print_matrix("BT", b);
  std::cout << "\n";
  //print_matrix("block", B);
  //print_matrix("simple", S);
  print_matrix("block transp", BT);
  print_matrix("strassen", ST);
  */
    getchar();
    return 0;
}

