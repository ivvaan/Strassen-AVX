#include "recur_inv.h"
#include "utils.h"
#include "strassen_mul.h"
#include "base_operations.h"
#include <iostream>
#include <fstream>
#include <boost/smart_ptr/scoped_array.hpp>

/*
Blockwise inverse matrix calculation using Strassen multiplication 

https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion 

!!!!
Numerically unstable!!!!
For test purpose only!!!
Not recommended for practical ussage!!!

*/

void  recur_inv_even(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
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
    recur_inv(sz, buf, AI, sz, A, AS); // AI - inverse A 
    auto T1 = buf; buf += subm_size;
    strassen_transp_and_mul(sz, buf, T1, sz, AI, sz, B, AS);
    auto T2 = buf; buf += subm_size;
    strassen_transp_and_mul(sz, buf, T2, sz, C, AS, T1, sz);
    matrix_sub(sz, T2, sz, T2, sz, D, AS); //T2=C*AI*B-D (T2=-Z where Z=D-C*AI*B)
    recur_inv(sz, buf, I11, IS, T2, sz); //-ZI calculated and stored in I11

    strassen_transp_and_mul(sz, buf, I01, IS, T1, sz, I11, IS);  //I01=-AI*B*ZI

    transp(sz, T2, AI, sz); // T2=AIT
    strassen_recur_mul_by_transposed(sz, buf, T1, sz, T2, sz, C, AS);  //T1=AIT*CT

    strassen_recur_mul_by_transposed(sz, buf, T2, sz, I01, IS, T1, sz); //T2=-AI*B*ZI*C*AI

    matrix_sub(sz, I00, IS, AI, sz, T2, sz);  // I00=AI+AI*B*ZI*C*AI
    strassen_recur_mul_by_transposed(sz, buf, I10, IS, I11, IS, T1, sz); // I10=-ZI*C*AI
    change_sign(sz, I11, IS);  //I11 = ZI

}


void print_matrix(char *s, unsigned SZ, double *M, unsigned MS = 0);
void print_matrix(char *s, int l, unsigned SZ, double *M, unsigned MS = 0);

//#define _PRA(M)    if(SZ==5){print_matrix(#M,__LINE__,sz_A,M); getchar();}

void  recur_inv_odd(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
{
    auto sz_D = SZ / 2;
    auto sz_A = SZ - sz_D;
    _ASSERT(sz_A == sz_D + 1);

    auto A = A_;
    auto B_enlarged = A + sz_D; //enlarged B has additional first column from A and its  size is sz_A*sz_A instead sz_D*sz_A
    auto C_enlarged = A + AS*sz_D;  //enlarged C has additional first row from A and its  size is sz_A*sz_A instead sz_A*sz_D
    auto D_actual = C_enlarged + AS + sz_A;   //  size sz_D*sz_D

    auto I00 = I;
    auto I01_enlarged = I00 + sz_D;  //enlarged I01 has additional first column from I00 and its  size is sz_A*sz_A 
    auto I10_enlarged = I00 + IS*sz_D; //enlarged I10 has additional first row from I00 and its  size is sz_A*sz_A
    auto I10_actual = I10_enlarged + IS;  //  size sz_A*sz_D
    auto I11_enlarged = I10_enlarged + sz_D; //enlarged I11 has additional first row and first column from I10 and I01 and its  size is sz_A*sz_A
    auto I11_actual = I10_actual + sz_A;  //  size sz_D*sz_D



      //    | I00  I01 |   | A   B_enlarged | -1
      //    |          | = |                |
      //    | I10  I11 |   | C_enlarged   D |

    auto subm_size_A = sz_A*sz_A;
    auto subm_size_D = sz_D*sz_D;
    auto sz_A_bytes = sz_A * sizeof(A[0]);

    auto AI = buf; buf += subm_size_A;
    recur_inv(sz_A, buf, AI, sz_A, A, AS); // AI - inverse A 
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
    memcpy(C_stored, C_enlarged, sz_A_bytes);
    memset(C_enlarged, 0, sz_A_bytes);
    strassen_transp_and_mul(sz_A, buf, CmulAImulB, sz_A, C_enlarged, AS, AImulB, sz_A);
    //_PRA(CmulAImulB);
    auto sz_A_plus_one = sz_A + 1;
    auto Z = CmulAImulB + sz_A_plus_one;

    matrix_sub(sz_D, Z, sz_A, Z, sz_A, D_actual, AS); //Z=C_enlarged*AI*B_enlarged-D 

    recur_inv(sz_D, buf, I11_actual, IS, Z, sz_A); //ZI calculated  and placed to I11

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
    strassen_recur_mul_by_transposed(sz_D, buf, I10_actual + 1, IS, I11_actual, IS, T1 + sz_A_plus_one, sz_A); // I10=ZI*C*AI
    matrA_mul_vectB(sz_D, I10_actual, IS, I11_actual, IS, T1 + 1);  // I10=ZI*C*AI

    change_sign(sz_D, I11_actual, IS);  //I11 = -ZI

    matrix_sub(sz_A, I00, IS, AI, sz_A, T2, sz_A);  // I00=AI-AI*B_enlarged*ZI_enlarged*C_enlarged*AI
}

typedef void(*FSimpleInvFunc)(double *, unsigned, double *, unsigned);

#define RESTRICTED_PTR __restrict
//#define RESTRICTED_PTR 
FSimpleInvFunc inv_funcs[] = {
    nullptr,
    [](double *RESTRICTED_PTR I,unsigned IS,double *RESTRICTED_PTR A,unsigned AS) {I[0] = 1 / A[0]; },
    [](double *RESTRICTED_PTR I,unsigned IS,double *RESTRICTED_PTR A,unsigned AS) {
    auto det = A[0] * A[AS + 1] - A[1] * A[AS];
    I[0] = A[AS + 1] / det; I[1] = -A[1] / det;
    I[IS] = -A[AS] / det; I[IS + 1] = A[0] / det;
},
[](double *RESTRICTED_PTR I,unsigned IS,double *RESTRICTED_PTR M,unsigned MS) {
    auto *RESTRICTED_PTR M1 = M + MS;
    auto *RESTRICTED_PTR M2 = M1 + MS;
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



void  recur_inv(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
{
    if (SZ < count_of(inv_funcs)) {
        inv_funcs[SZ](I, IS, A_, AS);
    }
    else {
        if (SZ % 2)
            recur_inv_odd(SZ, buf, I, IS, A_, AS);
        else recur_inv_even(SZ, buf, I, IS, A_, AS);
    }
};


void  invert(unsigned SZ_, double *I_, double *A_, int enl)
{
    auto SZ = SZ_;
    auto *I = I_;
    auto *A = A_;
    unsigned buf_size = (4 * SZ*SZ) / 3 + 8 * SZ + 1;
    boost::scoped_array<double> buf(new double[buf_size]);
    _SET_BUF_MAX(buf.get() + buf_size);
    recur_inv(SZ, buf.get(), I, SZ, A, SZ);
};
