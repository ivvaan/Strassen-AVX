#pragma once

void matrix_sum(unsigned sz, double * C, unsigned CS, double * A, unsigned AS, double * B, unsigned BS);

void matrix_sub(unsigned sz, double * C, unsigned CS, double * A, unsigned AS, double * B, unsigned BS);

void C_add_a_mul_B(unsigned sz, double * C, double a, double * B);

double vectA_mul_vectB(unsigned sz, double * A, double * B);

void matrA_mul_vectB(unsigned sz, double * C, unsigned CS, double * A, unsigned AS, double * B);

void transp(unsigned SZ, double * BT, double * B, unsigned BS);

void transp(unsigned SZ, double * BT, unsigned BTS, double * B, unsigned BS);

void change_sign(unsigned SZ, double * M, unsigned MS);

void inplace_transpose(unsigned SZ, double * B);

double matr_dif(unsigned SZ, double * A, double * B);

double matr_dif2(unsigned SZ, double * A, double * B);
