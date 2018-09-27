#pragma once

void get_num_calc(unsigned SZ, double(&num_calc)[12]);

double get_weight(unsigned SZ, unsigned enlarge);

unsigned get_best_enl(unsigned SZ, unsigned range);

void strassen_recur_mul_by_transposed(unsigned SZ, double * buf, double * C00, unsigned CS, double * A00, unsigned AS, double * B00, unsigned BS);

void strassen_transp_and_mul(unsigned SZ, double * buf, double * C, unsigned CS, double * A, unsigned AS, double * B, unsigned BS);

void strassen_mul(unsigned SZ_, double * C_, double * A_, double * B_, int enl);

void block_mul(unsigned SZ_, double * C_, double * A_, double * B_);
