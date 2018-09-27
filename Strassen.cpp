// Strassen.cpp : Defines the entry pounsigned for the console application.
//

//#include "stdafx.h"
#include "small_matr_funcs.h"
#include <stdio.h>
#include <tchar.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <boost/smart_ptr/scoped_array.hpp>
#include "utils.h"
#include "strassen_mul.h"
#include "recur_inv.h"
#include "base_operations.h"






#ifdef DIAGN_PRINT
#define _PRN(M) print_matrix(#M,matr_size,M);
#else
#define _PRN(M)
#endif




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
    unsigned enl = get_best_enl(matr_size, matr_size / 3);
    _REPEAT strassen_mul(matr_size, ST, A, B,enl);
    double d_strassen=static_cast<duration<double>>(high_resolution_clock::now() - start).count();

    start = high_resolution_clock::now();
    _REPEAT block_mul(matr_size, BL, A, B);
    double d_block = static_cast<duration<double>>(high_resolution_clock::now() - start).count();
    std::cout << "size=" << matr_size <<" enlarge="<<  enl<< " repeats=" << n_repeats << " block t=" << d_block / n_repeats;
    std::cout << " strassen t=" << d_strassen / n_repeats << " strassen predicted t=" << get_weight(matr_size, enl)*1e-9;
    std::cout << " diff=" << matr_dif(matr_size, BL, ST) << " ratio=" << d_block / d_strassen << "\n";

};

void compare_inv_strassen_mul(unsigned matr_size)
{
    ALLOC_RANDOM_MATR(A);
    ALLOC_MATR(AI);
    ALLOC_MATR(P);
    ALLOC_MATR(E);
    make_unit(matr_size, E);

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    invert(matr_size, AI, A);
    double d_inv = static_cast<duration<double>>(high_resolution_clock::now() - start).count();
    
    start = high_resolution_clock::now();
    strassen_mul(matr_size, P, AI, A, -1);
    double d_strassen = static_cast<duration<double>>(high_resolution_clock::now() - start).count();
    std::cout << "size=" << matr_size <<" inverse t" << d_inv <<" strassen t=" << d_strassen;
    std::cout << " diff E P " << matr_dif2(matr_size, E, P) << " ratio=" << d_inv / d_strassen << "\n";


}

void print_stat(unsigned SZ, double t, unsigned enl)
{
    double num_calc[12];
    get_num_calc(SZ, num_calc);
    std::cout << SZ-enl << "\t" << SZ << "\t" << t << "\t";
    for (unsigned i = 0; i < count_of(num_calc); i++)
        std::cout << num_calc[i] << "\t";
    std::cout << (enl ? SZ*SZ : 0) << "\t" << (enl ? 1 : 0) << "\n";
};

void strassen_stat(unsigned m_s)
{
    unsigned enl = randm() * 7;
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

void test_small_matrix_mul(unsigned bs)
{
    if (bs < 8 || bs>15) { std::cout << "wrong input, matrix size for this test must be 8..15"; return; }
    const unsigned repeats = 20000;
    unsigned matr_size = 350;
    ALLOC_RANDOM_MATR(A);
    ALLOC_RANDOM_MATR(B);
    ALLOC_MATR(C);
    int n_repeats = 0;

    using namespace std::chrono;
    FSimpleMatrProd mul = get_mul_func(bs);
    auto start = high_resolution_clock::now();
    for (unsigned k = 0; k<repeats; ++k)
        for (unsigned i = 0; i<matr_size - bs; i += bs)
            for (unsigned j = 0; j<matr_size - bs; j += bs)
            {
                mul(&C[i*matr_size + j], matr_size, &A[i*matr_size + j], matr_size, &B[i*matr_size + j], matr_size);
                ++n_repeats;
            }
    double d = static_cast<duration<double>>(high_resolution_clock::now() - start).count() / n_repeats;
    std::cout << bs << "\t" << d << "\n";

};

void(*test_fucs[])(unsigned) = { compare_strassen_block , strassen_stat , compare_inv_strassen_mul,test_small_matrix_mul };

int main(int argc, char* argv[])
{

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
    const unsigned test_type_default = 0;
    unsigned from(from_default), to(to_default);
    bool wait = false;
    unsigned test_type = test_type_default;
    if (argc == 1) { 
        std::cout << "program prints statistics for different matrix multiplication and inversion tests\n";
        std::cout << "usage: Strassen -fF -tT [-w] [-cC]\n";
        std::cout << "matrix size iterates from F to T-1\n";
        std::cout << "if -w presented programs gets stopped and waiting for 'enter' pressing before and after tests start\n";
        std::cout << "C means test type; default value is 0\n";
        std::cout << "  C==0 - compare block multiplication and Strassen\n";
        std::cout << "  C==1 - Strassen multiplication statistics; number of different operations made during calculations\n";
        std::cout << "  C==2 - recursive matrix inversion time compare to Strasssen multiplication time\n";
        std::cout << "  C==3 - time to make multiplication for small matrix having size 8..15; F,T params mast be in 8..16\n";
        return 0;
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
                case 'c':
                {
                    test_type = atoi(argv[i] + 2);
                    if (test_type>3)
                    {
                        test_type = test_type_default;
                        std::cout << "Some error in -c param " << test_type_default << " used instead.\n";
                    }
                }
                break;
                }
            }

    }
    if (test_type == 3) {
        if (from < 8) {
            from = 8;
            std::cout << "Some error in F param for this test " << 8 << " used instead\n";
        }
        if (to > 16) {
            to = 16;
            std::cout << "Some error in T param for this test " << 16 << " used instead\n";
        }
    }
    if (from >= to) {
        std::cout << "Somehow F>=T, to fix it T is increased to F+1\n";
        to = from + 1;
    }
    std::cout << "Finally command is\n Strassen -f"<<from<<" -t"<<to<<" -c"<<test_type;
  if (wait) std::cout << " -w\n"; else std::cout << "\n";
  if (wait)
  {
      std::cout << "press enter to start\n";
      getchar();
      std::cout << "wait...\n";
  }
  for (unsigned i = from; i < to; i++)  
      test_fucs[test_type](i);
  //for (unsigned i = 8; i < 16; i++) test_matrix_mul(i);
  if (wait) getchar();
    return 0; 
}

