/*

C11←A11−A21
C12←B22−B12
C21←C11×C12
C11←B12−B11
S11←B22−C11
C12←S11−B21
C22←A21+A22
S12←C22−A11
S21←A12−S12
S22←S12×S11
S11←S21×B22
S21←A22×C12
C12←C22×C11
C11←A11×B11
C22←A12×B21
S22←S22+C11
C21←S22+C21
C11←C11+C22
C22←C21+C12
C21←C21−S21
C12←C12+S22+S11


*/

/*#define KTH_TERM(z, n, IDX) BOOST_PP_IF(n,+,) IJ_MATR_ELEM(A,IDX/sub_matr_sz,n)*IJ_MATR_ELEM(B,n,IDX % sub_matr_sz)
#define GET_C_IJ(z,IDX,n) IJ_MATR_ELEM(C,IDX / sub_matr_sz,IDX % sub_matr_sz)=BOOST_PP_REPEAT_3RD(n,KTH_TERM,IDX);
#define GET_C_IJ_P(z,IDX,n) IJ_MATR_ELEM(C,IDX / sub_matr_sz,IDX % sub_matr_sz)+=BOOST_PP_REPEAT_3RD(n,KTH_TERM,IDX);

#define MATR_PROD_FUNC(z,n,P) ,[](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS)\
{const unsigned sub_matr_sz=n; BOOST_PP_REPEAT_2ND(BOOST_PP_MUL(n,n),GET_C_IJ##P,n)}

FSimpleMatrProd prod_funcs[] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
// BOOST_PP_REPEAT_FROM_TO_1ST(8,16,MATR_PROD_FUNC,) //MSVS C++13 compiler can't handle 8,16 maximum 8,15
MATR_PROD_FUNC(z, 8,)
MATR_PROD_FUNC(z, 9,)
MATR_PROD_FUNC(z, 10,)
MATR_PROD_FUNC(z, 11,)
MATR_PROD_FUNC(z, 12,)
MATR_PROD_FUNC(z, 13,)
MATR_PROD_FUNC(z, 14,)
MATR_PROD_FUNC(z, 15,)
//MATR_PROD_FUNC(z, 16,) //MSVS C++13 compiler can't handle 16; maximum 15
};*/

/*#define KTH_TERM(z, n, i_idx) BOOST_PP_IF(n,+,) IJ_MATR_ELEM(A,i_idx,n)*IJ_MATR_ELEM(B,n,J_IDX)
#define GET_C_I(z,i_idx,n) IJ_MATR_ELEM(C,i_idx,J_IDX)=BOOST_PP_REPEAT_3RD(n,KTH_TERM,i_idx);
#define GET_C_I_P(z,i_idx,n) IJ_MATR_ELEM(C,i_idx,J_IDX)+=BOOST_PP_REPEAT_3RD(n,KTH_TERM,i_idx);

#define GET_C_J(z,j_idx,n) {const unsigned J_IDX=j_idx; BOOST_PP_REPEAT_2ND(n,GET_C_I,n)}
#define GET_C_J_P(z,j_idx,n) {const unsigned J_IDX=j_idx; BOOST_PP_REPEAT_2ND(n,GET_C_I_P,n)}

#define MATR_PROD_FUNC(z,n,P) ,[](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS)\
{ BOOST_PP_REPEAT_1ST(n,GET_C_J##P,n)}*/




#define KTH_TERM(z, k, J_IDX) BOOST_PP_IF(k,+,) IJ_MATR_ELEM(A,i_idx,k)*IJ_MATR_ELEM(B,k,J_IDX)
#define IJK_CYCLE_K(z,J_IDX,n) IJ_MATR_ELEM(C,i_idx,J_IDX)=BOOST_PP_REPEAT_3RD(n,KTH_TERM,J_IDX);
#define IJK_CYCLE_K_P(z,J_IDX,n) IJ_MATR_ELEM(C,i_idx,J_IDX)+=BOOST_PP_REPEAT_3RD(n,KTH_TERM,J_IDX);

#define IJK_CYCLE_J(z,I_IDX,n) {const unsigned i_idx=I_IDX; BOOST_PP_REPEAT_2ND(n,IJK_CYCLE_K,n)}
#define IJK_CYCLE_J_P(z,I_IDX,n) {const unsigned i_idx=I_IDX; BOOST_PP_REPEAT_2ND(n,IJK_CYCLE_K_P,n)}

#define MATR_PROD_FUNC_IJK(z,n,P) [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS)\
{ BOOST_PP_REPEAT_1ST(n,IJK_CYCLE_J##P,n)}


#define ELEM_iK(z, n, K) IJ_MATR_ELEM(C,i_idx,n)=IJ_MATR_ELEM(A,i_idx,K)*IJ_MATR_ELEM(B,K,n);
#define ELEM_iK_P(z, n, K) IJ_MATR_ELEM(C,i_idx,n)+=IJ_MATR_ELEM(A,i_idx,K)*IJ_MATR_ELEM(B,K,n);
#define IKJ_CYCLE_J(z,K,n) BOOST_PP_REPEAT_3RD(n,ELEM_iK,K)
#define IKJ_CYCLE_J_P(z,K,n) BOOST_PP_REPEAT_3RD(n,ELEM_iK_P,K)

#define IKJ_CYCLE_K(z,I_IDX,n) {const unsigned i_idx=I_IDX; IKJ_CYCLE_J(z,0,n) BOOST_PP_REPEAT_FROM_TO_2ND(1,n,IKJ_CYCLE_J_P,n)}
#define IKJ_CYCLE_K_P(z,I_IDX,n) {const unsigned i_idx=I_IDX; BOOST_PP_REPEAT_2ND(n,IKJ_CYCLE_J_P,n)}

#define MATR_PROD_FUNC_IKJ(z,n,P) [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS)\
{ BOOST_PP_REPEAT_1ST(n,IKJ_CYCLE_K##P,n)}

// IJK faster when IKJ
#define MATR_PROD_FUNC(z,n,P) MATR_PROD_FUNC_IJK(z,n,P)

FSimpleMatrProd prod_funcs[] = { //prod_funcs[i] i=8..15 is function C=A*B for matrices ixi
    nullptr, nullptr, nullptr,
    MATR_PROD_FUNC(z, 3,),  // just to see how macros expanded
    nullptr, nullptr, nullptr, nullptr,
    MATR_PROD_FUNC(z, 8,),
    MATR_PROD_FUNC(z, 9,),
    MATR_PROD_FUNC(z, 10,),
    MATR_PROD_FUNC(z, 11,),
    MATR_PROD_FUNC(z, 12,),
    MATR_PROD_FUNC(z, 13,),
    MATR_PROD_FUNC(z, 14,),
    MATR_PROD_FUNC(z, 15,)
    //MATR_PROD_FUNC(z, 16,), 
};

FSimpleMatrProd prod_funcs_p[] = {  //prod_funcs_p[i] i=8..15 is function C+=A*B for matrices ixi
    nullptr, nullptr, nullptr,
    MATR_PROD_FUNC(z, 3,_P),  // just to see how macros expanded
    nullptr, nullptr, nullptr, nullptr,
    MATR_PROD_FUNC(z, 8,_P),
    MATR_PROD_FUNC(z, 9, _P),
    MATR_PROD_FUNC(z, 10, _P),
    MATR_PROD_FUNC(z, 11, _P),
    MATR_PROD_FUNC(z, 12, _P),
    MATR_PROD_FUNC(z, 13, _P),
    MATR_PROD_FUNC(z, 14, _P),
    MATR_PROD_FUNC(z, 15, _P)
    //MATR_PROD_FUNC_IJK(z, 16,_P), 
};

template <unsigned SZ>
void block_mul(double(*C)[SZ], double(*A)[SZ], double(*B)[SZ])
{
    FSimpleMatrProd f8 = prod_funcs[8];
    FSimpleMatrProd f8_p = prod_funcs_p[8];
    for (unsigned i = 0; i<SZ; i += 8)
        for (unsigned j = 0; j < SZ; j += 8)
        {
            //#define IJ_M_PTR(M,I,J) M+(I*SZ+J)
#define IJ_M_PTR(M,I,J) reinterpret_cast<double *>(&M[I][J])
            f8(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, 0), SZ, IJ_M_PTR(B, 0, j), SZ);
            for (unsigned k = 8; k<SZ; k += 8)
                f8_p(IJ_M_PTR(C, i, j), SZ, IJ_M_PTR(A, i, k), SZ, IJ_M_PTR(B, k, j), SZ);
        }
}


void STRASSEN_RECUR_MUL(unsignedSZ,double*buf,double*C00,unsignedCS,double*A00,unsignedAS,double*B00,unsignedBS)
{

if(SZ<16){prod_funcs[SZ](C00,CS,A00,AS,B00,BS);}
else{
autosz=SZ/2;

autoa10=AS*sz;
autob10=BS*sz;
autoc10=CS*sz;

autoA01=A00+sz;
autoA10=A00+AS*sz;
autoA11=A10+sz;

autoB01=B00+sz;
autoB10=B00+BS*sz;
autoB11=B10+sz;

autoC01=C00+sz;
autoC10=C00+CS*sz;
autoC11=C10+sz;

autosubm_size=sz*sz;

auto*S00=buf;
buf+=subm_size;
auto*S01=buf;
buf+=subm_size;
auto*S10=buf;
buf+=subm_size;
auto*S11=buf;
buf+=subm_size;

auto*T00=buf;
buf+=subm_size;
auto*T01=buf;
buf+=subm_size;
auto*T10=buf;
buf+=subm_size;
auto*T11=buf;
buf+=subm_size;


/*#defineMAKE_SIMPLE_OP(C,CS,PR,A,AS,OP,B,BS){autoA_=A;autoB_=B;autoC_=C;\
autoA_delta=AS-sz;autoB_delta=BS-sz;autoC_delta=CS-sz;\
double*C_last_row=C+CS*sz;\
while(C_<C_last_row){double*C_last_col=C_+sz;\
for(;C_<C_last_col;++A_,++B_,++C_){*C_PR*A_OP*B_;}\
A_+=A_delta;B_+=B_delta;C_+=C_delta;}}


MAKE_SIMPLE_OP(S00,sz,=,A10,AS,+,A11,AS);
MAKE_SIMPLE_OP(S01,sz,=,S00,sz,-,A00,AS);
MAKE_SIMPLE_OP(S10,sz,=,A00,AS,-,A10,AS);
MAKE_SIMPLE_OP(S11,sz,=,A01,AS,-,S01,sz);
MAKE_SIMPLE_OP(T00,sz,=,B01,BS,-,B00,BS);
STRASSEN_RECUR_MUL(sz,buf,T11,sz,S00,sz,T00,sz);
MAKE_SIMPLE_OP(S00,sz,=,B11,BS,-,T00,sz);
MAKE_SIMPLE_OP(T01,sz,=,B11,BS,-,B01,BS);
MAKE_SIMPLE_OP(T10,sz,=,S00,sz,-,B10,BS);
STRASSEN_RECUR_MUL(sz,buf,T00,sz,S10,sz,T01,sz);
STRASSEN_RECUR_MUL(sz,buf,T01,sz,S11,sz,B11,BS);
STRASSEN_RECUR_MUL(sz,buf,S10,sz,A11,AS,T10,sz);
STRASSEN_RECUR_MUL(sz,buf,S11,sz,S01,sz,S00,sz);
STRASSEN_RECUR_MUL(sz,buf,S00,sz,A00,AS,B00,BS);
MAKE_SIMPLE_OP(S01,sz,=,S00,sz,+,S11,sz);
MAKE_SIMPLE_OP(T10,sz,=,S01,sz,+,T00,sz);
MAKE_SIMPLE_OP(S11,sz,=,S01,sz,+,T11,sz);
STRASSEN_RECUR_MUL(sz,buf,T00,sz,A01,AS,B10,BS);
MAKE_SIMPLE_OP(C00,CS,=,S00,sz,+,T00,sz);
MAKE_SIMPLE_OP(C01,CS,=,S11,sz,+,T01,sz);
MAKE_SIMPLE_OP(C10,CS,=,T10,sz,-,S10,sz);
MAKE_SIMPLE_OP(C11,CS,=,T10,sz,+,T11,sz);*/
STRASSEN_SIMPLE_SUM(sz,S00,sz,A10,AS,A11,AS);
STRASSEN_SIMPLE_SUB(sz,S01,sz,S00,sz,A00,AS);
STRASSEN_SIMPLE_SUB(sz,S10,sz,A00,AS,A10,AS);
STRASSEN_SIMPLE_SUB(sz,S11,sz,A01,AS,S01,sz);
STRASSEN_SIMPLE_SUB(sz,T00,sz,B01,BS,B00,BS);
STRASSEN_RECUR_MUL(sz,buf,T11,sz,S00,sz,T00,sz);
STRASSEN_SIMPLE_SUB(sz,S00,sz,B11,BS,T00,sz);
STRASSEN_SIMPLE_SUB(sz,T01,sz,B11,BS,B01,BS);
STRASSEN_SIMPLE_SUB(sz,T10,sz,S00,sz,B10,BS);
STRASSEN_RECUR_MUL(sz,buf,T00,sz,S10,sz,T01,sz);
STRASSEN_RECUR_MUL(sz,buf,T01,sz,S11,sz,B11,BS);
STRASSEN_RECUR_MUL(sz,buf,S10,sz,A11,AS,T10,sz);
STRASSEN_RECUR_MUL(sz,buf,S11,sz,S01,sz,S00,sz);
STRASSEN_RECUR_MUL(sz,buf,S00,sz,A00,AS,B00,BS);
STRASSEN_SIMPLE_SUM(sz,S01,sz,S00,sz,S11,sz);
STRASSEN_SIMPLE_SUM(sz,T10,sz,S01,sz,T00,sz);
STRASSEN_SIMPLE_SUM(sz,S11,sz,S01,sz,T11,sz);
STRASSEN_RECUR_MUL(sz,buf,T00,sz,A01,AS,B10,BS);
STRASSEN_SIMPLE_SUM(sz,C00,CS,S00,sz,T00,sz);
STRASSEN_SIMPLE_SUM(sz,C01,CS,S11,sz,T01,sz);
STRASSEN_SIMPLE_SUB(sz,C10,CS,T10,sz,S10,sz);
STRASSEN_SIMPLE_SUM(sz,C11,CS,T10,sz,T11,sz);

};
}

#define  MAKE_SIMPLE_OP1(C,CS,PR,A,AS,OP,B,BS){ auto B_ = B; auto C_ = C;\
    auto B_delta = BS - sz;  auto C_delta = CS - sz;\
    double *C_last_row = C + CS*sz;\
	    while (C_ < C_last_row) { double *C_last_col = C_ + sz;\
        for(;C_ < C_last_col;++B_,++C_) {*C_ OP##= *B_;}\
        B_ += B_delta; C_ += C_delta;}}

#define  MAKE_SIMPLE_OP2(C,CS,PR,A,AS,OP,B,BS){ auto A_ = A; auto B_ = B; auto C_ = C;\
    auto A_delta = AS - sz; auto B_delta = BS - sz;  auto C_delta = CS - sz;\
    double *C_last_row = C + CS*sz;\
	    while (C_ < C_last_row) { double *C_last_col = C_ + sz;\
        for(;C_ < C_last_col;++A_,++B_,++C_) {*C_ PR *A_ OP *B_;}\
        A_ += A_delta; B_ += B_delta; C_ += C_delta;}}

#define B_DELTA_DECL_sz
#define C_DELTA_DECL_sz
#define B_DELTA_DECL_CS  auto B_delta = CS - sz;
#define C_DELTA_DECL_CS  auto C_delta = CS - sz;

#define B_INC_sz
#define C_INC_sz
#define B_INC_CS  B_ += B_delta;
#define C_INC_CS  C_ += C_delta;


#define  MAKE_SIMPLE_OP1(C,CS,PR,A,AS,OP,B,BS){ auto B_ = B; auto C_ = C;\
    B_DELTA_DECL_##BS  C_DELTA_DECL_##CS  auto C_last_row = C + CS*sz;\
	    while (C_ < C_last_row) {auto C_last_col = C_ + sz;\
        for(;C_ < C_last_col;++B_,++C_) {*C_ OP##= *B_;}\
        B_INC_##BS C_INC_##CS}}


void  STRASSEN_AUTO_SUM(unsigned sz, double *C, unsigned CS, double *B, unsigned BS)
{
    double *C_last_col;
    unsigned B_delta = BS - sz;
    unsigned C_delta = CS - sz;
    double *C_last_row = C + CS*sz;
    while (C < C_last_row) {
        C_last_col = C + sz;
        for (; C < C_last_col; ++B, ++C)  *C += *B;
        B += B_delta;
        C += C_delta;
    }
};



void  STRASSEN_AUTO_SUB(unsigned sz, double *C, unsigned CS, double *B, unsigned BS)
{
    double *C_last_col;
    unsigned B_delta = BS - sz;
    unsigned C_delta = CS - sz;
    double *C_last_row = C + CS*sz;
    while (C < C_last_row) {
        C_last_col = C + sz;
        for (; C < C_last_col; ++B, ++C)  *C -= *B;
        B += B_delta;
        C += C_delta;
    }
};




MAKE_SIMPLE_OP2(C00,CS,=,A00,AS,-,A10,AS);
MAKE_SIMPLE_OP2(C01,CS,=,B11,BS,-,B01,BS);
STRASSEN_RECUR_MUL(sz,buf,C10,CS,C00,CS,C01,CS);
MAKE_SIMPLE_OP2(C00,CS,=,B01,BS,-,B00,BS);
MAKE_SIMPLE_OP2(S00,sz,=,B11,BS,-,C00,CS);
MAKE_SIMPLE_OP2(C01,CS,=,S00,sz,-,B10,BS);
MAKE_SIMPLE_OP2(C11,CS,=,A10,AS,+,A11,AS);
MAKE_SIMPLE_OP2(S01,sz,=,C11,CS,-,A00,AS);
MAKE_SIMPLE_OP2(S10,sz,=,A01,AS,-,S01,sz);
STRASSEN_RECUR_MUL(sz,buf,S11,sz,S01,sz,S00,sz);
STRASSEN_RECUR_MUL(sz,buf,S00,sz,S10,sz,B11,BS);
STRASSEN_RECUR_MUL(sz,buf,S10,sz,A11,AS,C01,CS);
STRASSEN_RECUR_MUL(sz,buf,C01,CS,C11,CS,C00,CS);
STRASSEN_RECUR_MUL(sz,buf,C00,CS,A00,AS,B00,BS);
STRASSEN_RECUR_MUL(sz,buf,C11,CS,A01,AS,B10,BS);
MAKE_SIMPLE_OP1(S11,sz,=,S11,sz,+,C00,CS);
MAKE_SIMPLE_OP2(C10,CS,=,S11,sz,+,C10,CS);
MAKE_SIMPLE_OP1(C00,CS,=,C00,CS,+,C11,CS);
MAKE_SIMPLE_OP2(C11,CS,=,C10,CS,+,C01,CS);
MAKE_SIMPLE_OP1(C10,CS,=,C10,CS,-,S10,sz);
MAKE_SIMPLE_OP2(C01,CS,+=,S11,sz,+,S00,sz);
-----------------------------------------------------------------
STRASSEN_SIMPLE_SUB(sz,C00,CS,A00,AS,A10,AS);
STRASSEN_SIMPLE_SUB(sz,C01,CS,B11,BS,B01,BS);
STRASSEN_RECUR_MUL(sz,buf,C10,CS,C00,CS,C01,CS);
STRASSEN_SIMPLE_SUB(sz,C00,CS,B01,BS,B00,BS);
STRASSEN_SIMPLE_SUB(sz,S00,sz,B11,BS,C00,CS);
STRASSEN_SIMPLE_SUB(sz,C01,CS,S00,sz,B10,BS);
STRASSEN_SIMPLE_SUM(sz,C11,CS,A10,AS,A11,AS);
STRASSEN_SIMPLE_SUB(sz,S01,sz,C11,CS,A00,AS);
STRASSEN_SIMPLE_SUB(sz,S10,sz,A01,AS,S01,sz);
STRASSEN_RECUR_MUL(sz,buf,S11,sz,S01,sz,S00,sz);
STRASSEN_RECUR_MUL(sz,buf,S00,sz,S10,sz,B11,BS);
STRASSEN_RECUR_MUL(sz,buf,S10,sz,A11,AS,C01,CS);
STRASSEN_RECUR_MUL(sz,buf,C01,CS,C11,CS,C00,CS);
STRASSEN_RECUR_MUL(sz,buf,C00,CS,A00,AS,B00,BS);
STRASSEN_RECUR_MUL(sz,buf,C11,CS,A01,AS,B10,BS);
MAKE_SIMPLE_OP1(S11,sz,=,S11,sz,+,C00,CS);
STRASSEN_SIMPLE_SUM(sz,C10,CS,S11,sz,C10,CS);
MAKE_SIMPLE_OP1(C00,CS,=,C00,CS,+,C11,CS);
STRASSEN_SIMPLE_SUM(sz,C11,CS,C10,CS,C01,CS);
MAKE_SIMPLE_OP1(C10,CS,=,C10,CS,-,S10,sz);
//MAKE_SIMPLE_OP2(C01,CS,+=,S11,sz,+,S00,sz);
auto C01_delta = CS - sz;
auto C01_last_row = C01 + CS*sz;
while (C01 < C01_last_row) {
    auto C01_last_col = C01 + sz;
    for (; C01 < C01_last_col; ++S11, ++S00, ++C01) { *C01 += *S11 + *S00; }
    C01 += C01_delta;
}




STRASSEN_SIMPLE_SUB(sz, C00, CS, A00, AS, A10, AS);
STRASSEN_SIMPLE_SUB(sz, C01, CS, B11, BS, B01, BS);
STRASSEN_RECUR_MUL(sz, buf, C10, CS, C00, CS, C01, CS);
STRASSEN_SIMPLE_SUB(sz, C00, CS, B01, BS, B00, BS);
STRASSEN_SIMPLE_SUB(sz, S00, sz, B11, BS, C00, CS);
STRASSEN_SIMPLE_SUB(sz, C01, CS, S00, sz, B10, BS);
STRASSEN_SIMPLE_SUM(sz, C11, CS, A10, AS, A11, AS);
STRASSEN_SIMPLE_SUB(sz, S01, sz, C11, CS, A00, AS);
STRASSEN_SIMPLE_SUB(sz, S10, sz, A01, AS, S01, sz);
STRASSEN_RECUR_MUL(sz, buf, S11, sz, S01, sz, S00, sz);
STRASSEN_RECUR_MUL(sz, buf, S00, sz, S10, sz, B11, BS);
STRASSEN_RECUR_MUL(sz, buf, S10, sz, A11, AS, C01, CS);
STRASSEN_RECUR_MUL(sz, buf, C01, CS, C11, CS, C00, CS);
STRASSEN_RECUR_MUL(sz, buf, C00, CS, A00, AS, B00, BS);
STRASSEN_RECUR_MUL(sz, buf, C11, CS, A01, AS, B10, BS);
STRASSEN_SIMPLE_SUM(sz, S11, sz, S11, sz,  C00, CS);
STRASSEN_SIMPLE_SUM(sz, C10, CS, S11, sz, C10, CS);
STRASSEN_SIMPLE_SUM(sz, C00, CS, C00, CS,  C11, CS);
STRASSEN_SIMPLE_SUM(sz, C11, CS, C10, CS, C01, CS);
STRASSEN_SIMPLE_SUB(sz, C10, CS, C10, CS, S10, sz);
MAKE_SIMPLE_OP2(C01, CS, +=, S11, sz, +, S00, sz);



void  SIMPLE_RECUR_ADD_MUL(unsigned SZ, unsigned _SZ, double *C00, double *A00, double *B00)
{
    if (SZ < 16) { prod_funcs_p_t[SZ](C00, _SZ, A00, _SZ, B00, _SZ); }
    else {
        auto sz = SZ / 2;

        auto d10 = _SZ*sz;

        auto A01 = A00 + sz;
        auto A10 = A00 + d10;
        auto A11 = A10 + sz;

        // B - transposed matrix!!
        auto B10 = B00 + sz;
        auto B01 = B00 + d10;
        auto B11 = B01 + sz;

        auto C01 = C00 + sz;
        auto C10 = C00 + d10;
        auto C11 = C10 + sz;

        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C00, A00, B00);
        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C00, A01, B10);
        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C01, A00, B01);
        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C01, A01, B11);
        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C10, A10, B00);
        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C10, A11, B10);
        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C11, A10, B01);
        SIMPLE_RECUR_ADD_MUL(sz, _SZ, C11, A11, B11);
    }

};
template <unsigned SZ>
void  recur_mul(double(*C)[SZ], double(*A)[SZ], double(*B)[SZ])
{
    double *c = reinterpret_cast<double *>(C);
    double *c_end = c + SZ*SZ;
    while (c < c_end)*c++ = 0;
    transp(B);
    SIMPLE_RECUR_ADD_MUL(SZ, SZ, reinterpret_cast<double *>(C), reinterpret_cast<double *>(A), reinterpret_cast<double *>(B));
    transp(B);
};

typedef void(*FAdditiveOp)(double *, double *, double *);
#define ADDITIVE_OP(z, k, OP) C[k]= A[k] OP B[k];
#define UNROLL_FUNC(z,n,OP) [](double *C,double *A,double *B) {BOOST_PP_REPEAT_1ST(n,ADDITIVE_OP,OP)}
#define UNROLL_ELEM(z,n,OP) ,UNROLL_FUNC(z,n,OP)
#define DEFINE_UNROLL_FUNCS(name,n,OP) FAdditiveOp name []={nullptr BOOST_PP_REPEAT_FROM_TO_2ND(1,n,UNROLL_ELEM,OP)};

DEFINE_UNROLL_FUNCS(unroll_sum,65,+)
DEFINE_UNROLL_FUNCS(unroll_sub,65, -)

/*void  STRASSEN_SIMPLE_SUM(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
double *C_last_row = C + CS*sz;
while (C < C_last_row) {
for (unsigned i = 0; i<sz; i += 64) {
auto d = sz - i;
if (d > 64)d = 64;
unroll_sum[d](C + i, A + i, B + i);
}
A += AS;
B += BS;
C += CS;
}
};*/

/*void  STRASSEN_SIMPLE_SUB(unsigned sz, double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
double *C_last_row = C + CS*sz;
while (C < C_last_row) {
for(unsigned i=0;i<sz;i+=64){
auto d = sz - i;
if (d > 64)d = 64;
unroll_sub[d](C + i, A + i, B + i);
}
A += AS;
B += BS;
C += CS;
}

};*/


void STRASSEN_MUL_PREFIX_TB(unsigned sz, unsigned BS, double *T00, double *T01, double *T10, double *T11, double *B00, double *B01, double *B10, double *B11)
{

	auto B_delta = BS - sz;
	auto T_last_row = T00 + sz*sz;
	auto sz4 = sz % 4;
	while (T00 < T_last_row) {
		for (auto T_last_col = T00 + (sz - sz4); T00 < T_last_col; T00 += 4, T01 += 4, T10 += 4, T11 += 4, B00 += 4, B01 += 4, B10 += 4, B11 += 4)
		{
			__m256d ymm00 = _mm256_loadu_pd(B00);
			__m256d ymm01 = _mm256_loadu_pd(B01);
			__m256d ymm10;
			__m256d ymm11 = _mm256_loadu_pd(B11);

			//T10 = B11 - B01
			ymm10 = _mm256_sub_pd(ymm11, ymm01);
			_mm256_storeu_pd(T10, ymm10);
			ymm10 = _mm256_loadu_pd(B10);

			//T00 = B01 - B00
			ymm00 = _mm256_sub_pd(ymm01, ymm00);
			//T01 = B11 - T00
			ymm01 = _mm256_sub_pd(ymm11, ymm00);
			//T11 = T01 - B10
			ymm11 = _mm256_sub_pd(ymm01, ymm10);
			_mm256_storeu_pd(T00, ymm00);
			_mm256_storeu_pd(T01, ymm01);
			_mm256_storeu_pd(T11, ymm11);
		}
		for (auto T_last_col = T00 + sz; T00 < T_last_col; ++T00, ++T01, ++T10, ++T11, ++B00, ++B01, ++B10, ++B11)
		{
			*T00 = *B01 - *B00;
			*T01 = *B11 - *T00;
			*T10 = *B11 - *B01;
			*T11 = *T01 - *B10;

		}
		B00 += B_delta;
		B01 += B_delta;
		B10 += B_delta;
		B11 += B_delta;
	}

};

void STRASSEN_MUL_PREFIX_CA(unsigned sz, unsigned AS, unsigned CS, double *C00, double *C01, double *C10, double *C11, double *A00, double *A01, double *A10, double *A11)
{

	auto A_delta = AS - sz;
	auto C_delta = CS - sz;
	auto C_last_row = C00 + CS*sz;
	auto sz4 = sz % 4;
	while (C00 < C_last_row) {
		for (auto C_last_col = C00 + (sz - sz4); C00 < C_last_col; C00 += 4, C01 += 4, C10 += 4, C11 += 4, A00 += 4, A01 += 4, A10 += 4, A11 += 4)
		{
			__m256d ymm00 = _mm256_loadu_pd(A00);
			__m256d ymm01 = _mm256_loadu_pd(A01);
			__m256d ymm10 = _mm256_loadu_pd(A10);
			__m256d ymm11 = _mm256_loadu_pd(A11);

			//*C10 = *A00 - *A10;
			__m256d zmm10 = _mm256_sub_pd(ymm00, ymm10);
			//*C01 = *A10 + *A11;
			__m256d zmm01 = _mm256_add_pd(ymm10, ymm11);
			//*C00 = *C01 - *A00;
			ymm00 = _mm256_sub_pd(zmm01, ymm00);
			//*C11 = *C00 - *A01;
			ymm11 = _mm256_sub_pd(ymm00, ymm01);

			_mm256_storeu_pd(C00, ymm00);
			_mm256_storeu_pd(C01, zmm01);
			_mm256_storeu_pd(C10, zmm10);
			_mm256_storeu_pd(C11, ymm11);


		}
		for (auto C_last_col = C00 + sz4; C00 < C_last_col; ++C00, ++C01, ++C10, ++C11, ++A00, ++A01, ++A10, ++A11)
		{
			*C10 = *A00 - *A10;
			*C01 = *A10 + *A11;
			*C00 = *C01 - *A00;
			*C11 = *C00 - *A01;

		}
		A00 += A_delta;
		A01 += A_delta;
		A10 += A_delta;
		A11 += A_delta;
		C00 += C_delta;
		C01 += C_delta;
		C10 += C_delta;
		C11 += C_delta;
	}

};


void  STRASSEN_MUL(unsigned SZ, double *buf, double *C00, unsigned CS, double *A00, unsigned AS, double *B00, unsigned BS)
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

		auto *S00 = buf;
		buf += subm_size;
		auto *S01 = buf;
		buf += subm_size;
		auto *S10 = buf;
		buf += subm_size;
		auto *T00 = buf;
		buf += subm_size;
		auto *T01 = buf;
		buf += subm_size;
		auto *T10 = buf;
		buf += subm_size;
		auto *T11 = buf;
		buf += subm_size;
		STRASSEN_MUL_PREFIX_TB(sz, BS, T00, T01, T10, T11, B00, B01, B10, B11);
		STRASSEN_MUL_PREFIX_CA(sz, AS, CS, C00, C01, C10, C11, A00, A01, A10, A11);
		STRASSEN_MUL(sz, buf, S00, sz, A00, AS, B00, BS);
		STRASSEN_MUL(sz, buf, S01, sz, C11, CS, B11, BS);
		STRASSEN_MUL(sz, buf, S10, sz, C10, CS, T10, sz);
		STRASSEN_MUL(sz, buf, C10, CS, A11, AS, T11, sz);
		STRASSEN_MUL(sz, buf, C11, CS, C01, CS, T00, sz);
		STRASSEN_MUL(sz, buf, C01, CS, C00, CS, T01, sz);
		STRASSEN_MUL(sz, buf, C00, CS, A01, AS, B10, BS);
		STRASSEN_MUL_SUFFIX(sz, CS, C00, C01, C10, C11, S00, S01, S10);
	}
}




void mul8(double * C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
    double *C_last_row = C + CS * 8;
    while (C < C_last_row) {
        __m256d ymm0, ymm1, ymm2, ymm3, res;
        __m256d AmmF = _mm256_loadu_pd(A);
        __m256d Amm0 = _mm256_loadu_pd(A + 4);
        BM_AV_PROD4n(z, 0, 8);
        BM_AV_PROD4n(z, 1, 8);
        A += AS;
        C += CS;
    }

};

void mul8p(double * C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS)
{
    double *C_last_row = C + CS * 8;
    while (C < C_last_row) {
        __m256d ymm0, ymm1, ymm2, ymm3, res;
        __m256d AmmF = _mm256_loadu_pd(A);
        __m256d Amm0 = _mm256_loadu_pd(A + 4);
        BM_AV_PROD4n_P(z, 0, 8);
        BM_AV_PROD4n_P(z, 1, 8);
        A += AS;
        C += CS;
    }

};


int main()
{
#define ALLOC_MATR(M)	boost::scoped_array<row_t> M##_(new row_t[matr_size]); row_t *M = M##_.get()
    ALLOC_MATR(a);
    ALLOC_MATR(b);
    ALLOC_MATR(S);
    ALLOC_MATR(R);
    ALLOC_MATR(ST);
    //ALLOC_MATR(B);
    ALLOC_MATR(BT);
    srand(0 * (unsigned)time(NULL));
    for (unsigned i = 0; i<matr_size; i++)
        for (unsigned j = 0; j < matr_size; j++) {
            a[i][j] = rand() / 1000;
            b[i][j] = rand() / 1000;
        };
    std::cout << "press enter to start\n";
    getchar();
    std::cout << "wait...\n";
    int n_repeats = 1;
#define _CUB(a) ((a)*(a)*(a)+1)
#define _REPEAT for(n_repeats=0;n_repeats<_CUB(2.5*1024.0/matr_size);n_repeats++)
    //#define _REPEAT

    auto t_ = clock();
    //simple_mul(matr_size, reinterpret_cast<double *>(S), reinterpret_cast<double *>(a), reinterpret_cast<double *>(b));
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
    double calc_dif = 2000.0 / matr_size;
    calc_dif = calc_dif*calc_dif*calc_dif / n_repeats;

    std::cout << "size " << matr_size << "\n";
    std::cout << "n repeats " << n_repeats << "\n";

    //std::cout << "simple " << d_simple << "\n";
    //std::cout << "recur " << d_recur << "\n";
    //std::cout << "block " << d_block << "\n";
    std::cout << "block t " << d_block_t / n_repeats << "\n";
    std::cout << "strassen t " << d_strassen / n_repeats << "\n";
    std::cout << "block d " << d_block_t*calc_dif << "\n";
    std::cout << "strassen d " << d_strassen*calc_dif << "\n";
    //std::cout << "diff simple strassen " << matr_dif(S, ST) << "\n";

    //std::cout << "diff block block transp " << matr_dif(B, BT) << "\n";
    std::cout << "diff block transp strassen " << matr_dif(BT, ST) << "\n";
    //std::cout << "diff block recur " << matr_dif(B, R) << "\n";
    //std::cout << "diff block strassen " << matr_dif(B, ST) << "\n";
    //std::cout << "ratio simple/strassen  " << d_simple / d_strassen << "\n";
    std::cout << "ratio block transp/strassen  " << d_block_t / d_strassen << "\n";
    //std::cout << "ratio recur/strassen  " << d_recur / d_strassen << "\n";
    /*     std::cout << "\n";
    print_matrix("A", a);
    print_matrix("B", b);

    std::cout << "\n";
    //print_matrix("block", B);
    print_matrix("simple", S);
    print_matrix("block", BT);
    print_matrix("strassen", ST);
    */
    getchar();
    return 0;
}


void  strassen_recur_inv(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
{
    if (SZ < count_of(inv_funcs)) {
        inv_funcs[SZ](I, IS, A_, AS);
    }
    else {
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

        auto *AI = buf; buf += subm_size;
        strassen_recur_inv(sz, buf, AI, sz, A, AS); // AI - inverse A 
        auto *AImulB = buf; buf += subm_size;
        strassen_transp_and_mul(sz, buf, AImulB, sz, AI, sz, B, AS);
        auto *CmulAImulB = buf; buf += subm_size;
        strassen_transp_and_mul(sz, buf, CmulAImulB, sz, C, AS, AImulB, sz);
        auto Z = CmulAImulB; // storage for D-C*AI*B. Same location as C*AI*B
        matrix_sub(sz, Z, sz, CmulAImulB, sz, D, AS); //Z=C*AI*B-D
        strassen_recur_inv(sz, buf, I11, IS, Z, sz); //ZI calculated and stored in I11

        strassen_transp_and_mul(sz, buf, I01, IS, AImulB, sz, I11, IS);  //I01=AI*B*ZI

        auto AIT = Z; // storage for transpoze AI. Same location as Z
        transp(sz, AIT, AI, sz);
        auto AITmulCT = AImulB;  // storage for AIT*CT. Same location as AImulB (we dont need AImulB any more)
        strassen_recur_mul_by_transposed(sz, buf, AITmulCT, sz, AIT, sz, C, AS);
        auto I01mulCmulAI = AIT;


        strassen_recur_mul_by_transposed(sz, buf, I01mulCmulAI, sz, I01, IS, AITmulCT, sz); //I01mulCmulAI=AI*B*ZI*C*AI

        matrix_sub(sz, I00, IS, AI, sz, I01mulCmulAI, sz);  // I00=AI-AI*B*ZI*C*AI
        strassen_recur_mul_by_transposed(sz, buf, I10, IS, I11, IS, AITmulCT, sz); // I10=ZI*C*AI
        change_sign(sz, I11, IS);  //I11 = -ZI
    }
}


void trim_zero(unsigned SZ, double *M)
{
    memset(M + SZ*(SZ - 1), 0, SZ * sizeof(M[0]));//last row
                                                  //last col
    M += SZ - 1;

    for (auto M_end = M + SZ*SZ; M<M_end; M += SZ)
        *M = 0;
};


void  strassen_recur_inv_odd(unsigned SZ, double *buf, double *I, unsigned IS, double *A_, unsigned AS)
{
    auto sz_A = SZ / 2;
    auto sz_D = SZ - sz_A;
    //_ASSERT(sz_D==sz_A+1);

    auto A = A_;
    auto B = A + sz_A;
    auto C = A + AS*sz_A;
    auto D = C + sz_A;

    auto I00 = I;
    auto I01 = I00 + sz_A;
    auto I10 = I00 + IS*sz_A;
    auto I11 = I10 + sz_A;

    /*


    | I00  I01 |   | A   B | -1
    |          | = |       |
    | I10  I11 |   | C   D |
    see https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion for formulas
    */

    auto subm_size_A = sz_A*sz_A;
    auto subm_size_D = sz_D*sz_D;

    auto AI = buf; buf += subm_size_D;
    trim_zero(sz_D, AI);
    strassen_recur_inv(sz_A, buf, AI, sz_D, A, AS); // AI - inverse A sz_A*sz_A stored in enlarged matrix sz_D*sz_D
    auto AImulB = buf; buf += subm_size_D;

    auto BT = buf; buf += subm_size_D;
    //B has actual size sz_D*sz_A and  BT has size sz_D*sz_D, 
    //so we need to make zero last BT col   (sz_D=sz_A+1)
    transp_and_set_last_col_zero(sz_D, BT, B, AS);
    strassen_recur_mul_by_transposed(sz_D, buf, AImulB, sz_D, AI, sz_D, BT, sz_D);

    auto *CmulAImulB = buf; buf += subm_size_D;

    //store first column of D and make it zero because it is last column of enlaged C
    // C has actual size sz_A*sz_D and we need to add zeros to make it sz_D*sz_D
    auto D_stored = buf;
    buf += sz_D;
    for (auto D_first = D, D_end = D + AS*sz_D, D_stored_cur = D_stored; D_first < D_end; D_first += AS, D_stored_cur++) {
        *D_stored_cur = *D_first;
        *D_first = 0;
    }
    strassen_transp_and_mul(sz_D, buf, CmulAImulB, sz_D, C, AS, AImulB, sz_D);
    for (auto D_first = D, D_end = D + AS*sz_D, D_stored_cur = D_stored; D_first < D_end; D_first += AS, D_stored_cur++)
        *D_first = *D_stored_cur;  // restore first column of D 
    buf -= sz_D;
    auto Z = CmulAImulB;
    matrix_sub(sz_D, Z, sz_D, CmulAImulB, sz_D, D, AS); //Z=C*AI*B-D 
    auto ZI = buf; buf += subm_size_D;

    strassen_recur_inv(sz_D, buf, ZI, sz_D, Z, sz_D); //ZI calculated 


    strassen_transp_and_mul(sz_D, buf, I01, IS, AImulB, sz_D, ZI, sz_D);  //I01=AI*B*ZI

    auto AIT = Z;
    trim_zero(sz_D, AIT);
    transp(sz_A, AIT, AI, sz_A);
    auto T1 = AImulB;
    strassen_recur_mul_by_transposed(sz_D, buf, T1, sz_D, AIT, sz_D, C, AS);//T1=AIT*CT
    auto T2 = Z;
    strassen_recur_mul_by_transposed(sz_D, buf, T2, sz_D, I01, IS, T1, sz_D); //T2=-AI*B*ZI*C*AI

    matrix_sub(sz_A, I00, IS, AI, sz_A, T2, sz_A);  // I00=AI-AI*B*ZI*C*AI
    strassen_recur_mul_by_transposed(sz_D, buf, I10, IS, ZI, sz_D, T1, sz_D); // I10=ZI*C*AI
    copy_and_change_sign(sz_D, I11, IS, ZI, sz_D);  //I11 = -ZI
}


/*    [](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS) { *C = A[0] * B[0]; },
[](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS) {
{
auto B_ = B;
*C = A[0] * B_[0] + A[1] * B_[1];  C++; B_ += BS;
*C = A[0] * B_[0] + A[1] * B_[1];  C += CS - 1; A += AS;
}
*C = A[0] * B[0] + A[1] * B[1];   C++;  B += BS;
*C = A[0] * B[0] + A[1] * B[1];
},
[](double *C,unsigned CS,double *A,unsigned AS,double *B,unsigned BS) {
{
auto B_ = B;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C += CS - 2; A += AS;
}
{
auto B_ = B;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C += CS - 2; A += AS;
}
{
auto B_ = B;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
*C = A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2];
}
},   */

[](double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS) {
    {
        auto B_ = B;
        *C += A[0] * B_[0] + A[1] * B_[1];  C++; B_ += BS;
        *C += A[0] * B_[0] + A[1] * B_[1];  C += CS - 1; A += AS;
    }
    *C += A[0] * B[0] + A[1] * B[1];   C++;  B += BS;
    *C += A[0] * B[0] + A[1] * B[1];
},
[](double *C, unsigned CS, double *A, unsigned AS, double *B, unsigned BS) {
    {
        auto B_ = B;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C += CS - 2; A += AS;
    }
    {
        auto B_ = B;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C += CS - 2; A += AS;
    }
    {
        auto B_ = B;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2]; C++; B_ += BS;
        *C += A[0] * B_[0] + A[1] * B_[1] + A[2] * B_[2];
    }
},
/*
#define  a _A[0]
#define  b _A[1]
#define  c _A[2]
auto _B = _A + AS;
#define  d _B[0]
#define  e _B[1]
#define  f _B[2]
auto _C = _B + AS;
#define  g _C[0]
#define  h _C[1]
#define  i _C[2]
auto A = e*i - f*h;
auto D = c*h - b*i;
auto G = b*f - c*e;
auto B = f*g - d*i;
auto E = a*i - c*g;
auto H = c*d - a*f;
auto C = d*h - e*g;
auto F = b*g - a*h;
auto I = a*e - b*d;
auto det_rev = 1 / (a*A + b*B + c*C);
I[0] = det_rev*A;
I[1] = det_rev*D;
I[2] = det_rev*G;
I += IS;
I[0] = det_rev*B;
I[1] = det_rev*E;
I[2] = det_rev*H;
I += IS;
I[0] = det_rev*C;
I[1] = det_rev*F;
I[2] = det_rev*I;


#undef  a _A[0]
#undef  b _A[1]
#undef  c _A[2]
#undef  d _B[0]
#undef  e _B[1]
#undef  f _B[2]
#undef  j _C[0]
#undef  h _C[1]
#undef  i _C[2]  */
