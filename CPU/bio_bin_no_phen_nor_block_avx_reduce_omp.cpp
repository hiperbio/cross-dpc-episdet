#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <immintrin.h>
#include <float.h>
#include <limits.h>
#include <omp.h>

#define VEC_ELEMS 8
#define ALIGN 32
#define NUM_THREADS 72

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

int roundUp(int long long numToRound, int multiple) 
{
    return ((numToRound + multiple - 1) / multiple) * multiple;
}


#define TABLE_MAX_SIZE 748 //5060
#define TABLE_ERROR -0.0810 //-0.08104
float *addlogtable;

#if defined (ADVISOR)
    #include <ittnotify.h>
#endif

#if defined (LIKWID)
    #include <likwid.h>
#endif

#if defined (PAPI) || defined (PAPI_POW)
    #include <papi.h>

    int EventSet_pow[NUM_THREADS]; 
#endif

#define L1CACHESIZE 32768
#define L2CACHESIZE 262144
#define L3CACHESIZE 8388608

#define FREQ 2400000000
double POM[NUM_THREADS*(L1CACHESIZE+L2CACHESIZE+L3CACHESIZE)/8];

static inline void clear_cache_deep(){
	unsigned int i;
	for (i = 0; i < (NUM_THREADS*(L1CACHESIZE+L2CACHESIZE+L3CACHESIZE)/sizeof(double)); i++){
		POM[i] = 3.14*i;
	}
}

static inline long long read_tsc_start(){
	uint64_t d;
	uint64_t a;
	asm __volatile__ (
		"CPUID;"
		"rdtsc;"
		"movq %%rdx, %0;"
		"movq %%rax, %1;"
		: "=r" (d), "=r" (a)
		:
		: "%rax", "%rbx","%rcx", "%rdx"
	);

	return ((long long)d << 32 | a);
}

static inline long long read_tsc_end(){
	uint64_t d;
	uint64_t a;
	asm __volatile__ ("rdtscp;"
		"movq %%rdx, %0;"
		"movq %%rax, %1;"
		"CPUID;"
		: "=r" (d), "=r" (a)
		:
		: "%rax", "%rbx","%rcx", "%rdx"
	);

	return ((long long)d << 32 | a);
}

static inline void serialize(){
	asm volatile ( "xorl %%eax, %%eax \n cpuid " : : : "%eax","%ebx","%ecx","%edx" );
}

void generate_data(int long long N, int long long M, uint8_t **A, uint8_t **B){
    int i, j;

    srand(100);
    *A = (uint8_t *) _mm_malloc(N*M*sizeof(uint8_t), ALIGN);
    *B = (uint8_t *) _mm_malloc(N*sizeof(uint8_t), ALIGN);

    //Generate SNPs
    for (i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            //Generate Between 0 and 2
            (*A)[i*M + j] = rand() % 3;

        }
    }
    
        
    //Generate Phenotype
    /* for(i = 0; i < N; i++){
        //Generate Between 0 and 1
        (*B)[i] = rand() % 2;
    } */

    //Generate Phenotype 50/50
    for(i = 0; i < N; i++){
        //Generate Between 0 and 1
        (*B)[i] = 0;
    }
    for(i = 0; i < N/2; i++){
        int index = (int) (N * ((double) rand() / (RAND_MAX)));
		while((*B)[index] == 1){
            index = (int) (N * ((double) rand() / (RAND_MAX)));
        }
  		(*B)[index] = 1;
    }
}

void input_data(const char* path, uint8_t** data, uint8_t** phen, int long long* num_pac, int long long* num_snp)
{
	int i, j, temp;
	uint8_t *tdata, *tphen;
    int samplesize, locisize, nrows, ncols, data_col;

    std::string line;
	std::ifstream in(path);
	std::getline(in, line);
	std::istringstream test(line);
	std::string word;

	// get "data_col" and "locisize"
	i = 0;
	while(!test.eof())
	{
		std::getline(test, word, ',');
		i++;
	}
	data_col = i;
	locisize = i - 1;

	// get "samplesize"
	j = 0;
	while(!in.eof())
	{   
		std::getline(in, line);
		j++;
	}
	samplesize = j - 1;
	in.close();
	
	nrows = samplesize;
	ncols = locisize;

	// create "tdata" and "tphen"
    tdata = (uint8_t*) _mm_malloc(nrows * ncols * sizeof(uint8_t), 64);
    memset(tdata, 0, nrows * ncols * sizeof(uint8_t));
    tphen = (uint8_t*) _mm_malloc(nrows * sizeof(uint8_t), 64);
    memset(tphen, 0, nrows * sizeof(uint8_t));

	std::ifstream in1(path);
	std::getline(in1, line);
	std::istringstream test1(line);

	// fill "tdata" and "tphen"
	i = 0;
	while(!in1.eof())
	{
		if(i == nrows)
			break;
		std::getline(in1, line);
		std::istringstream values(line);
		j = 0;
		while(!values.eof())
		{
			std::getline(values, word, ',');
			std::istringstream int_iss(word);
			int_iss >> temp;
            if(j < ncols)
			    tdata[i * ncols + j] = temp;
            else
                tphen[i] = temp;
            j++;
		}
		i++;
	}
	in1.close();

    (*data) = tdata;
    (*phen) = tphen;
    (*num_pac) = nrows;
    (*num_snp) = ncols;
	// end
}

uint8_t * transpose_data(int long long N, int long long M, uint8_t * A){
    int i, j;
    
    uint8_t *A_trans = (uint8_t *) _mm_malloc(M*N*sizeof(uint8_t), ALIGN);
    
    for (i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            A_trans[j*N + i] = A[i*M+ + j];
        }
    }

    _mm_free(A);

    return A_trans;

}

void transposed_to_binary(uint8_t* original, uint8_t* original_ph, uint32_t** data_zeros, uint32_t** data_ones, int long long* phen_ones, int long long num_snp, int long long num_pac)
{

    int PP = ceil((1.0*num_pac)/32.0);
    uint32_t temp;
     int i, j, x_zeros, x_ones, n_zeros, n_ones;

    (*phen_ones) = 0;

    for(i = 0; i < num_pac; i++){
        if(original_ph[i] == 1){
            (*phen_ones) ++;
        }
    }

    int PP_ones = ceil((1.0*(*phen_ones))/32.0);
    int PP_zeros = ceil((1.0*(num_pac - (*phen_ones)))/32.0);

    // allocate data
    *data_zeros = (uint32_t*) _mm_malloc(num_snp*PP_zeros*2*sizeof(uint32_t), ALIGN);
    *data_ones = (uint32_t*) _mm_malloc(num_snp*PP_ones*2*sizeof(uint32_t), ALIGN);
    memset((*data_zeros), 0, num_snp*PP_zeros*2*sizeof(uint32_t));
    memset((*data_ones), 0, num_snp*PP_ones*2*sizeof(uint32_t));

    for(i = 0; i < num_snp; i++)
    {
        x_zeros = -1;
        x_ones = -1;
        n_zeros = 0;
        n_ones = 0;

        for(j = 0; j < num_pac; j++){
            temp = (uint32_t) original[i * num_pac + j];

            if(original_ph[j] == 1){
                if(n_ones%32 == 0){
                    x_ones ++;
                }
                // apply 1 shift left to 2 components
                (*data_ones)[i * PP_ones * 2 + x_ones*2 + 0] <<= 1;
                (*data_ones)[i * PP_ones * 2 + x_ones*2 + 1] <<= 1;
                // insert '1' in correct component
                if(temp == 0 || temp == 1){
                    (*data_ones)[i * PP_ones * 2 + x_ones*2 + temp ] |= 1;
                }
                n_ones ++;
            }else{
                if(n_zeros%32 == 0){
                    x_zeros ++;
                }
                // apply 1 shift left to 2 components
                (*data_zeros)[i * PP_zeros * 2 + x_zeros*2 + 0] <<= 1;
                (*data_zeros)[i * PP_zeros * 2 + x_zeros*2 + 1] <<= 1;
                // insert '1' in correct component
                if(temp == 0 || temp == 1){
                    (*data_zeros)[i * PP_zeros * 2 + x_zeros*2 + temp] |= 1;
                }
                n_zeros ++;
            }
        }
    }
}

uint32_t * transform_data_2_block(int long long N, int long long M, uint32_t * A, int block_pac, int block_snp){
    int i, j, ii, jj;

    int PP = ceil((1.0*(N))/32.0);

    int long long M_r = roundUp(M, block_snp); 
    int long long PP_r = roundUp(PP, block_pac);
    
    uint32_t *A_2_block = (uint32_t *) _mm_malloc(M_r*PP_r*2*sizeof(uint32_t), ALIGN);

    //PLACE SNPs in BLOCK MEMORY FORMAT
    for(i = 0; i < M; i+= block_snp){
        for(j = 0; j < PP; j+= block_pac){
            for(jj = 0; jj < block_pac && jj < PP - j; jj++){
                for(ii = 0; ii < block_snp && ii < M - i; ii++){
                    A_2_block[i*PP_r*2 + j*block_snp + ii*block_pac + jj] = A[(i+ii)*PP*2 + j*2 + jj*2 + 0];
                    A_2_block[i*PP_r*2 + block_snp*PP_r + j*block_snp + ii*block_pac + jj] = A[(i+ii)*PP*2 + j*2 + jj*2 + 1];
                }
            }
        }
    }

    _mm_free(A);

    return A_2_block;

}

// Returns value from addlog table or approximation, depending on number
float addlog(int n)
{
	if(n < TABLE_MAX_SIZE)
		return addlogtable[n];
	else
	{
		float x = (n + 0.5)*log(n) - (n - 1)*log(exp(1)) + TABLE_ERROR;
		return x;
	}
}

// Computes addlog table
float my_factorial(int n)
{
	float z = 0;

	if(n < 0)
	{
		printf("Error: n should be a non-negative number.\n");
		return 0;
	}
	if(n == 0)
		return 0;
	if(n == 1)
		return 0;

	z = addlogtable[n - 1] + log(n);
	return z;
}

void process_epi_bin_no_phen_nor_block(uint32_t* data_zeros, uint32_t* data_ones, int long long phen_ones, int dim_epi, int long long num_snp, int long long num_pac, int num_combs, int block_snp, int block_pac)
{

    float best_score = FLT_MAX;
    int long long cyc_s = LLONG_MAX, cyc_e = 0;
    omp_set_num_threads(NUM_THREADS);
    int best_snp_global[NUM_THREADS][3];
    int best_snp[3];

    #if defined (PAPI)
        uint64_t energy_s[4] = {LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX};
        uint64_t energy_e[4] = {0, 0, 0, 0};

        #pragma omp parallel reduction(min:cyc_s) reduction(max:cyc_e) reduction(min:energy_s) reduction(max:energy_e)
    #elif defined (PAPI_POW)
        uint64_t energy[4] = {0, 0, 0, 0};
        #pragma omp parallel reduction(min:cyc_s) reduction(max:cyc_e)
    #else
        #pragma omp parallel reduction(min:cyc_s) reduction(max:cyc_e)
    #endif
    {

    int counter = 0;

    int tid = omp_get_thread_num();
    
    int i, j, k, p, n;
    int m, mi, mj, mk;
    int ii, jj, pp, kk;
    int xii_p0, xii_p1, xi;
    int xjj_p0, xjj_p1, xj;
    int xkk_p0, xkk_p1, xk;
    int xft, xft0, xft00;
    int n_comb;

    int PP_ones = ceil((1.0*(phen_ones))/32.0);
    int PP_zeros = ceil((1.0*(num_pac - (phen_ones)))/32.0);

    int PP_ones_r = roundUp(PP_ones, block_pac);
    int PP_zeros_r = roundUp(PP_zeros, block_pac);

    int block_i, block_j, block_k, comb_ijk, comb_ij, comb_ii;

    int num_fts_I = block_snp*(block_snp -1)*(block_snp -2)/6;
    int num_fts_IJ = block_snp*block_snp*(block_snp-1)/2;
    int num_fts_IJK = block_snp*block_snp*block_snp;

    float best_score_local = FLT_MAX;

    #if defined (PAPI) || defined (PAPI_POW)
        uint64_t energy_s_local[4], energy_e_local[4];
    #endif

    uint32_t *SNPA_0, *SNPA_1, *SNPB_0, *SNPB_1, *SNPC_0, *SNPC_1;
    uint32_t dj2, di2, dk2;
    uint32_t t000, t001, t002, t010, t011, t012, t020, t021, t022, t100, t101, t102, t110, t111, t112, t120, t121, t122, t200, t201, t202, t210, t211, t212, t220, t221, t222;

    uint32_t ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13, ft14, ft15, ft16, ft17, ft18, ft19, ft20, ft21, ft22, ft23, ft24, ft25, ft26;

    //Creating MASK for non-existing pacients
    uint32_t mask_ones = 0xffffffff;
    uint32_t mask_zeros = 0xffffffff;

    int long long rem_pac = PP_ones*32 - phen_ones;

    mask_ones >>= (uint32_t) rem_pac;

    rem_pac = PP_zeros*32 - (num_pac - phen_ones);

    mask_zeros >>= (uint32_t) rem_pac; 

    __m256i v_SNPA_0, v_SNPA_1, v_SNPB_0, v_SNPB_1, v_SNPC_0, v_SNPC_1;
    __m256i v_dj2, v_di2, v_dk2;
    __m256i v_t000, v_t001, v_t002, v_t010, v_t011, v_t012, v_t020, v_t021, v_t022, v_t100, v_t101, v_t102, v_t110, v_t111, v_t112, v_t120, v_t121, v_t122, v_t200, v_t201, v_t202, v_t210, v_t211, v_t212, v_t220, v_t221, v_t222;

    __m256i v_ones;

    v_ones = _mm256_cmpeq_epi32 (v_ones, v_ones); //CREATE VECTOR OF ONES

    int v_elems;

    //Generate Frequency Table
    //uint32_t * freq_table = (uint32_t *) _mm_malloc(2*comb*sizeof(uint32_t), 64);
    //memset(freq_table, 0, 2*comb*sizeof(uint32_t));

    uint32_t* freq_table_I = (uint32_t*) _mm_malloc(2*num_fts_I*num_combs * sizeof(uint32_t), ALIGN);
    memset(freq_table_I, 0, 2*num_fts_I*num_combs*sizeof(uint32_t));
    
    uint32_t* freq_table_IJ = (uint32_t*) _mm_malloc(2 * num_fts_IJ * num_combs * sizeof(uint32_t), ALIGN);
    memset(freq_table_IJ, 0, 2*num_fts_IJ*num_combs*sizeof(uint32_t));

    uint32_t* freq_table_IJK = (uint32_t*) _mm_malloc(2 * num_fts_IJK * num_combs * sizeof(uint32_t), ALIGN);
    memset(freq_table_IJK, 0, 2*num_fts_IJK*num_combs*sizeof(uint32_t));

    int long long cyc_s_local, cyc_e_local;

    clear_cache_deep();

    serialize();

    #if defined (PAPI)
       PAPI_read(EventSet_pow[tid], (long long*) energy_s_local);
    #endif

    #if defined (LIKWID)
        LIKWID_MARKER_START("epistasis");
    #endif

    #if defined (PAPI_POW)
        PAPI_stop(EventSet_pow[tid], (long long*) energy_e_local);
    #endif

    cyc_s_local = read_tsc_start();

    #if defined (SDE)
        __SSC_MARK(0x111);
    #endif

    #if defined (ADVISOR)
        __itt_resume();
    #endif

    // fill frequency table
    #pragma omp for schedule(dynamic)
    for (ii = 0; ii < num_snp; ii+= block_snp){

        #if defined (PAPI_POW)
            PAPI_start(EventSet_pow[tid]);
        #endif

        xii_p0 = ii*PP_zeros_r*2;
        xii_p1 = ii*PP_ones_r*2;

        block_i = min(block_snp, num_snp - ii);

        for(jj = ii + block_snp; jj < num_snp; jj+= block_snp){
            xjj_p0 = jj*PP_zeros_r*2;
            xjj_p1 = jj*PP_ones_r*2;

            block_j = min(block_snp, num_snp - jj);

            for(kk = jj + block_snp; kk < num_snp; kk+= block_snp){
                xkk_p0 = kk*PP_zeros_r*2;
                xkk_p1 = kk*PP_ones_r*2;

                block_k = min(block_snp, num_snp - kk);

                comb_ijk = block_i*block_j*block_k;

                counter+=comb_ijk;

                //RESET FREQUENCY TABLES
                memset(freq_table_IJK, 0, 2*num_fts_IJK*num_combs*sizeof(uint32_t));

                //BETWEEN I, J and K

                v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;

                //Phenotype equal 0
                for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
                    xi = xii_p0 + pp*block_snp;
                    xj = xjj_p0 + pp*block_snp;
                    xk = xkk_p0 + pp*block_snp;

                    for(i = 0; i < block_i; i++){
                        SNPA_0 = &data_zeros[xi + i*block_pac];
                        SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
                        //SNPA = &data_zeros[xi + i*block_pac*2];
                        xft00 = i*block_j*block_k;
                        for(j = 0; j < block_j; j++){
                            xft0 = xft00 + j*block_k;
                            SNPB_0 = &data_zeros[xj + j*block_pac];
                            SNPB_1 = &data_zeros[xj + j*block_pac + block_snp*PP_zeros_r];
                            for(k = 0; k < block_k; k++){
                                xft = (xft0 + k)*num_combs;
                                //printf("xft00:%d\n", xft);
                                SNPC_0 = &data_zeros[xk + k*block_pac];
                                SNPC_1 = &data_zeros[xk + k*block_pac + block_snp*PP_zeros_r];

                                //RESET FT VARS
                                ft0 = ft0 ^ ft0;
                                ft1 = ft1 ^ ft1;
                                ft2 = ft2 ^ ft2;
                                ft3 = ft3 ^ ft3;
                                ft4 = ft4 ^ ft4;
                                ft5 = ft5 ^ ft5;
                                ft6 = ft6 ^ ft6;
                                ft7 = ft7 ^ ft7;
                                ft8 = ft8 ^ ft8;
                                ft9 = ft9 ^ ft9;
                                ft10 = ft10 ^ ft10;
                                ft11 = ft11 ^ ft11;
                                ft12 = ft12 ^ ft12;
                                ft13 = ft13 ^ ft13;
                                ft14 = ft14 ^ ft14;
                                ft15 = ft15 ^ ft15;
                                ft16 = ft16 ^ ft16;
                                ft17 = ft17 ^ ft17;
                                ft18 = ft18 ^ ft18;
                                ft19 = ft19 ^ ft19;
                                ft20 = ft20 ^ ft20;
                                ft21 = ft21 ^ ft21;
                                ft22 = ft22 ^ ft22;
                                ft23 = ft23 ^ ft23;
                                ft24 = ft24 ^ ft24;
                                ft25 = ft25 ^ ft25;
                                ft26 = ft26 ^ ft26;

                                for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                    //LOAD DATA
                                    v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                    v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                    v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                    v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                    v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                    v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                    //OR
                                    v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                    v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                    v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                    //NOT
                                    v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                    v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                    v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                    v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                    v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                    v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                    v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                    v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                    v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                    v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                    v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                    v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                    v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                    v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                    v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                    v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                    v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                    v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                    v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                    v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                    v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                    v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                    v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                    v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                    v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                    v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                    v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                    v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                    v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                    v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                                }
                                //Remaining non-vectorized elements
                                for(p = v_elems; p < block_pac; p++){
                                    di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                    dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                    dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                        
                                    t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                    t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                    t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                    t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                    t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                    t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                    t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                    t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                    t022 = SNPA_0[p] & dj2 & dk2;

                                    t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                    t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                    t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                    t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                    t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                    t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                    t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                    t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                    t122 = SNPA_1[p] & dj2 & dk2;

                                    t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                    t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                    t202 = di2 & SNPB_0[p] & dk2;
                                    t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                    t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                    t212 = di2 & SNPB_1[p] & dk2;
                                    t220 = di2 & dj2 & SNPC_0[p];
                                    t221 = di2 & dj2 & SNPC_1[p];
                                    t222 = di2 & dj2 & dk2;

                                    ft0 += _mm_popcnt_u32(t000);
                                    ft1 += _mm_popcnt_u32(t001);
                                    ft2 += _mm_popcnt_u32(t002);
                                    ft3 += _mm_popcnt_u32(t010);
                                    ft4 += _mm_popcnt_u32(t011);
                                    ft5 += _mm_popcnt_u32(t012);
                                    ft6 += _mm_popcnt_u32(t020);
                                    ft7 += _mm_popcnt_u32(t021);
                                    ft8 += _mm_popcnt_u32(t022);
                                    ft9 += _mm_popcnt_u32(t100);
                                    ft10 += _mm_popcnt_u32(t101);
                                    ft11 += _mm_popcnt_u32(t102);
                                    ft12 += _mm_popcnt_u32(t110);
                                    ft13 += _mm_popcnt_u32(t111);
                                    ft14 += _mm_popcnt_u32(t112);
                                    ft15 += _mm_popcnt_u32(t120);
                                    ft16 += _mm_popcnt_u32(t121);
                                    ft17 += _mm_popcnt_u32(t122);
                                    ft18 += _mm_popcnt_u32(t200);
                                    ft19 += _mm_popcnt_u32(t201);
                                    ft20 += _mm_popcnt_u32(t202);
                                    ft21 += _mm_popcnt_u32(t210);
                                    ft22 += _mm_popcnt_u32(t211);
                                    ft23 += _mm_popcnt_u32(t212);
                                    ft24 += _mm_popcnt_u32(t220);
                                    ft25 += _mm_popcnt_u32(t221);
                                    ft26 += _mm_popcnt_u32(t222);
                                }

                                freq_table_IJK[xft + 0] += ft0;
                                freq_table_IJK[xft + 1] += ft1;
                                freq_table_IJK[xft + 2] += ft2;
                                freq_table_IJK[xft + 3] += ft3;
                                freq_table_IJK[xft + 4] += ft4;
                                freq_table_IJK[xft + 5] += ft5;
                                freq_table_IJK[xft + 6] += ft6;
                                freq_table_IJK[xft + 7] += ft7;
                                freq_table_IJK[xft + 8] += ft8;
                                freq_table_IJK[xft + 9] += ft9;
                                freq_table_IJK[xft + 10] += ft10;
                                freq_table_IJK[xft + 11] += ft11;
                                freq_table_IJK[xft + 12] += ft12;
                                freq_table_IJK[xft + 13] += ft13;
                                freq_table_IJK[xft + 14] += ft14;
                                freq_table_IJK[xft + 15] += ft15;
                                freq_table_IJK[xft + 16] += ft16;
                                freq_table_IJK[xft + 17] += ft17;
                                freq_table_IJK[xft + 18] += ft18;
                                freq_table_IJK[xft + 19] += ft19;
                                freq_table_IJK[xft + 20] += ft20;
                                freq_table_IJK[xft + 21] += ft21;
                                freq_table_IJK[xft + 22] += ft22;
                                freq_table_IJK[xft + 23] += ft23;
                                freq_table_IJK[xft + 24] += ft24;
                                freq_table_IJK[xft + 25] += ft25;
                                freq_table_IJK[xft + 26] += ft26;
                            }
                        }
                    }
                }
                v_elems = ((PP_zeros - pp - 1)/VEC_ELEMS)*VEC_ELEMS;

                xi = xii_p0 + pp*block_snp;
                xj = xjj_p0 + pp*block_snp;
                xk = xkk_p0 + pp*block_snp;
                for(i = 0; i < block_i; i++){
                    SNPA_0 = &data_zeros[xi + i*block_pac];
                    SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
                    xft00 = i*block_j*block_k;
                    for(j = 0; j < block_j; j++){
                        xft0 = xft00 + j*block_k;
                        SNPB_0 = &data_zeros[xj + j*block_pac];
                        SNPB_1 = &data_zeros[xj + j*block_pac + block_snp*PP_zeros_r];
                        for(k = 0; k < block_k; k++){
                            xft = (xft0 + k)*num_combs;
                            SNPC_0 = &data_zeros[xk + k*block_pac];
                            SNPC_1 = &data_zeros[xk + k*block_pac + block_snp*PP_zeros_r];

                            //RESET FT VARS
                            ft0 = ft0 ^ ft0;
                            ft1 = ft1 ^ ft1;
                            ft2 = ft2 ^ ft2;
                            ft3 = ft3 ^ ft3;
                            ft4 = ft4 ^ ft4;
                            ft5 = ft5 ^ ft5;
                            ft6 = ft6 ^ ft6;
                            ft7 = ft7 ^ ft7;
                            ft8 = ft8 ^ ft8;
                            ft9 = ft9 ^ ft9;
                            ft10 = ft10 ^ ft10;
                            ft11 = ft11 ^ ft11;
                            ft12 = ft12 ^ ft12;
                            ft13 = ft13 ^ ft13;
                            ft14 = ft14 ^ ft14;
                            ft15 = ft15 ^ ft15;
                            ft16 = ft16 ^ ft16;
                            ft17 = ft17 ^ ft17;
                            ft18 = ft18 ^ ft18;
                            ft19 = ft19 ^ ft19;
                            ft20 = ft20 ^ ft20;
                            ft21 = ft21 ^ ft21;
                            ft22 = ft22 ^ ft22;
                            ft23 = ft23 ^ ft23;
                            ft24 = ft24 ^ ft24;
                            ft25 = ft25 ^ ft25;
                            ft26 = ft26 ^ ft26;

                            for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                //LOAD DATA
                                v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                //OR
                                v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                //NOT
                                v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                            }
                            //Remaining non-vectorized elements
                            for(p = v_elems; p < PP_zeros - pp - 1; p++){
                                di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                    
                                t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                t022 = SNPA_0[p] & dj2 & dk2;

                                t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                t122 = SNPA_1[p] & dj2 & dk2;

                                t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                t202 = di2 & SNPB_0[p] & dk2;
                                t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                t212 = di2 & SNPB_1[p] & dk2;
                                t220 = di2 & dj2 & SNPC_0[p];
                                t221 = di2 & dj2 & SNPC_1[p];
                                t222 = di2 & dj2 & dk2;

                                ft0 += _mm_popcnt_u32(t000);
                                ft1 += _mm_popcnt_u32(t001);
                                ft2 += _mm_popcnt_u32(t002);
                                ft3 += _mm_popcnt_u32(t010);
                                ft4 += _mm_popcnt_u32(t011);
                                ft5 += _mm_popcnt_u32(t012);
                                ft6 += _mm_popcnt_u32(t020);
                                ft7 += _mm_popcnt_u32(t021);
                                ft8 += _mm_popcnt_u32(t022);
                                ft9 += _mm_popcnt_u32(t100);
                                ft10 += _mm_popcnt_u32(t101);
                                ft11 += _mm_popcnt_u32(t102);
                                ft12 += _mm_popcnt_u32(t110);
                                ft13 += _mm_popcnt_u32(t111);
                                ft14 += _mm_popcnt_u32(t112);
                                ft15 += _mm_popcnt_u32(t120);
                                ft16 += _mm_popcnt_u32(t121);
                                ft17 += _mm_popcnt_u32(t122);
                                ft18 += _mm_popcnt_u32(t200);
                                ft19 += _mm_popcnt_u32(t201);
                                ft20 += _mm_popcnt_u32(t202);
                                ft21 += _mm_popcnt_u32(t210);
                                ft22 += _mm_popcnt_u32(t211);
                                ft23 += _mm_popcnt_u32(t212);
                                ft24 += _mm_popcnt_u32(t220);
                                ft25 += _mm_popcnt_u32(t221);
                                ft26 += _mm_popcnt_u32(t222);
                            }
                            //Do Remaining Elements
                            di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_zeros;
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_zeros;
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_zeros;
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);

                            freq_table_IJK[xft + 0] += ft0;
                            freq_table_IJK[xft + 1] += ft1;
                            freq_table_IJK[xft + 2] += ft2;
                            freq_table_IJK[xft + 3] += ft3;
                            freq_table_IJK[xft + 4] += ft4;
                            freq_table_IJK[xft + 5] += ft5;
                            freq_table_IJK[xft + 6] += ft6;
                            freq_table_IJK[xft + 7] += ft7;
                            freq_table_IJK[xft + 8] += ft8;
                            freq_table_IJK[xft + 9] += ft9;
                            freq_table_IJK[xft + 10] += ft10;
                            freq_table_IJK[xft + 11] += ft11;
                            freq_table_IJK[xft + 12] += ft12;
                            freq_table_IJK[xft + 13] += ft13;
                            freq_table_IJK[xft + 14] += ft14;
                            freq_table_IJK[xft + 15] += ft15;
                            freq_table_IJK[xft + 16] += ft16;
                            freq_table_IJK[xft + 17] += ft17;
                            freq_table_IJK[xft + 18] += ft18;
                            freq_table_IJK[xft + 19] += ft19;
                            freq_table_IJK[xft + 20] += ft20;
                            freq_table_IJK[xft + 21] += ft21;
                            freq_table_IJK[xft + 22] += ft22;
                            freq_table_IJK[xft + 23] += ft23;
                            freq_table_IJK[xft + 24] += ft24;
                            freq_table_IJK[xft + 25] += ft25;
                            freq_table_IJK[xft + 26] += ft26;
                        }
                    }
                }
                v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;

                //Phenotype equal 1
                for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
                    xi = xii_p1 + pp*block_snp;
                    xj = xjj_p1 + pp*block_snp;
                    xk = xkk_p1 + pp*block_snp;

                    for(i = 0; i < block_i; i++){
                        SNPA_0 = &data_ones[xi + i*block_pac];
                        SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
                        xft00 = i*block_j*block_k;
                        for(j = 0; j < block_j; j++){
                            xft0 = xft00 + j*block_k;
                            //xft = (comb_ij + xft0 + j)*num_combs;
                            SNPB_0 = &data_ones[xj + j*block_pac];
                            SNPB_1 = &data_ones[xj + j*block_pac + block_snp*PP_ones_r];
                            for(k = 0; k < block_k; k++){
                                xft = (comb_ijk + xft0 + k)*num_combs;
                                //printf("xft1:%d\n", xft);
                                SNPC_0 = &data_ones[xk + k*block_pac];
                                SNPC_1 = &data_ones[xk + k*block_pac + block_snp*PP_ones_r];

                                //RESET FT VARS
                                ft0 = ft0 ^ ft0;
                                ft1 = ft1 ^ ft1;
                                ft2 = ft2 ^ ft2;
                                ft3 = ft3 ^ ft3;
                                ft4 = ft4 ^ ft4;
                                ft5 = ft5 ^ ft5;
                                ft6 = ft6 ^ ft6;
                                ft7 = ft7 ^ ft7;
                                ft8 = ft8 ^ ft8;
                                ft9 = ft9 ^ ft9;
                                ft10 = ft10 ^ ft10;
                                ft11 = ft11 ^ ft11;
                                ft12 = ft12 ^ ft12;
                                ft13 = ft13 ^ ft13;
                                ft14 = ft14 ^ ft14;
                                ft15 = ft15 ^ ft15;
                                ft16 = ft16 ^ ft16;
                                ft17 = ft17 ^ ft17;
                                ft18 = ft18 ^ ft18;
                                ft19 = ft19 ^ ft19;
                                ft20 = ft20 ^ ft20;
                                ft21 = ft21 ^ ft21;
                                ft22 = ft22 ^ ft22;
                                ft23 = ft23 ^ ft23;
                                ft24 = ft24 ^ ft24;
                                ft25 = ft25 ^ ft25;
                                ft26 = ft26 ^ ft26;

                                for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                    //LOAD DATA
                                    v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                    v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                    v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                    v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                    v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                    v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                    //OR
                                    v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                    v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                    v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                    //NOT
                                    v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                    v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                    v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                    v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                    v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                    v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                    v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                    v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                    v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                    v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                    v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                    v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                    v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                    v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                    v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                    v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                    v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                    v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                    v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                    v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                    v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                    v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                    v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                    v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                    v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                    v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                    v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                    v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                    v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                    v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                    ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                    ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                    ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                    ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                    ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                    ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                    ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                    ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                    ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                    ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                    ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                    ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                    ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                    ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                    ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                    ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                    ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                    ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                    ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                    ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                    ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                    ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                    ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                    ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                    ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                    ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                    ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                                }
                                //Remaining non-vectorized elements
                                for(p = v_elems; p < block_pac; p++){
                                    di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                    dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                    dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                        
                                    t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                    t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                    t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                    t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                    t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                    t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                    t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                    t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                    t022 = SNPA_0[p] & dj2 & dk2;

                                    t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                    t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                    t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                    t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                    t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                    t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                    t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                    t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                    t122 = SNPA_1[p] & dj2 & dk2;

                                    t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                    t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                    t202 = di2 & SNPB_0[p] & dk2;
                                    t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                    t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                    t212 = di2 & SNPB_1[p] & dk2;
                                    t220 = di2 & dj2 & SNPC_0[p];
                                    t221 = di2 & dj2 & SNPC_1[p];
                                    t222 = di2 & dj2 & dk2;

                                    ft0 += _mm_popcnt_u32(t000);
                                    ft1 += _mm_popcnt_u32(t001);
                                    ft2 += _mm_popcnt_u32(t002);
                                    ft3 += _mm_popcnt_u32(t010);
                                    ft4 += _mm_popcnt_u32(t011);
                                    ft5 += _mm_popcnt_u32(t012);
                                    ft6 += _mm_popcnt_u32(t020);
                                    ft7 += _mm_popcnt_u32(t021);
                                    ft8 += _mm_popcnt_u32(t022);
                                    ft9 += _mm_popcnt_u32(t100);
                                    ft10 += _mm_popcnt_u32(t101);
                                    ft11 += _mm_popcnt_u32(t102);
                                    ft12 += _mm_popcnt_u32(t110);
                                    ft13 += _mm_popcnt_u32(t111);
                                    ft14 += _mm_popcnt_u32(t112);
                                    ft15 += _mm_popcnt_u32(t120);
                                    ft16 += _mm_popcnt_u32(t121);
                                    ft17 += _mm_popcnt_u32(t122);
                                    ft18 += _mm_popcnt_u32(t200);
                                    ft19 += _mm_popcnt_u32(t201);
                                    ft20 += _mm_popcnt_u32(t202);
                                    ft21 += _mm_popcnt_u32(t210);
                                    ft22 += _mm_popcnt_u32(t211);
                                    ft23 += _mm_popcnt_u32(t212);
                                    ft24 += _mm_popcnt_u32(t220);
                                    ft25 += _mm_popcnt_u32(t221);
                                    ft26 += _mm_popcnt_u32(t222);
                                }

                                freq_table_IJK[xft + 0] += ft0;
                                freq_table_IJK[xft + 1] += ft1;
                                freq_table_IJK[xft + 2] += ft2;
                                freq_table_IJK[xft + 3] += ft3;
                                freq_table_IJK[xft + 4] += ft4;
                                freq_table_IJK[xft + 5] += ft5;
                                freq_table_IJK[xft + 6] += ft6;
                                freq_table_IJK[xft + 7] += ft7;
                                freq_table_IJK[xft + 8] += ft8;
                                freq_table_IJK[xft + 9] += ft9;
                                freq_table_IJK[xft + 10] += ft10;
                                freq_table_IJK[xft + 11] += ft11;
                                freq_table_IJK[xft + 12] += ft12;
                                freq_table_IJK[xft + 13] += ft13;
                                freq_table_IJK[xft + 14] += ft14;
                                freq_table_IJK[xft + 15] += ft15;
                                freq_table_IJK[xft + 16] += ft16;
                                freq_table_IJK[xft + 17] += ft17;
                                freq_table_IJK[xft + 18] += ft18;
                                freq_table_IJK[xft + 19] += ft19;
                                freq_table_IJK[xft + 20] += ft20;
                                freq_table_IJK[xft + 21] += ft21;
                                freq_table_IJK[xft + 22] += ft22;
                                freq_table_IJK[xft + 23] += ft23;
                                freq_table_IJK[xft + 24] += ft24;
                                freq_table_IJK[xft + 25] += ft25;
                                freq_table_IJK[xft + 26] += ft26;
                            }
                        }
                    }
                }
                v_elems = ((PP_ones - pp - 1)/VEC_ELEMS)*VEC_ELEMS;

                xi = xii_p1 + pp*block_snp;
                xj = xjj_p1 + pp*block_snp;
                xk = xkk_p1 + pp*block_snp;

                for(i = 0; i < block_i; i++){
                    SNPA_0 = &data_ones[xi + i*block_pac];
                    SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
                    xft00 = i*block_j*block_k;
                    for(j = 0; j < block_j; j++){
                        xft0 = xft00 + j*block_k;
                        SNPB_0 = &data_ones[xj + j*block_pac];
                        SNPB_1 = &data_ones[xj + j*block_pac + block_snp*PP_ones_r];
                        for(k = 0; k < block_k; k++){
                            xft = (comb_ijk + xft0 + k)*num_combs;
                            //printf("xft1:%d\n", xft);
                            SNPC_0 = &data_ones[xk + k*block_pac];
                            SNPC_1 = &data_ones[xk + k*block_pac + block_snp*PP_ones_r];

                            //RESET FT VARS
                            ft0 = ft0 ^ ft0;
                            ft1 = ft1 ^ ft1;
                            ft2 = ft2 ^ ft2;
                            ft3 = ft3 ^ ft3;
                            ft4 = ft4 ^ ft4;
                            ft5 = ft5 ^ ft5;
                            ft6 = ft6 ^ ft6;
                            ft7 = ft7 ^ ft7;
                            ft8 = ft8 ^ ft8;
                            ft9 = ft9 ^ ft9;
                            ft10 = ft10 ^ ft10;
                            ft11 = ft11 ^ ft11;
                            ft12 = ft12 ^ ft12;
                            ft13 = ft13 ^ ft13;
                            ft14 = ft14 ^ ft14;
                            ft15 = ft15 ^ ft15;
                            ft16 = ft16 ^ ft16;
                            ft17 = ft17 ^ ft17;
                            ft18 = ft18 ^ ft18;
                            ft19 = ft19 ^ ft19;
                            ft20 = ft20 ^ ft20;
                            ft21 = ft21 ^ ft21;
                            ft22 = ft22 ^ ft22;
                            ft23 = ft23 ^ ft23;
                            ft24 = ft24 ^ ft24;
                            ft25 = ft25 ^ ft25;
                            ft26 = ft26 ^ ft26;

                            for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                //LOAD DATA
                                v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                //OR
                                v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                //NOT
                                v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                            }
                            //Remaining non-vectorized elements
                            for(p = v_elems; p < PP_ones - pp - 1; p++){
                                di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                    
                                t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                t022 = SNPA_0[p] & dj2 & dk2;

                                t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                t122 = SNPA_1[p] & dj2 & dk2;

                                t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                t202 = di2 & SNPB_0[p] & dk2;
                                t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                t212 = di2 & SNPB_1[p] & dk2;
                                t220 = di2 & dj2 & SNPC_0[p];
                                t221 = di2 & dj2 & SNPC_1[p];
                                t222 = di2 & dj2 & dk2;

                                ft0 += _mm_popcnt_u32(t000);
                                ft1 += _mm_popcnt_u32(t001);
                                ft2 += _mm_popcnt_u32(t002);
                                ft3 += _mm_popcnt_u32(t010);
                                ft4 += _mm_popcnt_u32(t011);
                                ft5 += _mm_popcnt_u32(t012);
                                ft6 += _mm_popcnt_u32(t020);
                                ft7 += _mm_popcnt_u32(t021);
                                ft8 += _mm_popcnt_u32(t022);
                                ft9 += _mm_popcnt_u32(t100);
                                ft10 += _mm_popcnt_u32(t101);
                                ft11 += _mm_popcnt_u32(t102);
                                ft12 += _mm_popcnt_u32(t110);
                                ft13 += _mm_popcnt_u32(t111);
                                ft14 += _mm_popcnt_u32(t112);
                                ft15 += _mm_popcnt_u32(t120);
                                ft16 += _mm_popcnt_u32(t121);
                                ft17 += _mm_popcnt_u32(t122);
                                ft18 += _mm_popcnt_u32(t200);
                                ft19 += _mm_popcnt_u32(t201);
                                ft20 += _mm_popcnt_u32(t202);
                                ft21 += _mm_popcnt_u32(t210);
                                ft22 += _mm_popcnt_u32(t211);
                                ft23 += _mm_popcnt_u32(t212);
                                ft24 += _mm_popcnt_u32(t220);
                                ft25 += _mm_popcnt_u32(t221);
                                ft26 += _mm_popcnt_u32(t222);
                            }
                            di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_ones;
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_ones;
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_ones;
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);

                            freq_table_IJK[xft + 0] += ft0;
                            freq_table_IJK[xft + 1] += ft1;
                            freq_table_IJK[xft + 2] += ft2;
                            freq_table_IJK[xft + 3] += ft3;
                            freq_table_IJK[xft + 4] += ft4;
                            freq_table_IJK[xft + 5] += ft5;
                            freq_table_IJK[xft + 6] += ft6;
                            freq_table_IJK[xft + 7] += ft7;
                            freq_table_IJK[xft + 8] += ft8;
                            freq_table_IJK[xft + 9] += ft9;
                            freq_table_IJK[xft + 10] += ft10;
                            freq_table_IJK[xft + 11] += ft11;
                            freq_table_IJK[xft + 12] += ft12;
                            freq_table_IJK[xft + 13] += ft13;
                            freq_table_IJK[xft + 14] += ft14;
                            freq_table_IJK[xft + 15] += ft15;
                            freq_table_IJK[xft + 16] += ft16;
                            freq_table_IJK[xft + 17] += ft17;
                            freq_table_IJK[xft + 18] += ft18;
                            freq_table_IJK[xft + 19] += ft19;
                            freq_table_IJK[xft + 20] += ft20;
                            freq_table_IJK[xft + 21] += ft21;
                            freq_table_IJK[xft + 22] += ft22;
                            freq_table_IJK[xft + 23] += ft23;
                            freq_table_IJK[xft + 24] += ft24;
                            freq_table_IJK[xft + 25] += ft25;
                            freq_table_IJK[xft + 26] += ft26;
                        }
                    }
                }
                
                for(mi = 0; mi < block_i; mi++){
                    for(mj = 0; mj < block_j; mj++){
                        for(mk = 0; mk < block_k; mk++){
                            m = mi*block_j*block_k + mj*block_k + mk;
                            float score = 0.0;
                            for(n = 0; n < num_combs; n++){
                                score += addlog(freq_table_IJK[m * num_combs + n] + freq_table_IJK[(comb_ijk + m) * num_combs + n] + 1) - addlog(freq_table_IJK[m * num_combs + n]) - addlog(freq_table_IJK[(comb_ijk + m) * num_combs + n]);
                            }
                            score = fabs(score);

                            if(score < best_score_local){
                                best_score_local = score;
                                best_snp_global[tid][0] = ii + mi;
                                best_snp_global[tid][1] = jj + mj;
                                best_snp_global[tid][2] = kk + mk;
                            }
                           // printf("%d, %d, %d - Score: %f\n", ii + mi, jj + mj, kk + mk, score);
                        }
                    }
                }
            } 
            //BETWEEN I AND J

            v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;

            comb_ij = (block_i*block_j*(block_j-1))/2;

            counter+=comb_ij;

            //RESET FREQUENCY TABLES
            memset(freq_table_IJ, 0, 2 * num_fts_IJ * num_combs * sizeof(uint32_t));

            //Phenotype equal 0
            for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
                xi = xii_p0 + pp*block_snp;
                xj = xjj_p0 + pp*block_snp;
                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i; i++){
                    SNPA_0 = &data_zeros[xi + i*block_pac];
                    SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
                    //xft0 = i*block_j;
                    for(j = 0; j < block_j-1; j++){
                        //xft = (xft0 + j)*num_combs;
                        SNPB_0 = &data_zeros[xj + j*block_pac];
                        SNPB_1 = &data_zeros[xj + j*block_pac + block_snp*PP_zeros_r];
                        for(k = j+1; k < block_j; k++){
                            xft = n_comb*num_combs;
                            SNPC_0 = &data_zeros[xj + k*block_pac];
                            SNPC_1 = &data_zeros[xj + k*block_pac + block_snp*PP_zeros_r];

                            //RESET FT VARS
                            ft0 = ft0 ^ ft0;
                            ft1 = ft1 ^ ft1;
                            ft2 = ft2 ^ ft2;
                            ft3 = ft3 ^ ft3;
                            ft4 = ft4 ^ ft4;
                            ft5 = ft5 ^ ft5;
                            ft6 = ft6 ^ ft6;
                            ft7 = ft7 ^ ft7;
                            ft8 = ft8 ^ ft8;
                            ft9 = ft9 ^ ft9;
                            ft10 = ft10 ^ ft10;
                            ft11 = ft11 ^ ft11;
                            ft12 = ft12 ^ ft12;
                            ft13 = ft13 ^ ft13;
                            ft14 = ft14 ^ ft14;
                            ft15 = ft15 ^ ft15;
                            ft16 = ft16 ^ ft16;
                            ft17 = ft17 ^ ft17;
                            ft18 = ft18 ^ ft18;
                            ft19 = ft19 ^ ft19;
                            ft20 = ft20 ^ ft20;
                            ft21 = ft21 ^ ft21;
                            ft22 = ft22 ^ ft22;
                            ft23 = ft23 ^ ft23;
                            ft24 = ft24 ^ ft24;
                            ft25 = ft25 ^ ft25;
                            ft26 = ft26 ^ ft26;

                            for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                //LOAD DATA
                                v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                //OR
                                v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                //NOT
                                v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                            }
                            //Remaining non-vectorized elements
                            for(p = v_elems; p < block_pac; p++){
                                di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                    
                                t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                t022 = SNPA_0[p] & dj2 & dk2;

                                t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                t122 = SNPA_1[p] & dj2 & dk2;

                                t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                t202 = di2 & SNPB_0[p] & dk2;
                                t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                t212 = di2 & SNPB_1[p] & dk2;
                                t220 = di2 & dj2 & SNPC_0[p];
                                t221 = di2 & dj2 & SNPC_1[p];
                                t222 = di2 & dj2 & dk2;

                                ft0 += _mm_popcnt_u32(t000);
                                ft1 += _mm_popcnt_u32(t001);
                                ft2 += _mm_popcnt_u32(t002);
                                ft3 += _mm_popcnt_u32(t010);
                                ft4 += _mm_popcnt_u32(t011);
                                ft5 += _mm_popcnt_u32(t012);
                                ft6 += _mm_popcnt_u32(t020);
                                ft7 += _mm_popcnt_u32(t021);
                                ft8 += _mm_popcnt_u32(t022);
                                ft9 += _mm_popcnt_u32(t100);
                                ft10 += _mm_popcnt_u32(t101);
                                ft11 += _mm_popcnt_u32(t102);
                                ft12 += _mm_popcnt_u32(t110);
                                ft13 += _mm_popcnt_u32(t111);
                                ft14 += _mm_popcnt_u32(t112);
                                ft15 += _mm_popcnt_u32(t120);
                                ft16 += _mm_popcnt_u32(t121);
                                ft17 += _mm_popcnt_u32(t122);
                                ft18 += _mm_popcnt_u32(t200);
                                ft19 += _mm_popcnt_u32(t201);
                                ft20 += _mm_popcnt_u32(t202);
                                ft21 += _mm_popcnt_u32(t210);
                                ft22 += _mm_popcnt_u32(t211);
                                ft23 += _mm_popcnt_u32(t212);
                                ft24 += _mm_popcnt_u32(t220);
                                ft25 += _mm_popcnt_u32(t221);
                                ft26 += _mm_popcnt_u32(t222);
                            }

                            freq_table_IJ[xft + 0] += ft0;
                            freq_table_IJ[xft + 1] += ft1;
                            freq_table_IJ[xft + 2] += ft2;
                            freq_table_IJ[xft + 3] += ft3;
                            freq_table_IJ[xft + 4] += ft4;
                            freq_table_IJ[xft + 5] += ft5;
                            freq_table_IJ[xft + 6] += ft6;
                            freq_table_IJ[xft + 7] += ft7;
                            freq_table_IJ[xft + 8] += ft8;
                            freq_table_IJ[xft + 9] += ft9;
                            freq_table_IJ[xft + 10] += ft10;
                            freq_table_IJ[xft + 11] += ft11;
                            freq_table_IJ[xft + 12] += ft12;
                            freq_table_IJ[xft + 13] += ft13;
                            freq_table_IJ[xft + 14] += ft14;
                            freq_table_IJ[xft + 15] += ft15;
                            freq_table_IJ[xft + 16] += ft16;
                            freq_table_IJ[xft + 17] += ft17;
                            freq_table_IJ[xft + 18] += ft18;
                            freq_table_IJ[xft + 19] += ft19;
                            freq_table_IJ[xft + 20] += ft20;
                            freq_table_IJ[xft + 21] += ft21;
                            freq_table_IJ[xft + 22] += ft22;
                            freq_table_IJ[xft + 23] += ft23;
                            freq_table_IJ[xft + 24] += ft24;
                            freq_table_IJ[xft + 25] += ft25;
                            freq_table_IJ[xft + 26] += ft26;

                            n_comb++;
                        }
                    }
                }
            }
            v_elems = ((PP_zeros - pp - 1)/VEC_ELEMS)*VEC_ELEMS;

            xi = xii_p0 + pp*block_snp;
            xj = xjj_p0 + pp*block_snp;

            n_comb = 0;
            //BETWEEN I and J
            for(i = 0; i < block_i; i++){
                SNPA_0 = &data_zeros[xi + i*block_pac];
                SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
                //xft0 = i*block_j;
                for(j = 0; j < block_j-1; j++){
                    //xft = (xft0 + j)*num_combs;
                    SNPB_0 = &data_zeros[xj + j*block_pac];
                    SNPB_1 = &data_zeros[xj + j*block_pac + block_snp*PP_zeros_r];
                    for(k = j+1; k < block_j; k++){
                        xft = n_comb*num_combs;
                        SNPC_0 = &data_zeros[xj + k*block_pac];
                        SNPC_1 = &data_zeros[xj + k*block_pac + block_snp*PP_zeros_r];

                        //RESET FT VARS
                        ft0 = ft0 ^ ft0;
                        ft1 = ft1 ^ ft1;
                        ft2 = ft2 ^ ft2;
                        ft3 = ft3 ^ ft3;
                        ft4 = ft4 ^ ft4;
                        ft5 = ft5 ^ ft5;
                        ft6 = ft6 ^ ft6;
                        ft7 = ft7 ^ ft7;
                        ft8 = ft8 ^ ft8;
                        ft9 = ft9 ^ ft9;
                        ft10 = ft10 ^ ft10;
                        ft11 = ft11 ^ ft11;
                        ft12 = ft12 ^ ft12;
                        ft13 = ft13 ^ ft13;
                        ft14 = ft14 ^ ft14;
                        ft15 = ft15 ^ ft15;
                        ft16 = ft16 ^ ft16;
                        ft17 = ft17 ^ ft17;
                        ft18 = ft18 ^ ft18;
                        ft19 = ft19 ^ ft19;
                        ft20 = ft20 ^ ft20;
                        ft21 = ft21 ^ ft21;
                        ft22 = ft22 ^ ft22;
                        ft23 = ft23 ^ ft23;
                        ft24 = ft24 ^ ft24;
                        ft25 = ft25 ^ ft25;
                        ft26 = ft26 ^ ft26;

                        for(p = 0; p < v_elems; p+=VEC_ELEMS){
                            //LOAD DATA
                            v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                            v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                            v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                            v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                            v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                            v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                            //OR
                            v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                            v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                            v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                            //NOT
                            v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                            v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                            v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                            v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                            v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                            v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                            v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                        }
                        //Remaining non-vectorized elements
                        for(p = v_elems; p < PP_zeros - pp - 1; p++){
                            di2 = ~(SNPA_0[p] | SNPA_1[p]);
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_zeros;
                        dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_zeros;
                        dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_zeros;
            
                        t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                        t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                        t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                        t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                        t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                        t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                        t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                        t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                        t022 = SNPA_0[p] & dj2 & dk2;

                        t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                        t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                        t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                        t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                        t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                        t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                        t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                        t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                        t122 = SNPA_1[p] & dj2 & dk2;

                        t200 = di2 & SNPB_0[p] & SNPC_0[p];
                        t201 = di2 & SNPB_0[p] & SNPC_1[p];
                        t202 = di2 & SNPB_0[p] & dk2;
                        t210 = di2 & SNPB_1[p] & SNPC_0[p];
                        t211 = di2 & SNPB_1[p] & SNPC_1[p];
                        t212 = di2 & SNPB_1[p] & dk2;
                        t220 = di2 & dj2 & SNPC_0[p];
                        t221 = di2 & dj2 & SNPC_1[p];
                        t222 = di2 & dj2 & dk2;

                        ft0 += _mm_popcnt_u32(t000);
                        ft1 += _mm_popcnt_u32(t001);
                        ft2 += _mm_popcnt_u32(t002);
                        ft3 += _mm_popcnt_u32(t010);
                        ft4 += _mm_popcnt_u32(t011);
                        ft5 += _mm_popcnt_u32(t012);
                        ft6 += _mm_popcnt_u32(t020);
                        ft7 += _mm_popcnt_u32(t021);
                        ft8 += _mm_popcnt_u32(t022);
                        ft9 += _mm_popcnt_u32(t100);
                        ft10 += _mm_popcnt_u32(t101);
                        ft11 += _mm_popcnt_u32(t102);
                        ft12 += _mm_popcnt_u32(t110);
                        ft13 += _mm_popcnt_u32(t111);
                        ft14 += _mm_popcnt_u32(t112);
                        ft15 += _mm_popcnt_u32(t120);
                        ft16 += _mm_popcnt_u32(t121);
                        ft17 += _mm_popcnt_u32(t122);
                        ft18 += _mm_popcnt_u32(t200);
                        ft19 += _mm_popcnt_u32(t201);
                        ft20 += _mm_popcnt_u32(t202);
                        ft21 += _mm_popcnt_u32(t210);
                        ft22 += _mm_popcnt_u32(t211);
                        ft23 += _mm_popcnt_u32(t212);
                        ft24 += _mm_popcnt_u32(t220);
                        ft25 += _mm_popcnt_u32(t221);
                        ft26 += _mm_popcnt_u32(t222);

                        freq_table_IJ[xft + 0] += ft0;
                        freq_table_IJ[xft + 1] += ft1;
                        freq_table_IJ[xft + 2] += ft2;
                        freq_table_IJ[xft + 3] += ft3;
                        freq_table_IJ[xft + 4] += ft4;
                        freq_table_IJ[xft + 5] += ft5;
                        freq_table_IJ[xft + 6] += ft6;
                        freq_table_IJ[xft + 7] += ft7;
                        freq_table_IJ[xft + 8] += ft8;
                        freq_table_IJ[xft + 9] += ft9;
                        freq_table_IJ[xft + 10] += ft10;
                        freq_table_IJ[xft + 11] += ft11;
                        freq_table_IJ[xft + 12] += ft12;
                        freq_table_IJ[xft + 13] += ft13;
                        freq_table_IJ[xft + 14] += ft14;
                        freq_table_IJ[xft + 15] += ft15;
                        freq_table_IJ[xft + 16] += ft16;
                        freq_table_IJ[xft + 17] += ft17;
                        freq_table_IJ[xft + 18] += ft18;
                        freq_table_IJ[xft + 19] += ft19;
                        freq_table_IJ[xft + 20] += ft20;
                        freq_table_IJ[xft + 21] += ft21;
                        freq_table_IJ[xft + 22] += ft22;
                        freq_table_IJ[xft + 23] += ft23;
                        freq_table_IJ[xft + 24] += ft24;
                        freq_table_IJ[xft + 25] += ft25;
                        freq_table_IJ[xft + 26] += ft26;

                        n_comb ++;
                    }
                }
            }

            v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;

            //Phenotype equal 1
            for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
                xi = xii_p1 + pp*block_snp;
                xj = xjj_p1 + pp*block_snp;

                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i; i++){
                    SNPA_0 = &data_ones[xi + i*block_pac];
                    SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
                    //xft0 = i*block_j;
                    for(j = 0; j < block_j-1; j++){
                        //xft = (comb_ij + xft0 + j)*num_combs;
                        SNPB_0 = &data_ones[xj + j*block_pac];
                        SNPB_1 = &data_ones[xj + j*block_pac + block_snp*PP_ones_r];
                        for(k = j+1; k < block_j; k++){
                            xft = (comb_ij + n_comb)*num_combs;
                            SNPC_0 = &data_ones[xj + k*block_pac];
                            SNPC_1 = &data_ones[xj + k*block_pac + block_snp*PP_ones_r];

                            //RESET FT VARS
                            ft0 = ft0 ^ ft0;
                            ft1 = ft1 ^ ft1;
                            ft2 = ft2 ^ ft2;
                            ft3 = ft3 ^ ft3;
                            ft4 = ft4 ^ ft4;
                            ft5 = ft5 ^ ft5;
                            ft6 = ft6 ^ ft6;
                            ft7 = ft7 ^ ft7;
                            ft8 = ft8 ^ ft8;
                            ft9 = ft9 ^ ft9;
                            ft10 = ft10 ^ ft10;
                            ft11 = ft11 ^ ft11;
                            ft12 = ft12 ^ ft12;
                            ft13 = ft13 ^ ft13;
                            ft14 = ft14 ^ ft14;
                            ft15 = ft15 ^ ft15;
                            ft16 = ft16 ^ ft16;
                            ft17 = ft17 ^ ft17;
                            ft18 = ft18 ^ ft18;
                            ft19 = ft19 ^ ft19;
                            ft20 = ft20 ^ ft20;
                            ft21 = ft21 ^ ft21;
                            ft22 = ft22 ^ ft22;
                            ft23 = ft23 ^ ft23;
                            ft24 = ft24 ^ ft24;
                            ft25 = ft25 ^ ft25;
                            ft26 = ft26 ^ ft26;

                            for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                //LOAD DATA
                                v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                //OR
                                v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                //NOT
                                v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                            }
                            //Remaining non-vectorized elements
                            for(p = v_elems; p < block_pac; p++){
                                di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                    
                                t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                t022 = SNPA_0[p] & dj2 & dk2;

                                t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                t122 = SNPA_1[p] & dj2 & dk2;

                                t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                t202 = di2 & SNPB_0[p] & dk2;
                                t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                t212 = di2 & SNPB_1[p] & dk2;
                                t220 = di2 & dj2 & SNPC_0[p];
                                t221 = di2 & dj2 & SNPC_1[p];
                                t222 = di2 & dj2 & dk2;

                                ft0 += _mm_popcnt_u32(t000);
                                ft1 += _mm_popcnt_u32(t001);
                                ft2 += _mm_popcnt_u32(t002);
                                ft3 += _mm_popcnt_u32(t010);
                                ft4 += _mm_popcnt_u32(t011);
                                ft5 += _mm_popcnt_u32(t012);
                                ft6 += _mm_popcnt_u32(t020);
                                ft7 += _mm_popcnt_u32(t021);
                                ft8 += _mm_popcnt_u32(t022);
                                ft9 += _mm_popcnt_u32(t100);
                                ft10 += _mm_popcnt_u32(t101);
                                ft11 += _mm_popcnt_u32(t102);
                                ft12 += _mm_popcnt_u32(t110);
                                ft13 += _mm_popcnt_u32(t111);
                                ft14 += _mm_popcnt_u32(t112);
                                ft15 += _mm_popcnt_u32(t120);
                                ft16 += _mm_popcnt_u32(t121);
                                ft17 += _mm_popcnt_u32(t122);
                                ft18 += _mm_popcnt_u32(t200);
                                ft19 += _mm_popcnt_u32(t201);
                                ft20 += _mm_popcnt_u32(t202);
                                ft21 += _mm_popcnt_u32(t210);
                                ft22 += _mm_popcnt_u32(t211);
                                ft23 += _mm_popcnt_u32(t212);
                                ft24 += _mm_popcnt_u32(t220);
                                ft25 += _mm_popcnt_u32(t221);
                                ft26 += _mm_popcnt_u32(t222);
                            }

                            freq_table_IJ[xft + 0] += ft0;
                            freq_table_IJ[xft + 1] += ft1;
                            freq_table_IJ[xft + 2] += ft2;
                            freq_table_IJ[xft + 3] += ft3;
                            freq_table_IJ[xft + 4] += ft4;
                            freq_table_IJ[xft + 5] += ft5;
                            freq_table_IJ[xft + 6] += ft6;
                            freq_table_IJ[xft + 7] += ft7;
                            freq_table_IJ[xft + 8] += ft8;
                            freq_table_IJ[xft + 9] += ft9;
                            freq_table_IJ[xft + 10] += ft10;
                            freq_table_IJ[xft + 11] += ft11;
                            freq_table_IJ[xft + 12] += ft12;
                            freq_table_IJ[xft + 13] += ft13;
                            freq_table_IJ[xft + 14] += ft14;
                            freq_table_IJ[xft + 15] += ft15;
                            freq_table_IJ[xft + 16] += ft16;
                            freq_table_IJ[xft + 17] += ft17;
                            freq_table_IJ[xft + 18] += ft18;
                            freq_table_IJ[xft + 19] += ft19;
                            freq_table_IJ[xft + 20] += ft20;
                            freq_table_IJ[xft + 21] += ft21;
                            freq_table_IJ[xft + 22] += ft22;
                            freq_table_IJ[xft + 23] += ft23;
                            freq_table_IJ[xft + 24] += ft24;
                            freq_table_IJ[xft + 25] += ft25;
                            freq_table_IJ[xft + 26] += ft26;

                            n_comb ++;
                        }
                    }
                }
            }
            v_elems = ((PP_ones - pp - 1)/VEC_ELEMS)*VEC_ELEMS;
            
            n_comb = 0;

            xi = xii_p1 + pp*block_snp;
            xj = xjj_p1 + pp*block_snp;
            //BETWEEN I and J
            for(i = 0; i < block_i; i++){
                SNPA_0 = &data_ones[xi + i*block_pac];
                SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
                //xft0 = i*block_j;
                for(j = 0; j < block_j-1; j++){
                    //xft = (comb_ij + xft0 + j)*num_combs;
                    SNPB_0 = &data_ones[xj + j*block_pac];
                    SNPB_1 = &data_ones[xj + j*block_pac + block_snp*PP_ones_r];
                    for(k = j+1; k < block_j; k++){
                        xft = (comb_ij + n_comb)*num_combs;
                        SNPC_0 = &data_ones[xj + k*block_pac];
                        SNPC_1 = &data_ones[xj + k*block_pac + block_snp*PP_ones_r];

                        //RESET FT VARS
                        ft0 = ft0 ^ ft0;
                        ft1 = ft1 ^ ft1;
                        ft2 = ft2 ^ ft2;
                        ft3 = ft3 ^ ft3;
                        ft4 = ft4 ^ ft4;
                        ft5 = ft5 ^ ft5;
                        ft6 = ft6 ^ ft6;
                        ft7 = ft7 ^ ft7;
                        ft8 = ft8 ^ ft8;
                        ft9 = ft9 ^ ft9;
                        ft10 = ft10 ^ ft10;
                        ft11 = ft11 ^ ft11;
                        ft12 = ft12 ^ ft12;
                        ft13 = ft13 ^ ft13;
                        ft14 = ft14 ^ ft14;
                        ft15 = ft15 ^ ft15;
                        ft16 = ft16 ^ ft16;
                        ft17 = ft17 ^ ft17;
                        ft18 = ft18 ^ ft18;
                        ft19 = ft19 ^ ft19;
                        ft20 = ft20 ^ ft20;
                        ft21 = ft21 ^ ft21;
                        ft22 = ft22 ^ ft22;
                        ft23 = ft23 ^ ft23;
                        ft24 = ft24 ^ ft24;
                        ft25 = ft25 ^ ft25;
                        ft26 = ft26 ^ ft26;

                        for(p = 0; p < v_elems; p+=VEC_ELEMS){
                            //LOAD DATA
                            v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                            v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                            v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                            v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                            v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                            v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                            //OR
                            v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                            v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                            v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                            //NOT
                            v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                            v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                            v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                            v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                            v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                            v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                            v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                        }
                        //Remaining non-vectorized elements
                        for(p = v_elems; p < PP_ones - pp - 1; p++){
                            di2 = ~(SNPA_0[p] | SNPA_1[p]);
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_ones;
                        dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_ones;
                        dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_ones;
            
                        t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                        t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                        t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                        t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                        t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                        t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                        t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                        t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                        t022 = SNPA_0[p] & dj2 & dk2;

                        t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                        t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                        t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                        t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                        t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                        t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                        t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                        t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                        t122 = SNPA_1[p] & dj2 & dk2;

                        t200 = di2 & SNPB_0[p] & SNPC_0[p];
                        t201 = di2 & SNPB_0[p] & SNPC_1[p];
                        t202 = di2 & SNPB_0[p] & dk2;
                        t210 = di2 & SNPB_1[p] & SNPC_0[p];
                        t211 = di2 & SNPB_1[p] & SNPC_1[p];
                        t212 = di2 & SNPB_1[p] & dk2;
                        t220 = di2 & dj2 & SNPC_0[p];
                        t221 = di2 & dj2 & SNPC_1[p];
                        t222 = di2 & dj2 & dk2;

                        ft0 += _mm_popcnt_u32(t000);
                        ft1 += _mm_popcnt_u32(t001);
                        ft2 += _mm_popcnt_u32(t002);
                        ft3 += _mm_popcnt_u32(t010);
                        ft4 += _mm_popcnt_u32(t011);
                        ft5 += _mm_popcnt_u32(t012);
                        ft6 += _mm_popcnt_u32(t020);
                        ft7 += _mm_popcnt_u32(t021);
                        ft8 += _mm_popcnt_u32(t022);
                        ft9 += _mm_popcnt_u32(t100);
                        ft10 += _mm_popcnt_u32(t101);
                        ft11 += _mm_popcnt_u32(t102);
                        ft12 += _mm_popcnt_u32(t110);
                        ft13 += _mm_popcnt_u32(t111);
                        ft14 += _mm_popcnt_u32(t112);
                        ft15 += _mm_popcnt_u32(t120);
                        ft16 += _mm_popcnt_u32(t121);
                        ft17 += _mm_popcnt_u32(t122);
                        ft18 += _mm_popcnt_u32(t200);
                        ft19 += _mm_popcnt_u32(t201);
                        ft20 += _mm_popcnt_u32(t202);
                        ft21 += _mm_popcnt_u32(t210);
                        ft22 += _mm_popcnt_u32(t211);
                        ft23 += _mm_popcnt_u32(t212);
                        ft24 += _mm_popcnt_u32(t220);
                        ft25 += _mm_popcnt_u32(t221);
                        ft26 += _mm_popcnt_u32(t222);

                        freq_table_IJ[xft + 0] += ft0;
                        freq_table_IJ[xft + 1] += ft1;
                        freq_table_IJ[xft + 2] += ft2;
                        freq_table_IJ[xft + 3] += ft3;
                        freq_table_IJ[xft + 4] += ft4;
                        freq_table_IJ[xft + 5] += ft5;
                        freq_table_IJ[xft + 6] += ft6;
                        freq_table_IJ[xft + 7] += ft7;
                        freq_table_IJ[xft + 8] += ft8;
                        freq_table_IJ[xft + 9] += ft9;
                        freq_table_IJ[xft + 10] += ft10;
                        freq_table_IJ[xft + 11] += ft11;
                        freq_table_IJ[xft + 12] += ft12;
                        freq_table_IJ[xft + 13] += ft13;
                        freq_table_IJ[xft + 14] += ft14;
                        freq_table_IJ[xft + 15] += ft15;
                        freq_table_IJ[xft + 16] += ft16;
                        freq_table_IJ[xft + 17] += ft17;
                        freq_table_IJ[xft + 18] += ft18;
                        freq_table_IJ[xft + 19] += ft19;
                        freq_table_IJ[xft + 20] += ft20;
                        freq_table_IJ[xft + 21] += ft21;
                        freq_table_IJ[xft + 22] += ft22;
                        freq_table_IJ[xft + 23] += ft23;
                        freq_table_IJ[xft + 24] += ft24;
                        freq_table_IJ[xft + 25] += ft25;
                        freq_table_IJ[xft + 26] += ft26;

                        n_comb ++;
                    }
                }
            }
 
            int base = 0;
            for(mi = 0; mi < block_i; mi++){
                for(mj = 0; mj < block_j - 1; mj++){
                    for(mk = mj + 1; mk < block_j; mk++){
                         m = base + (mk - (mj+1));
                        float score = 0.0;
                        for(n = 0; n < num_combs; n++){
                            score += addlog(freq_table_IJ[m * num_combs + n] + freq_table_IJ[(comb_ij + m) * num_combs + n] + 1) - addlog(freq_table_IJ[m * num_combs + n]) - addlog(freq_table_IJ[(comb_ij + m) * num_combs + n]);
                        }
                        score = fabs(score);

                        if(score < best_score_local){
                            best_score_local = score;
                            best_snp_global[tid][0] = ii + mi;
                            best_snp_global[tid][1] = jj + mj;
                            best_snp_global[tid][2] = jj + mk;
                        }
                        //printf("%d, %d, %d - Score: %f\n", ii + mi, jj + mj, jj + mk, score);
                    }
                    base += (block_j - (mj+1));
                }
            }

            v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;

            comb_ij = (block_i*block_j*(block_i-1))/2;

            counter+=comb_ij;

            //RESET FREQUENCY TABLES
            memset(freq_table_IJ, 0, 2 * num_fts_IJ * num_combs * sizeof(uint32_t));

            //Phenotype equal 0
            for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
                xi = xii_p0 + pp*block_snp;
                xj = xjj_p0 + pp*block_snp;
                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i-1; i++){
                    SNPA_0 = &data_zeros[xi + i*block_pac];
                    SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
                    //xft0 = i*block_j; 
                    for(j = i+1; j < block_i; j++){
                        //xft = (xft0 + j)*num_combs;
                        SNPB_0 = &data_zeros[xi + j*block_pac];
                        SNPB_1 = &data_zeros[xi + j*block_pac + block_snp*PP_zeros_r];
                        for(k = 0; k < block_j; k++){
                            xft = n_comb*num_combs;
                            SNPC_0 = &data_zeros[xj + k*block_pac];
                            SNPC_1 = &data_zeros[xj + k*block_pac + block_snp*PP_zeros_r];

                            //RESET FT VARS
                            ft0 = ft0 ^ ft0;
                            ft1 = ft1 ^ ft1;
                            ft2 = ft2 ^ ft2;
                            ft3 = ft3 ^ ft3;
                            ft4 = ft4 ^ ft4;
                            ft5 = ft5 ^ ft5;
                            ft6 = ft6 ^ ft6;
                            ft7 = ft7 ^ ft7;
                            ft8 = ft8 ^ ft8;
                            ft9 = ft9 ^ ft9;
                            ft10 = ft10 ^ ft10;
                            ft11 = ft11 ^ ft11;
                            ft12 = ft12 ^ ft12;
                            ft13 = ft13 ^ ft13;
                            ft14 = ft14 ^ ft14;
                            ft15 = ft15 ^ ft15;
                            ft16 = ft16 ^ ft16;
                            ft17 = ft17 ^ ft17;
                            ft18 = ft18 ^ ft18;
                            ft19 = ft19 ^ ft19;
                            ft20 = ft20 ^ ft20;
                            ft21 = ft21 ^ ft21;
                            ft22 = ft22 ^ ft22;
                            ft23 = ft23 ^ ft23;
                            ft24 = ft24 ^ ft24;
                            ft25 = ft25 ^ ft25;
                            ft26 = ft26 ^ ft26;

                            for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                //LOAD DATA
                                v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                //OR
                                v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                //NOT
                                v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                            }
                            //Remaining non-vectorized elements
                            for(p = v_elems; p < block_pac; p++){
                                di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                    
                                t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                t022 = SNPA_0[p] & dj2 & dk2;

                                t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                t122 = SNPA_1[p] & dj2 & dk2;

                                t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                t202 = di2 & SNPB_0[p] & dk2;
                                t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                t212 = di2 & SNPB_1[p] & dk2;
                                t220 = di2 & dj2 & SNPC_0[p];
                                t221 = di2 & dj2 & SNPC_1[p];
                                t222 = di2 & dj2 & dk2;

                                ft0 += _mm_popcnt_u32(t000);
                                ft1 += _mm_popcnt_u32(t001);
                                ft2 += _mm_popcnt_u32(t002);
                                ft3 += _mm_popcnt_u32(t010);
                                ft4 += _mm_popcnt_u32(t011);
                                ft5 += _mm_popcnt_u32(t012);
                                ft6 += _mm_popcnt_u32(t020);
                                ft7 += _mm_popcnt_u32(t021);
                                ft8 += _mm_popcnt_u32(t022);
                                ft9 += _mm_popcnt_u32(t100);
                                ft10 += _mm_popcnt_u32(t101);
                                ft11 += _mm_popcnt_u32(t102);
                                ft12 += _mm_popcnt_u32(t110);
                                ft13 += _mm_popcnt_u32(t111);
                                ft14 += _mm_popcnt_u32(t112);
                                ft15 += _mm_popcnt_u32(t120);
                                ft16 += _mm_popcnt_u32(t121);
                                ft17 += _mm_popcnt_u32(t122);
                                ft18 += _mm_popcnt_u32(t200);
                                ft19 += _mm_popcnt_u32(t201);
                                ft20 += _mm_popcnt_u32(t202);
                                ft21 += _mm_popcnt_u32(t210);
                                ft22 += _mm_popcnt_u32(t211);
                                ft23 += _mm_popcnt_u32(t212);
                                ft24 += _mm_popcnt_u32(t220);
                                ft25 += _mm_popcnt_u32(t221);
                                ft26 += _mm_popcnt_u32(t222);
                            }

                            freq_table_IJ[xft + 0] += ft0;
                            freq_table_IJ[xft + 1] += ft1;
                            freq_table_IJ[xft + 2] += ft2;
                            freq_table_IJ[xft + 3] += ft3;
                            freq_table_IJ[xft + 4] += ft4;
                            freq_table_IJ[xft + 5] += ft5;
                            freq_table_IJ[xft + 6] += ft6;
                            freq_table_IJ[xft + 7] += ft7;
                            freq_table_IJ[xft + 8] += ft8;
                            freq_table_IJ[xft + 9] += ft9;
                            freq_table_IJ[xft + 10] += ft10;
                            freq_table_IJ[xft + 11] += ft11;
                            freq_table_IJ[xft + 12] += ft12;
                            freq_table_IJ[xft + 13] += ft13;
                            freq_table_IJ[xft + 14] += ft14;
                            freq_table_IJ[xft + 15] += ft15;
                            freq_table_IJ[xft + 16] += ft16;
                            freq_table_IJ[xft + 17] += ft17;
                            freq_table_IJ[xft + 18] += ft18;
                            freq_table_IJ[xft + 19] += ft19;
                            freq_table_IJ[xft + 20] += ft20;
                            freq_table_IJ[xft + 21] += ft21;
                            freq_table_IJ[xft + 22] += ft22;
                            freq_table_IJ[xft + 23] += ft23;
                            freq_table_IJ[xft + 24] += ft24;
                            freq_table_IJ[xft + 25] += ft25;
                            freq_table_IJ[xft + 26] += ft26;

                            n_comb++;
                        }
                    }
                }
            }
            v_elems = ((PP_zeros - pp - 1)/VEC_ELEMS)*VEC_ELEMS;

            xi = xii_p0 + pp*block_snp;
            xj = xjj_p0 + pp*block_snp;

            n_comb = 0;
            //BETWEEN I and J
            for(i = 0; i < block_i-1; i++){
                SNPA_0 = &data_zeros[xi + i*block_pac];
                SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
                //xft0 = i*block_j;
                for(j = i+1; j < block_i; j++){
                    //xft = (xft0 + j)*num_combs;
                    SNPB_0 = &data_zeros[xi + j*block_pac];
                    SNPB_1 = &data_zeros[xi + j*block_pac + block_snp*PP_zeros_r];
                    for(k = 0; k < block_j; k++){
                        xft = n_comb*num_combs;
                        SNPC_0 = &data_zeros[xj + k*block_pac];
                        SNPC_1 = &data_zeros[xj + k*block_pac + block_snp*PP_zeros_r];

                        //RESET FT VARS
                        ft0 = ft0 ^ ft0;
                        ft1 = ft1 ^ ft1;
                        ft2 = ft2 ^ ft2;
                        ft3 = ft3 ^ ft3;
                        ft4 = ft4 ^ ft4;
                        ft5 = ft5 ^ ft5;
                        ft6 = ft6 ^ ft6;
                        ft7 = ft7 ^ ft7;
                        ft8 = ft8 ^ ft8;
                        ft9 = ft9 ^ ft9;
                        ft10 = ft10 ^ ft10;
                        ft11 = ft11 ^ ft11;
                        ft12 = ft12 ^ ft12;
                        ft13 = ft13 ^ ft13;
                        ft14 = ft14 ^ ft14;
                        ft15 = ft15 ^ ft15;
                        ft16 = ft16 ^ ft16;
                        ft17 = ft17 ^ ft17;
                        ft18 = ft18 ^ ft18;
                        ft19 = ft19 ^ ft19;
                        ft20 = ft20 ^ ft20;
                        ft21 = ft21 ^ ft21;
                        ft22 = ft22 ^ ft22;
                        ft23 = ft23 ^ ft23;
                        ft24 = ft24 ^ ft24;
                        ft25 = ft25 ^ ft25;
                        ft26 = ft26 ^ ft26;

                        for(p = 0; p < v_elems; p+=VEC_ELEMS){
                            //LOAD DATA
                            v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                            v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                            v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                            v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                            v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                            v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                            //OR
                            v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                            v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                            v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                            //NOT
                            v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                            v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                            v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                            v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                            v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                            v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                            v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                        }
                        //Remaining non-vectorized elements
                        for(p = v_elems; p < PP_zeros - pp - 1; p++){
                            di2 = ~(SNPA_0[p] | SNPA_1[p]);
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_zeros;
                        dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_zeros;
                        dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_zeros;
            
                        t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                        t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                        t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                        t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                        t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                        t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                        t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                        t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                        t022 = SNPA_0[p] & dj2 & dk2;

                        t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                        t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                        t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                        t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                        t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                        t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                        t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                        t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                        t122 = SNPA_1[p] & dj2 & dk2;

                        t200 = di2 & SNPB_0[p] & SNPC_0[p];
                        t201 = di2 & SNPB_0[p] & SNPC_1[p];
                        t202 = di2 & SNPB_0[p] & dk2;
                        t210 = di2 & SNPB_1[p] & SNPC_0[p];
                        t211 = di2 & SNPB_1[p] & SNPC_1[p];
                        t212 = di2 & SNPB_1[p] & dk2;
                        t220 = di2 & dj2 & SNPC_0[p];
                        t221 = di2 & dj2 & SNPC_1[p];
                        t222 = di2 & dj2 & dk2;

                        ft0 += _mm_popcnt_u32(t000);
                        ft1 += _mm_popcnt_u32(t001);
                        ft2 += _mm_popcnt_u32(t002);
                        ft3 += _mm_popcnt_u32(t010);
                        ft4 += _mm_popcnt_u32(t011);
                        ft5 += _mm_popcnt_u32(t012);
                        ft6 += _mm_popcnt_u32(t020);
                        ft7 += _mm_popcnt_u32(t021);
                        ft8 += _mm_popcnt_u32(t022);
                        ft9 += _mm_popcnt_u32(t100);
                        ft10 += _mm_popcnt_u32(t101);
                        ft11 += _mm_popcnt_u32(t102);
                        ft12 += _mm_popcnt_u32(t110);
                        ft13 += _mm_popcnt_u32(t111);
                        ft14 += _mm_popcnt_u32(t112);
                        ft15 += _mm_popcnt_u32(t120);
                        ft16 += _mm_popcnt_u32(t121);
                        ft17 += _mm_popcnt_u32(t122);
                        ft18 += _mm_popcnt_u32(t200);
                        ft19 += _mm_popcnt_u32(t201);
                        ft20 += _mm_popcnt_u32(t202);
                        ft21 += _mm_popcnt_u32(t210);
                        ft22 += _mm_popcnt_u32(t211);
                        ft23 += _mm_popcnt_u32(t212);
                        ft24 += _mm_popcnt_u32(t220);
                        ft25 += _mm_popcnt_u32(t221);
                        ft26 += _mm_popcnt_u32(t222);

                        freq_table_IJ[xft + 0] += ft0;
                        freq_table_IJ[xft + 1] += ft1;
                        freq_table_IJ[xft + 2] += ft2;
                        freq_table_IJ[xft + 3] += ft3;
                        freq_table_IJ[xft + 4] += ft4;
                        freq_table_IJ[xft + 5] += ft5;
                        freq_table_IJ[xft + 6] += ft6;
                        freq_table_IJ[xft + 7] += ft7;
                        freq_table_IJ[xft + 8] += ft8;
                        freq_table_IJ[xft + 9] += ft9;
                        freq_table_IJ[xft + 10] += ft10;
                        freq_table_IJ[xft + 11] += ft11;
                        freq_table_IJ[xft + 12] += ft12;
                        freq_table_IJ[xft + 13] += ft13;
                        freq_table_IJ[xft + 14] += ft14;
                        freq_table_IJ[xft + 15] += ft15;
                        freq_table_IJ[xft + 16] += ft16;
                        freq_table_IJ[xft + 17] += ft17;
                        freq_table_IJ[xft + 18] += ft18;
                        freq_table_IJ[xft + 19] += ft19;
                        freq_table_IJ[xft + 20] += ft20;
                        freq_table_IJ[xft + 21] += ft21;
                        freq_table_IJ[xft + 22] += ft22;
                        freq_table_IJ[xft + 23] += ft23;
                        freq_table_IJ[xft + 24] += ft24;
                        freq_table_IJ[xft + 25] += ft25;
                        freq_table_IJ[xft + 26] += ft26;

                        n_comb ++;
                    }
                }
            }

            v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;
            
            //Phenotype equal 1
            for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
                xi = xii_p1 + pp*block_snp;
                xj = xjj_p1 + pp*block_snp;

                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i - 1; i++){
                    SNPA_0 = &data_ones[xi + i*block_pac];
                    SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
                    //xft0 = i*block_j;
                    for(j = i + 1; j < block_i; j++){
                        //xft = (comb_ij + xft0 + j)*num_combs;
                        SNPB_0 = &data_ones[xi + j*block_pac];
                        SNPB_1 = &data_ones[xi + j*block_pac + block_snp*PP_ones_r];
                        for(k = 0; k < block_j; k++){
                            xft = (comb_ij + n_comb)*num_combs;
                            SNPC_0 = &data_ones[xj + k*block_pac];
                            SNPC_1 = &data_ones[xj + k*block_pac + block_snp*PP_ones_r];

                            //RESET FT VARS
                            ft0 = ft0 ^ ft0;
                            ft1 = ft1 ^ ft1;
                            ft2 = ft2 ^ ft2;
                            ft3 = ft3 ^ ft3;
                            ft4 = ft4 ^ ft4;
                            ft5 = ft5 ^ ft5;
                            ft6 = ft6 ^ ft6;
                            ft7 = ft7 ^ ft7;
                            ft8 = ft8 ^ ft8;
                            ft9 = ft9 ^ ft9;
                            ft10 = ft10 ^ ft10;
                            ft11 = ft11 ^ ft11;
                            ft12 = ft12 ^ ft12;
                            ft13 = ft13 ^ ft13;
                            ft14 = ft14 ^ ft14;
                            ft15 = ft15 ^ ft15;
                            ft16 = ft16 ^ ft16;
                            ft17 = ft17 ^ ft17;
                            ft18 = ft18 ^ ft18;
                            ft19 = ft19 ^ ft19;
                            ft20 = ft20 ^ ft20;
                            ft21 = ft21 ^ ft21;
                            ft22 = ft22 ^ ft22;
                            ft23 = ft23 ^ ft23;
                            ft24 = ft24 ^ ft24;
                            ft25 = ft25 ^ ft25;
                            ft26 = ft26 ^ ft26;

                            for(p = 0; p < v_elems; p+=VEC_ELEMS){
                                //LOAD DATA
                                v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                                v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                                v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                                v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                                v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                                v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                                //OR
                                v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                                v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                                v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                                //NOT
                                v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                                v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                                v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                                v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                                v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                                v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                                v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                                v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                                v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                                v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                                v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                                v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                                v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                                v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                                v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                                ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                                ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                                ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                                ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                                ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                                ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                                ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                                ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                                ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                                ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                                ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                                ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                                ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                                ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                                ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                                ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                                ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                                ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                                ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                                ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                                ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                                ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                                ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                                ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                                ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                                ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                                ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                            }
                            //Remaining non-vectorized elements
                            for(p = v_elems; p < block_pac; p++){
                                di2 = ~(SNPA_0[p] | SNPA_1[p]);
                                dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                                dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                    
                                t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                                t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                                t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                                t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                                t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                                t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                                t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                                t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                                t022 = SNPA_0[p] & dj2 & dk2;

                                t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                                t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                                t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                                t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                                t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                                t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                                t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                                t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                                t122 = SNPA_1[p] & dj2 & dk2;

                                t200 = di2 & SNPB_0[p] & SNPC_0[p];
                                t201 = di2 & SNPB_0[p] & SNPC_1[p];
                                t202 = di2 & SNPB_0[p] & dk2;
                                t210 = di2 & SNPB_1[p] & SNPC_0[p];
                                t211 = di2 & SNPB_1[p] & SNPC_1[p];
                                t212 = di2 & SNPB_1[p] & dk2;
                                t220 = di2 & dj2 & SNPC_0[p];
                                t221 = di2 & dj2 & SNPC_1[p];
                                t222 = di2 & dj2 & dk2;

                                ft0 += _mm_popcnt_u32(t000);
                                ft1 += _mm_popcnt_u32(t001);
                                ft2 += _mm_popcnt_u32(t002);
                                ft3 += _mm_popcnt_u32(t010);
                                ft4 += _mm_popcnt_u32(t011);
                                ft5 += _mm_popcnt_u32(t012);
                                ft6 += _mm_popcnt_u32(t020);
                                ft7 += _mm_popcnt_u32(t021);
                                ft8 += _mm_popcnt_u32(t022);
                                ft9 += _mm_popcnt_u32(t100);
                                ft10 += _mm_popcnt_u32(t101);
                                ft11 += _mm_popcnt_u32(t102);
                                ft12 += _mm_popcnt_u32(t110);
                                ft13 += _mm_popcnt_u32(t111);
                                ft14 += _mm_popcnt_u32(t112);
                                ft15 += _mm_popcnt_u32(t120);
                                ft16 += _mm_popcnt_u32(t121);
                                ft17 += _mm_popcnt_u32(t122);
                                ft18 += _mm_popcnt_u32(t200);
                                ft19 += _mm_popcnt_u32(t201);
                                ft20 += _mm_popcnt_u32(t202);
                                ft21 += _mm_popcnt_u32(t210);
                                ft22 += _mm_popcnt_u32(t211);
                                ft23 += _mm_popcnt_u32(t212);
                                ft24 += _mm_popcnt_u32(t220);
                                ft25 += _mm_popcnt_u32(t221);
                                ft26 += _mm_popcnt_u32(t222);
                            }

                            freq_table_IJ[xft + 0] += ft0;
                            freq_table_IJ[xft + 1] += ft1;
                            freq_table_IJ[xft + 2] += ft2;
                            freq_table_IJ[xft + 3] += ft3;
                            freq_table_IJ[xft + 4] += ft4;
                            freq_table_IJ[xft + 5] += ft5;
                            freq_table_IJ[xft + 6] += ft6;
                            freq_table_IJ[xft + 7] += ft7;
                            freq_table_IJ[xft + 8] += ft8;
                            freq_table_IJ[xft + 9] += ft9;
                            freq_table_IJ[xft + 10] += ft10;
                            freq_table_IJ[xft + 11] += ft11;
                            freq_table_IJ[xft + 12] += ft12;
                            freq_table_IJ[xft + 13] += ft13;
                            freq_table_IJ[xft + 14] += ft14;
                            freq_table_IJ[xft + 15] += ft15;
                            freq_table_IJ[xft + 16] += ft16;
                            freq_table_IJ[xft + 17] += ft17;
                            freq_table_IJ[xft + 18] += ft18;
                            freq_table_IJ[xft + 19] += ft19;
                            freq_table_IJ[xft + 20] += ft20;
                            freq_table_IJ[xft + 21] += ft21;
                            freq_table_IJ[xft + 22] += ft22;
                            freq_table_IJ[xft + 23] += ft23;
                            freq_table_IJ[xft + 24] += ft24;
                            freq_table_IJ[xft + 25] += ft25;
                            freq_table_IJ[xft + 26] += ft26;

                            n_comb ++;
                        }
                    }
                }
            }
            v_elems = ((PP_ones - pp - 1)/VEC_ELEMS)*VEC_ELEMS;


            n_comb = 0;

            xi = xii_p1 + pp*block_snp;
            xj = xjj_p1 + pp*block_snp;
            //BETWEEN I and J
            for(i = 0; i < block_i - 1; i++){
                SNPA_0 = &data_ones[xi + i*block_pac];
                SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
                //xft0 = i*block_j;
                for(j = i+1; j < block_i; j++){
                    //xft = (comb_ij + xft0 + j)*num_combs;
                    SNPB_0 = &data_ones[xi + j*block_pac];
                    SNPB_1 = &data_ones[xi + j*block_pac + block_snp*PP_ones_r];
                    for(k = 0; k < block_j; k++){
                        xft = (comb_ij + n_comb)*num_combs;
                        SNPC_0 = &data_ones[xj + k*block_pac];
                        SNPC_1 = &data_ones[xj + k*block_pac + block_snp*PP_ones_r];

                        //RESET FT VARS
                        ft0 = ft0 ^ ft0;
                        ft1 = ft1 ^ ft1;
                        ft2 = ft2 ^ ft2;
                        ft3 = ft3 ^ ft3;
                        ft4 = ft4 ^ ft4;
                        ft5 = ft5 ^ ft5;
                        ft6 = ft6 ^ ft6;
                        ft7 = ft7 ^ ft7;
                        ft8 = ft8 ^ ft8;
                        ft9 = ft9 ^ ft9;
                        ft10 = ft10 ^ ft10;
                        ft11 = ft11 ^ ft11;
                        ft12 = ft12 ^ ft12;
                        ft13 = ft13 ^ ft13;
                        ft14 = ft14 ^ ft14;
                        ft15 = ft15 ^ ft15;
                        ft16 = ft16 ^ ft16;
                        ft17 = ft17 ^ ft17;
                        ft18 = ft18 ^ ft18;
                        ft19 = ft19 ^ ft19;
                        ft20 = ft20 ^ ft20;
                        ft21 = ft21 ^ ft21;
                        ft22 = ft22 ^ ft22;
                        ft23 = ft23 ^ ft23;
                        ft24 = ft24 ^ ft24;
                        ft25 = ft25 ^ ft25;
                        ft26 = ft26 ^ ft26;

                        for(p = 0; p < v_elems; p+=VEC_ELEMS){
                            //LOAD DATA
                            v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                            v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                            v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                            v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                            v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                            v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                            //OR
                            v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                            v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                            v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                            //NOT
                            v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                            v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                            v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                            v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                            v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                            v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                            v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                        }
                        //Remaining non-vectorized elements
                        for(p = v_elems; p < PP_ones - pp - 1; p++){
                            di2 = ~(SNPA_0[p] | SNPA_1[p]);
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_ones;
                        dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_ones;
                        dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_ones;
            
                        t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                        t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                        t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                        t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                        t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                        t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                        t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                        t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                        t022 = SNPA_0[p] & dj2 & dk2;

                        t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                        t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                        t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                        t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                        t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                        t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                        t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                        t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                        t122 = SNPA_1[p] & dj2 & dk2;

                        t200 = di2 & SNPB_0[p] & SNPC_0[p];
                        t201 = di2 & SNPB_0[p] & SNPC_1[p];
                        t202 = di2 & SNPB_0[p] & dk2;
                        t210 = di2 & SNPB_1[p] & SNPC_0[p];
                        t211 = di2 & SNPB_1[p] & SNPC_1[p];
                        t212 = di2 & SNPB_1[p] & dk2;
                        t220 = di2 & dj2 & SNPC_0[p];
                        t221 = di2 & dj2 & SNPC_1[p];
                        t222 = di2 & dj2 & dk2;

                        ft0 += _mm_popcnt_u32(t000);
                        ft1 += _mm_popcnt_u32(t001);
                        ft2 += _mm_popcnt_u32(t002);
                        ft3 += _mm_popcnt_u32(t010);
                        ft4 += _mm_popcnt_u32(t011);
                        ft5 += _mm_popcnt_u32(t012);
                        ft6 += _mm_popcnt_u32(t020);
                        ft7 += _mm_popcnt_u32(t021);
                        ft8 += _mm_popcnt_u32(t022);
                        ft9 += _mm_popcnt_u32(t100);
                        ft10 += _mm_popcnt_u32(t101);
                        ft11 += _mm_popcnt_u32(t102);
                        ft12 += _mm_popcnt_u32(t110);
                        ft13 += _mm_popcnt_u32(t111);
                        ft14 += _mm_popcnt_u32(t112);
                        ft15 += _mm_popcnt_u32(t120);
                        ft16 += _mm_popcnt_u32(t121);
                        ft17 += _mm_popcnt_u32(t122);
                        ft18 += _mm_popcnt_u32(t200);
                        ft19 += _mm_popcnt_u32(t201);
                        ft20 += _mm_popcnt_u32(t202);
                        ft21 += _mm_popcnt_u32(t210);
                        ft22 += _mm_popcnt_u32(t211);
                        ft23 += _mm_popcnt_u32(t212);
                        ft24 += _mm_popcnt_u32(t220);
                        ft25 += _mm_popcnt_u32(t221);
                        ft26 += _mm_popcnt_u32(t222);

                        freq_table_IJ[xft + 0] += ft0;
                        freq_table_IJ[xft + 1] += ft1;
                        freq_table_IJ[xft + 2] += ft2;
                        freq_table_IJ[xft + 3] += ft3;
                        freq_table_IJ[xft + 4] += ft4;
                        freq_table_IJ[xft + 5] += ft5;
                        freq_table_IJ[xft + 6] += ft6;
                        freq_table_IJ[xft + 7] += ft7;
                        freq_table_IJ[xft + 8] += ft8;
                        freq_table_IJ[xft + 9] += ft9;
                        freq_table_IJ[xft + 10] += ft10;
                        freq_table_IJ[xft + 11] += ft11;
                        freq_table_IJ[xft + 12] += ft12;
                        freq_table_IJ[xft + 13] += ft13;
                        freq_table_IJ[xft + 14] += ft14;
                        freq_table_IJ[xft + 15] += ft15;
                        freq_table_IJ[xft + 16] += ft16;
                        freq_table_IJ[xft + 17] += ft17;
                        freq_table_IJ[xft + 18] += ft18;
                        freq_table_IJ[xft + 19] += ft19;
                        freq_table_IJ[xft + 20] += ft20;
                        freq_table_IJ[xft + 21] += ft21;
                        freq_table_IJ[xft + 22] += ft22;
                        freq_table_IJ[xft + 23] += ft23;
                        freq_table_IJ[xft + 24] += ft24;
                        freq_table_IJ[xft + 25] += ft25;
                        freq_table_IJ[xft + 26] += ft26;

                        n_comb ++;
                    }
                }
            }
 
            base = 0;
            for(mi = 0; mi < block_i-1; mi++){
                for(mj = mi+1; mj < block_i; mj++){
                    for(mk = 0; mk < block_j; mk++){
                        m = base + (mj - (mi+1))*block_j + mk;
                        float score = 0.0;
                        for(n = 0; n < num_combs; n++){
                            score += addlog(freq_table_IJ[m * num_combs + n] + freq_table_IJ[(comb_ij + m) * num_combs + n] + 1) - addlog(freq_table_IJ[m * num_combs + n]) - addlog(freq_table_IJ[(comb_ij + m) * num_combs + n]);
                        }
                        score = fabs(score);

                        if(score < best_score_local){
                            best_score_local = score;
                            best_snp_global[tid][0] = ii + mi;
                            best_snp_global[tid][1] = ii + mj;
                            best_snp_global[tid][2] = jj + mk;
                        }
                        //printf("%d, %d, %d - Score: %f\n", ii + mi, ii + mj, jj + mk, score);
                    }
                }
                base += block_j*(block_i - (mi + 1));
            }
        }

        v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;

        //RESET FREQUENCY TABLES
        memset(freq_table_I, 0, 2 * num_fts_I * num_combs * sizeof(uint32_t));
       
        xjj_p0 = jj*PP_zeros_r*2;
        xjj_p1 = jj*PP_ones_r*2;

        comb_ii = (block_i*(block_i -1)*(block_i -2))/6;

        counter+=comb_ii;

        //Phenotype = 0
        for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
            xi = xii_p0 + pp*block_snp;
            xj = xjj_p0 + pp*block_snp;
            n_comb = 0;
            //BLOCK II
            for(i = 0; i < block_i - 2; i++){
                SNPA_0 = &data_zeros[xi + i*block_pac];
                SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
                //INSIDE BLOCK I
                for(j = i+1; j < block_i - 1; j++){
                    SNPB_0 = &data_zeros[xi + j*block_pac];
                    SNPB_1 = &data_zeros[xi + j*block_pac + block_snp*PP_zeros_r];
                    for(k = j+1; k < block_i; k++){
                        SNPC_0 = &data_zeros[xi + k*block_pac];
                        SNPC_1 = &data_zeros[xi + k*block_pac + block_snp*PP_zeros_r];
                        xft = n_comb*num_combs;

                        //RESET FT VARS
                        ft0 = ft0 ^ ft0;
                        ft1 = ft1 ^ ft1;
                        ft2 = ft2 ^ ft2;
                        ft3 = ft3 ^ ft3;
                        ft4 = ft4 ^ ft4;
                        ft5 = ft5 ^ ft5;
                        ft6 = ft6 ^ ft6;
                        ft7 = ft7 ^ ft7;
                        ft8 = ft8 ^ ft8;
                        ft9 = ft9 ^ ft9;
                        ft10 = ft10 ^ ft10;
                        ft11 = ft11 ^ ft11;
                        ft12 = ft12 ^ ft12;
                        ft13 = ft13 ^ ft13;
                        ft14 = ft14 ^ ft14;
                        ft15 = ft15 ^ ft15;
                        ft16 = ft16 ^ ft16;
                        ft17 = ft17 ^ ft17;
                        ft18 = ft18 ^ ft18;
                        ft19 = ft19 ^ ft19;
                        ft20 = ft20 ^ ft20;
                        ft21 = ft21 ^ ft21;
                        ft22 = ft22 ^ ft22;
                        ft23 = ft23 ^ ft23;
                        ft24 = ft24 ^ ft24;
                        ft25 = ft25 ^ ft25;
                        ft26 = ft26 ^ ft26;

                        for(p = 0; p < v_elems; p+=VEC_ELEMS){
                            //LOAD DATA
                            v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                            v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                            v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                            v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                            v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                            v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                            //OR
                            v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                            v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                            v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                            //NOT
                            v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                            v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                            v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                            v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                            v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                            v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                            v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                        }
                        //Remaining non-vectorized elements
                        for(p = v_elems; p < block_pac; p++){
                            di2 = ~(SNPA_0[p] | SNPA_1[p]);
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);
                        }

                        freq_table_I[xft + 0] += ft0;
                        freq_table_I[xft + 1] += ft1;
                        freq_table_I[xft + 2] += ft2;
                        freq_table_I[xft + 3] += ft3;
                        freq_table_I[xft + 4] += ft4;
                        freq_table_I[xft + 5] += ft5;
                        freq_table_I[xft + 6] += ft6;
                        freq_table_I[xft + 7] += ft7;
                        freq_table_I[xft + 8] += ft8;
                        freq_table_I[xft + 9] += ft9;
                        freq_table_I[xft + 10] += ft10;
                        freq_table_I[xft + 11] += ft11;
                        freq_table_I[xft + 12] += ft12;
                        freq_table_I[xft + 13] += ft13;
                        freq_table_I[xft + 14] += ft14;
                        freq_table_I[xft + 15] += ft15;
                        freq_table_I[xft + 16] += ft16;
                        freq_table_I[xft + 17] += ft17;
                        freq_table_I[xft + 18] += ft18;
                        freq_table_I[xft + 19] += ft19;
                        freq_table_I[xft + 20] += ft20;
                        freq_table_I[xft + 21] += ft21;
                        freq_table_I[xft + 22] += ft22;
                        freq_table_I[xft + 23] += ft23;
                        freq_table_I[xft + 24] += ft24;
                        freq_table_I[xft + 25] += ft25;
                        freq_table_I[xft + 26] += ft26;

                        n_comb++;
                    }
                }
            }
        }
        v_elems = ((PP_zeros - pp - 1)/VEC_ELEMS)*VEC_ELEMS;


        xi = xii_p0 + pp*block_snp;
        xj = xjj_p0 + pp*block_snp;
        n_comb = 0;
        //BLOCK II
        for(i = 0; i < block_i - 2; i++){
            SNPA_0 = &data_zeros[xi + i*block_pac];
            SNPA_1 = &data_zeros[xi + i*block_pac + block_snp*PP_zeros_r];
            //INSIDE BLOCK I
            for(j = i+1; j < block_i - 1; j++){
                SNPB_0 = &data_zeros[xi + j*block_pac];
                SNPB_1 = &data_zeros[xi + j*block_pac + block_snp*PP_zeros_r];
                for(k = j+1; k < block_i; k++){
                    SNPC_0 = &data_zeros[xi + k*block_pac];
                    SNPC_1 = &data_zeros[xi + k*block_pac + block_snp*PP_zeros_r];
                    xft = n_comb*num_combs;

                    //RESET FT VARS
                    ft0 = ft0 ^ ft0;
                    ft1 = ft1 ^ ft1;
                    ft2 = ft2 ^ ft2;
                    ft3 = ft3 ^ ft3;
                    ft4 = ft4 ^ ft4;
                    ft5 = ft5 ^ ft5;
                    ft6 = ft6 ^ ft6;
                    ft7 = ft7 ^ ft7;
                    ft8 = ft8 ^ ft8;
                    ft9 = ft9 ^ ft9;
                    ft10 = ft10 ^ ft10;
                    ft11 = ft11 ^ ft11;
                    ft12 = ft12 ^ ft12;
                    ft13 = ft13 ^ ft13;
                    ft14 = ft14 ^ ft14;
                    ft15 = ft15 ^ ft15;
                    ft16 = ft16 ^ ft16;
                    ft17 = ft17 ^ ft17;
                    ft18 = ft18 ^ ft18;
                    ft19 = ft19 ^ ft19;
                    ft20 = ft20 ^ ft20;
                    ft21 = ft21 ^ ft21;
                    ft22 = ft22 ^ ft22;
                    ft23 = ft23 ^ ft23;
                    ft24 = ft24 ^ ft24;
                    ft25 = ft25 ^ ft25;
                    ft26 = ft26 ^ ft26;

                    for(p = 0; p < v_elems; p+=VEC_ELEMS){
                        //LOAD DATA
                        v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                        v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                        v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                        v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                        v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                        v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                        //OR
                        v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                        v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                        v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                        //NOT
                        v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                        v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                        v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                        v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                        v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                        v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                        v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                        v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                        v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                        v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                        v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                        v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                        v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                        v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                        v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                        v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                        v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                        v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                        v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                        v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                        v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                        v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                        v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                        v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                        v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                        v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                        v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                        v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                        v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                        v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                    }
                    //Remaining non-vectorized elements
                    for(p = v_elems; p < PP_zeros - pp - 1; p++){
                        di2 = ~(SNPA_0[p] | SNPA_1[p]);
                        dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                        dk2 = ~(SNPC_0[p] | SNPC_1[p]);
            
                        t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                        t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                        t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                        t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                        t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                        t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                        t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                        t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                        t022 = SNPA_0[p] & dj2 & dk2;

                        t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                        t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                        t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                        t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                        t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                        t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                        t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                        t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                        t122 = SNPA_1[p] & dj2 & dk2;

                        t200 = di2 & SNPB_0[p] & SNPC_0[p];
                        t201 = di2 & SNPB_0[p] & SNPC_1[p];
                        t202 = di2 & SNPB_0[p] & dk2;
                        t210 = di2 & SNPB_1[p] & SNPC_0[p];
                        t211 = di2 & SNPB_1[p] & SNPC_1[p];
                        t212 = di2 & SNPB_1[p] & dk2;
                        t220 = di2 & dj2 & SNPC_0[p];
                        t221 = di2 & dj2 & SNPC_1[p];
                        t222 = di2 & dj2 & dk2;

                        ft0 += _mm_popcnt_u32(t000);
                        ft1 += _mm_popcnt_u32(t001);
                        ft2 += _mm_popcnt_u32(t002);
                        ft3 += _mm_popcnt_u32(t010);
                        ft4 += _mm_popcnt_u32(t011);
                        ft5 += _mm_popcnt_u32(t012);
                        ft6 += _mm_popcnt_u32(t020);
                        ft7 += _mm_popcnt_u32(t021);
                        ft8 += _mm_popcnt_u32(t022);
                        ft9 += _mm_popcnt_u32(t100);
                        ft10 += _mm_popcnt_u32(t101);
                        ft11 += _mm_popcnt_u32(t102);
                        ft12 += _mm_popcnt_u32(t110);
                        ft13 += _mm_popcnt_u32(t111);
                        ft14 += _mm_popcnt_u32(t112);
                        ft15 += _mm_popcnt_u32(t120);
                        ft16 += _mm_popcnt_u32(t121);
                        ft17 += _mm_popcnt_u32(t122);
                        ft18 += _mm_popcnt_u32(t200);
                        ft19 += _mm_popcnt_u32(t201);
                        ft20 += _mm_popcnt_u32(t202);
                        ft21 += _mm_popcnt_u32(t210);
                        ft22 += _mm_popcnt_u32(t211);
                        ft23 += _mm_popcnt_u32(t212);
                        ft24 += _mm_popcnt_u32(t220);
                        ft25 += _mm_popcnt_u32(t221);
                        ft26 += _mm_popcnt_u32(t222);
                    }

                    //Do Remaining Elements
                    di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_zeros;
                    dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_zeros;
                    dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_zeros;
        
                    t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                    t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                    t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                    t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                    t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                    t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                    t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                    t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                    t022 = SNPA_0[p] & dj2 & dk2;

                    t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                    t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                    t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                    t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                    t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                    t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                    t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                    t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                    t122 = SNPA_1[p] & dj2 & dk2;

                    t200 = di2 & SNPB_0[p] & SNPC_0[p];
                    t201 = di2 & SNPB_0[p] & SNPC_1[p];
                    t202 = di2 & SNPB_0[p] & dk2;
                    t210 = di2 & SNPB_1[p] & SNPC_0[p];
                    t211 = di2 & SNPB_1[p] & SNPC_1[p];
                    t212 = di2 & SNPB_1[p] & dk2;
                    t220 = di2 & dj2 & SNPC_0[p];
                    t221 = di2 & dj2 & SNPC_1[p];
                    t222 = di2 & dj2 & dk2;

                    ft0 += _mm_popcnt_u32(t000);
                    ft1 += _mm_popcnt_u32(t001);
                    ft2 += _mm_popcnt_u32(t002);
                    ft3 += _mm_popcnt_u32(t010);
                    ft4 += _mm_popcnt_u32(t011);
                    ft5 += _mm_popcnt_u32(t012);
                    ft6 += _mm_popcnt_u32(t020);
                    ft7 += _mm_popcnt_u32(t021);
                    ft8 += _mm_popcnt_u32(t022);
                    ft9 += _mm_popcnt_u32(t100);
                    ft10 += _mm_popcnt_u32(t101);
                    ft11 += _mm_popcnt_u32(t102);
                    ft12 += _mm_popcnt_u32(t110);
                    ft13 += _mm_popcnt_u32(t111);
                    ft14 += _mm_popcnt_u32(t112);
                    ft15 += _mm_popcnt_u32(t120);
                    ft16 += _mm_popcnt_u32(t121);
                    ft17 += _mm_popcnt_u32(t122);
                    ft18 += _mm_popcnt_u32(t200);
                    ft19 += _mm_popcnt_u32(t201);
                    ft20 += _mm_popcnt_u32(t202);
                    ft21 += _mm_popcnt_u32(t210);
                    ft22 += _mm_popcnt_u32(t211);
                    ft23 += _mm_popcnt_u32(t212);
                    ft24 += _mm_popcnt_u32(t220);
                    ft25 += _mm_popcnt_u32(t221);
                    ft26 += _mm_popcnt_u32(t222);

                    freq_table_I[xft + 0] += ft0;
                    freq_table_I[xft + 1] += ft1;
                    freq_table_I[xft + 2] += ft2;
                    freq_table_I[xft + 3] += ft3;
                    freq_table_I[xft + 4] += ft4;
                    freq_table_I[xft + 5] += ft5;
                    freq_table_I[xft + 6] += ft6;
                    freq_table_I[xft + 7] += ft7;
                    freq_table_I[xft + 8] += ft8;
                    freq_table_I[xft + 9] += ft9;
                    freq_table_I[xft + 10] += ft10;
                    freq_table_I[xft + 11] += ft11;
                    freq_table_I[xft + 12] += ft12;
                    freq_table_I[xft + 13] += ft13;
                    freq_table_I[xft + 14] += ft14;
                    freq_table_I[xft + 15] += ft15;
                    freq_table_I[xft + 16] += ft16;
                    freq_table_I[xft + 17] += ft17;
                    freq_table_I[xft + 18] += ft18;
                    freq_table_I[xft + 19] += ft19;
                    freq_table_I[xft + 20] += ft20;
                    freq_table_I[xft + 21] += ft21;
                    freq_table_I[xft + 22] += ft22;
                    freq_table_I[xft + 23] += ft23;
                    freq_table_I[xft + 24] += ft24;
                    freq_table_I[xft + 25] += ft25;
                    freq_table_I[xft + 26] += ft26;

                    n_comb++;
                }
            }
        }

        v_elems = (block_pac/VEC_ELEMS)*VEC_ELEMS;

        //Phenotype = 1
        for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
            xi = xii_p1 + pp*block_snp;
            xj = xjj_p1 + pp*block_snp;
            n_comb = 0;
            //BLOCK II
            for(i = 0; i < block_i - 2; i++){
                SNPA_0 = &data_ones[xi + i*block_pac];
                SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
                //INSIDE BLOCK I
                for(j = i+1; j < block_i - 1; j++){
                    SNPB_0 = &data_ones[xi + j*block_pac];
                    SNPB_1 = &data_ones[xi + j*block_pac + block_snp*PP_ones_r];
                    for(k = j+1; k < block_i; k++){
                        SNPC_0 = &data_ones[xi + k*block_pac];
                        SNPC_1 = &data_ones[xi + k*block_pac + block_snp*PP_ones_r];
                        xft = (comb_ii + n_comb)*num_combs;

                        //RESET FT VARS
                        ft0 = ft0 ^ ft0;
                        ft1 = ft1 ^ ft1;
                        ft2 = ft2 ^ ft2;
                        ft3 = ft3 ^ ft3;
                        ft4 = ft4 ^ ft4;
                        ft5 = ft5 ^ ft5;
                        ft6 = ft6 ^ ft6;
                        ft7 = ft7 ^ ft7;
                        ft8 = ft8 ^ ft8;
                        ft9 = ft9 ^ ft9;
                        ft10 = ft10 ^ ft10;
                        ft11 = ft11 ^ ft11;
                        ft12 = ft12 ^ ft12;
                        ft13 = ft13 ^ ft13;
                        ft14 = ft14 ^ ft14;
                        ft15 = ft15 ^ ft15;
                        ft16 = ft16 ^ ft16;
                        ft17 = ft17 ^ ft17;
                        ft18 = ft18 ^ ft18;
                        ft19 = ft19 ^ ft19;
                        ft20 = ft20 ^ ft20;
                        ft21 = ft21 ^ ft21;
                        ft22 = ft22 ^ ft22;
                        ft23 = ft23 ^ ft23;
                        ft24 = ft24 ^ ft24;
                        ft25 = ft25 ^ ft25;
                        ft26 = ft26 ^ ft26;

                        for(p = 0; p < v_elems; p+=VEC_ELEMS){
                            //LOAD DATA
                            v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                            v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                            v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                            v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                            v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                            v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                            //OR
                            v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                            v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                            v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                            //NOT
                            v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                            v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                            v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                            v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                            v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                            v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                            v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                            v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                            v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                            v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                            v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                            v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                            v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                            v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                            v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                            ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                            ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                            ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                            ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                            ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                            ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                            ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                            ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                            ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                            ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                            ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                            ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                            ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                            ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                            ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                            ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                            ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                            ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                            ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                            ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                            ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                            ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                            ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                            ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                            ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                            ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                            ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                        }
                        //Remaining non-vectorized elements
                        for(p = v_elems; p < block_pac; p++){
                            di2 = ~(SNPA_0[p] | SNPA_1[p]);
                            dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                            dk2 = ~(SNPC_0[p] | SNPC_1[p]);
                
                            t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                            t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                            t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                            t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                            t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                            t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                            t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                            t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                            t022 = SNPA_0[p] & dj2 & dk2;

                            t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                            t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                            t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                            t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                            t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                            t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                            t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                            t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                            t122 = SNPA_1[p] & dj2 & dk2;

                            t200 = di2 & SNPB_0[p] & SNPC_0[p];
                            t201 = di2 & SNPB_0[p] & SNPC_1[p];
                            t202 = di2 & SNPB_0[p] & dk2;
                            t210 = di2 & SNPB_1[p] & SNPC_0[p];
                            t211 = di2 & SNPB_1[p] & SNPC_1[p];
                            t212 = di2 & SNPB_1[p] & dk2;
                            t220 = di2 & dj2 & SNPC_0[p];
                            t221 = di2 & dj2 & SNPC_1[p];
                            t222 = di2 & dj2 & dk2;

                            ft0 += _mm_popcnt_u32(t000);
                            ft1 += _mm_popcnt_u32(t001);
                            ft2 += _mm_popcnt_u32(t002);
                            ft3 += _mm_popcnt_u32(t010);
                            ft4 += _mm_popcnt_u32(t011);
                            ft5 += _mm_popcnt_u32(t012);
                            ft6 += _mm_popcnt_u32(t020);
                            ft7 += _mm_popcnt_u32(t021);
                            ft8 += _mm_popcnt_u32(t022);
                            ft9 += _mm_popcnt_u32(t100);
                            ft10 += _mm_popcnt_u32(t101);
                            ft11 += _mm_popcnt_u32(t102);
                            ft12 += _mm_popcnt_u32(t110);
                            ft13 += _mm_popcnt_u32(t111);
                            ft14 += _mm_popcnt_u32(t112);
                            ft15 += _mm_popcnt_u32(t120);
                            ft16 += _mm_popcnt_u32(t121);
                            ft17 += _mm_popcnt_u32(t122);
                            ft18 += _mm_popcnt_u32(t200);
                            ft19 += _mm_popcnt_u32(t201);
                            ft20 += _mm_popcnt_u32(t202);
                            ft21 += _mm_popcnt_u32(t210);
                            ft22 += _mm_popcnt_u32(t211);
                            ft23 += _mm_popcnt_u32(t212);
                            ft24 += _mm_popcnt_u32(t220);
                            ft25 += _mm_popcnt_u32(t221);
                            ft26 += _mm_popcnt_u32(t222);
                        }

                        freq_table_I[xft + 0] += ft0;
                        freq_table_I[xft + 1] += ft1;
                        freq_table_I[xft + 2] += ft2;
                        freq_table_I[xft + 3] += ft3;
                        freq_table_I[xft + 4] += ft4;
                        freq_table_I[xft + 5] += ft5;
                        freq_table_I[xft + 6] += ft6;
                        freq_table_I[xft + 7] += ft7;
                        freq_table_I[xft + 8] += ft8;
                        freq_table_I[xft + 9] += ft9;
                        freq_table_I[xft + 10] += ft10;
                        freq_table_I[xft + 11] += ft11;
                        freq_table_I[xft + 12] += ft12;
                        freq_table_I[xft + 13] += ft13;
                        freq_table_I[xft + 14] += ft14;
                        freq_table_I[xft + 15] += ft15;
                        freq_table_I[xft + 16] += ft16;
                        freq_table_I[xft + 17] += ft17;
                        freq_table_I[xft + 18] += ft18;
                        freq_table_I[xft + 19] += ft19;
                        freq_table_I[xft + 20] += ft20;
                        freq_table_I[xft + 21] += ft21;
                        freq_table_I[xft + 22] += ft22;
                        freq_table_I[xft + 23] += ft23;
                        freq_table_I[xft + 24] += ft24;
                        freq_table_I[xft + 25] += ft25;
                        freq_table_I[xft + 26] += ft26;

                        n_comb++;
                    }
                }
            }
        }

        v_elems = ((PP_ones - pp - 1)/VEC_ELEMS)*VEC_ELEMS;

        xi = xii_p1 + pp*block_snp;
        xj = xjj_p1 + pp*block_snp;
        n_comb = 0;
        //BLOCK II
        for(i = 0; i < block_i - 2; i++){
            SNPA_0 = &data_ones[xi + i*block_pac];
            SNPA_1 = &data_ones[xi + i*block_pac + block_snp*PP_ones_r];
            xft0 = i*block_snp;
            //INSIDE BLOCK I
            for(j = i+1; j < block_i - 1; j++){
                SNPB_0 = &data_ones[xi + j*block_pac];
                SNPB_1 = &data_ones[xi + j*block_pac + block_snp*PP_ones_r];
                for(k = j+1; k < block_i; k++){
                    SNPC_0 = &data_ones[xi + k*block_pac];
                    SNPC_1 = &data_ones[xi + k*block_pac + block_snp*PP_ones_r];
                    xft = (comb_ii + n_comb)*num_combs;

                    //RESET FT VARS
                    ft0 = ft0 ^ ft0;
                    ft1 = ft1 ^ ft1;
                    ft2 = ft2 ^ ft2;
                    ft3 = ft3 ^ ft3;
                    ft4 = ft4 ^ ft4;
                    ft5 = ft5 ^ ft5;
                    ft6 = ft6 ^ ft6;
                    ft7 = ft7 ^ ft7;
                    ft8 = ft8 ^ ft8;
                    ft9 = ft9 ^ ft9;
                    ft10 = ft10 ^ ft10;
                    ft11 = ft11 ^ ft11;
                    ft12 = ft12 ^ ft12;
                    ft13 = ft13 ^ ft13;
                    ft14 = ft14 ^ ft14;
                    ft15 = ft15 ^ ft15;
                    ft16 = ft16 ^ ft16;
                    ft17 = ft17 ^ ft17;
                    ft18 = ft18 ^ ft18;
                    ft19 = ft19 ^ ft19;
                    ft20 = ft20 ^ ft20;
                    ft21 = ft21 ^ ft21;
                    ft22 = ft22 ^ ft22;
                    ft23 = ft23 ^ ft23;
                    ft24 = ft24 ^ ft24;
                    ft25 = ft25 ^ ft25;
                    ft26 = ft26 ^ ft26;

                    for(p = 0; p < v_elems; p+=VEC_ELEMS){
                        //LOAD DATA
                        v_SNPA_0 = _mm256_loadu_si256 ((__m256i *) &SNPA_0[p]);
                        v_SNPA_1 = _mm256_loadu_si256 ((__m256i *) &SNPA_1[p]); 

                        v_SNPB_0 = _mm256_loadu_si256 ((__m256i *) &SNPB_0[p]);
                        v_SNPB_1 = _mm256_loadu_si256 ((__m256i *) &SNPB_1[p]);

                        v_SNPC_0 = _mm256_loadu_si256 ((__m256i *) &SNPC_0[p]);
                        v_SNPC_1 = _mm256_loadu_si256 ((__m256i *) &SNPC_1[p]);

                        //OR
                        v_di2 = _mm256_or_si256 (v_SNPA_0, v_SNPA_1);
                        v_dj2 = _mm256_or_si256 (v_SNPB_0, v_SNPB_1);
                        v_dk2 = _mm256_or_si256 (v_SNPC_0, v_SNPC_1);

                        //NOT
                        v_di2 = _mm256_xor_si256 (v_di2, v_ones);
                        v_dj2 = _mm256_xor_si256 (v_dj2, v_ones);
                        v_dk2 = _mm256_xor_si256 (v_dk2, v_ones);

                        v_t000 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                        v_t001 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_SNPC_1)); 
                        v_t002 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_0, v_dk2));
                        v_t010 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                        v_t011 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                        v_t012 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_SNPB_1, v_dk2));
                        v_t020 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_0));
                        v_t021 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                        v_t022 = _mm256_and_si256(v_SNPA_0, _mm256_and_si256(v_dj2, v_dk2));

                        v_t100 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                        v_t101 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                        v_t102 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_0, v_dk2));
                        v_t110 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                        v_t111 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                        v_t112 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_SNPB_1, v_dk2));
                        v_t120 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_0));
                        v_t121 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                        v_t122 = _mm256_and_si256(v_SNPA_1, _mm256_and_si256(v_dj2, v_dk2));

                        v_t200 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_0));
                        v_t201 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_SNPC_1));  
                        v_t202 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_0, v_dk2));
                        v_t210 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_0));
                        v_t211 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_SNPC_1)); 
                        v_t212 = _mm256_and_si256(v_di2, _mm256_and_si256(v_SNPB_1, v_dk2));
                        v_t220 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_0));
                        v_t221 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_SNPC_1)); 
                        v_t222 = _mm256_and_si256(v_di2, _mm256_and_si256(v_dj2, v_dk2));

                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 0));
                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 1));
                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 2));
                        ft0 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t000, 3));

                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 0));
                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 1));
                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 2));
                        ft1 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t001, 3));

                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 0));
                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 1));
                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 2));
                        ft2 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t002, 3));

                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 0));
                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 1));
                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 2));
                        ft3 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t010, 3));

                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 0));
                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 1));
                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 2));
                        ft4 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t011, 3));

                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 0));
                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 1));
                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 2));
                        ft5 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t012, 3));

                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 0));
                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 1));
                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 2));
                        ft6 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t020, 3));

                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 0));
                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 1));
                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 2));
                        ft7 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t021, 3));

                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 0));
                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 1));
                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 2));
                        ft8 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t022, 3));

                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 0));
                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 1));
                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 2));
                        ft9 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t100, 3));

                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 0));
                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 1));
                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 2));
                        ft10 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t101, 3));

                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 0));
                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 1));
                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 2));
                        ft11 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t102, 3));

                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 0));
                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 1));
                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 2));
                        ft12 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t110, 3));

                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 0));
                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 1));
                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 2));
                        ft13 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t111, 3));

                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 0));
                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 1));
                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 2));
                        ft14 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t112, 3));

                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 0));
                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 1));
                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 2));
                        ft15 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t120, 3));

                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 0));
                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 1));
                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 2));
                        ft16 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t121, 3));

                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 0));
                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 1));
                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 2));
                        ft17 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t122, 3));

                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 0));
                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 1));
                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 2));
                        ft18 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t200, 3));

                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 0));
                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 1));
                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 2));
                        ft19 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t201, 3));

                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 0));
                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 1));
                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 2));
                        ft20 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t202, 3));

                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 0));
                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 1));
                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 2));
                        ft21 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t210, 3));

                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 0));
                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 1));
                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 2));
                        ft22 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t211, 3));

                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 0));
                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 1));
                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 2));
                        ft23 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t212, 3));

                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 0));
                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 1));
                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 2));
                        ft24 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t220, 3));

                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 0));
                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 1));
                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 2));
                        ft25 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t221, 3));

                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 0));
                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 1));
                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 2));
                        ft26 += _mm_popcnt_u64 (_mm256_extract_epi64 (v_t222, 3));
                    }
                    //Remaining non-vectorized elements
                    for(p = v_elems; p < PP_ones - pp - 1; p++){
                        di2 = ~(SNPA_0[p] | SNPA_1[p]);
                        dj2 = ~(SNPB_0[p] | SNPB_1[p]);
                        dk2 = ~(SNPC_0[p] | SNPC_1[p]);
            
                        t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                        t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                        t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                        t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                        t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                        t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                        t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                        t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                        t022 = SNPA_0[p] & dj2 & dk2;

                        t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                        t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                        t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                        t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                        t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                        t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                        t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                        t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                        t122 = SNPA_1[p] & dj2 & dk2;

                        t200 = di2 & SNPB_0[p] & SNPC_0[p];
                        t201 = di2 & SNPB_0[p] & SNPC_1[p];
                        t202 = di2 & SNPB_0[p] & dk2;
                        t210 = di2 & SNPB_1[p] & SNPC_0[p];
                        t211 = di2 & SNPB_1[p] & SNPC_1[p];
                        t212 = di2 & SNPB_1[p] & dk2;
                        t220 = di2 & dj2 & SNPC_0[p];
                        t221 = di2 & dj2 & SNPC_1[p];
                        t222 = di2 & dj2 & dk2;

                        ft0 += _mm_popcnt_u32(t000);
                        ft1 += _mm_popcnt_u32(t001);
                        ft2 += _mm_popcnt_u32(t002);
                        ft3 += _mm_popcnt_u32(t010);
                        ft4 += _mm_popcnt_u32(t011);
                        ft5 += _mm_popcnt_u32(t012);
                        ft6 += _mm_popcnt_u32(t020);
                        ft7 += _mm_popcnt_u32(t021);
                        ft8 += _mm_popcnt_u32(t022);
                        ft9 += _mm_popcnt_u32(t100);
                        ft10 += _mm_popcnt_u32(t101);
                        ft11 += _mm_popcnt_u32(t102);
                        ft12 += _mm_popcnt_u32(t110);
                        ft13 += _mm_popcnt_u32(t111);
                        ft14 += _mm_popcnt_u32(t112);
                        ft15 += _mm_popcnt_u32(t120);
                        ft16 += _mm_popcnt_u32(t121);
                        ft17 += _mm_popcnt_u32(t122);
                        ft18 += _mm_popcnt_u32(t200);
                        ft19 += _mm_popcnt_u32(t201);
                        ft20 += _mm_popcnt_u32(t202);
                        ft21 += _mm_popcnt_u32(t210);
                        ft22 += _mm_popcnt_u32(t211);
                        ft23 += _mm_popcnt_u32(t212);
                        ft24 += _mm_popcnt_u32(t220);
                        ft25 += _mm_popcnt_u32(t221);
                        ft26 += _mm_popcnt_u32(t222);
                    }

                    //Do Remaining Elements
                    di2 = ~(SNPA_0[p] | SNPA_1[p]) & mask_ones;
                    dj2 = ~(SNPB_0[p] | SNPB_1[p]) & mask_ones;
                    dk2 = ~(SNPC_0[p] | SNPC_1[p]) & mask_ones;
        
                    t000 = SNPA_0[p] & SNPB_0[p] & SNPC_0[p]; 
                    t001 = SNPA_0[p] & SNPB_0[p] & SNPC_1[p];
                    t002 = SNPA_0[p] & SNPB_0[p] & dk2;
                    t010 = SNPA_0[p] & SNPB_1[p] & SNPC_0[p];
                    t011 = SNPA_0[p] & SNPB_1[p] & SNPC_1[p];
                    t012 = SNPA_0[p] & SNPB_1[p] & dk2;
                    t020 = SNPA_0[p] & dj2 & SNPC_0[p];
                    t021 = SNPA_0[p] & dj2 & SNPC_1[p];
                    t022 = SNPA_0[p] & dj2 & dk2;

                    t100 = SNPA_1[p] & SNPB_0[p] & SNPC_0[p];
                    t101 = SNPA_1[p] & SNPB_0[p] & SNPC_1[p];
                    t102 = SNPA_1[p] & SNPB_0[p] & dk2;
                    t110 = SNPA_1[p] & SNPB_1[p] & SNPC_0[p];
                    t111 = SNPA_1[p] & SNPB_1[p] & SNPC_1[p];
                    t112 = SNPA_1[p] & SNPB_1[p] & dk2;
                    t120 = SNPA_1[p] & dj2 & SNPC_0[p];
                    t121 = SNPA_1[p] & dj2 & SNPC_1[p];
                    t122 = SNPA_1[p] & dj2 & dk2;

                    t200 = di2 & SNPB_0[p] & SNPC_0[p];
                    t201 = di2 & SNPB_0[p] & SNPC_1[p];
                    t202 = di2 & SNPB_0[p] & dk2;
                    t210 = di2 & SNPB_1[p] & SNPC_0[p];
                    t211 = di2 & SNPB_1[p] & SNPC_1[p];
                    t212 = di2 & SNPB_1[p] & dk2;
                    t220 = di2 & dj2 & SNPC_0[p];
                    t221 = di2 & dj2 & SNPC_1[p];
                    t222 = di2 & dj2 & dk2;

                    ft0 += _mm_popcnt_u32(t000);
                    ft1 += _mm_popcnt_u32(t001);
                    ft2 += _mm_popcnt_u32(t002);
                    ft3 += _mm_popcnt_u32(t010);
                    ft4 += _mm_popcnt_u32(t011);
                    ft5 += _mm_popcnt_u32(t012);
                    ft6 += _mm_popcnt_u32(t020);
                    ft7 += _mm_popcnt_u32(t021);
                    ft8 += _mm_popcnt_u32(t022);
                    ft9 += _mm_popcnt_u32(t100);
                    ft10 += _mm_popcnt_u32(t101);
                    ft11 += _mm_popcnt_u32(t102);
                    ft12 += _mm_popcnt_u32(t110);
                    ft13 += _mm_popcnt_u32(t111);
                    ft14 += _mm_popcnt_u32(t112);
                    ft15 += _mm_popcnt_u32(t120);
                    ft16 += _mm_popcnt_u32(t121);
                    ft17 += _mm_popcnt_u32(t122);
                    ft18 += _mm_popcnt_u32(t200);
                    ft19 += _mm_popcnt_u32(t201);
                    ft20 += _mm_popcnt_u32(t202);
                    ft21 += _mm_popcnt_u32(t210);
                    ft22 += _mm_popcnt_u32(t211);
                    ft23 += _mm_popcnt_u32(t212);
                    ft24 += _mm_popcnt_u32(t220);
                    ft25 += _mm_popcnt_u32(t221);
                    ft26 += _mm_popcnt_u32(t222);

                    freq_table_I[xft + 0] += ft0;
                    freq_table_I[xft + 1] += ft1;
                    freq_table_I[xft + 2] += ft2;
                    freq_table_I[xft + 3] += ft3;
                    freq_table_I[xft + 4] += ft4;
                    freq_table_I[xft + 5] += ft5;
                    freq_table_I[xft + 6] += ft6;
                    freq_table_I[xft + 7] += ft7;
                    freq_table_I[xft + 8] += ft8;
                    freq_table_I[xft + 9] += ft9;
                    freq_table_I[xft + 10] += ft10;
                    freq_table_I[xft + 11] += ft11;
                    freq_table_I[xft + 12] += ft12;
                    freq_table_I[xft + 13] += ft13;
                    freq_table_I[xft + 14] += ft14;
                    freq_table_I[xft + 15] += ft15;
                    freq_table_I[xft + 16] += ft16;
                    freq_table_I[xft + 17] += ft17;
                    freq_table_I[xft + 18] += ft18;
                    freq_table_I[xft + 19] += ft19;
                    freq_table_I[xft + 20] += ft20;
                    freq_table_I[xft + 21] += ft21;
                    freq_table_I[xft + 22] += ft22;
                    freq_table_I[xft + 23] += ft23;
                    freq_table_I[xft + 24] += ft24;
                    freq_table_I[xft + 25] += ft25;
                    freq_table_I[xft + 26] += ft26;

                    n_comb++;
                }
            }
        }

        int base = 0;
        for(mi = 0; mi < block_i-2; mi++){
            for(mj = mi+1; mj < block_i-1; mj++){
                for(mk = mj+1; mk < block_i; mk++){
                    m = base + (mk - (mj+1));
                    float score = 0.0;
                    for(n = 0; n < num_combs; n++){
                        score += addlog(freq_table_I[m * num_combs + n] + freq_table_I[(comb_ii + m) * num_combs + n] + 1) - addlog(freq_table_I[m * num_combs + n]) - addlog(freq_table_I[(comb_ii + m) * num_combs + n]);
                    }
                    score = fabs(score);
                    
                    // compare score
                    if(score < best_score_local){
                        best_score_local = score;
                        best_snp_global[tid][0] = ii + mi;
                        best_snp_global[tid][1] = ii + mj;
                        best_snp_global[tid][2] = ii + mk;
                    }
                   // printf("%d, %d, %d - Score: %f\n", ii + mi, ii + mj, ii + mk, score);
                }
                base += (block_i - (mj+1));
            }
            
        }

        #if defined (PAPI_POW)
            PAPI_stop(EventSet_pow[tid], (long long*) energy_e_local);
            int papi_iter;
            if(tid == 0)
            {
                for(papi_iter = 0; papi_iter < POW_COUNTERS; papi_iter++){
                    energy[papi_iter] += energy_e_local[papi_iter];
                }
            }
        #endif
    }

    #pragma omp critical
    {
        if(best_score > best_score_local){
            best_score = best_score_local;
            best_snp[0] = best_snp_global[tid][0];
            best_snp[1] = best_snp_global[tid][1];
            best_snp[2] = best_snp_global[tid][2];
        }
        //printf("tid: %d - %d Combinations\n", tid, counter);
    }

    #if defined (ADVISOR)
        __itt_pause();
    #endif
            

    #if defined (SDE)
        __SSC_MARK(0x222);
    #endif
    
    cyc_e_local = read_tsc_end();

    #if defined (PAPI)
        PAPI_read(EventSet_pow[tid], (long long*) energy_e_local);
    #endif

    #if defined (LIKWID)
        LIKWID_MARKER_STOP("epistasis");
    #endif
    
    serialize();

    cyc_s = cyc_s_local;
    cyc_e = cyc_e_local;

    #if defined (PAPI)
        energy_e[0] = energy_e_local[0];
        energy_s[0] = energy_s_local[0];

        energy_e[1] = energy_e_local[1];
        energy_s[1] = energy_s_local[1];

        energy_e[2] = energy_e_local[2];
        energy_s[2] = energy_s_local[2];

        energy_e[3] = energy_e_local[3];
        energy_s[3] = energy_s_local[3];

        PAPI_stop(EventSet_pow[tid], (long long*) energy_e_local);

    #endif

    _mm_free(freq_table_IJK);
    _mm_free(freq_table_IJ);
    _mm_free(freq_table_I);

    }
    
    printf("Time bin_no_phen_nor_block avx omp: %f\n", (double) (cyc_e - cyc_s)/FREQ);
    
    //printf("Time bin_no_phen_nor_block_avx_omp: %f\n", (double) (cyc_e - cyc_s)/FREQ);

    #if defined (PAPI)
        double time = (double) (cyc_e - cyc_s)/FREQ;

        printf("Energy bin_no_phen_nor_block_avx_omp: PP0: %f, PKG: %f, DRAM: %f, PSYS: %f\n", (double) (energy_e[0] - energy_s[0])*1e-9, (double) (energy_e[1] - energy_s[1])*1e-9, (double) (energy_e[2] - energy_s[2])*1e-9, (double) (energy_e[3] - energy_s[3])*1e-9);
        printf("Power bin_no_phen_nor_block_avx_omp: PP0: %f, PKG: %f, DRAM: %f, PSYS: %f\n", (double) (energy_e[0] - energy_s[0])*1e-9/time, (energy_e[1] - energy_s[1])*1e-9/time, (energy_e[2] - energy_s[2])*1e-9/time, (energy_e[3] - energy_s[3])*1e-9/time);
    #endif

    #if defined (PAPI_POW)
        double time = (double) (cyc_e - cyc_s)/FREQ;

        printf("Energy bin_no_phen_nor_block_avx_omp: PP0: %f, PKG: %f, DRAM: %f, PSYS: %f\n", (double) (energy[0])*1e-9, (double) (energy[1])*1e-9, (double) (energy[2])*1e-9, (double) (energy[3])*1e-9);
        printf("Power bin_no_phen_nor_block_avx_omp: PP0: %f, PKG: %f, DRAM: %f, PSYS: %f\n", (double) (energy[0])*1e-9/time, (energy[1])*1e-9/time, (energy[2])*1e-9/time, (energy[3])*1e-9/time);
    #endif

    printf("bin_no_phen_nor_block avx omp Best: %d, %d, %d - Score: %f\n", best_snp[0], best_snp[1], best_snp[2], best_score);

    /* for( j = 0; j < num_combs; j++ ) 
		printf( " %d", ft[j] );
    printf("\n");
    for( j = 0; j < num_combs; j++ ) 
		printf( " %d", ft[num_combs + j] );
    printf("\n"); */   

    
}

int main(int argc, char **argv){

    #if defined (LIKWID)
        LIKWID_MARKER_INIT;

        #pragma omp parallel
        {
            LIKWID_MARKER_THREADINIT;
        }

        LIKWID_MARKER_REGISTER("epistasis");
    #endif

    #if defined (PAPI) || defined (PAPI_POW)
        char * Events_pow[] = {"rapl:::PP0_ENERGY:PACKAGE0", "rapl:::PACKAGE_ENERGY:PACKAGE0", "rapl:::DRAM_ENERGY:PACKAGE0", "rapl:::PSYS_ENERGY:PACKAGE0"};

        if ( PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT){
            fprintf(stderr,"PAPI library init error!\n");
            exit(1);
        }	

        // init PAPI thread
        if(PAPI_thread_init( (long unsigned int(*)(void)) omp_get_thread_num ) != PAPI_OK) {
            printf("PAPI thread init error! Line %d\n", __LINE__);
            exit(1);
        }	

        #pragma omp parallel
        {
            int i;
            int retval;
	        int native = 0x0;

            int tid = omp_get_thread_num();

            EventSet_pow[tid] = PAPI_NULL;
			
			if(PAPI_create_eventset(&(EventSet_pow[tid])) != PAPI_OK){
				printf("PAPI eventset create error!\n");
				exit(1);
			}

			for(i = 0; i < POW_COUNTERS; i++){
				if (PAPI_event_name_to_code(Events_pow[i],&native) != PAPI_OK){
					printf("Events_pow to code error: %s!\n", Events_pow[i]);
					exit(1);
				}

				if (PAPI_add_event(EventSet_pow[tid],native) != PAPI_OK){
					printf("Add event_power error: %s!\n", Events_pow[i]);
					exit(1);
				}
			}

            if(PAPI_start(EventSet_pow[tid]) != PAPI_OK){
                printf("PAPI eventset power start error\n");
                exit(1);
            }
        }
    #endif

    int long long num_snp, num_pac;
    int dim_epi;
    int block_snp, block_pac;
    int i, addlogsize;

    uint8_t *SNP_Data; 
    uint8_t *Ph_Data;
    uint32_t *bin_data_ones, *bin_data_zeros;
    int long long phen_ones;

    int long long num_snp_r, num_phen_zeros_r, num_phen_ones_r;

    dim_epi = 3; //atoi(argv[1]);
    block_pac = 96; //atoi(argv[4]);
    block_snp = 5; //atoi(argv[5]);
    int comb = (int)pow(3.0, dim_epi);

    if(argc == 3)
    {
        num_pac = atol(argv[1]);
        num_snp = atol(argv[2]);
        generate_data(num_pac, num_snp, &SNP_Data, &Ph_Data);
    }
    else if(argc == 2)
    {
        input_data(argv[1], &SNP_Data, &Ph_Data, &num_pac, &num_snp);
    }

    // create addlog table (up to TABLE_MAX_SIZE positions at max)
	addlogsize = TABLE_MAX_SIZE;
	addlogtable = new float[addlogsize];
	for(i = 0; i < addlogsize; i++)
		addlogtable[i] = my_factorial(i);

    SNP_Data = transpose_data(num_pac, num_snp, SNP_Data);
    transposed_to_binary(SNP_Data, Ph_Data, &bin_data_zeros, &bin_data_ones, &phen_ones, num_snp, num_pac);

    _mm_free(SNP_Data);
    _mm_free(Ph_Data);

    bin_data_zeros = transform_data_2_block(num_pac - phen_ones, num_snp, bin_data_zeros, block_pac, block_snp);
    bin_data_ones = transform_data_2_block(phen_ones, num_snp, bin_data_ones, block_pac, block_snp);

    process_epi_bin_no_phen_nor_block(bin_data_zeros, bin_data_ones, phen_ones, dim_epi, num_snp, num_pac, comb, block_snp, block_pac);

    _mm_free(bin_data_zeros);
    _mm_free(bin_data_ones);

    delete addlogtable;

    #if defined (LIKWID)
        LIKWID_MARKER_CLOSE;
    #endif

    #if defined (PAPI) || defined (PAPI_POW)
        #pragma omp parallel
        {
        int tid = omp_get_thread_num();

        PAPI_cleanup_eventset(EventSet_pow[tid]);
        PAPI_destroy_eventset(&EventSet_pow[tid]);
        }
    #endif

    return 0;
}
