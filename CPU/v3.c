#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <immintrin.h>
#include <float.h>
#include <climits>
#include <omp.h>

#ifndef __SSC_MARK
#define __SSC_MARK(tag)                                                        \
        __asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 "         \
                             ::"i"(tag) : "%ebx")
#endif

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

#define NUM_THREADS 72

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
    *A = (uint8_t *) _mm_malloc(N*M*sizeof(uint8_t), 64);
    *B = (uint8_t *) _mm_malloc(N*sizeof(uint8_t), 64);

    //Generate SNPs
    for (i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            //Generate Between 0 and 2
            (*A)[i*M + j] = rand() % 3;

        }
    }
    
        
    //Generate Phenotype
    for(i = 0; i < N; i++){
        //Generate Between 0 and 1
        (*B)[i] = rand() % 2;
    }
}

uint8_t * transpose_data(int long long N, int long long M, uint8_t * A){
    int i, j;
    
    uint8_t *A_trans = (uint8_t *) _mm_malloc(M*N*sizeof(uint8_t), 64);
    
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
    *data_zeros = (uint32_t*) _mm_malloc(num_snp*PP_zeros*2*sizeof(uint32_t), 64);
    *data_ones = (uint32_t*) _mm_malloc(num_snp*PP_ones*2*sizeof(uint32_t), 64);
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

    int long long PP = ceil((1.0*(N))/32.0);
    
    int long long M_r = roundUp(M, block_snp); 
    int long long PP_r = roundUp(PP, block_pac);
    
    uint32_t *A_2_block = (uint32_t *) _mm_malloc(M_r*PP_r*2*sizeof(uint32_t), 64);
   
    //PLACE SNPs in BLOCK MEMORY FORMAT
    for(i = 0; i < M; i+= block_snp){
        for(j = 0; j < PP; j+= block_pac){
            for(jj = 0; jj < block_pac && jj < PP - j; jj++){
                for(ii = 0; ii < block_snp && ii < M - i; ii++){
                    A_2_block[i*PP_r*2 + j*block_snp*2 + ii*block_pac*2 + jj*2 + 0] = A[(i+ii)*PP*2 + j*2 + jj*2 + 0];
                    A_2_block[i*PP_r*2 + j*block_snp*2 + ii*block_pac*2 + jj*2 + 1] = A[(i+ii)*PP*2 + j*2 + jj*2 + 1];
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

    #pragma omp parallel reduction(min:cyc_s) reduction(max:cyc_e)
    {

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

    uint32_t *SNPA, *SNPB, *SNPC;
    uint32_t dj2, di2, dk2;
    uint32_t t000, t001, t002, t010, t011, t012, t020, t021, t022, t100, t101, t102, t110, t111, t112, t120, t121, t122, t200, t201, t202, t210, t211, t212, t220, t221, t222;

    //Creating MASK for non-existing pacients
    uint32_t mask_ones = 0xffffffff;
    uint32_t mask_zeros = 0xffffffff;

    int long long rem_pac = PP_ones*32 - phen_ones;

    mask_ones >>= (uint32_t) rem_pac;

    rem_pac = PP_zeros*32 - (num_pac - phen_ones);

    mask_zeros >>= (uint32_t) rem_pac; 

    //Generate Frequency Table
    //uint32_t * freq_table = (uint32_t *) _mm_malloc(2*comb*sizeof(uint32_t), 64);
    //memset(freq_table, 0, 2*comb*sizeof(uint32_t));

    uint32_t* freq_table_I = (uint32_t*) _mm_malloc(2*num_fts_I*num_combs * sizeof(uint32_t), 64);
    memset(freq_table_I, 0, 2*num_fts_I*num_combs*sizeof(uint32_t));
    
    uint32_t* freq_table_IJ = (uint32_t*) _mm_malloc(2 * num_fts_IJ * num_combs * sizeof(uint32_t), 64);
    memset(freq_table_IJ, 0, 2*num_fts_IJ*num_combs*sizeof(uint32_t));

    uint32_t* freq_table_IJK = (uint32_t*) _mm_malloc(2 * num_fts_IJK * num_combs * sizeof(uint32_t), 64);
    memset(freq_table_IJK, 0, 2*num_fts_IJK*num_combs*sizeof(uint32_t));

    float best_score_local = FLT_MAX;
    int long long cyc_s_local, cyc_e_local;

    clear_cache_deep();

    serialize();

    cyc_s_local = read_tsc_start();

    #if defined (LIKWID)
        LIKWID_MARKER_START("epistasis");
    #endif

    #if defined (SDE)
        __SSC_MARK(0x111);
    #endif

    #if defined (ADVISOR)
        __itt_resume();
    #endif

    // fill frequency table
    #pragma omp for schedule(dynamic)
    for (ii = 0; ii < num_snp; ii+= block_snp){
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

                //RESET FREQUENCY TABLES
                memset(freq_table_IJK, 0, 2*num_fts_IJK*num_combs*sizeof(uint32_t));

                //BETWEEN I, J and K

                //Phenotype equal 0
                for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
                    xi = xii_p0 + 2*pp*block_snp;
                    xj = xjj_p0 + 2*pp*block_snp;
                    xk = xkk_p0 + 2*pp*block_snp;

                    for(i = 0; i < block_i; i++){
                        SNPA = &data_zeros[xi + i*block_pac*2];
                        xft00 = i*block_j*block_k;
                        for(j = 0; j < block_j; j++){
                            xft0 = xft00 + j*block_k;
                            SNPB = &data_zeros[xj + j*block_pac*2];
                            for(k = 0; k < block_k; k++){
                                xft = (xft0 + k)*num_combs;
                                //printf("xft00:%d\n", xft);
                                SNPC = &data_zeros[xk + k*block_pac*2];
                                for(p = 0; p < 2*block_pac; p+=2){
                                    di2 = ~(SNPA[p] | SNPA[p + 1]);
                                    dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                    dk2 = ~(SNPC[p] | SNPC[p + 1]);
                        
                                    t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                    t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                    t002 = SNPA[p] & SNPB[p] & dk2;
                                    t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                    t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                    t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                    t020 = SNPA[p] & dj2 & SNPC[p];
                                    t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                    t022 = SNPA[p] & dj2 & dk2;

                                    t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                    t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                    t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                    t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                    t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                    t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                    t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                    t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                    t122 = SNPA[p + 1] & dj2 & dk2;

                                    t200 = di2 & SNPB[p] & SNPC[p];
                                    t201 = di2 & SNPB[p] & SNPC[p + 1];
                                    t202 = di2 & SNPB[p] & dk2;
                                    t210 = di2 & SNPB[p + 1] & SNPC[p];
                                    t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                    t212 = di2 & SNPB[p + 1] & dk2;
                                    t220 = di2 & dj2 & SNPC[p];
                                    t221 = di2 & dj2 & SNPC[p + 1];
                                    t222 = di2 & dj2 & dk2;

                                    freq_table_IJK[xft + 0] += _mm_popcnt_u32(t000);
                                    freq_table_IJK[xft + 1] += _mm_popcnt_u32(t001);
                                    freq_table_IJK[xft + 2] += _mm_popcnt_u32(t002);
                                    freq_table_IJK[xft + 3] += _mm_popcnt_u32(t010);
                                    freq_table_IJK[xft + 4] += _mm_popcnt_u32(t011);
                                    freq_table_IJK[xft + 5] += _mm_popcnt_u32(t012);
                                    freq_table_IJK[xft + 6] += _mm_popcnt_u32(t020);
                                    freq_table_IJK[xft + 7] += _mm_popcnt_u32(t021);
                                    freq_table_IJK[xft + 8] += _mm_popcnt_u32(t022);
                                    freq_table_IJK[xft + 9] += _mm_popcnt_u32(t100);
                                    freq_table_IJK[xft + 10] += _mm_popcnt_u32(t101);
                                    freq_table_IJK[xft + 11] += _mm_popcnt_u32(t102);
                                    freq_table_IJK[xft + 12] += _mm_popcnt_u32(t110);
                                    freq_table_IJK[xft + 13] += _mm_popcnt_u32(t111);
                                    freq_table_IJK[xft + 14] += _mm_popcnt_u32(t112);
                                    freq_table_IJK[xft + 15] += _mm_popcnt_u32(t120);
                                    freq_table_IJK[xft + 16] += _mm_popcnt_u32(t121);
                                    freq_table_IJK[xft + 17] += _mm_popcnt_u32(t122);
                                    freq_table_IJK[xft + 18] += _mm_popcnt_u32(t200);
                                    freq_table_IJK[xft + 19] += _mm_popcnt_u32(t201);
                                    freq_table_IJK[xft + 20] += _mm_popcnt_u32(t202);
                                    freq_table_IJK[xft + 21] += _mm_popcnt_u32(t210);
                                    freq_table_IJK[xft + 22] += _mm_popcnt_u32(t211);
                                    freq_table_IJK[xft + 23] += _mm_popcnt_u32(t212);
                                    freq_table_IJK[xft + 24] += _mm_popcnt_u32(t220);
                                    freq_table_IJK[xft + 25] += _mm_popcnt_u32(t221);
                                    freq_table_IJK[xft + 26] += _mm_popcnt_u32(t222);
                                }
                            }
                        }
                    }
                }
                xi = xii_p0 + 2*pp*block_snp;
                xj = xjj_p0 + 2*pp*block_snp;
                xk = xkk_p0 + 2*pp*block_snp;
                for(i = 0; i < block_i; i++){
                    SNPA = &data_zeros[xi + i*block_pac*2];
                    xft00 = i*block_j*block_k;
                    for(j = 0; j < block_j; j++){
                        xft0 = xft00 + j*block_k;
                        SNPB = &data_zeros[xj + j*block_pac*2];
                        for(k = 0; k < block_k; k++){
                            xft = (xft0 + k)*num_combs;
                            SNPC = &data_zeros[xk + k*block_pac*2];
                            for(p = 0; p < 2*(PP_zeros - pp - 1); p+=2){
                                di2 = ~(SNPA[p] | SNPA[p + 1]);
                                dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                dk2 = ~(SNPC[p] | SNPC[p + 1]);
                    
                                t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                t002 = SNPA[p] & SNPB[p] & dk2;
                                t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                t020 = SNPA[p] & dj2 & SNPC[p];
                                t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                t022 = SNPA[p] & dj2 & dk2;

                                t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                t122 = SNPA[p + 1] & dj2 & dk2;

                                t200 = di2 & SNPB[p] & SNPC[p];
                                t201 = di2 & SNPB[p] & SNPC[p + 1];
                                t202 = di2 & SNPB[p] & dk2;
                                t210 = di2 & SNPB[p + 1] & SNPC[p];
                                t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                t212 = di2 & SNPB[p + 1] & dk2;
                                t220 = di2 & dj2 & SNPC[p];
                                t221 = di2 & dj2 & SNPC[p + 1];
                                t222 = di2 & dj2 & dk2;

                                freq_table_IJK[xft + 0] += _mm_popcnt_u32(t000);
                                freq_table_IJK[xft + 1] += _mm_popcnt_u32(t001);
                                freq_table_IJK[xft + 2] += _mm_popcnt_u32(t002);
                                freq_table_IJK[xft + 3] += _mm_popcnt_u32(t010);
                                freq_table_IJK[xft + 4] += _mm_popcnt_u32(t011);
                                freq_table_IJK[xft + 5] += _mm_popcnt_u32(t012);
                                freq_table_IJK[xft + 6] += _mm_popcnt_u32(t020);
                                freq_table_IJK[xft + 7] += _mm_popcnt_u32(t021);
                                freq_table_IJK[xft + 8] += _mm_popcnt_u32(t022);
                                freq_table_IJK[xft + 9] += _mm_popcnt_u32(t100);
                                freq_table_IJK[xft + 10] += _mm_popcnt_u32(t101);
                                freq_table_IJK[xft + 11] += _mm_popcnt_u32(t102);
                                freq_table_IJK[xft + 12] += _mm_popcnt_u32(t110);
                                freq_table_IJK[xft + 13] += _mm_popcnt_u32(t111);
                                freq_table_IJK[xft + 14] += _mm_popcnt_u32(t112);
                                freq_table_IJK[xft + 15] += _mm_popcnt_u32(t120);
                                freq_table_IJK[xft + 16] += _mm_popcnt_u32(t121);
                                freq_table_IJK[xft + 17] += _mm_popcnt_u32(t122);
                                freq_table_IJK[xft + 18] += _mm_popcnt_u32(t200);
                                freq_table_IJK[xft + 19] += _mm_popcnt_u32(t201);
                                freq_table_IJK[xft + 20] += _mm_popcnt_u32(t202);
                                freq_table_IJK[xft + 21] += _mm_popcnt_u32(t210);
                                freq_table_IJK[xft + 22] += _mm_popcnt_u32(t211);
                                freq_table_IJK[xft + 23] += _mm_popcnt_u32(t212);
                                freq_table_IJK[xft + 24] += _mm_popcnt_u32(t220);
                                freq_table_IJK[xft + 25] += _mm_popcnt_u32(t221);
                                freq_table_IJK[xft + 26] += _mm_popcnt_u32(t222);
                            }
                            //Do Remaining Elements
                            di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_zeros;
                            dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_zeros;
                            dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_zeros;
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_IJK[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_IJK[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_IJK[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_IJK[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_IJK[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_IJK[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_IJK[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_IJK[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_IJK[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_IJK[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_IJK[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_IJK[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_IJK[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_IJK[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_IJK[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_IJK[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_IJK[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_IJK[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_IJK[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_IJK[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_IJK[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_IJK[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_IJK[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_IJK[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_IJK[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_IJK[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_IJK[xft + 26] += _mm_popcnt_u32(t222);
                        }
                    }
                }
                //Phenotype equal 1
                for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
                    xi = xii_p1 + 2*pp*block_snp;
                    xj = xjj_p1 + 2*pp*block_snp;
                    xk = xkk_p1 + 2*pp*block_snp;

                    for(i = 0; i < block_i; i++){
                        SNPA = &data_ones[xi + i*block_pac*2];
                        xft00 = i*block_j*block_k;
                        for(j = 0; j < block_j; j++){
                            xft0 = xft00 + j*block_k;
                            //xft = (comb_ij + xft0 + j)*num_combs;
                            SNPB = &data_ones[xj + j*block_pac*2];
                            for(k = 0; k < block_k; k++){
                                xft = (comb_ijk + xft0 + k)*num_combs;
                                //printf("xft1:%d\n", xft);
                                SNPC = &data_ones[xk + k*block_pac*2];
                                for(p = 0; p < 2*block_pac; p+=2){
                                    di2 = ~(SNPA[p] | SNPA[p + 1]);
                                    dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                    dk2 = ~(SNPC[p] | SNPC[p + 1]);
                        
                                    t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                    t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                    t002 = SNPA[p] & SNPB[p] & dk2;
                                    t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                    t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                    t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                    t020 = SNPA[p] & dj2 & SNPC[p];
                                    t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                    t022 = SNPA[p] & dj2 & dk2;

                                    t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                    t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                    t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                    t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                    t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                    t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                    t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                    t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                    t122 = SNPA[p + 1] & dj2 & dk2;

                                    t200 = di2 & SNPB[p] & SNPC[p];
                                    t201 = di2 & SNPB[p] & SNPC[p + 1];
                                    t202 = di2 & SNPB[p] & dk2;
                                    t210 = di2 & SNPB[p + 1] & SNPC[p];
                                    t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                    t212 = di2 & SNPB[p + 1] & dk2;
                                    t220 = di2 & dj2 & SNPC[p];
                                    t221 = di2 & dj2 & SNPC[p + 1];
                                    t222 = di2 & dj2 & dk2;

                                    freq_table_IJK[xft + 0] += _mm_popcnt_u32(t000);
                                    freq_table_IJK[xft + 1] += _mm_popcnt_u32(t001);
                                    freq_table_IJK[xft + 2] += _mm_popcnt_u32(t002);
                                    freq_table_IJK[xft + 3] += _mm_popcnt_u32(t010);
                                    freq_table_IJK[xft + 4] += _mm_popcnt_u32(t011);
                                    freq_table_IJK[xft + 5] += _mm_popcnt_u32(t012);
                                    freq_table_IJK[xft + 6] += _mm_popcnt_u32(t020);
                                    freq_table_IJK[xft + 7] += _mm_popcnt_u32(t021);
                                    freq_table_IJK[xft + 8] += _mm_popcnt_u32(t022);
                                    freq_table_IJK[xft + 9] += _mm_popcnt_u32(t100);
                                    freq_table_IJK[xft + 10] += _mm_popcnt_u32(t101);
                                    freq_table_IJK[xft + 11] += _mm_popcnt_u32(t102);
                                    freq_table_IJK[xft + 12] += _mm_popcnt_u32(t110);
                                    freq_table_IJK[xft + 13] += _mm_popcnt_u32(t111);
                                    freq_table_IJK[xft + 14] += _mm_popcnt_u32(t112);
                                    freq_table_IJK[xft + 15] += _mm_popcnt_u32(t120);
                                    freq_table_IJK[xft + 16] += _mm_popcnt_u32(t121);
                                    freq_table_IJK[xft + 17] += _mm_popcnt_u32(t122);
                                    freq_table_IJK[xft + 18] += _mm_popcnt_u32(t200);
                                    freq_table_IJK[xft + 19] += _mm_popcnt_u32(t201);
                                    freq_table_IJK[xft + 20] += _mm_popcnt_u32(t202);
                                    freq_table_IJK[xft + 21] += _mm_popcnt_u32(t210);
                                    freq_table_IJK[xft + 22] += _mm_popcnt_u32(t211);
                                    freq_table_IJK[xft + 23] += _mm_popcnt_u32(t212);
                                    freq_table_IJK[xft + 24] += _mm_popcnt_u32(t220);
                                    freq_table_IJK[xft + 25] += _mm_popcnt_u32(t221);
                                    freq_table_IJK[xft + 26] += _mm_popcnt_u32(t222);
                                }
                            }
                        }
                    }
                }
                xi = xii_p1 + 2*pp*block_snp;
                xj = xjj_p1 + 2*pp*block_snp;
                xk = xkk_p1 + 2*pp*block_snp;

                for(i = 0; i < block_i; i++){
                    SNPA = &data_ones[xi + i*block_pac*2];
                    xft00 = i*block_j*block_k;
                    for(j = 0; j < block_j; j++){
                        xft0 = xft00 + j*block_k;
                        SNPB = &data_ones[xj + j*block_pac*2];
                        for(k = 0; k < block_k; k++){
                            xft = (comb_ijk + xft0 + k)*num_combs;
                            //printf("xft1:%d\n", xft);
                            SNPC = &data_ones[xk + k*block_pac*2];
                            for(p = 0; p < 2*(PP_ones - pp - 1); p+=2){
                                di2 = ~(SNPA[p] | SNPA[p + 1]);
                                dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                dk2 = ~(SNPC[p] | SNPC[p + 1]);
                    
                                t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                t002 = SNPA[p] & SNPB[p] & dk2;
                                t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                t020 = SNPA[p] & dj2 & SNPC[p];
                                t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                t022 = SNPA[p] & dj2 & dk2;

                                t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                t122 = SNPA[p + 1] & dj2 & dk2;

                                t200 = di2 & SNPB[p] & SNPC[p];
                                t201 = di2 & SNPB[p] & SNPC[p + 1];
                                t202 = di2 & SNPB[p] & dk2;
                                t210 = di2 & SNPB[p + 1] & SNPC[p];
                                t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                t212 = di2 & SNPB[p + 1] & dk2;
                                t220 = di2 & dj2 & SNPC[p];
                                t221 = di2 & dj2 & SNPC[p + 1];
                                t222 = di2 & dj2 & dk2;

                                freq_table_IJK[xft + 0] += _mm_popcnt_u32(t000);
                                freq_table_IJK[xft + 1] += _mm_popcnt_u32(t001);
                                freq_table_IJK[xft + 2] += _mm_popcnt_u32(t002);
                                freq_table_IJK[xft + 3] += _mm_popcnt_u32(t010);
                                freq_table_IJK[xft + 4] += _mm_popcnt_u32(t011);
                                freq_table_IJK[xft + 5] += _mm_popcnt_u32(t012);
                                freq_table_IJK[xft + 6] += _mm_popcnt_u32(t020);
                                freq_table_IJK[xft + 7] += _mm_popcnt_u32(t021);
                                freq_table_IJK[xft + 8] += _mm_popcnt_u32(t022);
                                freq_table_IJK[xft + 9] += _mm_popcnt_u32(t100);
                                freq_table_IJK[xft + 10] += _mm_popcnt_u32(t101);
                                freq_table_IJK[xft + 11] += _mm_popcnt_u32(t102);
                                freq_table_IJK[xft + 12] += _mm_popcnt_u32(t110);
                                freq_table_IJK[xft + 13] += _mm_popcnt_u32(t111);
                                freq_table_IJK[xft + 14] += _mm_popcnt_u32(t112);
                                freq_table_IJK[xft + 15] += _mm_popcnt_u32(t120);
                                freq_table_IJK[xft + 16] += _mm_popcnt_u32(t121);
                                freq_table_IJK[xft + 17] += _mm_popcnt_u32(t122);
                                freq_table_IJK[xft + 18] += _mm_popcnt_u32(t200);
                                freq_table_IJK[xft + 19] += _mm_popcnt_u32(t201);
                                freq_table_IJK[xft + 20] += _mm_popcnt_u32(t202);
                                freq_table_IJK[xft + 21] += _mm_popcnt_u32(t210);
                                freq_table_IJK[xft + 22] += _mm_popcnt_u32(t211);
                                freq_table_IJK[xft + 23] += _mm_popcnt_u32(t212);
                                freq_table_IJK[xft + 24] += _mm_popcnt_u32(t220);
                                freq_table_IJK[xft + 25] += _mm_popcnt_u32(t221);
                                freq_table_IJK[xft + 26] += _mm_popcnt_u32(t222);
                            }
                            di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_ones;
                            dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_ones;
                            dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_ones;
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_IJK[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_IJK[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_IJK[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_IJK[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_IJK[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_IJK[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_IJK[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_IJK[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_IJK[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_IJK[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_IJK[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_IJK[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_IJK[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_IJK[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_IJK[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_IJK[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_IJK[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_IJK[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_IJK[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_IJK[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_IJK[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_IJK[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_IJK[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_IJK[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_IJK[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_IJK[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_IJK[xft + 26] += _mm_popcnt_u32(t222);
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
                            // compare score
                            if(score < best_score_local){
                                best_score_local = score;
                                best_snp_global[tid][0] = ii + mi;
                                best_snp_global[tid][1] = jj + mj;
                                best_snp_global[tid][2] = kk + mk;
                            }
                        }
                    }
                }
            } 
            //BETWEEN I AND J

            comb_ij = (block_i*block_j*(block_j-1))/2;

            //RESET FREQUENCY TABLES
            memset(freq_table_IJ, 0, 2 * num_fts_IJ * num_combs * sizeof(uint32_t));

            //Phenotype equal 0
            for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
                xi = xii_p0 + 2*pp*block_snp;
                xj = xjj_p0 + 2*pp*block_snp;
                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i; i++){
                    SNPA = &data_zeros[xi + i*block_pac*2];
                    //xft0 = i*block_j;
                    for(j = 0; j < block_j-1; j++){
                        //xft = (xft0 + j)*num_combs;
                        SNPB = &data_zeros[xj + j*block_pac*2];
                        for(k = j+1; k < block_j; k++){
                            xft = n_comb*num_combs;
                            SNPC = &data_zeros[xj + k*block_pac*2];
                            for(p = 0; p < 2*block_pac; p+=2){
                                di2 = ~(SNPA[p] | SNPA[p + 1]);
                                dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                dk2 = ~(SNPC[p] | SNPC[p + 1]);
                    
                                t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                t002 = SNPA[p] & SNPB[p] & dk2;
                                t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                t020 = SNPA[p] & dj2 & SNPC[p];
                                t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                t022 = SNPA[p] & dj2 & dk2;

                                t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                t122 = SNPA[p + 1] & dj2 & dk2;

                                t200 = di2 & SNPB[p] & SNPC[p];
                                t201 = di2 & SNPB[p] & SNPC[p + 1];
                                t202 = di2 & SNPB[p] & dk2;
                                t210 = di2 & SNPB[p + 1] & SNPC[p];
                                t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                t212 = di2 & SNPB[p + 1] & dk2;
                                t220 = di2 & dj2 & SNPC[p];
                                t221 = di2 & dj2 & SNPC[p + 1];
                                t222 = di2 & dj2 & dk2;

                                freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                                freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                                freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                                freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                                freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                                freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                                freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                                freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                                freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                                freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                                freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                                freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                                freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                                freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                                freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                                freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                                freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                                freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                                freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                                freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                                freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                                freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                                freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                                freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                                freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                                freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                                freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                            }
                            n_comb++;
                        }
                    }
                }
            }
            xi = xii_p0 + 2*pp*block_snp;
            xj = xjj_p0 + 2*pp*block_snp;

            n_comb = 0;
            //BETWEEN I and J
            for(i = 0; i < block_i; i++){
                SNPA = &data_zeros[xi + i*block_pac*2];
                //xft0 = i*block_j;
                for(j = 0; j < block_j-1; j++){
                    //xft = (xft0 + j)*num_combs;
                    SNPB = &data_zeros[xj + j*block_pac*2];
                    for(k = j+1; k < block_j; k++){
                        xft = n_comb*num_combs;
                        SNPC = &data_zeros[xj + k*block_pac*2];
                        for(p = 0; p < 2*(PP_zeros - pp - 1); p+=2){
                            di2 = ~(SNPA[p] | SNPA[p + 1]);
                            dj2 = ~(SNPB[p] | SNPB[p + 1]);
                            dk2 = ~(SNPC[p] | SNPC[p + 1]);
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_zeros;
                        dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_zeros;
                        dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_zeros;
            
                        t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                        t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                        t002 = SNPA[p] & SNPB[p] & dk2;
                        t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                        t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                        t012 = SNPA[p] & SNPB[p + 1] & dk2;
                        t020 = SNPA[p] & dj2 & SNPC[p];
                        t021 = SNPA[p] & dj2 & SNPC[p + 1];
                        t022 = SNPA[p] & dj2 & dk2;

                        t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                        t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                        t102 = SNPA[p + 1] & SNPB[p] & dk2;
                        t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                        t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                        t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                        t120 = SNPA[p + 1] & dj2 & SNPC[p];
                        t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                        t122 = SNPA[p + 1] & dj2 & dk2;

                        t200 = di2 & SNPB[p] & SNPC[p];
                        t201 = di2 & SNPB[p] & SNPC[p + 1];
                        t202 = di2 & SNPB[p] & dk2;
                        t210 = di2 & SNPB[p + 1] & SNPC[p];
                        t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                        t212 = di2 & SNPB[p + 1] & dk2;
                        t220 = di2 & dj2 & SNPC[p];
                        t221 = di2 & dj2 & SNPC[p + 1];
                        t222 = di2 & dj2 & dk2;

                        freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                        freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                        freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                        freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                        freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                        freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                        freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                        freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                        freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                        freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                        freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                        freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                        freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                        freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                        freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                        freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                        freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                        freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                        freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                        freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                        freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                        freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                        freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                        freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                        freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                        freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                        freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);

                        n_comb ++;
                    }
                }
            }
            //Phenotype equal 1
            for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
                xi = xii_p1 + 2*pp*block_snp;
                xj = xjj_p1 + 2*pp*block_snp;

                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i; i++){
                    SNPA = &data_ones[xi + i*block_pac*2];
                    //xft0 = i*block_j;
                    for(j = 0; j < block_j-1; j++){
                        //xft = (comb_ij + xft0 + j)*num_combs;
                        SNPB = &data_ones[xj + j*block_pac*2];
                        for(k = j+1; k < block_j; k++){
                            xft = (comb_ij + n_comb)*num_combs;
                            SNPC = &data_ones[xj + k*block_pac*2];
                            for(p = 0; p < 2*block_pac; p+=2){
                                di2 = ~(SNPA[p] | SNPA[p + 1]);
                                dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                dk2 = ~(SNPC[p] | SNPC[p + 1]);
                    
                                t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                t002 = SNPA[p] & SNPB[p] & dk2;
                                t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                t020 = SNPA[p] & dj2 & SNPC[p];
                                t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                t022 = SNPA[p] & dj2 & dk2;

                                t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                t122 = SNPA[p + 1] & dj2 & dk2;

                                t200 = di2 & SNPB[p] & SNPC[p];
                                t201 = di2 & SNPB[p] & SNPC[p + 1];
                                t202 = di2 & SNPB[p] & dk2;
                                t210 = di2 & SNPB[p + 1] & SNPC[p];
                                t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                t212 = di2 & SNPB[p + 1] & dk2;
                                t220 = di2 & dj2 & SNPC[p];
                                t221 = di2 & dj2 & SNPC[p + 1];
                                t222 = di2 & dj2 & dk2;

                                freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                                freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                                freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                                freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                                freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                                freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                                freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                                freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                                freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                                freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                                freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                                freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                                freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                                freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                                freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                                freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                                freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                                freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                                freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                                freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                                freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                                freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                                freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                                freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                                freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                                freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                                freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                            }
                            n_comb ++;
                        }
                    }
                }
            }
            n_comb = 0;

            xi = xii_p1 + 2*pp*block_snp;
            xj = xjj_p1 + 2*pp*block_snp;
            //BETWEEN I and J
            for(i = 0; i < block_i; i++){
                SNPA = &data_ones[xi + i*block_pac*2];
                //xft0 = i*block_j;
                for(j = 0; j < block_j-1; j++){
                    //xft = (comb_ij + xft0 + j)*num_combs;
                    SNPB = &data_ones[xj + j*block_pac*2];
                    for(k = j+1; k < block_j; k++){
                        xft = (comb_ij + n_comb)*num_combs;
                        SNPC = &data_ones[xj + k*block_pac*2];
                        for(p = 0; p < 2*(PP_ones - pp - 1); p+=2){
                            di2 = ~(SNPA[p] | SNPA[p + 1]);
                            dj2 = ~(SNPB[p] | SNPB[p + 1]);
                            dk2 = ~(SNPC[p] | SNPC[p + 1]);
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_ones;
                        dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_ones;
                        dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_ones;
            
                        t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                        t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                        t002 = SNPA[p] & SNPB[p] & dk2;
                        t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                        t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                        t012 = SNPA[p] & SNPB[p + 1] & dk2;
                        t020 = SNPA[p] & dj2 & SNPC[p];
                        t021 = SNPA[p] & dj2 & SNPC[p + 1];
                        t022 = SNPA[p] & dj2 & dk2;

                        t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                        t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                        t102 = SNPA[p + 1] & SNPB[p] & dk2;
                        t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                        t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                        t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                        t120 = SNPA[p + 1] & dj2 & SNPC[p];
                        t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                        t122 = SNPA[p + 1] & dj2 & dk2;

                        t200 = di2 & SNPB[p] & SNPC[p];
                        t201 = di2 & SNPB[p] & SNPC[p + 1];
                        t202 = di2 & SNPB[p] & dk2;
                        t210 = di2 & SNPB[p + 1] & SNPC[p];
                        t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                        t212 = di2 & SNPB[p + 1] & dk2;
                        t220 = di2 & dj2 & SNPC[p];
                        t221 = di2 & dj2 & SNPC[p + 1];
                        t222 = di2 & dj2 & dk2;

                        freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                        freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                        freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                        freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                        freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                        freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                        freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                        freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                        freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                        freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                        freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                        freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                        freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                        freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                        freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                        freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                        freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                        freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                        freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                        freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                        freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                        freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                        freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                        freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                        freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                        freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                        freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);

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
                        // compare score
                        if(score < best_score_local){
                            best_score_local = score;
                            best_snp_global[tid][0] = ii + mi;
                            best_snp_global[tid][1] = jj + mj;
                            best_snp_global[tid][2] = jj + mk;
                        }
                    }
                    base += (block_j - (mj+1));
                }
            }

            //RESET FREQUENCY TABLES
            memset(freq_table_IJ, 0, 2 * num_fts_IJ * num_combs * sizeof(uint32_t));

            comb_ij = (block_i*block_j*(block_i-1))/2;

            //Phenotype equal 0
            for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
                xi = xii_p0 + 2*pp*block_snp;
                xj = xjj_p0 + 2*pp*block_snp;
                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i-1; i++){
                    SNPA = &data_zeros[xi + i*block_pac*2];
                    //xft0 = i*block_j; 
                    for(j = i+1; j < block_i; j++){
                        //xft = (xft0 + j)*num_combs;
                        SNPB = &data_zeros[xi + j*block_pac*2];
                        for(k = 0; k < block_j; k++){
                            xft = n_comb*num_combs;
                            SNPC = &data_zeros[xj + k*block_pac*2];
                            for(p = 0; p < 2*block_pac; p+=2){
                                di2 = ~(SNPA[p] | SNPA[p + 1]);
                                dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                dk2 = ~(SNPC[p] | SNPC[p + 1]);
                    
                                t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                t002 = SNPA[p] & SNPB[p] & dk2;
                                t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                t020 = SNPA[p] & dj2 & SNPC[p];
                                t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                t022 = SNPA[p] & dj2 & dk2;

                                t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                t122 = SNPA[p + 1] & dj2 & dk2;

                                t200 = di2 & SNPB[p] & SNPC[p];
                                t201 = di2 & SNPB[p] & SNPC[p + 1];
                                t202 = di2 & SNPB[p] & dk2;
                                t210 = di2 & SNPB[p + 1] & SNPC[p];
                                t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                t212 = di2 & SNPB[p + 1] & dk2;
                                t220 = di2 & dj2 & SNPC[p];
                                t221 = di2 & dj2 & SNPC[p + 1];
                                t222 = di2 & dj2 & dk2;

                                freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                                freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                                freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                                freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                                freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                                freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                                freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                                freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                                freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                                freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                                freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                                freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                                freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                                freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                                freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                                freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                                freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                                freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                                freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                                freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                                freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                                freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                                freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                                freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                                freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                                freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                                freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                            }
                            n_comb++;
                        }
                    }
                }
            }
            xi = xii_p0 + 2*pp*block_snp;
            xj = xjj_p0 + 2*pp*block_snp;

            n_comb = 0;
            //BETWEEN I and J
            for(i = 0; i < block_i-1; i++){
                SNPA = &data_zeros[xi + i*block_pac*2];
                //xft0 = i*block_j;
                for(j = i+1; j < block_i; j++){
                    //xft = (xft0 + j)*num_combs;
                    SNPB = &data_zeros[xi + j*block_pac*2];
                    for(k = 0; k < block_j; k++){
                        xft = n_comb*num_combs;
                        SNPC = &data_zeros[xj + k*block_pac*2];
                        for(p = 0; p < 2*(PP_zeros - pp - 1); p+=2){
                            di2 = ~(SNPA[p] | SNPA[p + 1]);
                            dj2 = ~(SNPB[p] | SNPB[p + 1]);
                            dk2 = ~(SNPC[p] | SNPC[p + 1]);
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_zeros;
                        dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_zeros;
                        dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_zeros;
            
                        t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                        t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                        t002 = SNPA[p] & SNPB[p] & dk2;
                        t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                        t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                        t012 = SNPA[p] & SNPB[p + 1] & dk2;
                        t020 = SNPA[p] & dj2 & SNPC[p];
                        t021 = SNPA[p] & dj2 & SNPC[p + 1];
                        t022 = SNPA[p] & dj2 & dk2;

                        t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                        t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                        t102 = SNPA[p + 1] & SNPB[p] & dk2;
                        t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                        t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                        t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                        t120 = SNPA[p + 1] & dj2 & SNPC[p];
                        t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                        t122 = SNPA[p + 1] & dj2 & dk2;

                        t200 = di2 & SNPB[p] & SNPC[p];
                        t201 = di2 & SNPB[p] & SNPC[p + 1];
                        t202 = di2 & SNPB[p] & dk2;
                        t210 = di2 & SNPB[p + 1] & SNPC[p];
                        t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                        t212 = di2 & SNPB[p + 1] & dk2;
                        t220 = di2 & dj2 & SNPC[p];
                        t221 = di2 & dj2 & SNPC[p + 1];
                        t222 = di2 & dj2 & dk2;

                        freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                        freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                        freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                        freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                        freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                        freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                        freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                        freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                        freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                        freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                        freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                        freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                        freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                        freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                        freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                        freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                        freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                        freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                        freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                        freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                        freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                        freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                        freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                        freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                        freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                        freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                        freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);

                        n_comb ++;
                    }
                }
            }
            //Phenotype equal 1
            for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
                xi = xii_p1 + 2*pp*block_snp;
                xj = xjj_p1 + 2*pp*block_snp;

                n_comb = 0;
                //BETWEEN I and J
                for(i = 0; i < block_i - 1; i++){
                    SNPA = &data_ones[xi + i*block_pac*2];
                    //xft0 = i*block_j;
                    for(j = i + 1; j < block_i; j++){
                        //xft = (comb_ij + xft0 + j)*num_combs;
                        SNPB = &data_ones[xi + j*block_pac*2];
                        for(k = 0; k < block_j; k++){
                            xft = (comb_ij + n_comb)*num_combs;
                            SNPC = &data_ones[xj + k*block_pac*2];
                            for(p = 0; p < 2*block_pac; p+=2){
                                di2 = ~(SNPA[p] | SNPA[p + 1]);
                                dj2 = ~(SNPB[p] | SNPB[p + 1]);
                                dk2 = ~(SNPC[p] | SNPC[p + 1]);
                    
                                t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                                t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                                t002 = SNPA[p] & SNPB[p] & dk2;
                                t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                                t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                                t012 = SNPA[p] & SNPB[p + 1] & dk2;
                                t020 = SNPA[p] & dj2 & SNPC[p];
                                t021 = SNPA[p] & dj2 & SNPC[p + 1];
                                t022 = SNPA[p] & dj2 & dk2;

                                t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                                t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                                t102 = SNPA[p + 1] & SNPB[p] & dk2;
                                t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                                t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                                t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                                t120 = SNPA[p + 1] & dj2 & SNPC[p];
                                t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                                t122 = SNPA[p + 1] & dj2 & dk2;

                                t200 = di2 & SNPB[p] & SNPC[p];
                                t201 = di2 & SNPB[p] & SNPC[p + 1];
                                t202 = di2 & SNPB[p] & dk2;
                                t210 = di2 & SNPB[p + 1] & SNPC[p];
                                t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                                t212 = di2 & SNPB[p + 1] & dk2;
                                t220 = di2 & dj2 & SNPC[p];
                                t221 = di2 & dj2 & SNPC[p + 1];
                                t222 = di2 & dj2 & dk2;

                                freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                                freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                                freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                                freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                                freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                                freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                                freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                                freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                                freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                                freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                                freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                                freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                                freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                                freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                                freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                                freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                                freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                                freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                                freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                                freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                                freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                                freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                                freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                                freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                                freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                                freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                                freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                            }
                            n_comb ++;
                        }
                    }
                }
            }
            n_comb = 0;

            xi = xii_p1 + 2*pp*block_snp;
            xj = xjj_p1 + 2*pp*block_snp;
            //BETWEEN I and J
            for(i = 0; i < block_i - 1; i++){
                SNPA = &data_ones[xi + i*block_pac*2];
                //xft0 = i*block_j;
                for(j = i+1; j < block_i; j++){
                    //xft = (comb_ij + xft0 + j)*num_combs;
                    SNPB = &data_ones[xi + j*block_pac*2];
                    for(k = 0; k < block_j; k++){
                        xft = (comb_ij + n_comb)*num_combs;
                        SNPC = &data_ones[xj + k*block_pac*2];
                        for(p = 0; p < 2*(PP_ones - pp - 1); p+=2){
                            di2 = ~(SNPA[p] | SNPA[p + 1]);
                            dj2 = ~(SNPB[p] | SNPB[p + 1]);
                            dk2 = ~(SNPC[p] | SNPC[p + 1]);
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);
                        }
                        //Do Remaining Elements
                        di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_ones;
                        dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_ones;
                        dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_ones;
            
                        t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                        t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                        t002 = SNPA[p] & SNPB[p] & dk2;
                        t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                        t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                        t012 = SNPA[p] & SNPB[p + 1] & dk2;
                        t020 = SNPA[p] & dj2 & SNPC[p];
                        t021 = SNPA[p] & dj2 & SNPC[p + 1];
                        t022 = SNPA[p] & dj2 & dk2;

                        t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                        t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                        t102 = SNPA[p + 1] & SNPB[p] & dk2;
                        t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                        t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                        t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                        t120 = SNPA[p + 1] & dj2 & SNPC[p];
                        t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                        t122 = SNPA[p + 1] & dj2 & dk2;

                        t200 = di2 & SNPB[p] & SNPC[p];
                        t201 = di2 & SNPB[p] & SNPC[p + 1];
                        t202 = di2 & SNPB[p] & dk2;
                        t210 = di2 & SNPB[p + 1] & SNPC[p];
                        t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                        t212 = di2 & SNPB[p + 1] & dk2;
                        t220 = di2 & dj2 & SNPC[p];
                        t221 = di2 & dj2 & SNPC[p + 1];
                        t222 = di2 & dj2 & dk2;

                        freq_table_IJ[xft + 0] += _mm_popcnt_u32(t000);
                        freq_table_IJ[xft + 1] += _mm_popcnt_u32(t001);
                        freq_table_IJ[xft + 2] += _mm_popcnt_u32(t002);
                        freq_table_IJ[xft + 3] += _mm_popcnt_u32(t010);
                        freq_table_IJ[xft + 4] += _mm_popcnt_u32(t011);
                        freq_table_IJ[xft + 5] += _mm_popcnt_u32(t012);
                        freq_table_IJ[xft + 6] += _mm_popcnt_u32(t020);
                        freq_table_IJ[xft + 7] += _mm_popcnt_u32(t021);
                        freq_table_IJ[xft + 8] += _mm_popcnt_u32(t022);
                        freq_table_IJ[xft + 9] += _mm_popcnt_u32(t100);
                        freq_table_IJ[xft + 10] += _mm_popcnt_u32(t101);
                        freq_table_IJ[xft + 11] += _mm_popcnt_u32(t102);
                        freq_table_IJ[xft + 12] += _mm_popcnt_u32(t110);
                        freq_table_IJ[xft + 13] += _mm_popcnt_u32(t111);
                        freq_table_IJ[xft + 14] += _mm_popcnt_u32(t112);
                        freq_table_IJ[xft + 15] += _mm_popcnt_u32(t120);
                        freq_table_IJ[xft + 16] += _mm_popcnt_u32(t121);
                        freq_table_IJ[xft + 17] += _mm_popcnt_u32(t122);
                        freq_table_IJ[xft + 18] += _mm_popcnt_u32(t200);
                        freq_table_IJ[xft + 19] += _mm_popcnt_u32(t201);
                        freq_table_IJ[xft + 20] += _mm_popcnt_u32(t202);
                        freq_table_IJ[xft + 21] += _mm_popcnt_u32(t210);
                        freq_table_IJ[xft + 22] += _mm_popcnt_u32(t211);
                        freq_table_IJ[xft + 23] += _mm_popcnt_u32(t212);
                        freq_table_IJ[xft + 24] += _mm_popcnt_u32(t220);
                        freq_table_IJ[xft + 25] += _mm_popcnt_u32(t221);
                        freq_table_IJ[xft + 26] += _mm_popcnt_u32(t222);

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
                        // compare score
                        if(score < best_score_local){
                            best_score_local = score;
                            best_snp_global[tid][0] = ii + mi;
                            best_snp_global[tid][1] = ii + mj;
                            best_snp_global[tid][2] = jj + mk;
                        }
                    }
                }
                base += block_j*(block_i - (mi + 1));
            }
        }

        //RESET FREQUENCY TABLES
        memset(freq_table_I, 0, 2 * num_fts_I * num_combs * sizeof(uint32_t));
       
        comb_ii = (block_i*(block_i -1)*(block_i -2))/6;

        //Phenotype = 0
        for(pp = 0; pp < PP_zeros - block_pac; pp+=block_pac){
            xi = xii_p0 + 2*pp*block_snp;
            n_comb = 0;
            //BLOCK II
            for(i = 0; i < block_i - 2; i++){
                SNPA = &data_zeros[xi + i*block_pac*2];
                //INSIDE BLOCK I
                for(j = i+1; j < block_i - 1; j++){
                    SNPB = &data_zeros[xi + j*block_pac*2];
                    for(k = j+1; k < block_i; k++){
                        SNPC = &data_zeros[xi + k*block_pac*2];
                        xft = n_comb*num_combs;
                        for(p = 0; p < 2*block_pac; p+=2){
                            di2 = ~(SNPA[p] | SNPA[p + 1]);
                            dj2 = ~(SNPB[p] | SNPB[p + 1]);
                            dk2 = ~(SNPC[p] | SNPC[p + 1]);
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_I[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_I[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_I[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_I[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_I[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_I[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_I[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_I[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_I[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_I[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_I[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_I[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_I[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_I[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_I[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_I[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_I[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_I[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_I[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_I[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_I[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_I[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_I[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_I[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_I[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_I[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_I[xft + 26] += _mm_popcnt_u32(t222);
                        }
                        n_comb++;
                    }
                }
            }
        }
        xi = xii_p0 + 2*pp*block_snp;
        n_comb = 0;
        //BLOCK II
        for(i = 0; i < block_i - 2; i++){
            SNPA = &data_zeros[xi + i*block_pac*2];
            //INSIDE BLOCK I
            for(j = i+1; j < block_i - 1; j++){
                SNPB = &data_zeros[xi + j*block_pac*2];
                for(k = j+1; k < block_i; k++){
                    SNPC = &data_zeros[xi + k*block_pac*2];
                    xft = n_comb*num_combs;
                    for(p = 0; p < 2*(PP_zeros - pp - 1); p+=2){
                        di2 = ~(SNPA[p] | SNPA[p + 1]);
                        dj2 = ~(SNPB[p] | SNPB[p + 1]);
                        dk2 = ~(SNPC[p] | SNPC[p + 1]);
            
                        t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                        t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                        t002 = SNPA[p] & SNPB[p] & dk2;
                        t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                        t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                        t012 = SNPA[p] & SNPB[p + 1] & dk2;
                        t020 = SNPA[p] & dj2 & SNPC[p];
                        t021 = SNPA[p] & dj2 & SNPC[p + 1];
                        t022 = SNPA[p] & dj2 & dk2;

                        t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                        t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                        t102 = SNPA[p + 1] & SNPB[p] & dk2;
                        t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                        t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                        t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                        t120 = SNPA[p + 1] & dj2 & SNPC[p];
                        t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                        t122 = SNPA[p + 1] & dj2 & dk2;

                        t200 = di2 & SNPB[p] & SNPC[p];
                        t201 = di2 & SNPB[p] & SNPC[p + 1];
                        t202 = di2 & SNPB[p] & dk2;
                        t210 = di2 & SNPB[p + 1] & SNPC[p];
                        t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                        t212 = di2 & SNPB[p + 1] & dk2;
                        t220 = di2 & dj2 & SNPC[p];
                        t221 = di2 & dj2 & SNPC[p + 1];
                        t222 = di2 & dj2 & dk2;

                        freq_table_I[xft + 0] += _mm_popcnt_u32(t000);
                        freq_table_I[xft + 1] += _mm_popcnt_u32(t001);
                        freq_table_I[xft + 2] += _mm_popcnt_u32(t002);
                        freq_table_I[xft + 3] += _mm_popcnt_u32(t010);
                        freq_table_I[xft + 4] += _mm_popcnt_u32(t011);
                        freq_table_I[xft + 5] += _mm_popcnt_u32(t012);
                        freq_table_I[xft + 6] += _mm_popcnt_u32(t020);
                        freq_table_I[xft + 7] += _mm_popcnt_u32(t021);
                        freq_table_I[xft + 8] += _mm_popcnt_u32(t022);
                        freq_table_I[xft + 9] += _mm_popcnt_u32(t100);
                        freq_table_I[xft + 10] += _mm_popcnt_u32(t101);
                        freq_table_I[xft + 11] += _mm_popcnt_u32(t102);
                        freq_table_I[xft + 12] += _mm_popcnt_u32(t110);
                        freq_table_I[xft + 13] += _mm_popcnt_u32(t111);
                        freq_table_I[xft + 14] += _mm_popcnt_u32(t112);
                        freq_table_I[xft + 15] += _mm_popcnt_u32(t120);
                        freq_table_I[xft + 16] += _mm_popcnt_u32(t121);
                        freq_table_I[xft + 17] += _mm_popcnt_u32(t122);
                        freq_table_I[xft + 18] += _mm_popcnt_u32(t200);
                        freq_table_I[xft + 19] += _mm_popcnt_u32(t201);
                        freq_table_I[xft + 20] += _mm_popcnt_u32(t202);
                        freq_table_I[xft + 21] += _mm_popcnt_u32(t210);
                        freq_table_I[xft + 22] += _mm_popcnt_u32(t211);
                        freq_table_I[xft + 23] += _mm_popcnt_u32(t212);
                        freq_table_I[xft + 24] += _mm_popcnt_u32(t220);
                        freq_table_I[xft + 25] += _mm_popcnt_u32(t221);
                        freq_table_I[xft + 26] += _mm_popcnt_u32(t222);
                    }
                    //Do Remaining Elements
                    di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_zeros;
                    dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_zeros;
                    dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_zeros;
        
                    t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                    t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                    t002 = SNPA[p] & SNPB[p] & dk2;
                    t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                    t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                    t012 = SNPA[p] & SNPB[p + 1] & dk2;
                    t020 = SNPA[p] & dj2 & SNPC[p];
                    t021 = SNPA[p] & dj2 & SNPC[p + 1];
                    t022 = SNPA[p] & dj2 & dk2;

                    t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                    t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                    t102 = SNPA[p + 1] & SNPB[p] & dk2;
                    t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                    t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                    t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                    t120 = SNPA[p + 1] & dj2 & SNPC[p];
                    t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                    t122 = SNPA[p + 1] & dj2 & dk2;

                    t200 = di2 & SNPB[p] & SNPC[p];
                    t201 = di2 & SNPB[p] & SNPC[p + 1];
                    t202 = di2 & SNPB[p] & dk2;
                    t210 = di2 & SNPB[p + 1] & SNPC[p];
                    t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                    t212 = di2 & SNPB[p + 1] & dk2;
                    t220 = di2 & dj2 & SNPC[p];
                    t221 = di2 & dj2 & SNPC[p + 1];
                    t222 = di2 & dj2 & dk2;

                    freq_table_I[xft + 0] += _mm_popcnt_u32(t000);
                    freq_table_I[xft + 1] += _mm_popcnt_u32(t001);
                    freq_table_I[xft + 2] += _mm_popcnt_u32(t002);
                    freq_table_I[xft + 3] += _mm_popcnt_u32(t010);
                    freq_table_I[xft + 4] += _mm_popcnt_u32(t011);
                    freq_table_I[xft + 5] += _mm_popcnt_u32(t012);
                    freq_table_I[xft + 6] += _mm_popcnt_u32(t020);
                    freq_table_I[xft + 7] += _mm_popcnt_u32(t021);
                    freq_table_I[xft + 8] += _mm_popcnt_u32(t022);
                    freq_table_I[xft + 9] += _mm_popcnt_u32(t100);
                    freq_table_I[xft + 10] += _mm_popcnt_u32(t101);
                    freq_table_I[xft + 11] += _mm_popcnt_u32(t102);
                    freq_table_I[xft + 12] += _mm_popcnt_u32(t110);
                    freq_table_I[xft + 13] += _mm_popcnt_u32(t111);
                    freq_table_I[xft + 14] += _mm_popcnt_u32(t112);
                    freq_table_I[xft + 15] += _mm_popcnt_u32(t120);
                    freq_table_I[xft + 16] += _mm_popcnt_u32(t121);
                    freq_table_I[xft + 17] += _mm_popcnt_u32(t122);
                    freq_table_I[xft + 18] += _mm_popcnt_u32(t200);
                    freq_table_I[xft + 19] += _mm_popcnt_u32(t201);
                    freq_table_I[xft + 20] += _mm_popcnt_u32(t202);
                    freq_table_I[xft + 21] += _mm_popcnt_u32(t210);
                    freq_table_I[xft + 22] += _mm_popcnt_u32(t211);
                    freq_table_I[xft + 23] += _mm_popcnt_u32(t212);
                    freq_table_I[xft + 24] += _mm_popcnt_u32(t220);
                    freq_table_I[xft + 25] += _mm_popcnt_u32(t221);
                    freq_table_I[xft + 26] += _mm_popcnt_u32(t222);

                    n_comb++;
                }
            }
        }

        //Phenotype = 1
        for(pp = 0; pp < PP_ones - block_pac; pp+=block_pac){
            xi = xii_p1 + 2*pp*block_snp;
            n_comb = 0;
            //BLOCK II
            for(i = 0; i < block_i - 2; i++){
                SNPA = &data_ones[xi + i*block_pac*2];
                //INSIDE BLOCK I
                for(j = i+1; j < block_i - 1; j++){
                    SNPB = &data_ones[xi + j*block_pac*2];
                    for(k = j+1; k < block_i; k++){
                        SNPC = &data_ones[xi + k*block_pac*2];
                        xft = (comb_ii + n_comb)*num_combs;
                        for(p = 0; p < 2*block_pac; p+=2){
                            di2 = ~(SNPA[p] | SNPA[p + 1]);
                            dj2 = ~(SNPB[p] | SNPB[p + 1]);
                            dk2 = ~(SNPC[p] | SNPC[p + 1]);
                
                            t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                            t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                            t002 = SNPA[p] & SNPB[p] & dk2;
                            t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                            t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                            t012 = SNPA[p] & SNPB[p + 1] & dk2;
                            t020 = SNPA[p] & dj2 & SNPC[p];
                            t021 = SNPA[p] & dj2 & SNPC[p + 1];
                            t022 = SNPA[p] & dj2 & dk2;

                            t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                            t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                            t102 = SNPA[p + 1] & SNPB[p] & dk2;
                            t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                            t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                            t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                            t120 = SNPA[p + 1] & dj2 & SNPC[p];
                            t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                            t122 = SNPA[p + 1] & dj2 & dk2;

                            t200 = di2 & SNPB[p] & SNPC[p];
                            t201 = di2 & SNPB[p] & SNPC[p + 1];
                            t202 = di2 & SNPB[p] & dk2;
                            t210 = di2 & SNPB[p + 1] & SNPC[p];
                            t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                            t212 = di2 & SNPB[p + 1] & dk2;
                            t220 = di2 & dj2 & SNPC[p];
                            t221 = di2 & dj2 & SNPC[p + 1];
                            t222 = di2 & dj2 & dk2;

                            freq_table_I[xft + 0] += _mm_popcnt_u32(t000);
                            freq_table_I[xft + 1] += _mm_popcnt_u32(t001);
                            freq_table_I[xft + 2] += _mm_popcnt_u32(t002);
                            freq_table_I[xft + 3] += _mm_popcnt_u32(t010);
                            freq_table_I[xft + 4] += _mm_popcnt_u32(t011);
                            freq_table_I[xft + 5] += _mm_popcnt_u32(t012);
                            freq_table_I[xft + 6] += _mm_popcnt_u32(t020);
                            freq_table_I[xft + 7] += _mm_popcnt_u32(t021);
                            freq_table_I[xft + 8] += _mm_popcnt_u32(t022);
                            freq_table_I[xft + 9] += _mm_popcnt_u32(t100);
                            freq_table_I[xft + 10] += _mm_popcnt_u32(t101);
                            freq_table_I[xft + 11] += _mm_popcnt_u32(t102);
                            freq_table_I[xft + 12] += _mm_popcnt_u32(t110);
                            freq_table_I[xft + 13] += _mm_popcnt_u32(t111);
                            freq_table_I[xft + 14] += _mm_popcnt_u32(t112);
                            freq_table_I[xft + 15] += _mm_popcnt_u32(t120);
                            freq_table_I[xft + 16] += _mm_popcnt_u32(t121);
                            freq_table_I[xft + 17] += _mm_popcnt_u32(t122);
                            freq_table_I[xft + 18] += _mm_popcnt_u32(t200);
                            freq_table_I[xft + 19] += _mm_popcnt_u32(t201);
                            freq_table_I[xft + 20] += _mm_popcnt_u32(t202);
                            freq_table_I[xft + 21] += _mm_popcnt_u32(t210);
                            freq_table_I[xft + 22] += _mm_popcnt_u32(t211);
                            freq_table_I[xft + 23] += _mm_popcnt_u32(t212);
                            freq_table_I[xft + 24] += _mm_popcnt_u32(t220);
                            freq_table_I[xft + 25] += _mm_popcnt_u32(t221);
                            freq_table_I[xft + 26] += _mm_popcnt_u32(t222);
                        }
                        n_comb++;
                    }
                }
            }
        }
        xi = xii_p1 + 2*pp*block_snp;
        n_comb = 0;
        //BLOCK II
        for(i = 0; i < block_i - 2; i++){
            SNPA = &data_ones[xi + i*block_pac*2];
            xft0 = i*block_snp;
            //INSIDE BLOCK I
            for(j = i+1; j < block_i - 1; j++){
                SNPB = &data_ones[xi + j*block_pac*2];
                for(k = j+1; k < block_i; k++){
                    SNPC = &data_ones[xi + k*block_pac*2];
                    xft = (comb_ii + n_comb)*num_combs;
                    for(p = 0; p < 2*(PP_ones - pp - 1); p+=2){
                        di2 = ~(SNPA[p] | SNPA[p + 1]);
                        dj2 = ~(SNPB[p] | SNPB[p + 1]);
                        dk2 = ~(SNPC[p] | SNPC[p + 1]);
            
                        t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                        t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                        t002 = SNPA[p] & SNPB[p] & dk2;
                        t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                        t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                        t012 = SNPA[p] & SNPB[p + 1] & dk2;
                        t020 = SNPA[p] & dj2 & SNPC[p];
                        t021 = SNPA[p] & dj2 & SNPC[p + 1];
                        t022 = SNPA[p] & dj2 & dk2;

                        t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                        t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                        t102 = SNPA[p + 1] & SNPB[p] & dk2;
                        t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                        t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                        t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                        t120 = SNPA[p + 1] & dj2 & SNPC[p];
                        t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                        t122 = SNPA[p + 1] & dj2 & dk2;

                        t200 = di2 & SNPB[p] & SNPC[p];
                        t201 = di2 & SNPB[p] & SNPC[p + 1];
                        t202 = di2 & SNPB[p] & dk2;
                        t210 = di2 & SNPB[p + 1] & SNPC[p];
                        t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                        t212 = di2 & SNPB[p + 1] & dk2;
                        t220 = di2 & dj2 & SNPC[p];
                        t221 = di2 & dj2 & SNPC[p + 1];
                        t222 = di2 & dj2 & dk2;

                        freq_table_I[xft + 0] += _mm_popcnt_u32(t000);
                        freq_table_I[xft + 1] += _mm_popcnt_u32(t001);
                        freq_table_I[xft + 2] += _mm_popcnt_u32(t002);
                        freq_table_I[xft + 3] += _mm_popcnt_u32(t010);
                        freq_table_I[xft + 4] += _mm_popcnt_u32(t011);
                        freq_table_I[xft + 5] += _mm_popcnt_u32(t012);
                        freq_table_I[xft + 6] += _mm_popcnt_u32(t020);
                        freq_table_I[xft + 7] += _mm_popcnt_u32(t021);
                        freq_table_I[xft + 8] += _mm_popcnt_u32(t022);
                        freq_table_I[xft + 9] += _mm_popcnt_u32(t100);
                        freq_table_I[xft + 10] += _mm_popcnt_u32(t101);
                        freq_table_I[xft + 11] += _mm_popcnt_u32(t102);
                        freq_table_I[xft + 12] += _mm_popcnt_u32(t110);
                        freq_table_I[xft + 13] += _mm_popcnt_u32(t111);
                        freq_table_I[xft + 14] += _mm_popcnt_u32(t112);
                        freq_table_I[xft + 15] += _mm_popcnt_u32(t120);
                        freq_table_I[xft + 16] += _mm_popcnt_u32(t121);
                        freq_table_I[xft + 17] += _mm_popcnt_u32(t122);
                        freq_table_I[xft + 18] += _mm_popcnt_u32(t200);
                        freq_table_I[xft + 19] += _mm_popcnt_u32(t201);
                        freq_table_I[xft + 20] += _mm_popcnt_u32(t202);
                        freq_table_I[xft + 21] += _mm_popcnt_u32(t210);
                        freq_table_I[xft + 22] += _mm_popcnt_u32(t211);
                        freq_table_I[xft + 23] += _mm_popcnt_u32(t212);
                        freq_table_I[xft + 24] += _mm_popcnt_u32(t220);
                        freq_table_I[xft + 25] += _mm_popcnt_u32(t221);
                        freq_table_I[xft + 26] += _mm_popcnt_u32(t222);
                    }
                    //Do Remaining Elements
                    di2 = ~(SNPA[p] | SNPA[p + 1]) & mask_ones;
                    dj2 = ~(SNPB[p] | SNPB[p + 1]) & mask_ones;
                    dk2 = ~(SNPC[p] | SNPC[p + 1]) & mask_ones;
        
                    t000 = SNPA[p] & SNPB[p] & SNPC[p]; 
                    t001 = SNPA[p] & SNPB[p] & SNPC[p + 1];
                    t002 = SNPA[p] & SNPB[p] & dk2;
                    t010 = SNPA[p] & SNPB[p + 1] & SNPC[p];
                    t011 = SNPA[p] & SNPB[p + 1] & SNPC[p + 1];
                    t012 = SNPA[p] & SNPB[p + 1] & dk2;
                    t020 = SNPA[p] & dj2 & SNPC[p];
                    t021 = SNPA[p] & dj2 & SNPC[p + 1];
                    t022 = SNPA[p] & dj2 & dk2;

                    t100 = SNPA[p + 1] & SNPB[p] & SNPC[p];
                    t101 = SNPA[p + 1] & SNPB[p] & SNPC[p + 1];
                    t102 = SNPA[p + 1] & SNPB[p] & dk2;
                    t110 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p];
                    t111 = SNPA[p + 1] & SNPB[p + 1] & SNPC[p + 1];
                    t112 = SNPA[p + 1] & SNPB[p + 1] & dk2;
                    t120 = SNPA[p + 1] & dj2 & SNPC[p];
                    t121 = SNPA[p + 1] & dj2 & SNPC[p + 1];
                    t122 = SNPA[p + 1] & dj2 & dk2;

                    t200 = di2 & SNPB[p] & SNPC[p];
                    t201 = di2 & SNPB[p] & SNPC[p + 1];
                    t202 = di2 & SNPB[p] & dk2;
                    t210 = di2 & SNPB[p + 1] & SNPC[p];
                    t211 = di2 & SNPB[p + 1] & SNPC[p + 1];
                    t212 = di2 & SNPB[p + 1] & dk2;
                    t220 = di2 & dj2 & SNPC[p];
                    t221 = di2 & dj2 & SNPC[p + 1];
                    t222 = di2 & dj2 & dk2;

                    freq_table_I[xft + 0] += _mm_popcnt_u32(t000);
                    freq_table_I[xft + 1] += _mm_popcnt_u32(t001);
                    freq_table_I[xft + 2] += _mm_popcnt_u32(t002);
                    freq_table_I[xft + 3] += _mm_popcnt_u32(t010);
                    freq_table_I[xft + 4] += _mm_popcnt_u32(t011);
                    freq_table_I[xft + 5] += _mm_popcnt_u32(t012);
                    freq_table_I[xft + 6] += _mm_popcnt_u32(t020);
                    freq_table_I[xft + 7] += _mm_popcnt_u32(t021);
                    freq_table_I[xft + 8] += _mm_popcnt_u32(t022);
                    freq_table_I[xft + 9] += _mm_popcnt_u32(t100);
                    freq_table_I[xft + 10] += _mm_popcnt_u32(t101);
                    freq_table_I[xft + 11] += _mm_popcnt_u32(t102);
                    freq_table_I[xft + 12] += _mm_popcnt_u32(t110);
                    freq_table_I[xft + 13] += _mm_popcnt_u32(t111);
                    freq_table_I[xft + 14] += _mm_popcnt_u32(t112);
                    freq_table_I[xft + 15] += _mm_popcnt_u32(t120);
                    freq_table_I[xft + 16] += _mm_popcnt_u32(t121);
                    freq_table_I[xft + 17] += _mm_popcnt_u32(t122);
                    freq_table_I[xft + 18] += _mm_popcnt_u32(t200);
                    freq_table_I[xft + 19] += _mm_popcnt_u32(t201);
                    freq_table_I[xft + 20] += _mm_popcnt_u32(t202);
                    freq_table_I[xft + 21] += _mm_popcnt_u32(t210);
                    freq_table_I[xft + 22] += _mm_popcnt_u32(t211);
                    freq_table_I[xft + 23] += _mm_popcnt_u32(t212);
                    freq_table_I[xft + 24] += _mm_popcnt_u32(t220);
                    freq_table_I[xft + 25] += _mm_popcnt_u32(t221);
                    freq_table_I[xft + 26] += _mm_popcnt_u32(t222);

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
                }
                base += (block_i - (mj+1));
            }
            
        }
    }

    #pragma omp critical
    {
        if(best_score > best_score_local){
            best_score = best_score_local;
            best_snp[0] = best_snp_global[tid][0];
            best_snp[1] = best_snp_global[tid][1];
            best_snp[2] = best_snp_global[tid][2];
        }
    }

    #if defined (ADVISOR)
        __itt_pause();
    #endif  

    #if defined (SDE)
        __SSC_MARK(0x222);
    #endif

    #if defined (LIKWID)
        LIKWID_MARKER_STOP("epistasis");
    #endif
    
    cyc_e_local = read_tsc_end();
    
    serialize();

    cyc_s = cyc_s_local;
    cyc_e = cyc_e_local;

    _mm_free(freq_table_IJK);
    _mm_free(freq_table_IJ);
    _mm_free(freq_table_I);

    }
    
    printf("Time bin_no_phen_nor_block: %f\n", (double) (cyc_e - cyc_s)/FREQ);

    printf("bin_no_phen_nor_block Best: %d, %d, %d - Score: %f\n", best_snp[0], best_snp[1], best_snp[2], best_score);

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
    #endif

    #if defined (LIKWID)
        LIKWID_MARKER_REGISTER("epistasis");
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

    dim_epi = atoi(argv[1]);
    num_pac = atol(argv[2]);
    num_snp = atol(argv[3]);
    block_pac = atoi(argv[4]);
    block_snp = atoi(argv[5]);

    int comb = (int)pow(3.0, dim_epi);

    generate_data(num_pac, num_snp, &SNP_Data, &Ph_Data);

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

    return 0;
}

