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

#define TABLE_MAX_SIZE 748 
#define TABLE_ERROR -0.0810 
float *addlogtable;

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif 


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
    
        
    //Generate Phenotype 50/50
    for(i = 0; i < N; i++){
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

void process_epi_bin_no_phen_nor(uint32_t* data_zeros, uint32_t* data_ones, int long long phen_ones, int dim_epi, int long long num_snp, int long long num_pac, int num_combs)
{

    float best_score = FLT_MAX;
    int long long cyc_s = LLONG_MAX, cyc_e = 0;
    omp_set_num_threads(NUM_THREADS);
    int best_snp_global[NUM_THREADS][3];
    int best_snp[3];

    #pragma omp parallel reduction(min:cyc_s) reduction(max:cyc_e)
    {

    int tid = omp_get_thread_num();
    
    int i, j, k, p, m;

    int PP_ones = ceil((1.0*(phen_ones))/32.0);
    int PP_zeros = ceil((1.0*(num_pac - (phen_ones)))/32.0);

    uint32_t *SNPA_p0, *SNPA_p1, *SNPB_p0, *SNPB_p1, *SNPC_p0, *SNPC_p1;
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
    uint32_t * ft = (uint32_t *) _mm_malloc(2*num_combs*sizeof(uint32_t), 64);
    memset(ft, 0, 2*num_combs*sizeof(uint32_t));

    float best_score_local = FLT_MAX;
    int long long cyc_s_local, cyc_e_local;

    serialize();

    cyc_s_local = read_tsc_start();

    // fill frequency table
    #pragma omp for schedule(dynamic)
    for(i = 0; i < num_snp - 2; i++){
        SNPA_p0 = &data_zeros[i*PP_zeros*2];
        SNPA_p1 = &data_ones[i*PP_ones*2];
    
        for(j = i + 1; j < num_snp - 1; j++){
            SNPB_p0 = &data_zeros[j*PP_zeros*2];
            SNPB_p1 = &data_ones[j*PP_ones*2];

            for(k = j + 1; k < num_snp; k++){
                SNPC_p0 = &data_zeros[k*PP_zeros*2];
                SNPC_p1 = &data_ones[k*PP_ones*2];

                // reset frequency table
                memset(ft, 0, 2*num_combs*sizeof(uint32_t));
                
                //Phenotype equal 0
                for(p = 0; p < 2*(PP_zeros - 1); p+=2){

                    di2 = ~(SNPA_p0[p] | SNPA_p0[p + 1]);
                    dj2 = ~(SNPB_p0[p] | SNPB_p0[p + 1]);
                    dk2 = ~(SNPC_p0[p] | SNPC_p0[p + 1]);
                    
                    t000 = SNPA_p0[p] & SNPB_p0[p] & SNPC_p0[p]; 
                    t001 = SNPA_p0[p] & SNPB_p0[p] & SNPC_p0[p + 1];
                    t002 = SNPA_p0[p] & SNPB_p0[p] & dk2;
                    t010 = SNPA_p0[p] & SNPB_p0[p + 1] & SNPC_p0[p];
                    t011 = SNPA_p0[p] & SNPB_p0[p + 1] & SNPC_p0[p + 1];
                    t012 = SNPA_p0[p] & SNPB_p0[p + 1] & dk2;
                    t020 = SNPA_p0[p] & dj2 & SNPC_p0[p];
                    t021 = SNPA_p0[p] & dj2 & SNPC_p0[p + 1];
                    t022 = SNPA_p0[p] & dj2 & dk2;

                    t100 = SNPA_p0[p + 1] & SNPB_p0[p] & SNPC_p0[p];
                    t101 = SNPA_p0[p + 1] & SNPB_p0[p] & SNPC_p0[p + 1];
                    t102 = SNPA_p0[p + 1] & SNPB_p0[p] & dk2;
                    t110 = SNPA_p0[p + 1] & SNPB_p0[p + 1] & SNPC_p0[p];
                    t111 = SNPA_p0[p + 1] & SNPB_p0[p + 1] & SNPC_p0[p + 1];
                    t112 = SNPA_p0[p + 1] & SNPB_p0[p + 1] & dk2;
                    t120 = SNPA_p0[p + 1] & dj2 & SNPC_p0[p];
                    t121 = SNPA_p0[p + 1] & dj2 & SNPC_p0[p + 1];
                    t122 = SNPA_p0[p + 1] & dj2 & dk2;

                    t200 = di2 & SNPB_p0[p] & SNPC_p0[p];
                    t201 = di2 & SNPB_p0[p] & SNPC_p0[p + 1];
                    t202 = di2 & SNPB_p0[p] & dk2;
                    t210 = di2 & SNPB_p0[p + 1] & SNPC_p0[p];
                    t211 = di2 & SNPB_p0[p + 1] & SNPC_p0[p + 1];
                    t212 = di2 & SNPB_p0[p + 1] & dk2;
                    t220 = di2 & dj2 & SNPC_p0[p];
                    t221 = di2 & dj2 & SNPC_p0[p + 1];
                    t222 = di2 & dj2 & dk2;

                    ft[0] += _mm_popcnt_u32(t000);
                    ft[1] += _mm_popcnt_u32(t001);
                    ft[2] += _mm_popcnt_u32(t002);
                    ft[3] += _mm_popcnt_u32(t010);
                    ft[4] += _mm_popcnt_u32(t011);
                    ft[5] += _mm_popcnt_u32(t012);
                    ft[6] += _mm_popcnt_u32(t020);
                    ft[7] += _mm_popcnt_u32(t021);
                    ft[8] += _mm_popcnt_u32(t022);
                    ft[9] += _mm_popcnt_u32(t100);
                    ft[10] += _mm_popcnt_u32(t101);
                    ft[11] += _mm_popcnt_u32(t102);
                    ft[12] += _mm_popcnt_u32(t110);
                    ft[13] += _mm_popcnt_u32(t111);
                    ft[14] += _mm_popcnt_u32(t112);
                    ft[15] += _mm_popcnt_u32(t120);
                    ft[16] += _mm_popcnt_u32(t121);
                    ft[17] += _mm_popcnt_u32(t122);
                    ft[18] += _mm_popcnt_u32(t200);
                    ft[19] += _mm_popcnt_u32(t201);
                    ft[20] += _mm_popcnt_u32(t202);
                    ft[21] += _mm_popcnt_u32(t210);
                    ft[22] += _mm_popcnt_u32(t211);
                    ft[23] += _mm_popcnt_u32(t212);
                    ft[24] += _mm_popcnt_u32(t220);
                    ft[25] += _mm_popcnt_u32(t221);
                    ft[26] += _mm_popcnt_u32(t222);
                }
                //Do Remaining Elements
                di2 = ~(SNPA_p0[p] | SNPA_p0[p + 1]) & mask_zeros;
                dj2 = ~(SNPB_p0[p] | SNPB_p0[p + 1]) & mask_zeros;
                dk2 = ~(SNPC_p0[p] | SNPC_p0[p + 1]) & mask_zeros;
                
                t000 = SNPA_p0[p] & SNPB_p0[p] & SNPC_p0[p]; 
                t001 = SNPA_p0[p] & SNPB_p0[p] & SNPC_p0[p + 1];
                t002 = SNPA_p0[p] & SNPB_p0[p] & dk2;
                t010 = SNPA_p0[p] & SNPB_p0[p + 1] & SNPC_p0[p];
                t011 = SNPA_p0[p] & SNPB_p0[p + 1] & SNPC_p0[p + 1];
                t012 = SNPA_p0[p] & SNPB_p0[p + 1] & dk2;
                t020 = SNPA_p0[p] & dj2 & SNPC_p0[p];
                t021 = SNPA_p0[p] & dj2 & SNPC_p0[p + 1];
                t022 = SNPA_p0[p] & dj2 & dk2;
                t100 = SNPA_p0[p + 1] & SNPB_p0[p] & SNPC_p0[p];
                t101 = SNPA_p0[p + 1] & SNPB_p0[p] & SNPC_p0[p + 1];
                t102 = SNPA_p0[p + 1] & SNPB_p0[p] & dk2;
                t110 = SNPA_p0[p + 1] & SNPB_p0[p + 1] & SNPC_p0[p];
                t111 = SNPA_p0[p + 1] & SNPB_p0[p + 1] & SNPC_p0[p + 1];
                t112 = SNPA_p0[p + 1] & SNPB_p0[p + 1] & dk2;
                t120 = SNPA_p0[p + 1] & dj2 & SNPC_p0[p];
                t121 = SNPA_p0[p + 1] & dj2 & SNPC_p0[p + 1];
                t122 = SNPA_p0[p + 1] & dj2 & dk2;
                t200 = di2 & SNPB_p0[p] & SNPC_p0[p];
                t201 = di2 & SNPB_p0[p] & SNPC_p0[p + 1];
                t202 = di2 & SNPB_p0[p] & dk2;
                t210 = di2 & SNPB_p0[p + 1] & SNPC_p0[p];
                t211 = di2 & SNPB_p0[p + 1] & SNPC_p0[p + 1];
                t212 = di2 & SNPB_p0[p + 1] & dk2;
                t220 = di2 & dj2 & SNPC_p0[p];
                t221 = di2 & dj2 & SNPC_p0[p + 1];
                t222 = di2 & dj2 & dk2;



                ft[0] += _mm_popcnt_u32(t000);
                ft[1] += _mm_popcnt_u32(t001);
                ft[2] += _mm_popcnt_u32(t002);
                ft[3] += _mm_popcnt_u32(t010);
                ft[4] += _mm_popcnt_u32(t011);
                ft[5] += _mm_popcnt_u32(t012);
                ft[6] += _mm_popcnt_u32(t020);
                ft[7] += _mm_popcnt_u32(t021);
                ft[8] += _mm_popcnt_u32(t022);
                ft[9] += _mm_popcnt_u32(t100);
                ft[10] += _mm_popcnt_u32(t101);
                ft[11] += _mm_popcnt_u32(t102);
                ft[12] += _mm_popcnt_u32(t110);
                ft[13] += _mm_popcnt_u32(t111);
                ft[14] += _mm_popcnt_u32(t112);
                ft[15] += _mm_popcnt_u32(t120);
                ft[16] += _mm_popcnt_u32(t121);
                ft[17] += _mm_popcnt_u32(t122);
                ft[18] += _mm_popcnt_u32(t200);
                ft[19] += _mm_popcnt_u32(t201);
                ft[20] += _mm_popcnt_u32(t202);
                ft[21] += _mm_popcnt_u32(t210);
                ft[22] += _mm_popcnt_u32(t211);
                ft[23] += _mm_popcnt_u32(t212);
                ft[24] += _mm_popcnt_u32(t220);
                ft[25] += _mm_popcnt_u32(t221);
                ft[26] += _mm_popcnt_u32(t222);

                //Phenotype equal 1
                for(p = 0; p < 2*(PP_ones-1); p+=2){

                    di2 = ~(SNPA_p1[p] | SNPA_p1[p + 1]);
                    dj2 = ~(SNPB_p1[p] | SNPB_p1[p + 1]);
                    dk2 = ~(SNPC_p1[p] | SNPC_p1[p + 1]);
                    
                    t000 = SNPA_p1[p] & SNPB_p1[p] & SNPC_p1[p]; 
                    t001 = SNPA_p1[p] & SNPB_p1[p] & SNPC_p1[p + 1];
                    t002 = SNPA_p1[p] & SNPB_p1[p] & dk2;
                    t010 = SNPA_p1[p] & SNPB_p1[p + 1] & SNPC_p1[p];
                    t011 = SNPA_p1[p] & SNPB_p1[p + 1] & SNPC_p1[p + 1];
                    t012 = SNPA_p1[p] & SNPB_p1[p + 1] & dk2;
                    t020 = SNPA_p1[p] & dj2 & SNPC_p1[p];
                    t021 = SNPA_p1[p] & dj2 & SNPC_p1[p + 1];
                    t022 = SNPA_p1[p] & dj2 & dk2;
                    t100 = SNPA_p1[p + 1] & SNPB_p1[p] & SNPC_p1[p];
                    t101 = SNPA_p1[p + 1] & SNPB_p1[p] & SNPC_p1[p + 1];
                    t102 = SNPA_p1[p + 1] & SNPB_p1[p] & dk2;
                    t110 = SNPA_p1[p + 1] & SNPB_p1[p + 1] & SNPC_p1[p];
                    t111 = SNPA_p1[p + 1] & SNPB_p1[p + 1] & SNPC_p1[p + 1];
                    t112 = SNPA_p1[p + 1] & SNPB_p1[p + 1] & dk2;
                    t120 = SNPA_p1[p + 1] & dj2 & SNPC_p1[p];
                    t121 = SNPA_p1[p + 1] & dj2 & SNPC_p1[p + 1];
                    t122 = SNPA_p1[p + 1] & dj2 & dk2;
                    t200 = di2 & SNPB_p1[p] & SNPC_p1[p];
                    t201 = di2 & SNPB_p1[p] & SNPC_p1[p + 1];
                    t202 = di2 & SNPB_p1[p] & dk2;
                    t210 = di2 & SNPB_p1[p + 1] & SNPC_p1[p];
                    t211 = di2 & SNPB_p1[p + 1] & SNPC_p1[p + 1];
                    t212 = di2 & SNPB_p1[p + 1] & dk2;
                    t220 = di2 & dj2 & SNPC_p1[p];
                    t221 = di2 & dj2 & SNPC_p1[p + 1];
                    t222 = di2 & dj2 & dk2;



                    ft[num_combs + 0] += _mm_popcnt_u32(t000);
                    ft[num_combs + 1] += _mm_popcnt_u32(t001);
                    ft[num_combs + 2] += _mm_popcnt_u32(t002);
                    ft[num_combs + 3] += _mm_popcnt_u32(t010);
                    ft[num_combs + 4] += _mm_popcnt_u32(t011);
                    ft[num_combs + 5] += _mm_popcnt_u32(t012);
                    ft[num_combs + 6] += _mm_popcnt_u32(t020);
                    ft[num_combs + 7] += _mm_popcnt_u32(t021);
                    ft[num_combs + 8] += _mm_popcnt_u32(t022);
                    ft[num_combs + 9] += _mm_popcnt_u32(t100);
                    ft[num_combs + 10] += _mm_popcnt_u32(t101);
                    ft[num_combs + 11] += _mm_popcnt_u32(t102);
                    ft[num_combs + 12] += _mm_popcnt_u32(t110);
                    ft[num_combs + 13] += _mm_popcnt_u32(t111);
                    ft[num_combs + 14] += _mm_popcnt_u32(t112);
                    ft[num_combs + 15] += _mm_popcnt_u32(t120);
                    ft[num_combs + 16] += _mm_popcnt_u32(t121);
                    ft[num_combs + 17] += _mm_popcnt_u32(t122);
                    ft[num_combs + 18] += _mm_popcnt_u32(t200);
                    ft[num_combs + 19] += _mm_popcnt_u32(t201);
                    ft[num_combs + 20] += _mm_popcnt_u32(t202);
                    ft[num_combs + 21] += _mm_popcnt_u32(t210);
                    ft[num_combs + 22] += _mm_popcnt_u32(t211);
                    ft[num_combs + 23] += _mm_popcnt_u32(t212);
                    ft[num_combs + 24] += _mm_popcnt_u32(t220);
                    ft[num_combs + 25] += _mm_popcnt_u32(t221);
                    ft[num_combs + 26] += _mm_popcnt_u32(t222);
                }
                //Do Remaining Elements
                di2 = ~(SNPA_p1[p] | SNPA_p1[p + 1]) & mask_ones;
                dj2 = ~(SNPB_p1[p] | SNPB_p1[p + 1]) & mask_ones;
                dk2 = ~(SNPC_p1[p] | SNPC_p1[p + 1]) & mask_ones;
                
                t000 = SNPA_p1[p] & SNPB_p1[p] & SNPC_p1[p]; 
                t001 = SNPA_p1[p] & SNPB_p1[p] & SNPC_p1[p + 1];
                t002 = SNPA_p1[p] & SNPB_p1[p] & dk2;
                t010 = SNPA_p1[p] & SNPB_p1[p + 1] & SNPC_p1[p];
                t011 = SNPA_p1[p] & SNPB_p1[p + 1] & SNPC_p1[p + 1];
                t012 = SNPA_p1[p] & SNPB_p1[p + 1] & dk2;
                t020 = SNPA_p1[p] & dj2 & SNPC_p1[p];
                t021 = SNPA_p1[p] & dj2 & SNPC_p1[p + 1];
                t022 = SNPA_p1[p] & dj2 & dk2;
                t100 = SNPA_p1[p + 1] & SNPB_p1[p] & SNPC_p1[p];
                t101 = SNPA_p1[p + 1] & SNPB_p1[p] & SNPC_p1[p + 1];
                t102 = SNPA_p1[p + 1] & SNPB_p1[p] & dk2;
                t110 = SNPA_p1[p + 1] & SNPB_p1[p + 1] & SNPC_p1[p];
                t111 = SNPA_p1[p + 1] & SNPB_p1[p + 1] & SNPC_p1[p + 1];
                t112 = SNPA_p1[p + 1] & SNPB_p1[p + 1] & dk2;
                t120 = SNPA_p1[p + 1] & dj2 & SNPC_p1[p];
                t121 = SNPA_p1[p + 1] & dj2 & SNPC_p1[p + 1];
                t122 = SNPA_p1[p + 1] & dj2 & dk2;
                t200 = di2 & SNPB_p1[p] & SNPC_p1[p];
                t201 = di2 & SNPB_p1[p] & SNPC_p1[p + 1];
                t202 = di2 & SNPB_p1[p] & dk2;
                t210 = di2 & SNPB_p1[p + 1] & SNPC_p1[p];
                t211 = di2 & SNPB_p1[p + 1] & SNPC_p1[p + 1];
                t212 = di2 & SNPB_p1[p + 1] & dk2;
                t220 = di2 & dj2 & SNPC_p1[p];
                t221 = di2 & dj2 & SNPC_p1[p + 1];
                t222 = di2 & dj2 & dk2;



                ft[num_combs + 0] += _mm_popcnt_u32(t000);
                ft[num_combs + 1] += _mm_popcnt_u32(t001);
                ft[num_combs + 2] += _mm_popcnt_u32(t002);
                ft[num_combs + 3] += _mm_popcnt_u32(t010);
                ft[num_combs + 4] += _mm_popcnt_u32(t011);
                ft[num_combs + 5] += _mm_popcnt_u32(t012);
                ft[num_combs + 6] += _mm_popcnt_u32(t020);
                ft[num_combs + 7] += _mm_popcnt_u32(t021);
                ft[num_combs + 8] += _mm_popcnt_u32(t022);
                ft[num_combs + 9] += _mm_popcnt_u32(t100);
                ft[num_combs + 10] += _mm_popcnt_u32(t101);
                ft[num_combs + 11] += _mm_popcnt_u32(t102);
                ft[num_combs + 12] += _mm_popcnt_u32(t110);
                ft[num_combs + 13] += _mm_popcnt_u32(t111);
                ft[num_combs + 14] += _mm_popcnt_u32(t112);
                ft[num_combs + 15] += _mm_popcnt_u32(t120);
                ft[num_combs + 16] += _mm_popcnt_u32(t121);
                ft[num_combs + 17] += _mm_popcnt_u32(t122);
                ft[num_combs + 18] += _mm_popcnt_u32(t200);
                ft[num_combs + 19] += _mm_popcnt_u32(t201);
                ft[num_combs + 20] += _mm_popcnt_u32(t202);
                ft[num_combs + 21] += _mm_popcnt_u32(t210);
                ft[num_combs + 22] += _mm_popcnt_u32(t211);
                ft[num_combs + 23] += _mm_popcnt_u32(t212);
                ft[num_combs + 24] += _mm_popcnt_u32(t220);
                ft[num_combs + 25] += _mm_popcnt_u32(t221);
                ft[num_combs + 26] += _mm_popcnt_u32(t222);

                //Objective Function
                float score = 0.0;
                for(m = 0; m < num_combs; m++)
                    score += addlog(ft[m] + ft[num_combs + m] + 1) - addlog(ft[m]) - addlog(ft[num_combs + m]);
                score = fabs(score);

                // compare score
                if(score < best_score_local){
                    best_score_local = score;
                    best_snp_global[tid][0] = i;
                    best_snp_global[tid][1] = j;
                    best_snp_global[tid][2] = k;
                }
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
    
    cyc_e_local = read_tsc_end();
    
    serialize();

    cyc_s = cyc_s_local;
    cyc_e = cyc_e_local;

    _mm_free(ft);
    
    }
    
    printf("Time: %f\n", (double) (cyc_e - cyc_s)/FREQ);

    printf("Best SNPs: %d, %d, %d - Score: %f\n", best_snp[0], best_snp[1], best_snp[2], best_score);
}

int main(int argc, char **argv){

    int long long num_snp, num_pac;
    int dim_epi;
    int block_snp, block_pac;
    int i, addlogsize;

    uint8_t *SNP_Data; 
    uint8_t *Ph_Data;
    uint32_t *bin_data_ones, *bin_data_zeros;
    int long long phen_ones;

    dim_epi = 3;
    num_pac = atol(argv[1]);
    num_snp = atol(argv[2]);

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

    process_epi_bin_no_phen_nor(bin_data_zeros, bin_data_ones, phen_ones, dim_epi, num_snp, num_pac, comb);

    _mm_free(bin_data_zeros);
    _mm_free(bin_data_ones);

    delete addlogtable;

    return 0;
}

