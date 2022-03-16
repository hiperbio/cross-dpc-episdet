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

void transposed_to_binary(uint8_t* original, uint8_t* original_ph, uint32_t** data_f, uint32_t** phen_f, int long long num_snp, int long long num_pac)
{
    int PP = ceil((1.0*num_pac)/32.0);
    uint32_t temp;
    int i, j, x;

    // allocate data
    uint32_t *data = (uint32_t*) _mm_malloc(num_snp * PP * 3  * sizeof(uint32_t), 64);
    uint32_t *phen = (uint32_t*) _mm_malloc(PP * sizeof(uint32_t), 64);
    memset(data, 0, num_snp * PP * 3 * sizeof(uint32_t));
    memset(phen, 0, PP * sizeof(uint32_t));

    for(i = 0; i < num_snp; i++)
    {
        int new_j = 0;
        for(j = 0; j < num_pac; j += 32)
        {
            data[(i * PP + new_j) * 3 + 0] = 0;
            data[(i * PP + new_j) * 3 + 1] = 0;
            data[(i * PP + new_j) * 3 + 2] = 0;
            for(x = j; x < num_pac && x < j + 32; x++)
            {
                // apply 1 shift left to 3 components
                data[(i * PP + new_j) * 3 + 0] <<= 1;
                data[(i * PP + new_j) * 3 + 1] <<= 1;
                data[(i * PP + new_j) * 3 + 2] <<= 1;
                // insert '1' in correct component
                temp = (uint32_t) original[i * num_pac + x];
                data[(i * PP + new_j) * 3 + temp] |= 1;
            }
            new_j++;
        }

    }
    int new_j = 0;
    for(j = 0; j < num_pac; j += 32)
    {
        phen[new_j] = 0;
        for(x = j; x < num_pac && x < j + 32; x++)
        {
            // apply 1 shift left
            phen[new_j] <<= 1;
            // insert new value
            phen[new_j] |= original_ph[x];
        }
        new_j++;
    }

    *data_f = data;
    *phen_f = phen;
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

void process_epi_bin(uint32_t* data, uint32_t* phen, int dim_epi, int long long num_snp, int long long num_pac, int num_combs)
{
    
    
    float best_score = FLT_MAX;
    int long long cyc_s = LLONG_MAX, cyc_e = 0;
    omp_set_num_threads(NUM_THREADS);
    int best_snp_global[NUM_THREADS][3];
    int best_snp[3];

    #pragma omp parallel reduction(min:cyc_s) reduction(max:cyc_e)
    {

    uint32_t *SNPA, *SNPB, *SNPC;

    int tid = omp_get_thread_num();

    //Generate Frequency Table
    uint32_t * ft = (uint32_t *) _mm_malloc(2*num_combs*sizeof(uint32_t), 64);
    memset(ft, 0, 2*num_combs*sizeof(uint32_t));

    float best_score_local = FLT_MAX;
    int long long cyc_s_local, cyc_e_local;

    int PP = ceil((1.0*num_pac)/32.0);
    int i, j, k, p, m;
    int igt, jgt, kgt;
    int aux_i, aux_j, aux_k;
    int index;

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
    for(i = 0; i < num_snp - 2; i++){
        aux_i = i*PP*3;
        for(j = i + 1; j < num_snp-1; j++){
            aux_j = j*PP*3;
            for(k = j + 1; k < num_snp; k++){
                aux_k = k*PP*3;
                
                // reset frequency table
                memset(ft, 0, 2*num_combs*sizeof(uint32_t));
                
                for(p = 0; p < PP; p++){

                    uint32_t state = phen[p]; 
                    SNPA = &data[aux_i + 3*p];
                    SNPB = &data[aux_j + 3*p];
                    SNPC = &data[aux_k + 3*p];
                    index = 0;
                    for(igt = 0; igt < 3; igt++){
                        for(jgt = 0; jgt < 3; jgt++){
                            for(kgt = 0; kgt < 3; kgt++){                   
                                int res = SNPA[igt] & SNPB[jgt] & SNPC[kgt];                        
                                int res0 = res & ~state;                 
                                int res1 = res & state;                  
                                ft[index] += _mm_popcnt_u32(res0);            
                                ft[num_combs + index] += _mm_popcnt_u32(res1);
                                index++;              
                            }      
                        }
                    }
                }

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

    _mm_free(ft);

    }
    
    printf("Time bin: %f\n", (double) (cyc_e - cyc_s)/FREQ);

    printf("bin Best: %d, %d, %d - Score: %f\n", best_snp[0], best_snp[1], best_snp[2], best_score);

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
    uint32_t *bin_data, *bin_phen;

    dim_epi = atoi(argv[1]);
    num_pac = atol(argv[2]);
    num_snp = atol(argv[3]);

    int comb = (int)pow(3.0, dim_epi);

    generate_data(num_pac, num_snp, &SNP_Data, &Ph_Data);

    // create addlog table (up to TABLE_MAX_SIZE positions at max)
	addlogsize = TABLE_MAX_SIZE;
	addlogtable = new float[addlogsize];
	for(i = 0; i < addlogsize; i++)
		addlogtable[i] = my_factorial(i);

    SNP_Data = transpose_data(num_pac, num_snp, SNP_Data);
    
    transposed_to_binary(SNP_Data, Ph_Data, &bin_data, &bin_phen, num_snp, num_pac);

    _mm_free(SNP_Data);
    _mm_free(Ph_Data);

    process_epi_bin(bin_data, bin_phen, dim_epi, num_snp, num_pac, comb);

    _mm_free(bin_data);
    _mm_free(bin_phen);

    delete addlogtable;

    #if defined (LIKWID)
        LIKWID_MARKER_CLOSE;
    #endif

    return 0;
}

