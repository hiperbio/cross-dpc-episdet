// V2

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include <float.h>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>

#define TABLE_MAX_SIZE 748
#define TABLE_ERROR -0.0810
#define FREQ 3500000000 

static inline long long read_tsc_start()
{
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

static inline long long read_tsc_end()
{
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

float gammafunction(int n, float* addlogtable)
{
	if(n < TABLE_MAX_SIZE)
		return addlogtable[n];
	else
	{
		float x = (n + 0.5f) * cl::sycl::log((float) n) - (n - 1) * cl::sycl::log(cl::sycl::exp((float) 1.0f)) + TABLE_ERROR;
		return x;
	}
}

float my_factorial(int n, float* addlogtable)
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

uint8_t * transpose_data(int long long N, int long long M, uint8_t * A){
    int i, j;
    
    uint8_t *A_trans = (uint8_t *) _mm_malloc(M*N*sizeof(uint8_t), 64);
    
    for (i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            A_trans[j*N + i] = A[i*M+ + j];
        }
    }

    return A_trans;
}

void transposed_to_binary(uint8_t* original, uint8_t* original_ph, uint32_t** data_zeros, uint32_t** data_ones, int* phen_ones, int num_snp, int num_pac)
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

int main(int argc, char **argv)
{
    int num_snp, num_snp_m, num_pac, phen_ones;
    int dim_epi;
    int block_snp, block_pac;
    int x;

    uint8_t* SNP_Data; 
    uint8_t* Ph_Data;
    uint32_t* bin_data_ones;
    uint32_t* bin_data_zeros;

    dim_epi = 3;
    num_pac = atoi(argv[1]);
    num_snp = atoi(argv[2]);
    block_snp = 64;

    generate_data(num_pac, num_snp, &SNP_Data, &Ph_Data);
    SNP_Data = transpose_data(num_pac, num_snp, SNP_Data);
    transposed_to_binary(SNP_Data, Ph_Data, &bin_data_zeros, &bin_data_ones, &phen_ones, num_snp, num_pac);

    int PP = ceil(1.0*num_pac/32.0);
    int PP_ones = ceil((1.0 * phen_ones)/32.0);
    int PP_zeros = ceil((1.0*(num_pac - phen_ones))/32.0);
    int comb = (int)pow(3.0, dim_epi);
    num_snp_m = num_snp;
    while(num_snp_m % block_snp != 0)
        num_snp_m++;

    uint mask_zeros, mask_ones;
    mask_zeros = 0xFFFFFFFF;
    for(x = num_pac - phen_ones; x < PP_zeros * 32; x++)
        mask_zeros = mask_zeros >> 1;
    mask_ones = 0xFFFFFFFF;
    for(x = phen_ones; x < PP_ones * 32; x++)
        mask_ones = mask_ones >> 1;

    ///////////////////////////////

    // create command queue (enable profiling and use GPU as device)
    auto property_list = cl::sycl::property_list{cl::sycl::property::queue::enable_profiling(), cl::sycl::property::queue::in_order()};
    auto device_selector = cl::sycl::gpu_selector{};
    cl::sycl::queue queue(device_selector, property_list);

    // get device and context
    auto device = queue.get_device();
    auto context = queue.get_context();
    std::cout << "Selected device: " << device.get_info<cl::sycl::info::device::name>() << std::endl;
    
    // create USM buffers
    uint* dev_data_zeros = (uint*) cl::sycl::malloc_shared(num_snp * PP_zeros * 2 * sizeof(uint), device, context);
    uint* dev_data_ones  = (uint*) cl::sycl::malloc_shared(num_snp * PP_ones  * 2 * sizeof(uint), device, context);
    float* dev_scores = (float*) cl::sycl::malloc_shared(num_snp_m * num_snp * num_snp * sizeof(float), device, context);
    float* dev_addlogtable = (float*) cl::sycl::malloc_shared(TABLE_MAX_SIZE * sizeof(float), device, context);

    // create addlog table
	for(x = 0; x < TABLE_MAX_SIZE; x++)
		dev_addlogtable[x] = my_factorial(x, dev_addlogtable);

    // copy buffers from host to device
    for(x = 0; x < num_snp_m * num_snp * num_snp; x++)
        dev_scores[x] = FLT_MAX;
	for(x = 0; x < num_snp * PP_zeros * 2; x++)
        dev_data_zeros[x] = bin_data_zeros[x];
    for(x = 0; x < num_snp * PP_ones * 2; x++)
        dev_data_ones[x] = bin_data_ones[x];
    
    // setup kernel ND-range
    cl::sycl::range<3> global_epi(num_snp_m, num_snp, num_snp);
    cl::sycl::range<3> local_epi(block_snp, 1, 1);

    // DPC++ kernel call
    queue.wait();
    double start = read_tsc_start();
    queue.submit([&](cl::sycl::handler& h)
    {
        h.parallel_for<class kernel_epi>(cl::sycl::nd_range<3>(global_epi, local_epi), [=](cl::sycl::nd_item<3> id)
        {
            int i, j, t, tid, p, k;
            float score = FLT_MAX;

            i = id.get_global_id(0);
            j = id.get_global_id(1);
            t = id.get_global_id(2);
            tid = i * num_snp_m * num_snp + j * num_snp + t;

            if(j > i && t > j && t < num_snp)
            {
                uint t000, t001, t002, t010, t011, t012, t020, t021, t022,
                    t100, t101, t102, t110, t111, t112, t120, t121, t122,
                    t200, t201, t202, t210, t211, t212, t220, t221, t222;
                uint di2, dj2, dt2;

                // create frequency table
                uint ft[2 * 27];
                for(k = 0; k < 2 * 27; k++)
                    ft[k] = 0;
                
                // Phenotype 0
                uint* SNPi = &dev_data_zeros[i * PP_zeros * 2];
                uint* SNPj = &dev_data_zeros[j * PP_zeros * 2];
                uint* SNPt = &dev_data_zeros[t * PP_zeros * 2];
                for(p = 0; p < 2 * PP_zeros - 2; p += 2)
                {           
                    di2 = ~(SNPi[p] | SNPi[p + 1]);
                    dj2 = ~(SNPj[p] | SNPj[p + 1]);
                    dt2 = ~(SNPt[p] | SNPt[p + 1]);

                    t000 = SNPi[p] & SNPj[p] & SNPt[p];
                    t001 = SNPi[p] & SNPj[p] & SNPt[p + 1];
                    t002 = SNPi[p] & SNPj[p] & dt2;
                    t010 = SNPi[p] & SNPj[p + 1] & SNPt[p];
                    t011 = SNPi[p] & SNPj[p + 1] & SNPt[p + 1];
                    t012 = SNPi[p] & SNPj[p + 1] & dt2;
                    t020 = SNPi[p] & dj2 & SNPt[p];
                    t021 = SNPi[p] & dj2 & SNPt[p + 1];
                    t022 = SNPi[p] & dj2 & dt2;
                    t100 = SNPi[p + 1] & SNPj[p] & SNPt[p];
                    t101 = SNPi[p + 1] & SNPj[p] & SNPt[p + 1];
                    t102 = SNPi[p + 1] & SNPj[p] & dt2;
                    t110 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p];
                    t111 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p + 1];
                    t112 = SNPi[p + 1] & SNPj[p + 1] & dt2;
                    t120 = SNPi[p + 1] & dj2 & SNPt[p];
                    t121 = SNPi[p + 1] & dj2 & SNPt[p + 1];
                    t122 = SNPi[p + 1] & dj2 & dt2;
                    t200 = di2 & SNPj[p] & SNPt[p];
                    t201 = di2 & SNPj[p] & SNPt[p + 1];
                    t202 = di2 & SNPj[p] & dt2;
                    t210 = di2 & SNPj[p + 1] & SNPt[p];
                    t211 = di2 & SNPj[p + 1] & SNPt[p + 1];
                    t212 = di2 & SNPj[p + 1] & dt2;
                    t220 = di2 & dj2 & SNPt[p];
                    t221 = di2 & dj2 & SNPt[p + 1];
                    t222 = di2 & dj2 & dt2;         

                    ft[0]  += cl::sycl::popcount(t000);
                    ft[1]  += cl::sycl::popcount(t001);
                    ft[2]  += cl::sycl::popcount(t002);
                    ft[3]  += cl::sycl::popcount(t010);
                    ft[4]  += cl::sycl::popcount(t011);
                    ft[5]  += cl::sycl::popcount(t012);
                    ft[6]  += cl::sycl::popcount(t020);
                    ft[7]  += cl::sycl::popcount(t021);
                    ft[8]  += cl::sycl::popcount(t022);
                    ft[9]  += cl::sycl::popcount(t100);
                    ft[10] += cl::sycl::popcount(t101);
                    ft[11] += cl::sycl::popcount(t102);
                    ft[12] += cl::sycl::popcount(t110);
                    ft[13] += cl::sycl::popcount(t111);
                    ft[14] += cl::sycl::popcount(t112);
                    ft[15] += cl::sycl::popcount(t120);
                    ft[16] += cl::sycl::popcount(t121);
                    ft[17] += cl::sycl::popcount(t122);
                    ft[18] += cl::sycl::popcount(t200);
                    ft[19] += cl::sycl::popcount(t201);
                    ft[20] += cl::sycl::popcount(t202);
                    ft[21] += cl::sycl::popcount(t210);
                    ft[22] += cl::sycl::popcount(t211);
                    ft[23] += cl::sycl::popcount(t212);
                    ft[24] += cl::sycl::popcount(t220);
                    ft[25] += cl::sycl::popcount(t221);
                    ft[26] += cl::sycl::popcount(t222);
                }
                di2 = ~(SNPi[p] | SNPi[p + 1]);
                dj2 = ~(SNPj[p] | SNPj[p + 1]);
                dt2 = ~(SNPt[p] | SNPt[p + 1]);
                di2 = di2 & mask_zeros;
                dj2 = dj2 & mask_zeros;
                dt2 = dt2 & mask_zeros;

                t000 = SNPi[p] & SNPj[p] & SNPt[p];
                t001 = SNPi[p] & SNPj[p] & SNPt[p + 1];
                t002 = SNPi[p] & SNPj[p] & dt2;
                t010 = SNPi[p] & SNPj[p + 1] & SNPt[p];
                t011 = SNPi[p] & SNPj[p + 1] & SNPt[p + 1];
                t012 = SNPi[p] & SNPj[p + 1] & dt2;
                t020 = SNPi[p] & dj2 & SNPt[p];
                t021 = SNPi[p] & dj2 & SNPt[p + 1];
                t022 = SNPi[p] & dj2 & dt2;
                t100 = SNPi[p + 1] & SNPj[p] & SNPt[p];
                t101 = SNPi[p + 1] & SNPj[p] & SNPt[p + 1];
                t102 = SNPi[p + 1] & SNPj[p] & dt2;
                t110 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p];
                t111 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p + 1];
                t112 = SNPi[p + 1] & SNPj[p + 1] & dt2;
                t120 = SNPi[p + 1] & dj2 & SNPt[p];
                t121 = SNPi[p + 1] & dj2 & SNPt[p + 1];
                t122 = SNPi[p + 1] & dj2 & dt2;
                t200 = di2 & SNPj[p] & SNPt[p];
                t201 = di2 & SNPj[p] & SNPt[p + 1];
                t202 = di2 & SNPj[p] & dt2;
                t210 = di2 & SNPj[p + 1] & SNPt[p];
                t211 = di2 & SNPj[p + 1] & SNPt[p + 1];
                t212 = di2 & SNPj[p + 1] & dt2;
                t220 = di2 & dj2 & SNPt[p];
                t221 = di2 & dj2 & SNPt[p + 1];
                t222 = di2 & dj2 & dt2;         

                ft[0]  += cl::sycl::popcount(t000);
                ft[1]  += cl::sycl::popcount(t001);
                ft[2]  += cl::sycl::popcount(t002);
                ft[3]  += cl::sycl::popcount(t010);
                ft[4]  += cl::sycl::popcount(t011);
                ft[5]  += cl::sycl::popcount(t012);
                ft[6]  += cl::sycl::popcount(t020);
                ft[7]  += cl::sycl::popcount(t021);
                ft[8]  += cl::sycl::popcount(t022);
                ft[9]  += cl::sycl::popcount(t100);
                ft[10] += cl::sycl::popcount(t101);
                ft[11] += cl::sycl::popcount(t102);
                ft[12] += cl::sycl::popcount(t110);
                ft[13] += cl::sycl::popcount(t111);
                ft[14] += cl::sycl::popcount(t112);
                ft[15] += cl::sycl::popcount(t120);
                ft[16] += cl::sycl::popcount(t121);
                ft[17] += cl::sycl::popcount(t122);
                ft[18] += cl::sycl::popcount(t200);
                ft[19] += cl::sycl::popcount(t201);
                ft[20] += cl::sycl::popcount(t202);
                ft[21] += cl::sycl::popcount(t210);
                ft[22] += cl::sycl::popcount(t211);
                ft[23] += cl::sycl::popcount(t212);
                ft[24] += cl::sycl::popcount(t220);
                ft[25] += cl::sycl::popcount(t221);
                ft[26] += cl::sycl::popcount(t222);

                // Phenotype 1
                SNPi = &dev_data_ones[i * PP_ones * 2];
                SNPj = &dev_data_ones[j * PP_ones * 2];
                SNPt = &dev_data_ones[t * PP_ones * 2];
                for(p = 0; p < 2 * PP_ones - 2; p += 2)
                {           
                    di2 = ~(SNPi[p] | SNPi[p + 1]);
                    dj2 = ~(SNPj[p] | SNPj[p + 1]);
                    dt2 = ~(SNPt[p] | SNPt[p + 1]);

                    t000 = SNPi[p] & SNPj[p] & SNPt[p];
                    t001 = SNPi[p] & SNPj[p] & SNPt[p + 1];
                    t002 = SNPi[p] & SNPj[p] & dt2;
                    t010 = SNPi[p] & SNPj[p + 1] & SNPt[p];
                    t011 = SNPi[p] & SNPj[p + 1] & SNPt[p + 1];
                    t012 = SNPi[p] & SNPj[p + 1] & dt2;
                    t020 = SNPi[p] & dj2 & SNPt[p];
                    t021 = SNPi[p] & dj2 & SNPt[p + 1];
                    t022 = SNPi[p] & dj2 & dt2;
                    t100 = SNPi[p + 1] & SNPj[p] & SNPt[p];
                    t101 = SNPi[p + 1] & SNPj[p] & SNPt[p + 1];
                    t102 = SNPi[p + 1] & SNPj[p] & dt2;
                    t110 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p];
                    t111 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p + 1];
                    t112 = SNPi[p + 1] & SNPj[p + 1] & dt2;
                    t120 = SNPi[p + 1] & dj2 & SNPt[p];
                    t121 = SNPi[p + 1] & dj2 & SNPt[p + 1];
                    t122 = SNPi[p + 1] & dj2 & dt2;
                    t200 = di2 & SNPj[p] & SNPt[p];
                    t201 = di2 & SNPj[p] & SNPt[p + 1];
                    t202 = di2 & SNPj[p] & dt2;
                    t210 = di2 & SNPj[p + 1] & SNPt[p];
                    t211 = di2 & SNPj[p + 1] & SNPt[p + 1];
                    t212 = di2 & SNPj[p + 1] & dt2;
                    t220 = di2 & dj2 & SNPt[p];
                    t221 = di2 & dj2 & SNPt[p + 1];
                    t222 = di2 & dj2 & dt2;

                    ft[27] += cl::sycl::popcount(t000);
                    ft[28] += cl::sycl::popcount(t001);
                    ft[29] += cl::sycl::popcount(t002);
                    ft[30] += cl::sycl::popcount(t010);
                    ft[31] += cl::sycl::popcount(t011);
                    ft[32] += cl::sycl::popcount(t012);
                    ft[33] += cl::sycl::popcount(t020);
                    ft[34] += cl::sycl::popcount(t021);
                    ft[35] += cl::sycl::popcount(t022);
                    ft[36] += cl::sycl::popcount(t100);
                    ft[37] += cl::sycl::popcount(t101);
                    ft[38] += cl::sycl::popcount(t102);
                    ft[39] += cl::sycl::popcount(t110);
                    ft[40] += cl::sycl::popcount(t111);
                    ft[41] += cl::sycl::popcount(t112);
                    ft[42] += cl::sycl::popcount(t120);
                    ft[43] += cl::sycl::popcount(t121);
                    ft[44] += cl::sycl::popcount(t122);
                    ft[45] += cl::sycl::popcount(t200);
                    ft[46] += cl::sycl::popcount(t201);
                    ft[47] += cl::sycl::popcount(t202);
                    ft[48] += cl::sycl::popcount(t210);
                    ft[49] += cl::sycl::popcount(t211);
                    ft[50] += cl::sycl::popcount(t212);
                    ft[51] += cl::sycl::popcount(t220);
                    ft[52] += cl::sycl::popcount(t221);
                    ft[53] += cl::sycl::popcount(t222);
                }
                di2 = ~(SNPi[p] | SNPi[p + 1]);
                dj2 = ~(SNPj[p] | SNPj[p + 1]);
                dt2 = ~(SNPt[p] | SNPt[p + 1]);
                di2 = di2 & mask_ones;
                dj2 = dj2 & mask_ones;
                dt2 = dt2 & mask_ones;

                t000 = SNPi[p] & SNPj[p] & SNPt[p];
                t001 = SNPi[p] & SNPj[p] & SNPt[p + 1];
                t002 = SNPi[p] & SNPj[p] & dt2;
                t010 = SNPi[p] & SNPj[p + 1] & SNPt[p];
                t011 = SNPi[p] & SNPj[p + 1] & SNPt[p + 1];
                t012 = SNPi[p] & SNPj[p + 1] & dt2;
                t020 = SNPi[p] & dj2 & SNPt[p];
                t021 = SNPi[p] & dj2 & SNPt[p + 1];
                t022 = SNPi[p] & dj2 & dt2;
                t100 = SNPi[p + 1] & SNPj[p] & SNPt[p];
                t101 = SNPi[p + 1] & SNPj[p] & SNPt[p + 1];
                t102 = SNPi[p + 1] & SNPj[p] & dt2;
                t110 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p];
                t111 = SNPi[p + 1] & SNPj[p + 1] & SNPt[p + 1];
                t112 = SNPi[p + 1] & SNPj[p + 1] & dt2;
                t120 = SNPi[p + 1] & dj2 & SNPt[p];
                t121 = SNPi[p + 1] & dj2 & SNPt[p + 1];
                t122 = SNPi[p + 1] & dj2 & dt2;
                t200 = di2 & SNPj[p] & SNPt[p];
                t201 = di2 & SNPj[p] & SNPt[p + 1];
                t202 = di2 & SNPj[p] & dt2;
                t210 = di2 & SNPj[p + 1] & SNPt[p];
                t211 = di2 & SNPj[p + 1] & SNPt[p + 1];
                t212 = di2 & SNPj[p + 1] & dt2;
                t220 = di2 & dj2 & SNPt[p];
                t221 = di2 & dj2 & SNPt[p + 1];
                t222 = di2 & dj2 & dt2;

                ft[27] += cl::sycl::popcount(t000);
                ft[28] += cl::sycl::popcount(t001);
                ft[29] += cl::sycl::popcount(t002);
                ft[30] += cl::sycl::popcount(t010);
                ft[31] += cl::sycl::popcount(t011);
                ft[32] += cl::sycl::popcount(t012);
                ft[33] += cl::sycl::popcount(t020);
                ft[34] += cl::sycl::popcount(t021);
                ft[35] += cl::sycl::popcount(t022);
                ft[36] += cl::sycl::popcount(t100);
                ft[37] += cl::sycl::popcount(t101);
                ft[38] += cl::sycl::popcount(t102);
                ft[39] += cl::sycl::popcount(t110);
                ft[40] += cl::sycl::popcount(t111);
                ft[41] += cl::sycl::popcount(t112);
                ft[42] += cl::sycl::popcount(t120);
                ft[43] += cl::sycl::popcount(t121);
                ft[44] += cl::sycl::popcount(t122);
                ft[45] += cl::sycl::popcount(t200);
                ft[46] += cl::sycl::popcount(t201);
                ft[47] += cl::sycl::popcount(t202);
                ft[48] += cl::sycl::popcount(t210);
                ft[49] += cl::sycl::popcount(t211);
                ft[50] += cl::sycl::popcount(t212);
                ft[51] += cl::sycl::popcount(t220);
                ft[52] += cl::sycl::popcount(t221);
                ft[53] += cl::sycl::popcount(t222);

                // compute score
                score = 0.0;
                for(k = 0; k < 27; k++)
                    score += gammafunction(ft[k] + ft[27 + k] + 1, dev_addlogtable) - gammafunction(ft[k], dev_addlogtable) - gammafunction(ft[27 + k], dev_addlogtable);
                score = cl::sycl::fabs((float) score);
                if(score <= 0)
                    score = FLT_MAX;
                dev_scores[tid] = score;
            }
            // end kernel
        });
    });
    queue.wait();
    
    float score = FLT_MAX;
    uint solution;
    for(x = 0; x < num_snp_m * num_snp * num_snp; x++)
    {
        if(dev_scores[x] < score)
        {
            score = dev_scores[x];
            solution = x;
        }
    }
    double end = read_tsc_end();
    std::cout << "Time: " << (double) (end - start)/FREQ << std::endl;
    std::cout << "Score: " << score << std::endl;
std::cout << "Solution: " << solution / (num_snp_m * num_snp) << " " << (solution % (num_snp_m * num_snp)) / num_snp << " " << (solution % (num_snp_m * num_snp)) % num_snp << std::endl;

    _mm_free(SNP_Data);
    _mm_free(Ph_Data);
    _mm_free(bin_data_zeros);
    _mm_free(bin_data_ones);
    cl::sycl::free(dev_data_zeros, context);
    cl::sycl::free(dev_data_ones, context);
    cl::sycl::free(dev_scores, context);
    cl::sycl::free(dev_addlogtable, context);

    return 0;
}
