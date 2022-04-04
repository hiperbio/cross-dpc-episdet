// V1

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

void transposed_to_binary(uint8_t* original, uint8_t* original_ph, uint32_t** data_f, uint32_t** phen_f, int long long num_snp, int long long num_pac)
{
    int PP = ceil((1.0f * num_pac) / 32.0f);
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

int main(int argc, char **argv)
{
    int num_snp, num_snp_m, num_pac, PP;
    int dim_epi;
    int block_snp, block_pac;
    int x;

    uint8_t* SNP_Data; 
    uint8_t* Ph_Data;
    uint32_t* bin_data;
    uint32_t* bin_phen;

    dim_epi = 3;
    num_pac = atoi(argv[1]);
    num_snp = atoi(argv[2]);
    block_snp = 64;
    PP = ceil(1.0f * num_pac / 32.0f);
    num_snp_m = num_snp;
    while(num_snp_m % block_snp != 0)
        num_snp_m++;

    int comb = (int) pow(3.0f, dim_epi);

    generate_data(num_pac, num_snp, &SNP_Data, &Ph_Data);
    SNP_Data = transpose_data(num_pac, num_snp, SNP_Data);
    transposed_to_binary(SNP_Data, Ph_Data, &bin_data, &bin_phen, num_snp, num_pac);

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
    uint* dev_data = (uint*) cl::sycl::malloc_shared(num_snp * PP * 3 * sizeof(uint), device, context);
    uint* dev_phen = (uint*) cl::sycl::malloc_shared(PP * sizeof(uint), device, context);
    float* dev_scores = (float*) cl::sycl::malloc_shared(num_snp_m * num_snp * num_snp * sizeof(float), device, context);
    float* dev_addlogtable = (float*) cl::sycl::malloc_shared(TABLE_MAX_SIZE * sizeof(float), device, context);

    // create addlog table
	for(x = 0; x < TABLE_MAX_SIZE; x++)
		dev_addlogtable[x] = my_factorial(x, dev_addlogtable);

    // copy buffers from host to device
    for(x = 0; x < num_snp_m * num_snp * num_snp; x++)
        dev_scores[x] = FLT_MAX;
	for(x = 0; x < num_snp * PP * 3; x++)
        dev_data[x] = bin_data[x];
    for(x = 0; x < PP; x++)
        dev_phen[x] = bin_phen[x];
    
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
            int i, j, t, tid, igt, jgt, tgt, p, k;
            int index;
            float score = FLT_MAX;

            i = id.get_global_id(0);
            j = id.get_global_id(1);
            t = id.get_global_id(2);
            tid = i * num_snp_m * num_snp + j * num_snp + t;

            if(j > i && t > j && t < num_snp)
            {
                // create frequency table
                uint ft[27][2];
                for(k = 0; k < comb; k++)
                {
                    ft[k][0] = 0;
                    ft[k][1] = 0;
                }
                
                // get pointer to data
                uint* SNPi_ini = &dev_data[i * PP * 3];
                uint* SNPj_ini = &dev_data[j * PP * 3];
                uint* SNPt_ini = &dev_data[t * PP * 3];
                for(p = 0; p < PP; p++)
                {   
                    index = 0;
                    uint* SNPi = &SNPi_ini[p * 3];
                    uint* SNPj = &SNPj_ini[p * 3];
                    uint* SNPt = &SNPt_ini[p * 3];
                    uint state = dev_phen[p];
                    for(igt = 0; igt < 3; igt++)
                    {
                        for(jgt = 0; jgt < 3; jgt++)
                        {
                            for(tgt = 0; tgt < 3; tgt++)
                            {                    
                                uint res = SNPi[igt] & SNPj[jgt] & SNPt[tgt];
                                uint res0 = res & ~state;
                                uint res1 = res & state;
                                ft[index][0] += cl::sycl::popcount(res0);
                                ft[index][1] += cl::sycl::popcount(res1);
                                index++;
                            }
                        }
                    }
                }
                // compute score
                score = 0.0f;
                for(k = 0; k < comb; k++)
                    score += gammafunction(ft[k][0] + ft[k][1] + 1, dev_addlogtable) - gammafunction(ft[k][0], dev_addlogtable) - gammafunction(ft[k][1], dev_addlogtable);
                score = cl::sycl::fabs(score);
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
    _mm_free(bin_data);
    _mm_free(bin_phen);
    cl::sycl::free(dev_data, context);
    cl::sycl::free(dev_phen, context);
    cl::sycl::free(dev_scores, context);
    cl::sycl::free(dev_addlogtable, context);

    return 0;
}
