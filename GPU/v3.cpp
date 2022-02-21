// V3

#define SYCL_SIMPLE_SWIZZLES
#define SYCL_INTEL_group_algorithms	
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
#include <omp.h>

float gammafunction(uint n)
{   
    if(n == 0)
        return 0.0f;
    float x = (n + 0.5f) * cl::sycl::log((float) n) - (n - 1) * cl::sycl::log(cl::sycl::exp((float) 1.0f));
    return x;
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

    // generate phenotype
    for(i = 0; i < N; i++)
    {
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
    auto device_selector = cl::sycl::cpu_selector{};
    cl::sycl::queue queue(device_selector, property_list);

    // get device and context
    auto device = queue.get_device();
    auto context = queue.get_context();
    std::cout << "Selected device: " << device.get_info<cl::sycl::info::device::name>() << std::endl;

    ///////////////////////////////
    
    // create USM buffers
    uint* dev_data = (uint*) cl::sycl::malloc_shared(num_snp * PP * 3 * sizeof(uint), device, context);
    uint* dev_phen = (uint*) cl::sycl::malloc_shared(PP * sizeof(uint), device, context);
    float* dev_scores = (float*) cl::sycl::malloc_shared(num_snp_m * num_snp * num_snp * sizeof(float), device, context);

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
    cl::sycl::range<1> global_red(256*3);
    cl::sycl::range<1> local_red(256);

    // DPC++ kernel call
    queue.wait();
    double start = omp_get_wtime();
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
                    for(igt = 0; igt < 3; igt++)
                    {
                        for(jgt = 0; jgt < 3; jgt++)
                        {
                            for(tgt = 0; tgt < 3; tgt++)
                            {                    
                                uint res = SNPi[igt] & SNPj[jgt] & SNPt[tgt];
                                uint state = dev_phen[p];
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
                    score += gammafunction(ft[k][0] + ft[k][1] + 1) - gammafunction(ft[k][0]) - gammafunction(ft[k][1]);
                score = cl::sycl::fabs(score);
                if(score <= 0)
                    score = FLT_MAX;
                dev_scores[tid] = score;
            }
            // end kernel
        });
    });
    
    queue.submit([&](cl::sycl::handler& h)
    {
        h.parallel_for<class kernel_red>(cl::sycl::nd_range<1>(global_red, local_red), [=](cl::sycl::nd_item<1> id)
        {
            cl::sycl::group<1> gr = id.get_group();
            int size = num_snp_m * num_snp * num_snp;
            int lid = id.get_local_id(0);
            int grid = gr.get_id(0);
            int gid = id.get_global_id(0);
            int gsize = 256*3;
            int i;
            float a, b;

            a = dev_scores[gid];
            for(i = gid + gsize; i < size; i += gsize)
            {
                b = dev_scores[i];
                a = (float) cl::sycl::fmin(a, b);
            }
            id.barrier(cl::sycl::access::fence_space::local_space);
            a = sycl::ONEAPI::reduce<cl::sycl::group<1>, float, sycl::ONEAPI::minimum<float>>(gr, a, sycl::ONEAPI::minimum<float>());
            id.barrier(cl::sycl::access::fence_space::local_space);

            if(lid == 0)
                dev_scores[grid] = a;
            // end kernel
        });
    });
    
    queue.wait();
    
    float score = FLT_MAX;
    for(x = 0; x < 3; x++)
    {
        if(dev_scores[x] < score)
            score = dev_scores[x];
    }
    double end = omp_get_wtime();
    std::cout << "Time: " << end - start << std::endl;
    std::cout << "Score: " << score << std::endl;

    _mm_free(SNP_Data);
    _mm_free(Ph_Data);
    _mm_free(bin_data);
    _mm_free(bin_phen);
    cl::sycl::free(dev_data, context);
    cl::sycl::free(dev_phen, context);
    cl::sycl::free(dev_scores, context);

    return 0;
}
