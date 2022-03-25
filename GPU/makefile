CC = dpcpp
FLAGS = -g -O2

ifeq ($(gpu),nvidia_titan_xp)
	FLAGS+=-std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend --offload-arch=sm_61 -Xcuda-ptxas -dlcm=ca
endif
ifeq ($(gpu),nvidia)
	FLAGS+=-std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
endif
ifeq ($(gpu),amd_vega_20)
	FLAGS+=-std=c++17 -fsycl -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx906
endif
ifeq ($(gpu),amd_instinct_mi100)
	FLAGS+=-std=c++17 -fsycl -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx908
endif
ifeq ($(gpu),amd_navi_21)
	FLAGS+=-std=c++17 -fsycl -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx1030
endif

all: v1 v2 v3 v4

v1: v1.cpp
	$(CC) -o $@ $^ $(FLAGS)
v2: v2.cpp
	$(CC) -o $@ $^ $(FLAGS)
v3: v3.cpp
	$(CC) -o $@ $^ $(FLAGS)
v4: v4.cpp
	$(CC) -o $@ $^ $(FLAGS)

clean:
	rm v1 v2 v3 v4
