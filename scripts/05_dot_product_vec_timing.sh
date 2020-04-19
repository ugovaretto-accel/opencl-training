#!/bin/bash
#g++ ../src/05_dot_product_vec_timing.cpp ../src/clutil.cpp -I../src -lOpenCL \
# -o 05_dot_product_vec_timing -DUSE_DOUBLE -pthread
#no double --> no convergence, no pthread --> no parallel processing on CPU
platform="AMD Accelerated Parallel Processing"
#platform="Portable Computing Language"
device_id="0"
type="gpu"
size=$(((2**19) * 1024))
echo "size: $size"
local_size=256
vec_element_width=1
./05_dot_product_vec_timing "$platform" \
$type $device_id ../src/kernels/05_dot_product_vec.cl dotprod $size $local_size \
$vec_element_width


#Number of platforms: 2
#
#-----------
#Platform 0
#-----------
#Vendor: Advanced Micro Devices, Inc.
#Profile: FULL_PROFILE
#Version: OpenCL 2.1 AMD-APP (3075.10)
#Name: AMD Accelerated Parallel Processing
#Extensions: cl_khr_icd cl_amd_event_callback cl_amd_offline_devices 
#Number of devices: 2
#Device 0
#  Type: GPU 
#  Name: gfx1010
#  Version: OpenCL 2.0 AMD-APP (3075.10)
#  Vendor: Advanced Micro Devices, Inc.
#  Profile: FULL_PROFILE
#  Compute units: 18
#  Max work item dim: 3
#  Work item sizes: 1024 1024 1024
#  Max clock freq: 1650 MHz
#  Global memory: 6425673728 bytes
#  Local memory: 65536 bytes
#Device 1
#  Type: GPU 
#  Name: gfx1010
#  Version: OpenCL 2.0 AMD-APP (3075.10)
#  Vendor: Advanced Micro Devices, Inc.
#  Profile: FULL_PROFILE
#  Compute units: 18
#  Max work item dim: 3
#  Work item sizes: 1024 1024 1024
#  Max clock freq: 1650 MHz
#  Global memory: 6425673728 bytes
#  Local memory: 65536 bytes

#All results double precision on a < AUD 500 RX 5600 XT card
#Peak power consumption 18W:
# ========================ROCm System Management Interface======================
# ==============================================================================
# GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%  
# 0    37.0c  9.0W    800Mhz  100Mhz  0.0%  auto  150.0W    0%   0%    
# 1    43.0c  18.0W   800Mhz  100Mhz  0.0%  auto  150.0W    8%   0%    
# ==============================================================================

############################
# AMD GPU:
#Size:          8290304
#Local size:    256
#Element width: 1
#2.36358e+08 2.36358e+08
#PASSED
#kernel:         5.5247ms
#host reduction: 0.201358ms
#total:          5.72606ms
#transfer:       0.528093ms
#
#host:           76.2224ms

#AMD GPU w/ -pthread!
#kernel:         1.77053ms
#host reduction: 0.206006ms
#total:          1.97654ms
#transfer:       0.576252ms
#
#host:           78.6203ms

###########################
# AMD GPU w/ pthread!
#Size:          16777216
#Local size:    256
#Element width: 1
#3.39788e+08 3.39788e+08
#PASSED
#kernel:         8.57888ms
#host reduction: 0.610502ms
#total:          9.18939ms
#transfer:       0.726634ms
#
#host:              147.032ms


###########################
# AMD GPU w/ pthread! 2GB arrayes
#size: 268435456
#Size:          268435456
#Local size:    256
#Element width: 1
#5.43562e+09 5.43562e+09
#PASSED
#kernel:         32.1953ms
#host reduction: 6.54658ms
#total:          38.7418ms
#transfer:       1.94638ms
#
#host:              2323.27ms

###########################
# AMD GPU w/ pthread! 4GB arrayes
# size: 536870912
# Size:          536870912
# Local size:    256
# Element width: 1
# 1.08715e+10 1.08715e+10
# PASSED
# kernel:         151.873ms
# host reduction: 13.3555ms
# total:          165.229ms
# transfer:       2.08787ms

# host:              4582.05ms



###########################
# POCL, -pthread
#Size:          8290304
#Local size:    256
#Element width: 1
#2.36261e+08 2.36261e+08
#PASSED
#kernel:         11.1046ms
#host reduction: 0.211331ms
#total:          11.3159ms
#transfer:       0.045545ms
#
#host:           72.4735ms

###########################
# POCL, NO -pthread!
#Size:          8290304
#Local size:    256
#Element width: 1
#2.36223e+08 2.36223e+08
#PASSED
#kernel:         88.5211ms
#host reduction: 0.209976ms
#total:          88.731ms
#transfer:       0.048571ms
#
#host:           79.2825ms
