// Memcopy example w/ bandwidth tests. Author: Ugo Varetto Note: page-locked
// memory transfers might not work properly on systems sharing the same memory
// for both host and device (e.g. CPU), does work on POCL
// g++ ../src/clutil.cpp ../src/etc/09_memcpy_bw_test.cpp -I ../src/ -lOpenCL \
//-o 09_memcpy -DCL_TARGET_OPENCL_VERSION=110
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "clutil.h"

using namespace std;

typedef vector<uint8_t> ByteArray;

//------------------------------------------------------------------------------
double profile_time_s(cl_event event) {
    cl_ulong tstart = 0;
    cl_ulong tend = 0;
    cl_int ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &tstart, 0);
    check_cl_error(ret, "clGetEventProfilingInfo");
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &tend, 0);
    check_cl_error(ret, "clGetEventProfilingInfo");
    return (tend - tstart) / 1E9;
}

//------------------------------------------------------------------------------
double copy_host_to_device(cl_context context, cl_command_queue commandQueue,
                           size_t size, bool mapped, bool pinned) {
    cl_int status = -1;
    uint8_t *hostMem = nullptr;
    cl_mem hostPinnedMem = nullptr;
    cl_mem deviceMem = nullptr;
    cl_event profileEvent;
    // Allocate host memory
    if (pinned) {
        // Create a host buffer
        hostPinnedMem =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                           size, 0, &status);
        check_cl_error(status, to_string(__LINE__).c_str());

        // Get a mapped pointer to page locked memeory
        hostMem = (uint8_t *)clEnqueueMapBuffer(commandQueue, hostPinnedMem,
                                                CL_FALSE, CL_MAP_WRITE, 0, size,
                                                0, nullptr, nullptr, &status);
        check_cl_error(status, to_string(__LINE__).c_str());

        // initialize
        for (int i = 0; i != size; ++i) {
            hostMem[i] = (uint8_t)i;
        }

        // unmap: disconnect host pointer from page locked memory,
        // to access memory from host map again
        status = clEnqueueUnmapMemObject(commandQueue, hostPinnedMem,
                                         (void *)hostMem, 0, nullptr, nullptr);
        check_cl_error(status, to_string(__LINE__).c_str());
        hostMem = nullptr;  // buffer unmapped set to null
    } else {
        // host allocation
        hostMem = (uint8_t *)malloc(size);

        // initialize
        for (unsigned int i = 0; i != size; i++) {
            hostMem[i] = (uint8_t)i;
        }
    }

    // allocate device memory
    deviceMem = clCreateBuffer(context, CL_MEM_READ_ONLY, size, 0, &status);
    check_cl_error(status, to_string(__LINE__).c_str());

    // sync queue
    clFinish(commandQueue);

    if (!mapped) {
        if (pinned) {
            // map again page locked memory to accessible host pointer
            // cl_event pe;
            hostMem = (uint8_t *)clEnqueueMapBuffer(
                commandQueue, hostPinnedMem, CL_FALSE, CL_MAP_READ, 0, size, 0,
                nullptr, nullptr /*&pe*/, &status);
            // clWaitForEvents(1, &pe);
            // cout << profile_time_s(pe) << endl; //pe time == 0
            check_cl_error(status, to_string(__LINE__).c_str());
        }

        // copy data from page locked host buffer into device memory
        clEnqueueWriteBuffer(commandQueue, deviceMem, CL_FALSE, 0, size,
                             hostMem, 0, nullptr, &profileEvent);
        check_cl_error(status, to_string(__LINE__).c_str());

        check_cl_error(clWaitForEvents(1, &profileEvent), "clWaitForEvents");
        status = clFinish(commandQueue);
        check_cl_error(status, to_string(__LINE__).c_str());
    } else {
        // mapped access: data is moved between two host memory pointers and
        // copied into GPU memory during the unmap operation

        // map device memory into host memory
        void *hostMappedDeviceMem =
            clEnqueueMapBuffer(commandQueue, deviceMem, CL_FALSE, CL_MAP_WRITE,
                               0, size, 0, nullptr, nullptr, &status);
        check_cl_error(status, to_string(__LINE__).c_str());
        if (pinned) {
            // if pinned buffer available map host page locked buffer to
            // host memory pointer
            hostMem = (uint8_t *)clEnqueueMapBuffer(
                commandQueue, hostPinnedMem, CL_FALSE, CL_MAP_READ, 0, size, 0,
                nullptr, nullptr, &status);
            check_cl_error(status, to_string(__LINE__).c_str());
        }

        // copy data from host memory to host memory mapping device memory
        memcpy(hostMappedDeviceMem, hostMem, size);

        // DATA is moved from host to device in unmap operation, this is the
        // only operation that should be timed to compute host to device
        // bandwidth
        status = clEnqueueUnmapMemObject(commandQueue, deviceMem,
                                         hostMappedDeviceMem, 0, nullptr,
                                         &profileEvent);
        check_cl_error(status, to_string(__LINE__).c_str());
        clWaitForEvents(1, &profileEvent);
        status = clFinish(commandQueue);
        check_cl_error(status, to_string(__LINE__).c_str());
    }

    const double elapsedTimeSec = profile_time_s(profileEvent);
    const double bandwidthMB =
        ((double)size / (elapsedTimeSec * (double)(1 << 20)));

    // clean up memory
    if (deviceMem) clReleaseMemObject(deviceMem);
    if (hostPinnedMem) {
        clReleaseMemObject(hostPinnedMem);
    } else {
        free(hostMem);
    }
    return bandwidthMB;
}

//------------------------------------------------------------------------------
double copy_device_to_host(cl_context context, cl_command_queue commandQueue,
                           size_t size, bool mapped, bool pinned) {
    cl_int status = -1;
    uint8_t *hostMem = nullptr;
    cl_mem hostPinnedMem = nullptr;
    cl_mem deviceMem = nullptr;
    cl_event profileEvent;
    // Allocate host memory
    if (pinned) {
        // Create a host buffer, buffer is read/write because we also use it
        // to initialise GPU memory before timing the transfer back
        hostPinnedMem =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                           size, 0, &status);
        check_cl_error(status, to_string(__LINE__).c_str());

        // Map pinned buffer to host pointer
        hostMem = (uint8_t *)clEnqueueMapBuffer(commandQueue, hostPinnedMem,
                                                CL_TRUE, CL_MAP_WRITE, 0, size,
                                                0, 0, 0, &status);
        check_cl_error(status, to_string(__LINE__).c_str());

        // initialize
        for (int i = 0; i != size; ++i) {
            hostMem[i] = (uint8_t)(i);
        }

        // unmap paage locked memory region
        status = clEnqueueUnmapMemObject(commandQueue, hostPinnedMem,
                                         (void *)hostMem, 0, 0, 0);
        hostMem = nullptr;
        check_cl_error(status, to_string(__LINE__).c_str());

    } else {
        // standard host alloc
        hostMem = (uint8_t *)malloc(size);

        // initialize
        for (int i = 0; i != size; ++i) {
            hostMem[i] = (uint8_t)(i);
        }
    }

    // allocate device memory to be written by kernel
    deviceMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, 0, &status);
    check_cl_error(status, to_string(__LINE__).c_str());

    // initialize device memory: copy data from host to device
    if (pinned) {
        // get a mapped pointer
        hostMem = (uint8_t *)clEnqueueMapBuffer(commandQueue, hostPinnedMem,
                                                CL_TRUE, CL_MAP_WRITE, 0, size,
                                                0, 0, 0, &status);

        // copy data from host to device
        status = clEnqueueWriteBuffer(commandQueue, deviceMem, CL_FALSE, 0,
                                      size, hostMem, 0, 0, 0);
        check_cl_error(status, to_string(__LINE__).c_str());

        clEnqueueUnmapMemObject(commandQueue, hostPinnedMem, (void *)hostMem, 0,
                                0, 0);
        check_cl_error(status, to_string(__LINE__).c_str());

    } else {
        // no mapping here just copy from host to device directly
        status = clEnqueueWriteBuffer(commandQueue, deviceMem, CL_FALSE, 0,
                                      size, hostMem, 0, 0, 0);
        check_cl_error(status, to_string(__LINE__).c_str());
    }
    check_cl_error(status, to_string(__LINE__).c_str());

    // sync queue
    status = clFinish(commandQueue);

    if (!mapped) {
        // DIRECT:  API access to device buffee

        if (pinned) {
            // get a mapped pointer
            hostMem = (uint8_t *)clEnqueueMapBuffer(commandQueue, hostPinnedMem,
                                                    CL_FALSE, CL_MAP_WRITE, 0,
                                                    size, 0, 0, 0, &status);
            check_cl_error(status, to_string(__LINE__).c_str());
        }

        status = clEnqueueReadBuffer(commandQueue, deviceMem, CL_FALSE, 0, size,
                                     hostMem, 0, 0, &profileEvent);

        check_cl_error(status, to_string(__LINE__).c_str());
        check_cl_error(clWaitForEvents(1, &profileEvent), "clWaitForEvents");

        if (pinned) {
            clEnqueueUnmapMemObject(commandQueue, hostPinnedMem,
                                    (void *)hostMem, 0, 0, 0);
            check_cl_error(status, to_string(__LINE__).c_str());
        }

        status = clFinish(commandQueue);
        check_cl_error(status, to_string(__LINE__).c_str());
    } else {
        // MAPPED: mapped pointers to device buffer for conventional pointer
        // access
        void *hostMappedDeviceMem =
            clEnqueueMapBuffer(commandQueue, deviceMem, CL_FALSE, CL_MAP_WRITE,
                               0, size, 0, 0, 0, &status);
        check_cl_error(status, to_string(__LINE__).c_str());

        memcpy(hostMem, hostMappedDeviceMem, size);
        status = clEnqueueUnmapMemObject(
            commandQueue, deviceMem, hostMappedDeviceMem, 0, 0, &profileEvent);
        check_cl_error(status, to_string(__LINE__).c_str());
        check_cl_error(clWaitForEvents(1, &profileEvent), "clWaitForEvents");
    }
    const double elapsedTimeSec = profile_time_s(profileEvent);
    // calculate bandwidth in MB/s
    const double bandwidthMB =
        ((double)size / (elapsedTimeSec * (double)(1 << 20)));

    // clean up memory
    if (deviceMem) clReleaseMemObject(deviceMem);
    if (hostPinnedMem) {
        clReleaseMemObject(hostPinnedMem);
    } else {
        free(hostMem);
    }

    return bandwidthMB;
}

//------------------------------------------------------------------------------
double copy_device_to_device(cl_context context, cl_command_queue commandQueue,
                             size_t size) {
    cl_int status = -1;
    uint8_t *hostMem = nullptr;
    cl_mem deviceMem = nullptr;
    cl_event profileEvent;

    // allocate and initialize host memory
    hostMem = (uint8_t *)malloc(size);

    for (int i = 0; i != size; ++i) {
        hostMem[i] = (uint8_t)(i);
    }

    // allocate source and destination device buffers
    cl_mem devSrc =
        clCreateBuffer(context, CL_MEM_READ_ONLY, size, nullptr, &status);
    check_cl_error(status, to_string(__LINE__).c_str());

    cl_mem devDest =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr, &status);
    check_cl_error(status, to_string(__LINE__).c_str());

    // initialize device buffer with host data
    status = clEnqueueWriteBuffer(commandQueue, devSrc, CL_TRUE, 0, size,
                                  hostMem, 0, nullptr, nullptr);
    check_cl_error(status, to_string(__LINE__).c_str());

    // Sync queue to host,
    clFinish(commandQueue);

    // copy data on device from src to dest buffer
    status = clEnqueueCopyBuffer(commandQueue, devSrc, devDest, 0, 0, size, 0,
                                 nullptr, &profileEvent);
    check_cl_error(status, to_string(__LINE__).c_str());
    check_cl_error(clWaitForEvents(1, &profileEvent), "clWaitForEvents");

    // sync
    clFinish(commandQueue);

    const double elapsedTimeSec = profile_time_s(profileEvent);
    const double bandwidthMB =
        ((double)size / (elapsedTimeSec * (double)(1 << 20)));

    // clean up memory on host and device
    free(hostMem);
    clReleaseMemObject(devSrc);
    clReleaseMemObject(devDest);

    return bandwidthMB;
}

//------------------------------------------------------------
int main(int argc, char **argv) {
    vector<string> cmdline;
    copy(argv, argv + argc, back_inserter(cmdline));

    if (cmdline.size() < 5) {
        cerr << "usage: " << cmdline[0]
             << " <platform name>"
                " <device type: default | cpu | gpu | acc>"
                " <device num>"
                " <size>"
                " [mapped (default = not mapped)]"
                " [pinned (default = not pinned)]"
                " [--iterations <num iterations, default=1>]"
             << endl;
        exit(EXIT_FAILURE);
    }
    const bool PINNED =
        find(begin(cmdline), end(cmdline), "pinned") != end(cmdline);
    const bool MAPPED =
        find(begin(cmdline), end(cmdline), "mapped") != end(cmdline);
    int iterations = 1;
    if (find(begin(cmdline), end(cmdline), "--iterations") != end(cmdline)) {
        auto i = find(begin(cmdline), end(cmdline), "--iterations");
        if (++i == end(cmdline)) {
            cerr << "missing number of iterations" << endl;
            exit(EXIT_FAILURE);
        }
        iterations = atoi(i->c_str());
        if (iterations <= 0) {
            std::cerr << "number of iterations must be greater that zero"
                      << endl;
        }
    }
    const int platformID = atoi(argv[1]);
    cl_device_type deviceType;
    const string dt(argv[2]);
    if (dt == "default")
        deviceType = CL_DEVICE_TYPE_DEFAULT;
    else if (dt == "cpu")
        deviceType = CL_DEVICE_TYPE_CPU;
    else if (dt == "gpu")
        deviceType = CL_DEVICE_TYPE_GPU;
    else if (dt == "acc")
        deviceType = CL_DEVICE_TYPE_ACCELERATOR;
    else {
        cerr << "ERROR - unrecognized device type " << dt << endl;
        exit(EXIT_FAILURE);
    }
    const int deviceID = atoi(argv[3]);
    const size_t SIZE = atoll(argv[4]);

    CLEnv clenv = create_clenv(argv[1], argv[2], atoi(argv[3]), true);

    cl_int status;

    ByteArray data(SIZE);

    double H2D_BW_MB = 0.0;
    for (int i = 0; i != iterations; ++i) {
        H2D_BW_MB += copy_host_to_device(clenv.context, clenv.commandQueue,
                                         SIZE, MAPPED, PINNED);
    }
    double D2H_BW_MB = 0.0;
    for (int i = 0; i != iterations; ++i) {
        D2H_BW_MB += copy_device_to_host(clenv.context, clenv.commandQueue,
                                         SIZE, MAPPED, PINNED);
    }
    D2H_BW_MB /= iterations;
    double D2D_BW_MB = 0.0;
    for (int i = 0; i != iterations; ++i) {
        D2D_BW_MB +=
            copy_device_to_device(clenv.context, clenv.commandQueue, SIZE);
    }
    D2D_BW_MB /= iterations;

    cout << "Memory configuration:\n";
    cout << (PINNED ? "  PINNED" : "  NOT PINNED") << endl;
    cout << (MAPPED ? "  MAPPED" : "  NOT MAPPED") << endl;
    cout << "Bandwidth (MB/s):\n"
         << "  Host to device:   " << H2D_BW_MB << endl
         << "  Device to host:   " << D2H_BW_MB << endl
         << "  Device to device: " << D2D_BW_MB << endl;

    return 0;
}