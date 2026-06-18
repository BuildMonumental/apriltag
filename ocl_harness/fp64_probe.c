#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

// Measures whether per-cluster sequential double-precision moment
// accumulation (the compute_lfps + fit_line pattern from fit_quad) is fast
// enough on the iGPU to justify a GPU fit_quads port. Each work-item plays
// the role of one cluster: a serial scan of N points accumulating the six
// line-fit moments in double, with one image byte sampled per point.
static const char *probeSource =
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "__kernel void lfpsSim(__global const uchar *im, int imBytes, int ptsPerCluster,\n"
    "                      __global double *out) {\n"
    "    int cluster = get_global_id(0);\n"
    "    uint seed = (uint)cluster * 2654435761u + 1u;\n"
    "    double Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, W = 0;\n"
    "    for (int i = 0; i < ptsPerCluster; i++) {\n"
    "        seed = seed * 1664525u + 1013904223u;\n"
    "        double x = (double)(seed & 0xFFFu);\n"
    "        double y = (double)((seed >> 12) & 0xFFFu);\n"
    "        double w = (double)im[seed % (uint)imBytes] + 1.0;\n"
    "        Mx += w * x; My += w * y;\n"
    "        Mxx += w * x * x; Mxy += w * x * y; Myy += w * y * y;\n"
    "        W += w;\n"
    "    }\n"
    "    double Ex = Mx / W, Ey = My / W;\n"
    "    double Cxx = Mxx / W - Ex * Ex;\n"
    "    double Cxy = Mxy / W - Ex * Ey;\n"
    "    double Cyy = Myy / W - Ey * Ey;\n"
    "    double eig = 0.5 * (Cxx + Cyy - sqrt((Cxx - Cyy) * (Cxx - Cyy) + 4.0 * Cxy * Cxy));\n"
    "    out[cluster] = eig;\n"
    "}\n";

static double nowMs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    const int nClusters = 9000;
    const int ptsPerCluster = 300;
    const int imBytes = 6 * 1024 * 1024;

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS) {
        fprintf(stderr, "no GPU\n");
        return 1;
    }
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, NULL, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &probeSource, NULL, &err);
    if (clBuildProgram(program, 1, &device, "", NULL, NULL) != CL_SUCCESS) {
        char log[4096] = { 0 };
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "lfpsSim", &err);

    unsigned char *imageHost = malloc(imBytes);
    for (int i = 0; i < imBytes; i++)
        imageHost[i] = (unsigned char)(i * 31);
    cl_mem bufIm = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imBytes, imageHost, &err);
    cl_mem bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nClusters * 8, NULL, &err);

    const cl_int cImBytes = imBytes, cPts = ptsPerCluster;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufIm);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &cImBytes);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &cPts);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufOut);

    const size_t global[1] = { (size_t)nClusters };
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
    clFinish(queue);

    double best = 1e9;
    for (int iter = 0; iter < 10; iter++) {
        double start = nowMs();
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
        clFinish(queue);
        double elapsed = nowMs() - start;
        if (elapsed < best)
            best = elapsed;
    }
    printf("fp64 lfps simulation: %d clusters x %d pts: best %.2f ms\n", nClusters, ptsPerCluster, best);
    return 0;
}
