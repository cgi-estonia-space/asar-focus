

#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <memory>


// Wrappers around cuda APIs for exception safety
struct CudaMallocDeleter {
    void operator()(void *d_ptr) { cudaFree(d_ptr); }
};

template<class T>
using CudaMallocTypeCleanup = std::unique_ptr<T, CudaMallocDeleter>;

using CudaMallocCleanup = std::unique_ptr<void, CudaMallocDeleter>;

class CufftPlanCleanup {
public:
    explicit CufftPlanCleanup(cufftHandle plan) : plan_(plan) {}

    CufftPlanCleanup(const CufftPlanCleanup &) = delete;

    CufftPlanCleanup &operator=(const CufftPlanCleanup &) = delete;

    ~CufftPlanCleanup() { cufftDestroy(plan_); }

private:
    cufftHandle plan_;
};
