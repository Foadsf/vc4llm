#pragma once
#include <cstring>
#include <iostream>
#include <vector>

typedef int cl_int;
typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_mem* cl_mem;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
typedef unsigned long cl_bitfield;
typedef unsigned long cl_device_type;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU 1
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_MEM_READ_ONLY 1
#define CL_MEM_WRITE_ONLY 2
#define CL_MEM_USE_HOST_PTR 4
#define CL_MEM_COPY_HOST_PTR 8
#define CL_FALSE 0
#define CL_TRUE 1
#define CL_PROGRAM_BUILD_LOG 0x1183

// Stub functions
static inline cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    if (platforms) *platforms = (cl_platform_id)1;
    if (num_platforms) *num_platforms = 1;
    return CL_SUCCESS;
}

static inline cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) {
    if (devices) *devices = (cl_device_id)1;
    if (num_devices) *num_devices = 1;
    return CL_SUCCESS;
}

static inline cl_int clGetDeviceInfo(cl_device_id device, cl_uint param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    if (param_name == CL_DEVICE_NAME) {
        strncpy((char*)param_value, "Mock Device", param_value_size);
    } else if (param_name == CL_DEVICE_EXTENSIONS) {
        strncpy((char*)param_value, "cl_arm_integer_dot_product_int8", param_value_size);
    }
    return CL_SUCCESS;
}

static inline cl_context clCreateContext(const void *properties, cl_uint num_devices, const cl_device_id *devices, void *pfn_notify, void *user_data, cl_int *errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_context)1;
}

static inline cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_bitfield properties, cl_int *errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_command_queue)1;
}

static inline cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_program)1;
}

static inline cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void *pfn_notify, void *user_data) {
    return CL_SUCCESS;
}

static inline cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_uint param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_SUCCESS;
}

static inline cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_kernel)1;
}

static inline cl_mem clCreateBuffer(cl_context context, cl_bitfield flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_mem)1;
}

static inline cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    return CL_SUCCESS;
}

static inline cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const void *event_wait_list, void *event) {
    return CL_SUCCESS;
}

static inline cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_int blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const void *event_wait_list, void *event) {
    // In mock, we don't actually compute anything, so we might return zeros or not touch the buffer.
    // This is just to satisfy the linker.
    return CL_SUCCESS;
}

static inline cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_int blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const void *event_wait_list, void *event) {
    return CL_SUCCESS;
}

static inline cl_int clReleaseMemObject(cl_mem memobj) {
    return CL_SUCCESS;
}

static inline cl_int clReleaseKernel(cl_kernel kernel) {
    return CL_SUCCESS;
}

static inline cl_int clReleaseProgram(cl_program program) {
    return CL_SUCCESS;
}

static inline cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    return CL_SUCCESS;
}

static inline cl_int clReleaseContext(cl_context context) {
    return CL_SUCCESS;
}
