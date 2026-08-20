/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cublasLt.h>
#include <string>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace comfy {
class CublasLtRuntime {
public:
    // Function pointer types for cuBLASLt functions we need
    using cublasLtCreate_t = cublasStatus_t (*)(cublasLtHandle_t*);
    using cublasLtDestroy_t = cublasStatus_t (*)(cublasLtHandle_t);
    using cublasLtMatmul_t = cublasStatus_t (*)(
        cublasLtHandle_t,
        cublasLtMatmulDesc_t,
        const void*,
        const void*,
        cublasLtMatrixLayout_t,
        const void*,
        cublasLtMatrixLayout_t,
        const void*,
        const void*,
        cublasLtMatrixLayout_t,
        void*,
        cublasLtMatrixLayout_t,
        const cublasLtMatmulAlgo_t*,
        void*,
        size_t,
        cudaStream_t
    );
    using cublasLtMatmulDescCreate_t = cublasStatus_t (*)(
        cublasLtMatmulDesc_t*,
        cublasComputeType_t,
        cudaDataType_t
    );
    using cublasLtMatmulDescDestroy_t = cublasStatus_t (*)(cublasLtMatmulDesc_t);
    using cublasLtMatmulDescSetAttribute_t = cublasStatus_t (*)(
        cublasLtMatmulDesc_t,
        cublasLtMatmulDescAttributes_t,
        const void*,
        size_t
    );
    using cublasLtMatrixLayoutCreate_t = cublasStatus_t (*)(
        cublasLtMatrixLayout_t*,
        cudaDataType_t,
        uint64_t,
        uint64_t,
        int64_t
    );
    using cublasLtMatrixLayoutDestroy_t = cublasStatus_t (*)(cublasLtMatrixLayout_t);
    using cublasLtMatmulPreferenceCreate_t = cublasStatus_t (*)(cublasLtMatmulPreference_t*);
    using cublasLtMatmulPreferenceDestroy_t = cublasStatus_t (*)(cublasLtMatmulPreference_t);
    using cublasLtMatmulPreferenceSetAttribute_t = cublasStatus_t (*)(
        cublasLtMatmulPreference_t,
        cublasLtMatmulPreferenceAttributes_t,
        const void*,
        size_t
    );
    using cublasLtMatmulAlgoGetHeuristic_t = cublasStatus_t (*)(
        cublasLtHandle_t,
        cublasLtMatmulDesc_t,
        cublasLtMatrixLayout_t,
        cublasLtMatrixLayout_t,
        cublasLtMatrixLayout_t,
        cublasLtMatrixLayout_t,
        cublasLtMatmulPreference_t,
        int,
        cublasLtMatmulHeuristicResult_t*,
        int*
    );

    static CublasLtRuntime& instance() {
        static CublasLtRuntime runtime;
        return runtime;
    }

    bool is_available() const { return available_; }
    const std::string& error_message() const { return error_message_; }

    // Function pointers - only valid if is_available() returns true
    cublasLtCreate_t cublasLtCreate = nullptr;
    cublasLtDestroy_t cublasLtDestroy = nullptr;
    cublasLtMatmul_t cublasLtMatmul = nullptr;
    cublasLtMatmulDescCreate_t cublasLtMatmulDescCreate = nullptr;
    cublasLtMatmulDescDestroy_t cublasLtMatmulDescDestroy = nullptr;
    cublasLtMatmulDescSetAttribute_t cublasLtMatmulDescSetAttribute = nullptr;
    cublasLtMatrixLayoutCreate_t cublasLtMatrixLayoutCreate = nullptr;
    cublasLtMatrixLayoutDestroy_t cublasLtMatrixLayoutDestroy = nullptr;
    cublasLtMatmulPreferenceCreate_t cublasLtMatmulPreferenceCreate = nullptr;
    cublasLtMatmulPreferenceDestroy_t cublasLtMatmulPreferenceDestroy = nullptr;
    cublasLtMatmulPreferenceSetAttribute_t cublasLtMatmulPreferenceSetAttribute = nullptr;
    cublasLtMatmulAlgoGetHeuristic_t cublasLtMatmulAlgoGetHeuristic = nullptr;

private:
    CublasLtRuntime() {
        load();
    }

    ~CublasLtRuntime() {
        unload();
    }

    // Delete copy/move
    CublasLtRuntime(const CublasLtRuntime&) = delete;
    CublasLtRuntime& operator=(const CublasLtRuntime&) = delete;

    void load() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (load_attempted_) return;
        load_attempted_ = true;

#ifdef _WIN32
        // Windows: cuBLAS 13.x (CUDA 13+)
        const char* lib_names[] = {
            "cublasLt64_13.dll",
            "cublasLt64.dll",       // Fallback (may be 13.x)
            nullptr
        };
        
        for (const char** name = lib_names; *name != nullptr; ++name) {
            handle_ = LoadLibraryA(*name);
            if (handle_) break;
        }
        
        if (!handle_) {
            error_message_ = "cuBLASLt 13.x library not found (requires CUDA 13+)";
            return;
        }
#else
        // Linux: cuBLAS 13.x (CUDA 13+)
        const char* lib_names[] = {
            "libcublasLt.so.13",
            "libcublasLt.so",       // Fallback (may be 13.x)
            nullptr
        };
        
        for (const char** name = lib_names; *name != nullptr; ++name) {
            handle_ = dlopen(*name, RTLD_NOW | RTLD_GLOBAL);
            if (handle_) break;
        }
        
        if (!handle_) {
            error_message_ = std::string("cuBLASLt 13.x library not found (requires CUDA 13+): ") + dlerror();
            return;
        }
#endif

        // Load all required function pointers
        if (!load_symbols()) {
            unload();
            return;
        }

        available_ = true;
    }

    bool load_symbols() {
#ifdef _WIN32
#define LOAD_SYMBOL(name) \
        name = reinterpret_cast<name##_t>(GetProcAddress(static_cast<HMODULE>(handle_), #name)); \
        if (!name) { \
            error_message_ = "Failed to load symbol: " #name; \
            return false; \
        }
#else
#define LOAD_SYMBOL(name) \
        name = reinterpret_cast<name##_t>(dlsym(handle_, #name)); \
        if (!name) { \
            error_message_ = std::string("Failed to load symbol: " #name ": ") + dlerror(); \
            return false; \
        }
#endif

        LOAD_SYMBOL(cublasLtCreate);
        LOAD_SYMBOL(cublasLtDestroy);
        LOAD_SYMBOL(cublasLtMatmul);
        LOAD_SYMBOL(cublasLtMatmulDescCreate);
        LOAD_SYMBOL(cublasLtMatmulDescDestroy);
        LOAD_SYMBOL(cublasLtMatmulDescSetAttribute);
        LOAD_SYMBOL(cublasLtMatrixLayoutCreate);
        LOAD_SYMBOL(cublasLtMatrixLayoutDestroy);
        LOAD_SYMBOL(cublasLtMatmulPreferenceCreate);
        LOAD_SYMBOL(cublasLtMatmulPreferenceDestroy);
        LOAD_SYMBOL(cublasLtMatmulPreferenceSetAttribute);
        LOAD_SYMBOL(cublasLtMatmulAlgoGetHeuristic);

#undef LOAD_SYMBOL
        return true;
    }

    void unload() {
        if (handle_) {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
            handle_ = nullptr;
        }
        available_ = false;
        
        // Clear function pointers
        cublasLtCreate = nullptr;
        cublasLtDestroy = nullptr;
        cublasLtMatmul = nullptr;
        cublasLtMatmulDescCreate = nullptr;
        cublasLtMatmulDescDestroy = nullptr;
        cublasLtMatmulDescSetAttribute = nullptr;
        cublasLtMatrixLayoutCreate = nullptr;
        cublasLtMatrixLayoutDestroy = nullptr;
        cublasLtMatmulPreferenceCreate = nullptr;
        cublasLtMatmulPreferenceDestroy = nullptr;
        cublasLtMatmulPreferenceSetAttribute = nullptr;
        cublasLtMatmulAlgoGetHeuristic = nullptr;
    }

    void* handle_ = nullptr;
    bool available_ = false;
    bool load_attempted_ = false;
    std::string error_message_;
    std::mutex mutex_;
};


} // namespace comfy

