#pragma once
#include <minidl/dtype.h>

#include <cstddef>

namespace minidl {
class Allocator {
   public:
    Allocator() = default;
    virtual ~Allocator() = default;

    virtual void* allocate(std::size_t nbytes) = 0;
    virtual void deallocate(void* data) = 0;

    // delete copy
    Allocator(const Allocator& other) = delete;
    Allocator& operator=(const Allocator& other) = delete;
    Allocator(Allocator&& other) = delete;
    Allocator& operator=(Allocator&& othre) = delete;
};
}  // namespace minidl
