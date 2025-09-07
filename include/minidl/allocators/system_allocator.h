#pragma once
#include <minidl/allocator.h>

#include <new>

namespace minidl {

class SystemAllocator final : public Allocator {
    void* allocate(std::size_t nbytes) override {
        if (nbytes == 0) return nullptr;
        return ::operator new(nbytes);
    }
    void deallocate(void* data) override { ::operator delete(data); }
};

}  // namespace minidl
