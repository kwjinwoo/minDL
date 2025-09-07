#pragma once
#include <minidl/allocator.h>

#include <new>

namespace minidl {

class SystemAllocator final : public Allocator {
    void* allocate(std::size_t nbytes) override;
    void deallocate(void* data) override;
};

}  // namespace minidl
