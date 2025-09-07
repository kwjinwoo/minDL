#pragma once
#include <memory>

#include "minidl/allocators/system_allocator.h"
namespace minidl {
inline std::shared_ptr<Allocator> get_default_allocator() {
    static std::shared_ptr<Allocator> a = std::make_shared<SystemAllocator>();
    return a;
}
}  // namespace minidl
