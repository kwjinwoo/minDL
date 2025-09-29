#include <cstddef>
#include <vector>

namespace minidl::kernels {

template <typename T>
void add_contig(T* __restrict z, const T* __restrict x, const T* __restrict y, std::size_t n) noexcept;

template <typename T>
void add_same_shape_strided(T* __restrict z, const T* __restrict x, const T* __restrict y,
                            const std::vector<std::size_t>& shape, const std::vector<std::size_t>& xs,
                            const std::vector<std::size_t>& ys) noexcept;

template <typename T>
void add_broadcast(T* __restrict z, const T* __restrict x, const T* __restrict y,
                   const std::vector<std::size_t>& out_shape, const std::vector<std::size_t>& xs,
                   const std::vector<std::size_t>& ys) noexcept;

}  // namespace minidl::kernels