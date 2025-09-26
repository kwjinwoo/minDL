#include <gtest/gtest.h>
#include <minidl/detail/iter.h>

using namespace minidl;

TEST(NdCounterNext, TotalIterCountOneDim) {
    std::vector<std::size_t> shape({3});

    const std::size_t numel = 3;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, TotalIterCountTwodims) {
    std::vector<std::size_t> shape({3, 2});

    const std::size_t numel = 6;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, TotalIterCountThreeDims) {
    std::vector<std::size_t> shape({3, 2, 4});

    const std::size_t numel = 24;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, TotalIterCountScalar) {
    std::vector<std::size_t> shape({});

    const std::size_t numel = 1;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, ScalarVisitsExactlyOnce) {
    std::vector<std::size_t> shape({});
    detail::NdCounter c(shape);

    ASSERT_FALSE(c.done());
    c.next();
    ASSERT_TRUE(c.done());
}

TEST(NdCounterNext, RowMajorOrder3x2) {
    std::vector<std::size_t> shape({3, 2});
    detail::NdCounter c(shape);

    std::vector<std::array<std::size_t, 2>> seen;
    for (; !c.done(); c.next()) {
        const auto& idx = c.idx;
        seen.push_back({idx[0], idx[1]});
    }

    std::vector<std::array<std::size_t, 2>> expected = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1},
    };
    EXPECT_EQ(seen, expected);
}

TEST(NdCounterNext, ZeroSizedDimNoIteration) {
    std::vector<std::size_t> shape({3, 0, 2});
    detail::NdCounter c(shape);

    size_t counted = 0;
    for (; !c.done(); c.next()) counted++;
    EXPECT_EQ(counted, 0u);  // numel == 0
}

TEST(OffsetElems, OneDimOffset) {
    std::vector<size_t> idx({2});
    std::vector<std::size_t> strides({1});

    std::size_t expected = 2;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, TwoDimsOffset) {
    // (2, 3) shape
    std::vector<std::size_t> idx({1, 2});
    std::vector<std::size_t> strides({3, 1});

    std::size_t expected = 5;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, ThreeDimsOffset) {
    // (2, 3, 4) shape
    std::vector<std::size_t> idx({1, 2, 2});
    std::vector<std::size_t> strides({12, 4, 1});

    std::size_t expected = 22;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, ScalarOffset) {
    std::vector<std::size_t> idx;
    std::vector<std::size_t> strides;

    std::size_t expected = 0;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, ZeroStrideBroadcast) {
    std::vector<std::size_t> strides({0, 1});

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            std::size_t out = detail::offset_elems({i, j}, strides);
            EXPECT_EQ(out, j);
        }
    }
}

TEST(OffsetElems, NonContiguousGeneral) {
    std::vector<std::size_t> strides({1, 3});

    // (i,j) -> i*1 + j*3
    EXPECT_EQ(detail::offset_elems({0, 0}, strides), 0u);
    EXPECT_EQ(detail::offset_elems({2, 0}, strides), 2u);
    EXPECT_EQ(detail::offset_elems({0, 2}, strides), 6u);
    EXPECT_EQ(detail::offset_elems({2, 3}, strides), 11u);
}
