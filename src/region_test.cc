#include "distributions.hpp"
#include "gtest/gtest.h"

#include "exp/user/stb/slam/region.hh"

namespace {
using namespace BOOM;
using namespace slam;
using std::cout;
using std::endl;

class RegionTest : public ::testing::Test {
protected:
  RegionTest() {
    BOOM::GlobalRng::rng.seed(8675309);
    epsilon_ = 1e-10;
  }
  double epsilon_;
};

class PixelIteratorRange {
public:
  PixelIteratorRange(const Region &region) : region_(region) {}
  std::ostream &print(std::ostream &out) const {
    for (const auto &px : region_) {
      out << px << std::endl;
    }
    return out;
  }

private:
  const Region &region_;
};

inline std::ostream &operator<<(std::ostream &out,
                                const PixelIteratorRange &range) {
  return range.print(out);
}

bool EqualPoints(const Point2D &p1, const Point2D &p2, double epsilon) {
  double square_distance = square(p1.x() - p2.x()) + square(p1.y() - p2.y());
  return square_distance < epsilon;
}

TEST_F(RegionTest, BoundingBox) {
  BoundingBox unit(0, 1, 0, 1);
  BoundingBox three(0, 3, 0, 3);

  EXPECT_TRUE(three.intersects(unit, true));
  EXPECT_TRUE(unit.intersects(three, true));

  BoundingBox far(4, 5, 4, 5);
  EXPECT_FALSE(unit.intersects(far, true));
  EXPECT_FALSE(far.intersects(unit, true));

  EXPECT_TRUE(far.intersection(unit).empty());

  EXPECT_TRUE(three.contains(Point2D(1, 1)));
  EXPECT_TRUE(three.contains(Point2D(0, 0)));
  EXPECT_TRUE(three.contains(Point2D(3, 0)));
  EXPECT_TRUE(three.contains(Point2D(0, 3)));
  EXPECT_TRUE(three.contains(Point2D(3, 3)));
  EXPECT_TRUE(three.contains(Point2D(2.8, 0.1)));

  EXPECT_FALSE(three.contains(-0.1, 0));
  EXPECT_FALSE(three.contains(3.1, 3.1));
  EXPECT_FALSE(three.contains(3.1, 1.0));

  BoundingBox u2 = three.intersection(unit);
  EXPECT_TRUE(u2.intersects(unit, true));
  EXPECT_NEAR(u2.left(), unit.left(), epsilon_);
  EXPECT_NEAR(u2.right(), unit.right(), epsilon_);
  EXPECT_NEAR(u2.bottom(), unit.bottom(), epsilon_);
  EXPECT_NEAR(u2.top(), unit.top(), epsilon_);

  BoundingBox super = unit.super_box(far);
  EXPECT_DOUBLE_EQ(super.left(), unit.left());
  EXPECT_DOUBLE_EQ(super.bottom(), unit.bottom());
  EXPECT_DOUBLE_EQ(super.right(), far.right());
  EXPECT_DOUBLE_EQ(super.top(), far.top());

  BoundingBox empty;
  EXPECT_DOUBLE_EQ(0, empty.area());
  EXPECT_DOUBLE_EQ(0, empty.width());
  EXPECT_DOUBLE_EQ(0, empty.height());
  EXPECT_TRUE(empty.begin() == empty.end());
}

TEST_F(RegionTest, BoundingBoxIterator) {
  BoundingBox box(0, 3, 0, 2);
  auto it = box.begin();
  EXPECT_TRUE(EqualPoints(*it, Point2D(0, 0), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(0, 1), epsilon_));
  ++it;
  EXPECT_FALSE(it == box.end());
  EXPECT_TRUE(EqualPoints(*it, Point2D(0, 2), epsilon_));
  ++it;

  EXPECT_TRUE(EqualPoints(*it, Point2D(1, 0), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(1, 1), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(1, 2), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(2, 0), epsilon_));
  ++it;
  EXPECT_FALSE(it == box.end());
  EXPECT_TRUE(EqualPoints(*it, Point2D(2, 1), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(2, 2), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(3, 0), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(3, 1), epsilon_));
  ++it;
  EXPECT_TRUE(EqualPoints(*it, Point2D(3, 2), epsilon_));
  ++it;
  EXPECT_TRUE(it == box.end());
  ++it;
  ++it;
  ++it;
  ++it;
  ++it;
  EXPECT_TRUE(it == box.end());
}

TEST_F(RegionTest, EmptyBoxUnion) {
  BoxUnion empty;
  EXPECT_TRUE(empty.boundingBox().empty());
  EXPECT_TRUE(empty.empty());
  BoundingBox box(1.2, 3.7, 0.1, 1.9);
  EXPECT_FALSE(empty.intersects(box, true));
  EXPECT_FALSE(empty.intersects(box, false));
  EXPECT_TRUE(empty.begin() == empty.end());
  BoxUnion overlap = intersection(empty, box);
  EXPECT_TRUE(overlap.empty());

  EXPECT_DOUBLE_EQ(0.0, empty.area());
  EXPECT_FALSE(empty.contains(Point2D(1, 4)));
  EXPECT_FALSE(empty.contains(Point2D(0, 0)));
}

TEST_F(RegionTest, BoxUnion) {
  BoundingBox unit(0, 1, 0, 1);
  BoundingBox two(0, 2, 0, 2);
  BoundingBox right(3, 9, 4, 7);

  BoxUnion region;
  region.addBox(unit);
  region.addBox(two);
  region.addBox(right);
  EXPECT_DOUBLE_EQ(region.area(), 4 + 18);
  BoundingBox empty;
  region.addBox(empty);
  EXPECT_DOUBLE_EQ(region.area(), 4 + 18);

  // The (0, 2, 0, 2) region has 9 grid points.  The 'right' region has 28
  // grid points.  When iterating through, we should hit 37 grid points.
  EXPECT_EQ(37, std::distance(region.begin(), region.end()))
      << "Iterator did not produce the expected number of grid points."
      << std::endl
      << region << PixelIteratorRange(region);

  BoundingBox shifted_unit(.5, 1.5, .5, 1.5);
  BoundingBox shifted_two(.5, 2.5, .5, 2.5);
  BoundingBox shifted_right(3.5, 9.5, 4.5, 7.5);
  BoxUnion shifted_region;
  shifted_region.addBox(shifted_unit);
  shifted_region.addBox(shifted_two);
  shifted_region.addBox(shifted_right);
  EXPECT_DOUBLE_EQ(shifted_region.area(), 4 + 18);
  shifted_region.addBox(empty);
  EXPECT_DOUBLE_EQ(shifted_region.area(), 4 + 18);
  EXPECT_EQ(37, std::distance(shifted_region.begin(), shifted_region.end()))
      << "Iterator did not produce the expected number of grid points"
      << " in the shifted region." << std::endl
      << shifted_region << PixelIteratorRange(shifted_region);
  EXPECT_TRUE(shifted_region.contains(Point2D(4.5, 7.5)));
}

inline int expectedPixelCount(const BoundingBox &box) {
  int left = lround(floor(box.left()));
  int bottom = lround(floor(box.bottom()));
  int right = lround(ceil(box.right()));
  double w = box.width();
  if (w - floor(w) < std::numeric_limits<double>::epsilon()) {
    --right;
  }
  int top = lround(ceil(box.top()));
  double h = box.height();
  if (h - floor(h) < std::numeric_limits<double>::epsilon()) {
    --top;
  }
  return (right - left) * (top - bottom);
}

inline int expectedPixelCount(const BoxUnion &region) {
  int ans = 0;
  for (const auto &box : region.boxes()) {
    ans += expectedPixelCount(box);
  }
  return ans;
}

TEST_F(RegionTest, AreasMatch) {
  BoundingBox box1(27.272077938642145, 29.727922061357859, 27.272077938642145,
                   29.727922061357859);
  BoundingBox box2(150.14199338072009, 152.59783750343578, 12.054388463491492,
                   14.510232586207202);

  BoxUnion region(box1, box2);
  EXPECT_DOUBLE_EQ(region.area(), box1.area() + box2.area());
  EXPECT_EQ(expectedPixelCount(box1), 9);
  EXPECT_EQ(expectedPixelCount(box1) + expectedPixelCount(box2),
            expectedPixelCount(region));

  EXPECT_EQ(std::distance(region.begin(), region.end()),
            expectedPixelCount(region))
      << "Number of pixels differs from expectation" << endl
      << PixelIteratorRange(region);

  EXPECT_GE(std::distance(region.begin(), region.end()), region.area())
      << "Number of pixels should be at least as large as the area."
      << std::endl
      << region << std::endl
      << PixelIteratorRange(region);

  EXPECT_LE(
      std::distance(region.begin(), region.end()),
      (box1.right() + 1 - box1.left()) * (box1.top() + 1 - box1.bottom()) +
          (box2.right() + 1 - box2.left()) * (box2.top() + 1 - box2.bottom()))
      << "Number of pixels should not exceed the area when box dimensions "
      << "are inflated by 1.";
}

/** Given two maps that should contain identical elements, find those that are
 * missing from each.
 *
 * @param big A map that should contain everything.
 * @param partition A map obtained after processing big somehow.
 *
 */
std::string missingPixels(const std::map<Point2D, int> &big,
                          const std::map<Point2D, int> &partition) {
  std::ostringstream out;
  out << "In 'partition' but not 'big'" << endl;
  for (const auto &kv : partition) {
    if (big.find(kv.first) == big.end()) {
      out << kv.first << endl;
    }
  }

  out << "In 'big' but not 'partition'" << endl;
  for (const auto &kv : big) {
    if (partition.find(kv.first) == partition.end()) {
      out << kv.first << endl;
    }
  }
  return out.str();
}

/** Convert a real valued point to its pixel coordinates.
 */
inline Point2D pixelize(const Point2D &pt) {
  return Point2D(floor(pt.x()), floor(pt.y()));
}

/** Convert a map associating two sets of points to a string, for logging.
 */
std::string toString(std::map<Point2D, Point2D> &elements) {
  std::ostringstream out;
  for (const auto &kv : elements) {
    out << kv.first << " " << kv.second << endl;
  }
  return out.str();
}

//---------------------------------------------------------------------------
// Checks that each pixel in the full region shows up exactly once in the
// partition.
void verifyPartition(const Region &full, const Region &partition) {
  std::map<Point2D, int> fullPixelCount;
  std::map<Point2D, Point2D> fullDuplicatePixels;
  for (const auto &px : full) {
    fullPixelCount[pixelize(px)] += 1;
    if (fullPixelCount[pixelize(px)] > 1) {
      fullDuplicatePixels[px] = pixelize(px);
    }
  }
  EXPECT_TRUE(fullDuplicatePixels.empty())
      << "Some pixels in 'full' occurred more than once" << endl
      << toString(fullDuplicatePixels);

  std::map<Point2D, int> partitionPixelCount;
  std::map<Point2D, Point2D> partitionDuplicatePixels;
  for (const auto &px : partition) {
    partitionPixelCount[pixelize(px)] += 1;
    if (partitionPixelCount[pixelize(px)] > 1) {
      partitionDuplicatePixels[px] = pixelize(px);
    }
  }
  EXPECT_TRUE(partitionDuplicatePixels.empty())
      << "Some pixels in the partition occurred more than once." << endl
      << toString(partitionDuplicatePixels);

  EXPECT_EQ(fullPixelCount.size(), partitionPixelCount.size())
      << missingPixels(fullPixelCount, partitionPixelCount);
}

TEST_F(RegionTest, GridEachPixelExactlyOnce) {
  BoundingBox bigRegion(0, 100, 0, 100);
  ImageGrid grid(bigRegion, 12);
  verifyPartition(bigRegion, grid);
}

TEST_F(RegionTest, BoxPartitionEachPixelExactlyOnce) {
  BoundingBox box(0, 100, 0, 100);
  BoundingBox integerPixels(10, 20, 30, 40);
  verifyPartition(box, BoxUnion(box, integerPixels));

  BoundingBox fractionalPixels(9.5, 20.2, 30.8, 37.9);
  verifyPartition(box, BoxUnion(box, fractionalPixels));
}

} // namespace
