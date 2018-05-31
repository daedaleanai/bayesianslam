/** @file Contains tools for modeling subsets of a screen.  The primary tools
 * are bounding boxes, and unions and intersections of bounding boxes.  Each of
 * these provides an iterator class that can be used to iterate over all the
 * pixels in the region.
 *
 * NOTE: When we move on to modeling images in 3D space we will need separate
 * notions for pixel-related bounding boxes and physical-space related bounding
 * boxes.  Perspective implies a mapping from one to another, but it may be a
 * complicated mapping.
 */
#pragma once

#include "exp/user/stb/slam/point.hh"
#include <memory>
#include <set>
#include <vector>

/**
 * @namespace
 * The slam namespace is for the experimental slam algorithm.
 */
namespace slam {
class BoundingBox;
class BoxUnion;
class BoundingBoxPixelIterator;
class BoxUnionPixelIterator;
class Screen2D;

//===========================================================================
/** @class PixelIteratorImpl
 * A based class defining the interface to pixel iterators.
 */
class PixelIteratorImpl {
public:
  virtual void increment() = 0;
  virtual Point2D value() const = 0;
  virtual void moveToEnd() = 0;
  virtual bool equals(const PixelIteratorImpl &rhs) const = 0;
  virtual bool equals(const BoundingBoxPixelIterator &rhs) const = 0;
  virtual bool equals(const BoxUnionPixelIterator &rhs) const = 0;
};
//---------------------------------------------------------------------------
/** @class PixelIterator
 * Implements the pixel iterator used to iterate over an
 * arbitrary region of a screen.
 */
class PixelIterator {
public:
  typedef Point2D value_type;
  typedef int64_t difference_type;
  typedef value_type *pointer;
  typedef const value_type &reference;
  typedef std::forward_iterator_tag iterator_category;

  PixelIterator(const std::vector<BoundingBox> &boxes);
  PixelIterator(const BoundingBox &box);

  PixelIterator &operator++() {
    impl_->increment();
    return *this;
  }
  Point2D operator*() const { return impl_->value(); }
  void moveToEnd() { impl_->moveToEnd(); }

  bool operator==(const PixelIterator &rhs) const {
    return impl_->equals(*rhs.impl_);
  }
  bool operator!=(const PixelIterator &rhs) const { return !(*this == rhs); }

private:
  std::shared_ptr<PixelIteratorImpl> impl_;
};

//===========================================================================
/** @class Region
 * A subset of a screen representable by unions and intersections of boxes.
 */
class Region {
public:
  virtual bool empty() const = 0;
  virtual PixelIterator begin() const = 0;
  virtual PixelIterator end() const = 0;

  virtual double area() const = 0;

  /** Check whether this region intersects another.
   * @param[region] The region to check for an intersection with *this.
   * @param[strict] If true then ignore intersections that only occur on the
   *   boundary.
   *
   * @return true if this region intersects the function argument.
   */
  virtual bool intersects(const Region &region, bool strict) const = 0;
  virtual bool intersects(const BoundingBox &region, bool strict) const = 0;
  virtual bool intersects(const BoxUnion &region, bool strict) const = 0;

  virtual std::ostream &print(std::ostream &out) const = 0;

  virtual void computeIntersection(const Region &rhs,
                                   BoxUnion &result) const = 0;
  virtual void computeIntersection(const BoundingBox &rhs,
                                   BoxUnion &result) const = 0;
  virtual void computeIntersection(const BoxUnion &rhs,
                                   BoxUnion &result) const = 0;
};

//---------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &out, const Region &box) {
  return box.print(out);
}

//===========================================================================
/** @class BoundingBox
 * A rectangle aligned to the horizontal and vertical axes, modeling a
 * rectangular subset of a screen.
 */
class BoundingBox : public Region {
public:
  // An empty bounding box.
  BoundingBox();

  // A bounding box that covers the whole screen.
  BoundingBox(const Screen2D &screen);

  // Args:
  //   left, right, bottom, top: The coordinates of the box, measured in
  //     pixels.  It is the caller's responsibility to ensure left <= right
  //     and bottom <= top.
  //
  // When iterating through the pixels in a bounding box, the top and right
  // boundaries are considered not to be owned by the box.  Thus if creating a
  // box manually, it is wise to add a 1-pixel buffer zone on the top and
  // right.
  //
  // Omitting the top and right boundaries means that boxes can have immediate
  // vertical and horizontal neighbors without duplicating the boundary
  // pixels.
  BoundingBox(double left, double right, double bottom, double top);

  // Predicate checking whether the bounding box is empty.
  bool empty() const override { return (left_ == right_) || (top_ == bottom_); }

  // The less-than operator must be defined so that bounding boxes can be used
  // with std::map.
  bool operator<(const BoundingBox &rhs) const;

  PixelIterator begin() const override;
  PixelIterator end() const override;

  bool operator==(const BoundingBox &rhs) const {
    return left_ == rhs.left_ && right_ == rhs.right_ &&
           bottom_ == rhs.bottom_ && top_ == rhs.top_;
  }

  bool operator!=(const BoundingBox &rhs) const { return !((*this) == rhs); }

  // Check if the point (x, y) is contained within the box, either in the
  // interior or on the lower boundary.  The box does not own the upper
  // boundary.
  bool contains(double x, double y) const {
    return x >= left_ && x <= right_ && y <= top_ && y >= bottom_;
  }
  bool contains(const Point2D &point) const {
    return contains(point.x(), point.y());
  }

  double width() const { return empty() ? 0 : right_ - left_; }
  double height() const { return empty() ? 0 : top_ - bottom_; }
  double volume() const { return width() * height(); }
  double area() const override { return volume(); }

  // Check whether this box intersects with rhs.  Two boxes intersect if they
  // intersect in both the horizontal and vertical directions.
  //
  // Args:
  //   rhs:  A box that might intersect with this.
  //   strict: If true an intersection is only declared if it has positive
  //     area.  If false then the intersection can be at a point or an edge.
  bool intersects(const BoundingBox &rhs, bool strict) const override {
    return intervals_overlap(left_, right_, rhs.left_, rhs.right_, strict) &&
           intervals_overlap(bottom_, top_, rhs.bottom_, rhs.top_, strict);
  }
  bool intersects(const Region &region, bool strict) const override {
    // Double dispatch.
    return region.intersects(*this, strict);
  }
  bool intersects(const BoxUnion &rhs, bool strict) const override;

  // Compute (rather than check the existence of) the bounding box formed by
  // the intersection of *this and rhs.  If there is no intersection then an
  // empty box is returned.
  BoundingBox intersection(const BoundingBox &rhs) const;

  void computeIntersection(const Region &rhs, BoxUnion &result) const override;
  void computeIntersection(const BoundingBox &rhs,
                           BoxUnion &result) const override;
  void computeIntersection(const BoxUnion &rhs,
                           BoxUnion &result) const override;

  // Returns the smallest box containing both *this and rhs.
  BoundingBox super_box(const BoundingBox &rhs) const;

  double left() const { return left_; }
  double right() const { return right_; }
  double bottom() const { return bottom_; }
  double top() const { return top_; }

  std::ostream &print(std::ostream &out) const override;

public:
  // Returns true iff the interval [lo0, hi0] intersects the interval [lo1,
  // hi1].  The intersection can be at a single point.
  //
  // This function assumes lo0 <= hi0 and lo1 <= hi1.  It is the caller's
  // responsibility to ensure this assumption is valid.
  bool intervals_overlap(double lo0, double hi0, double lo1, double hi1,
                         bool strict) const {
    if (strict) {
      return (hi0 > lo1) && (hi1 > lo0);
    } else {
      return (hi0 >= lo1) && (hi1 >= lo0);
    }
  }

  double left_;
  double right_;
  double bottom_;
  double top_;
};
//---------------------------------------------------------------------------
// For iterating through the pixels in a BoundingBox.
class BoundingBoxPixelIterator : public PixelIteratorImpl {
public:
  BoundingBoxPixelIterator();
  BoundingBoxPixelIterator(const BoundingBox &box);
  void increment() override;
  void operator++() { increment(); }
  Point2D value() const override { return Point2D(x_, y_); }
  void moveToEnd() override;
  bool atEnd() const;

  bool equals(const PixelIteratorImpl &rhs) const override {
    return rhs.equals(*this);
  }
  bool equals(const BoundingBoxPixelIterator &rhs) const override;
  bool equals(const BoxUnionPixelIterator &rhs) const override { return false; }

private:
  BoundingBox box_;
  double x_;
  double y_;
};

//===========================================================================
// A set union of bounding boxes, for modeling complex regions of the screen.
class BoxUnion : public Region {
public:
  BoxUnion() {}
  BoxUnion(const BoundingBox &box1, const BoundingBox &box2);

  PixelIterator begin() const override;
  PixelIterator end() const override;

  std::ostream &print(std::ostream &out) const override;

  // The bounding box of the region contianed in this object.
  BoundingBox boundingBox() const;

  // Add a box to the union.  If the box intersects any current boxes it will
  // be chopped into disjoint pieces, and those will be added to the union
  // instead.
  void addBox(const BoundingBox &box) { addBox(box, 0); }

  // Augment the current set with another box union, creating a larger union.
  void addUnion(const BoxUnion &box_union);

  bool empty() const override { return boxes_.empty(); }

  bool intersects(const Region &region, bool strict) const override {
    return region.intersects(*this, strict);
  }
  bool intersects(const BoundingBox &region, bool strict) const override;
  bool intersects(const BoxUnion &region, bool strict) const override;

  BoxUnion intersection(const BoxUnion &rhs) const;
  BoxUnion intersection(const BoundingBox &rhs) const;

  void computeIntersection(const Region &rhs, BoxUnion &result) const override;
  void computeIntersection(const BoundingBox &rhs,
                           BoxUnion &result) const override;
  void computeIntersection(const BoxUnion &rhs,
                           BoxUnion &result) const override;

  double area() const override;
  bool contains(const Point2D &point) const;

  const std::vector<BoundingBox> &boxes() const { return boxes_; }

protected:
  // Add a box to the set of boxes, with intersection checking starting from a
  // particular index.  The box being added must be disjoint from all boxes
  // prior to the starting point.
  void addBox(const BoundingBox &box, int start_from);

private:
  // The set of non-overlapping boxes comprising the union.
  std::vector<BoundingBox> boxes_;
};

//---------------------------------------------------------------------------
class BoxUnionPixelIterator : public PixelIteratorImpl {
public:
  // The pixel iterator starts at the beginning.
  BoxUnionPixelIterator(const std::vector<BoundingBox> &boxes);

  void increment() override;
  Point2D value() const override { return position_.value(); }

  bool atEnd() const;
  void moveToEnd() override;

  bool equals(const PixelIteratorImpl &rhs) const override {
    return rhs.equals(*this);
  }
  bool equals(const BoundingBoxPixelIterator &rhs) const override {
    return false;
  }
  bool equals(const BoxUnionPixelIterator &rhs) const override;

private:
  // Values of the current pixel.  Pixels start at the bottom left of the box.
  // They work their way up y, then move to the next x and start at the bottom
  // again.
  const std::vector<BoundingBox> &boxes_;
  int currentBox_;
  BoundingBoxPixelIterator position_;

  // Keeps track of the pixels that have already been visited, to make sure the
  // same pixel does not get served twice.
  std::set<Point2D> visited_;
};

//===========================================================================
// An image grid is a grid of square (maybe rectangle at the boundary) cells
// partitioning an image.
class ImageGrid : public BoxUnion {
public:
  // Partition a region into square cells of a given size.
  ImageGrid(const BoundingBox &region, int cell_size);

  // The grid is laid out like a matrix.  The first row (row 0) is on top.
  const BoundingBox &operator()(int i, int j) const;

  // Number of grid rows.
  int nrow() const { return nrow_; }

  // Number of grid colums.
  int ncol() const { return ncol_; }

private:
  int nrow_;
  int ncol_;
};

//---------------------------------------------------------------------------
// Return a vector of BoundingBox objects created by removing the intersection
// of box1 and box 2 from box 1.  The vector can contain as few as 0 and as
// many as 4 boxes.
//
// Args:
//   box1, box2:  The boxes used to construct the set difference.
//   neighbors: A set of flags indicating whether the intersection of box1 and
//     box2 has neighbors on the left, right, bottom, or top.
// Returns:
//   A vector containing between 0 and 4 disjoint bounding boxes, whose union is
//   the set of points in box1 but not in box 2.  Some boxes will overlap along
//   a set of zero area (e.g. the boundary lines between two boxes).
std::vector<BoundingBox> setDifference(const BoundingBox &box1,
                                       const BoundingBox &box2);

BoxUnion intersection(const Region &region1, const Region &region2);

} // namespace slam
