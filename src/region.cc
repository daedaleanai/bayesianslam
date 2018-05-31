#include "exp/user/stb/slam/region.hh"
#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"

#include "cpputil/report_error.hpp"

namespace slam {
using namespace BOOM;

BoundingBox::BoundingBox()
    : left_(BOOM::negative_infinity()), right_(BOOM::negative_infinity()),
      bottom_(BOOM::negative_infinity()), top_(BOOM::negative_infinity()) {}

BoundingBox::BoundingBox(double left, double right, double bottom, double top)
    : left_(left), right_(right), bottom_(bottom), top_(top) {
  if (right_ < left_) {
    BOOM::report_error(
        "Right edge of the bounding box must be larger than left.");
  } else if (top_ < bottom_) {
    BOOM::report_error(
        "Top edge of the bounding box must be larger than the bottom.");
  }
}

BoundingBox::BoundingBox(const Screen2D &screen)
    : BoundingBox(screen.left(), screen.right(), screen.bottom(),
                  screen.top()) {}

PixelIterator BoundingBox::begin() const { return PixelIterator(*this); }

PixelIterator BoundingBox::end() const {
  PixelIterator ans(*this);
  ans.moveToEnd();
  return ans;
}

bool BoundingBox::intersects(const BoxUnion &rhs, bool strict) const {
  return rhs.intersects(*this, strict);
}

BoundingBox BoundingBox::intersection(const BoundingBox &rhs) const {
  double left = std::max(left_, rhs.left_);
  double bottom = std::max(bottom_, rhs.bottom_);

  double right = std::min(right_, rhs.right_);
  double top = std::min(top_, rhs.top_);
  if (left > right || top < bottom) {
    return BoundingBox();
  } else {
    return BoundingBox(left, right, bottom, top);
  }
}

void BoundingBox::computeIntersection(const Region &rhs,
                                      BoxUnion &result) const {
  rhs.computeIntersection(*this, result);
}

void BoundingBox::computeIntersection(const BoundingBox &rhs,
                                      BoxUnion &result) const {
  result.addBox(intersection(rhs));
}

void BoundingBox::computeIntersection(const BoxUnion &rhs,
                                      BoxUnion &result) const {
  result = rhs.intersection(*this);
}

bool BoundingBox::operator<(const BoundingBox &rhs) const {
  if (left_ < rhs.left_) {
    return true;
  } else if (left_ == rhs.left_) {
    if (bottom_ < rhs.bottom_)
      return true;
    else if (bottom_ == rhs.bottom_) {
      if (right_ < rhs.right_)
        return true;
      else if (right_ == rhs.right_) {
        if (top_ < rhs.top_)
          return true;
      }
    }
  }
  return false;
}

BoundingBox BoundingBox::super_box(const BoundingBox &rhs) const {
  return BoundingBox(std::min(left_, rhs.left_), std::max(right_, rhs.right_),
                     std::min(bottom_, rhs.bottom_), std::max(top_, rhs.top_));
}

std::ostream &BoundingBox::print(std::ostream &out) const {
  out << endl
      << "left:   " << setw(10) << left_ << " right:  " << right_ << endl
      << "bottom: " << setw(10) << bottom_ << " top:    " << top_ << endl;
  return out;
}

//===========================================================================
PixelIterator::PixelIterator(const std::vector<BoundingBox> &boxes)
    : impl_(new BoxUnionPixelIterator(boxes)) {}

PixelIterator::PixelIterator(const BoundingBox &box)
    : impl_(new BoundingBoxPixelIterator(box)) {}

//---------------------------------------------------------------------------
BoundingBoxPixelIterator::BoundingBoxPixelIterator()
    : x_(infinity()), y_(infinity()) {}

BoundingBoxPixelIterator::BoundingBoxPixelIterator(const BoundingBox &box)
    : box_(box) {
  x_ = floor(box_.left());
  y_ = floor(box_.bottom());
  if (box_.empty()) {
    moveToEnd();
  }
}

void BoundingBoxPixelIterator::increment() {
  if (atEnd()) {
    moveToEnd();
    return;
  }
  ++y_;
  if (y_ > box_.top()) {
    ++x_;
    if (x_ > box_.right()) {
      moveToEnd();
      return;
    }
    y_ = floor(box_.bottom());
  }
}

bool BoundingBoxPixelIterator::equals(
    const BoundingBoxPixelIterator &rhs) const {
  return x_ == rhs.x_ && y_ == rhs.y_ && box_ == rhs.box_;
}

void BoundingBoxPixelIterator::moveToEnd() {
  x_ = infinity();
  y_ = infinity();
}

bool BoundingBoxPixelIterator::atEnd() const {
  return box_.empty() || x_ > box_.right();
}

//---------------------------------------------------------------------------
BoxUnionPixelIterator::BoxUnionPixelIterator(
    const std::vector<BoundingBox> &boxes)
    : boxes_(boxes), currentBox_(0) {
  if (boxes_.empty()) {
    moveToEnd();
  } else {
    position_ = BoundingBoxPixelIterator(boxes_[0]);
    visited_.insert(value());
  }
}

void BoxUnionPixelIterator::increment() {
  if (atEnd()) {
    moveToEnd();
    return;
  }
  ++position_;
  if (position_.atEnd()) {
    ++currentBox_;
    if (currentBox_ < boxes_.size()) {
      position_ = BoundingBoxPixelIterator(boxes_[currentBox_]);
    } else {
      moveToEnd();
    }
  }

  if (!atEnd()) {
    Point2D pixel = position_.value();
    if (visited_.find(pixel) != visited_.end()) {
      // This pixel has already been shown.  Increment to the next one.
      increment();
    } else {
      visited_.insert(pixel);
    }
  }
}

bool BoxUnionPixelIterator::atEnd() const {
  return currentBox_ == boxes_.size();
}

void BoxUnionPixelIterator::moveToEnd() {
  currentBox_ = boxes_.size();
  position_ = BoundingBoxPixelIterator();
}

bool BoxUnionPixelIterator::equals(const BoxUnionPixelIterator &other) const {
  // The vector comparison is last because it is the most expensive.
  return currentBox_ == other.currentBox_ &&
         position_.equals(other.position_) && boxes_ == other.boxes_;
}

//===========================================================================
BoxUnion::BoxUnion(const BoundingBox &box1, const BoundingBox &box2)
    : boxes_(1, box1) {
  addBox(box2);
}

PixelIterator BoxUnion::begin() const { return PixelIterator(boxes_); }

PixelIterator BoxUnion::end() const {
  PixelIterator ans(boxes_);
  ans.moveToEnd();
  return ans;
}

std::ostream &BoxUnion::print(std::ostream &out) const {
  for (const auto &box : boxes_) {
    out << "   " << box;
  }
  return out;
}

void BoxUnion::addBox(const BoundingBox &box, int startFrom) {
  if (box.empty() || box.area() == 0.0) {
    return;
  }
  for (int i = startFrom; i < boxes_.size(); ++i) {
    if (box.intersects(boxes_[i], true)) {
      // If the box being added intersects with any existing boxes, then
      // 1-A) Partition the existing box into the intersection, and a vector of
      //      smaller disjoint boxes.
      // 1-B) Replace the current box with the intersection, and put the
      // disjoint
      //      boxes at the end.
      // 2-A) Partition the added box into the intersection with the existing
      // box,
      //      and a set of disjoint boxes.
      // 2-B) Discard the intersection and call addBox on the remaining boxes.
      //      You know they don't intersect any box before component i.
      BoundingBox overlap = boxes_[i].intersection(box);
      std::vector<BoundingBox> alreadyIncluded =
          setDifference(boxes_[i], overlap);

      std::vector<BoundingBox> needsInclusion = setDifference(box, overlap);
      boxes_.insert(boxes_.end(), alreadyIncluded.begin(),
                    alreadyIncluded.end());
      boxes_[i] = overlap;

      for (int j = 0; j < needsInclusion.size(); ++j) {
        if (!needsInclusion[j].empty()) {
          addBox(needsInclusion[j], i);
        }
      }
      // A return statement is needed here because all the pieces of the box
      // have been added.
      return;
    }
  }
  // If the code reaches this point then there is no intersection between
  // 'box' and any item in boxes_, so box should be added to the union.
  boxes_.push_back(box);
}

//-----------------------------------------------------------------
void BoxUnion::addUnion(const BoxUnion &box_union) {
  for (const auto &box : box_union.boxes_) {
    addBox(box);
  }
}

bool BoxUnion::intersects(const BoundingBox &box, bool strict) const {
  for (const auto &b : boxes_) {
    if (box.intersects(b, strict)) {
      return true;
    }
  }
  return false;
}

bool BoxUnion::intersects(const BoxUnion &region, bool strict) const {
  for (const auto &b1 : boxes_) {
    for (const auto &b2 : region.boxes_) {
      if (b1.intersects(b2, strict)) {
        return true;
      }
    }
  }
  return false;
}

BoxUnion BoxUnion::intersection(const BoundingBox &box) const {
  std::vector<BoundingBox> smaller_boxes;
  for (const auto &b : boxes_) {
    BoundingBox this_box = b.intersection(box);
    if (!this_box.empty()) {
      smaller_boxes.push_back(this_box);
    }
  }
  BoxUnion ans;
  for (int i = 0; i < smaller_boxes.size(); ++i) {
    ans.addBox(smaller_boxes[i], i);
  }
  return ans;
}

BoxUnion BoxUnion::intersection(const BoxUnion &rhs) const {
  std::vector<BoundingBox> smaller_boxes;
  for (const auto &b1 : boxes_) {
    for (const auto &b2 : rhs.boxes_) {
      if (b1.intersects(b2, true)) {
        smaller_boxes.emplace_back(b1.intersection(b2));
      }
    }
  }
  BoxUnion ans;
  for (int i = 0; i < smaller_boxes.size(); ++i) {
    ans.addBox(smaller_boxes[i], i);
  }
  return ans;
}

BoundingBox BoxUnion::boundingBox() const {
  if (boxes_.empty()) {
    return BoundingBox();
  }
  double left = BOOM::infinity();
  double bottom = left;
  double top = BOOM::negative_infinity();
  double right = top;
  for (const auto &box : boxes_) {
    left = std::min(left, box.left());
    bottom = std::min(bottom, box.bottom());
    right = std::max(right, box.right());
    top = std::max(top, box.top());
  }
  return BoundingBox(left, right, bottom, top);
}

void BoxUnion::computeIntersection(const Region &rhs, BoxUnion &result) const {
  rhs.computeIntersection(*this, result);
}

void BoxUnion::computeIntersection(const BoundingBox &rhs,
                                   BoxUnion &result) const {
  result = intersection(rhs);
}

void BoxUnion::computeIntersection(const BoxUnion &rhs,
                                   BoxUnion &result) const {
  result = intersection(rhs);
}

double BoxUnion::area() const {
  double ans = 0;
  for (const auto &box : boxes_) {
    ans += box.area();
  }
  return ans;
}

bool BoxUnion::contains(const Point2D &point) const {
  for (const auto &box : boxes_) {
    if (box.contains(point))
      return true;
  }
  return false;
}

//===========================================================================
// Return the region of points in box1 but not box2.
std::vector<BoundingBox> setDifference(const BoundingBox &box1,
                                       const BoundingBox &box2) {
  std::vector<BoundingBox> ans;

  BoundingBox overlap = box1.intersection(box2);
  if (overlap == box1) {
    // If box1 is a proper subset of box2 then return ans empty vector.
    return ans;
  }

  if (overlap.empty()) {
    // If there is no intersection, then the set difference is just box1.
    ans.push_back(box1);
    return ans;
  }

  if (overlap.left() > box1.left()) {
    // Remove a vertical strip on the left.
    BoundingBox leftBox(box1.left(), overlap.left(), box1.bottom(), box1.top());
    if (leftBox.area() > 0) {
      ans.push_back(leftBox);
    }
  }

  if (box1.right() > overlap.right()) {
    // Remove a vertical strip on the right.
    BoundingBox rightBox(overlap.right(), box1.right(), box1.bottom(),
                         box1.top());
    if (rightBox.area() > 0) {
      ans.push_back(rightBox);
    }
  }

  if (overlap.bottom() > box1.bottom()) {
    // Remove the box below the intersection.  The left and right strips have
    // aleady been removed.
    BoundingBox bottomBox(overlap.left(), overlap.right(), box1.bottom(),
                          overlap.bottom());
    if (bottomBox.area() > 0) {
      ans.push_back(bottomBox);
    }
  }

  if (box1.top() > overlap.top()) {
    BoundingBox topBox(overlap.left(), overlap.right(), overlap.top(),
                       box1.top());
    if (topBox.area() > 0) {
      ans.push_back(topBox);
    }
  }
  return ans;
}
//===========================================================================
BoxUnion intersection(const Region &r1, const Region &r2) {
  BoxUnion ans;
  r1.computeIntersection(r2, ans);
  return ans;
}
//===========================================================================

// Vertical pixels move fastest, so grid cells are stored in column major
// order.  The bottom row is first, then moving to the top, then move right
// one strip and repeat.
ImageGrid::ImageGrid(const BoundingBox &region, int cell_size) {
  ncol_ = 0;
  for (double left = region.left(); left < region.right(); left += cell_size) {
    double right = std::min(region.right(), left + cell_size - .01);
    if (right > left) {
      ++ncol_;
      for (double bottom = region.bottom(); bottom < region.top();
           bottom += cell_size) {
        double top = std::min(region.top(), bottom + cell_size - .01);
        if (top > bottom) {
          addBox(BoundingBox(left, right, bottom, top), boxes().size());
        }
      }
    }
  }
  if (ncol_ > 0) {
    nrow_ = boxes().size() / ncol_;
  } else {
    nrow_ = 0;
  }
}

const BoundingBox &ImageGrid::operator()(int i, int j) const {
  if (i < 0 || j < 0 || i >= nrow_ || j >= ncol_) {
    report_error("Index out of bounds.");
  }
  return boxes()[nrow_ * j + i];
}
} // namespace slam
