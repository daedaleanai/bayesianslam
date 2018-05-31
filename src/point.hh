#pragma once

#include <cmath>
#include <ostream>

namespace slam {
//===========================================================================
// A point in two-dimensional space.
class Point2D {
public:
  Point2D() : x_(0), y_(0) {}
  Point2D(double x, double y) : x_(x), y_(y) {}
  double x() const { return x_; }
  double y() const { return y_; }

  bool operator<(const Point2D &rhs) const {
    if (x_ < rhs.x()) {
      return true;
    } else if (x_ > rhs.x()) {
      return false;
    } else {
      return y_ < rhs.y();
    }
  }

  bool operator==(const Point2D &rhs) const {
    return x_ == rhs.x_ && y_ == rhs.y_;
  }

private:
  double x_;
  double y_;
};

inline std::ostream &operator<<(std::ostream &out, const Point2D &point) {
  out << "(" << point.x() << ", " << point.y() << ")";
  return out;
}

inline double distance(const Point2D &p1, const Point2D &p2) {
  return std::hypot(p1.x() - p2.x(), p1.y() - p2.y());
}

} // namespace slam
