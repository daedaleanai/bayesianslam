#pragma once

#include <memory>
#include <vector>

#include "exp/user/stb/slam/object_sampler.hh"
#include "exp/user/stb/slam/objects.hh"

namespace slam {

// An element in the particle filter that maintains the belief state over the
// environment.
class Particle {
public:
  // How to initialize objects and background??
  Particle();
  Particle(const Particle &rhs);
  Particle *clone() const { return new Particle(*this); }

  // The number of objects being tracked.
  int number_ofObjects() const { return objects_.size(); }

  std::vector<std::shared_ptr<Object2D>> &objects() { return objects_; }
  const std::vector<std::shared_ptr<Object2D>> &objects() const {
    return objects_;
  }

  // Identify objects that have passed off the screen and remove them from
  // particles.
  void remove_passedObjects(const Screen2D &screen);

private:
  // Note that objects are ordered.  If objects overlap, then earlier objects
  // obscure later ones.
  std::vector<std::shared_ptr<Object2D>> objects_;
};

//======================================================================
class AircraftPose2D {
public:
  AircraftPose2D(double x, double y, double velocity_x, double velocity_y)
      : x_(x), y_(y), velocity_x_(velocity_x), velocity_y_(velocity_y) {}

  // In this version of things we know the true state of the aircraft.  Right
  // now we're just trying to learn about obstacles.  In the future advancing
  // the pose will involve a filtering step, where the propagation step is
  // determined by position and velocity (and maybe random noise), and the
  // updating step is determined by the sensor image in the new location.
  void advance() {
    x_ += velocity_x_;
    y_ += velocity_y_;
  }
  double x() const { return x_; }
  double y() const { return y_; }
  double velocity_x() const { return velocity_x_; }
  double velocity_y() const { return velocity_y_; }

private:
  double x_;
  double y_;
  double velocity_x_;
  double velocity_y_;
  //    std::vector<EnvironmentParticle> belief_;
};
//======================================================================
class Aircraft2D {
public:
  Aircraft2D(int number_of_particles,
             const std::shared_ptr<Background2D> &background,
             const std::shared_ptr<ObjectSampler> &sampler);

  // Observe the data on the screen and use it to update the belief state
  // represented by the particles.
  void updateBelief(BOOM::RNG &rng, const Screen2D &screen);

  std::vector<std::shared_ptr<Particle>> &particles() { return particles_; }
  const std::vector<std::shared_ptr<Particle>> &particles() const {
    return particles_;
  }

  //----------------------------------------------------------------------
  // This section contains implementation steps for updateBelief.

  // Update the pose of the aircraft.
  // Args:
  //   screen:  The screen showing the aircraft's view of the world.
  //
  // Returns:
  //   The bounding box giving the previously unobserved pixels on the screen.
  BoundingBox updatePose(const Screen2D &screen);

  // Find objects that are no longer on the screen, and remove them from any
  // particles that were tracking them.  Maybe pass them to some sort of long
  // term storage in case we revist the same point on the map later.
  void removePassedObjects(const Screen2D &screen);

  // Resample particles conditional on the new screen view.
  void updateParticles(const Screen2D &screen);
  void updateParticle(Particle &particle, const BoundingBox &box,
                      const Screen2D &screen,
                      const std::vector<Particle> &originalParticles,
                      const std::shared_ptr<Background2D> &background);

  double logLikelihood(const Screen2D &screen, const BoundingBox &box,
                       const std::vector<std::shared_ptr<Object2D>> &objects,
                       const std::shared_ptr<Background2D> &background) const;

  void setCirclePrior(Ptr<GaussianModelGivenSigma> &meanPrior,
                      Ptr<ChisqModel> &precisionPrior);
  void setSquarePrior(Ptr<GaussianModelGivenSigma> &meanPrior,
                      Ptr<ChisqModel> &precisionPrior);

  ObjectSampler *objectSampler() { return objectSampler_.get(); }

private:
  // The set of particles responsible for maintaining the belief state of the
  // aircraft.
  std::vector<std::shared_ptr<Particle>> particles_;
  std::shared_ptr<AircraftPose2D> pose_;
  std::shared_ptr<Background2D> background_;

  Ptr<GaussianModelGivenSigma> squareColorMeanPrior_;
  Ptr<ChisqModel> squareColorPrecisionPrior_;

  Ptr<GaussianModelGivenSigma> circleColorMeanPrior_;
  Ptr<ChisqModel> circleColorPrecisionPrior_;

  std::shared_ptr<ObjectSampler> objectSampler_;
};

} // namespace slam
