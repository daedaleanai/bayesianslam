#include "exp/user/stb/slam/aircraft.hh"
#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"
#include <functional> // for lambdas

#include "distributions.hpp"

namespace slam {
using namespace BOOM;

Particle::Particle() {}

Particle::Particle(const Particle &rhs) {
  for (const auto &el : rhs.objects_) {
    objects_.emplace_back(el->clone());
  }
}

Aircraft2D::Aircraft2D(int numberOfParticles,
                       const std::shared_ptr<Background2D> &background,
                       const std::shared_ptr<ObjectSampler> &sampler)
    : pose_(new AircraftPose2D(0, 0, 0, 0)), background_(background),
      objectSampler_(sampler) {
  particles_.reserve(numberOfParticles);
  for (int i = 0; i < numberOfParticles; ++i) {
    particles_.emplace_back(new Particle);
  }
}

void Aircraft2D::updateBelief(RNG &rng, const Screen2D &screen) {
  BoundingBox new_pixels = updatePose(screen);
  removePassedObjects(screen);
  objectSampler_->scanForNewObjects(rng, screen, new_pixels, 100);
  updateParticles(screen);
}

// Right now we're assuming the pose is deterministic.  The return value is a
// bounding box containing the new pixels that appear on the screen as a
// result of the aircraft's movement.
BoundingBox Aircraft2D::updatePose(const Screen2D &screen) {
  pose_->advance();
  return BoundingBox(screen.left(), screen.right() - 1,
                     screen.top() - pose_->velocity_y(), screen.top());
}

void Aircraft2D::removePassedObjects(const Screen2D &screen) {
  BoundingBox screenBox(screen);
  for (int i = 0; i < particles_.size(); ++i) {
    std::remove_if(
        particles_[i]->objects().begin(), particles_[i]->objects().end(),
        [screenBox](const std::shared_ptr<Object2D> &object) {
          return object->boundingBox().intersection(screenBox).empty();
        });
  }
}

void Aircraft2D::updateParticles(const Screen2D &screen) {
  ////////// Need to deep copy the particles....
  std::vector<std::shared_ptr<Particle>> originalParticles(particles_);
  // for (auto &particle : particles_) {
  //   for (const auto &objectp : particle.objects()) {
  //     updateParticle(i, box, screen, originalParticles);
  //   }
  // }
}

void Aircraft2D::updateParticle(
    Particle &particle, const BoundingBox &box, const Screen2D &screen,
    const std::vector<Particle> &originalParticles,
    const std::shared_ptr<Background2D> &background) {
  Vector log_weights(particles_.size());
  for (int i = 0; i < originalParticles.size(); ++i) {
    log_weights[i] =
        logLikelihood(screen, box, originalParticles[i].objects(), background);
  }
}

//===========================================================================
// Compute the conditional log likelihood of an image subset.
// Args:
//   screen: The screen containing the image seen by the aircraft.
//   box: A bounding box identifying a subset of the screen over which to
//     compute the likelihood.
//   objects:  An ordered sequence of objects from the particle being evaluated.
//   background:  The model for the background clutter.
//
// Returns:
//   The log likelihood of the subset of the image in the box.
double Aircraft2D::logLikelihood(
    const Screen2D &screen, const BoundingBox &box,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) const {
  // Find the portion of 'box' that is actually on the screen.
  const BoundingBox visibleBox(BoundingBox(screen).intersection(box));
  if (visibleBox.empty()) {
    return 0;
  }

  // Locate the set of objects whose bounding box intersects box.
  std::vector<std::shared_ptr<Object2D>> candidates;
  for (const auto &objectptr : objects) {
    if (objectptr->boundingBox().intersects(visibleBox, true)) {
      candidates.push_back(objectptr);
    }
  }

  double loglike = 0;
  for (int x = visibleBox.left(); x <= visibleBox.right(); ++x) {
    for (double y = visibleBox.bottom(); y <= visibleBox.top(); ++y) {
      double observation = screen(x, y, Screen2D::LOGIT);
      double logLikelihoodContribution = negative_infinity();
      for (const auto &objectptr : candidates) {
        if (objectptr->contains(Point2D(x, y))) {
          logLikelihoodContribution = dnorm(observation, objectptr->colorMean(),
                                            objectptr->colorSd(), true);
          break;
        }
      }
      if (logLikelihoodContribution == negative_infinity()) {
        logLikelihoodContribution = dnorm(observation, background->colorMean(),
                                          background->colorSd(), true);
      }
      loglike += logLikelihoodContribution;
    }
  }
  return loglike;
}

void Aircraft2D::setCirclePrior(Ptr<GaussianModelGivenSigma> &meanPrior,
                                Ptr<ChisqModel> &precisionPrior) {
  circleColorMeanPrior_ = meanPrior;
  circleColorPrecisionPrior_ = precisionPrior;
}

void Aircraft2D::setSquarePrior(Ptr<GaussianModelGivenSigma> &meanPrior,
                                Ptr<ChisqModel> &precisionPrior) {
  squareColorMeanPrior_ = meanPrior;
  squareColorPrecisionPrior_ = precisionPrior;
}

} // namespace slam
