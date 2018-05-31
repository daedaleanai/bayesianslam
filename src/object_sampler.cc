#include "exp/user/stb/slam/object_sampler.hh"

namespace slam {
using namespace BOOM;

ObjectSampler::ObjectSampler(const std::shared_ptr<Background2D> &background,
                             int gridSearchSize)
    : background_(background), gridSearchSize_(gridSearchSize) {}

void ObjectSampler::scanForNewObjects(RNG &rng, const Screen2D &screen,
                                      const BoundingBox &region, int niter) {
  // Fill the region with randomly chosen objects by dividing the region into
  // search squares, and placing a randomly chosen object in each search
  // square.
  BoundingBox visible_region = region.intersection(BoundingBox(screen));
  std::vector<std::shared_ptr<Object2D>> objects =
      initializeObjectSearch(rng, screen, visible_region);

  for (int i = 0; i < niter; ++i) {
    std::cout << "iteration " << i << std::endl;
    for (int j = 0; j < 10; ++j) {
      addRemoveObjects(rng, screen, visible_region, objects);
    }

    if (objectPriors_.size() >= 2) {
      for (int j = 0; j < objects.size(); ++j) {
        morphSingleObject(j, rng, screen, region, objects);
      }
    }
    std::cout << "There are " << objects.size() << " objects." << endl;
    moveAndResizeObjects(rng, screen, visible_region, objects);
    //      splitCombineObjects(rng, screen, visible_region, objects);

    recordObjects(objects);
    std::cout << mhMoveResults_.to_matrix() << endl;
  }
}

void ObjectSampler::addObjectType(const std::shared_ptr<ObjectPrior2D> &prior) {
  objectPriors_.push_back(prior);
  priorLocations_[prior->type()] = objectPriors_.size() - 1;
  if (objectPriors_.size() != priorLocations_.size()) {
    report_error("Duplicate values of ObjectPrior2D.");
  }
}

std::shared_ptr<ObjectPrior2D> ObjectSampler::getPrior(ObjectType type) const {
  auto location = priorLocations_.find(type);
  if (location == priorLocations_.end()) {
    return nullptr;
  }
  return objectPriors_[location->second];
}

//===========================================================================
void ObjectSampler::moveAndResizeObjects(
    RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  MoveTimer timer = mhMoveResults_.start_time("moveAndResize");
  for (int i = 0; i < objects.size(); ++i) {
    std::cout << "sampling object " << i << " of " << objects.size() << ": "
              << *objects[i] << std::endl;
    objects[i]->samplePosterior(rng, screen, region, objects, background_);
  }
}

//===========================================================================
void ObjectSampler::splitCombineObjects(
    RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  double u = runif_mt(rng);
  if (u < .5) {
    splitMove(rng, screen, region, objects);
  } else {
    combineMove(rng, screen, region, objects);
  }
}

//===========================================================================
void ObjectSampler::splitMove(RNG &rng, const Screen2D &screen,
                              const BoundingBox &region,
                              std::vector<std::shared_ptr<Object2D>> &objects) {
  MoveTimer timer = mhMoveResults_.start_time("split");
  int index = random_int_mt(rng, 0, objects.size() - 1);
  std::shared_ptr<Object2D> original = objects[index];

  BoundingBox box = original->boundingBox().intersection(region);
  std::shared_ptr<Object2D> candidate = newRandomObject(rng, box);

  // localObjects are all the objects with centers contained by the bounding
  // box of original.
  std::vector<std::shared_ptr<Object2D>> localObjects;
  double objectCount = 0;
  for (const auto &el : objects) {
    if (box.contains(el->center())) {
      localObjects.push_back(el);
    }
    if (el->type() == candidate->type()) {
      ++objectCount;
    }
  }

  double loglikeOriginal =
      Object2D::integratedLogLikelihood(screen, box, localObjects, background_);
  localObjects.push_back(candidate);
  double loglikeCand =
      Object2D::integratedLogLikelihood(screen, box, localObjects, background_);

  double log_MH_ratio =
      loglikeCand - loglikeOriginal +
      log(objects.size() * overallIntensity() /
          ((1.0 + objects.size()) * localObjects.size() * (objectCount + 1)));

  double log_u = log(runif_mt(rng));
  if (log_u < log_MH_ratio) {
    objects.push_back(candidate);
    mhMoveResults_.record_acceptance("split");
  } else {
    mhMoveResults_.record_rejection("split");
  }
}

//===========================================================================
void ObjectSampler::combineMove(
    RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  MoveTimer timer = mhMoveResults_.start_time("combine");
  int index = random_int_mt(rng, 0, objects.size() - 1);
  std::shared_ptr<Object2D> original = objects[index];

  BoundingBox box = original->boundingBox().intersection(region);
  std::vector<std::shared_ptr<Object2D>> localObjects;
  for (const auto &el : objects) {
    if (box.contains(el->center())) {
      localObjects.push_back(el);
    }
  }
  if (localObjects.size() < 2) {
    return;
  }

  double loglikeOriginal =
      Object2D::integratedLogLikelihood(screen, box, localObjects, background_);

  int local_index = -1;
  do {
    local_index = random_int_mt(rng, 0, localObjects.size() - 1);
  } while (localObjects[local_index] == original);
  std::shared_ptr<Object2D> candidate = localObjects[local_index];

  double objectCount = 0;
  for (const auto &el : objects) {
    if (el->type() == candidate->type()) {
      ++objectCount;
    }
  }

  localObjects.erase(localObjects.begin() + local_index);
  double loglikeCand =
      Object2D::integratedLogLikelihood(screen, box, localObjects, background_);

  double log_MH_ratio =
      loglikeCand - loglikeOriginal -
      log(objects.size() * overallIntensity() / (objects.size() + 1) *
          objectCount * localObjects.size());

  double log_u = log(runif_mt(rng));
  if (log_u < log_MH_ratio) {
    objects.erase(std::find(objects.begin(), objects.end(), candidate));
    mhMoveResults_.record_acceptance("combine");
  } else {
    mhMoveResults_.record_rejection("combine");
  }
}

//===========================================================================
void ObjectSampler::morphObjects(
    RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  int index = random_int_mt(rng, 0, objects.size() - 1);
  morphSingleObject(index, rng, screen, region, objects);
}

//===========================================================================
void ObjectSampler::morphSingleObject(
    int index, RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  MoveTimer timer = mhMoveResults_.start_time("morph");
  std::shared_ptr<Object2D> originalObject = objects[index];
  std::shared_ptr<Object2D> candidate = randomMorph(rng, originalObject);

  ObjectType originalObjectType = originalObject->type();
  ObjectType candidateObjectType = candidate->type();

  double candidate_logPrior = getPrior(candidateObjectType)->logpri(*candidate);
  if (!std::isfinite(candidate_logPrior)) {
    mhMoveResults_.record_rejection("morph");
    return;
  }

  int originalObjectCount = 0;
  int candidateObjectCount = 0;
  for (int i = 0; i < objects.size(); ++i) {
    if (objects[i]->type() == originalObjectType)
      ++originalObjectCount;
    if (objects[i]->type() == candidateObjectType)
      ++candidateObjectCount;
  }
  double originalIntensity = getPrior(originalObjectType)->intensity();
  double candidateIntensity = getPrior(candidateObjectType)->intensity();
  double overallIntensity = this->overallIntensity();

  // Now compute MH ratio.
  BoundingBox box =
      originalObject->boundingBox().super_box(candidate->boundingBox());

  double loglikeOriginal =
      Object2D::integratedLogLikelihood(screen, box, objects, background_);
  objects[index] = candidate;
  double loglikeCand =
      Object2D::integratedLogLikelihood(screen, box, objects, background_);

  double log_MH_ratio = loglikeCand - loglikeOriginal +
                        log((overallIntensity - originalIntensity) /
                            (overallIntensity - candidateIntensity)) +
                        log(originalObjectCount / (candidateObjectCount + 1.0));

  double log_u = std::log(runif_mt(rng));
  std::ostringstream logMessage;
  if (log_u < log_MH_ratio) {
    logMessage << originalObject->logMessages() << std::endl
               << "Morph accepted: New object is " << *candidate;
    candidate->addLogMessage(logMessage.str());
    mhMoveResults_.record_acceptance("morph");
  } else {
    // Reject the draw and keep the original object in place.
    logMessage << "Morph rejected.  Proposal: " << *candidate
               << " original object: " << *originalObject;
    originalObject->addLogMessage(logMessage.str());
    mhMoveResults_.record_rejection("morph");
    objects[index] = originalObject;
  }
}

//===========================================================================
void ObjectSampler::addRemoveObjects(
    RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  double u = runif_mt(rng);
  if (u < .5) {
    addObjectMove(rng, screen, region, objects);
  } else {
    removeObjectMove(rng, screen, region, objects);
  }
}

void ObjectSampler::addObjectMove(
    RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  MoveTimer timer = mhMoveResults_.start_time("add");
  std::shared_ptr<Object2D> candidate = newRandomObject(rng, region);
  BoundingBox affected_region = candidate->boundingBox().intersection(region);

  double original_number_ofObjects = objects.size();
  double candidateObjectTypeCount = 0;
  for (int i = 0; i < objects.size(); ++i) {
    if (objects[i]->type() == candidate->type()) {
      ++candidateObjectTypeCount;
    }
  }

  double loglikeOriginal = Object2D::integratedLogLikelihood(
      screen, affected_region, objects, background_);
  objects.push_back(candidate);
  double loglikeCand = Object2D::integratedLogLikelihood(
      screen, affected_region, objects, background_);

  double log_MH_ratio =
      loglikeCand - loglikeOriginal +
      log(overallIntensity() * (original_number_ofObjects + 1) /
          (candidateObjectTypeCount + 1));

  double log_u = log(runif_mt(rng));
  if (log_u < log_MH_ratio) {
    std::ostringstream logMessage;
    logMessage << "MH add move successful.  New object: " << *candidate;
    candidate->addLogMessage(logMessage.str());
    mhMoveResults_.record_acceptance("add");
  } else {
    mhMoveResults_.record_rejection("add");
    objects.pop_back();
  }
}
//===========================================================================

void ObjectSampler::removeObjectMove(
    RNG &rng, const Screen2D &screen, const BoundingBox &region,
    std::vector<std::shared_ptr<Object2D>> &objects) {
  MoveTimer timer = mhMoveResults_.start_time("remove");
  int index = random_int_mt(rng, 0, objects.size() - 1);
  std::shared_ptr<Object2D> candidate = objects[index];

  // The number of objects with the same type as the deletion candidate,
  // before it is deleted.
  double originalObjectCount = 0;
  for (const auto &el : objects) {
    if (el->type() == candidate->type()) {
      ++originalObjectCount;
    }
  }

  double original_number_ofObjects = objects.size();

  BoundingBox affected_region = candidate->boundingBox().intersection(region);
  double loglikeOriginal = Object2D::integratedLogLikelihood(
      screen, affected_region, objects, background_);
  objects.erase(objects.begin() + index);
  double loglikeCand = Object2D::integratedLogLikelihood(
      screen, affected_region, objects, background_);

  double log_MH_ratio = loglikeCand - loglikeOriginal -
                        log(overallIntensity() * original_number_ofObjects /
                            (originalObjectCount));

  double log_u = log(runif_mt(rng));
  if (log_u < log_MH_ratio) {
    mhMoveResults_.record_acceptance("remove");
  } else {
    std::ostringstream logMessage;
    logMessage << "MH 'remove' move failed for " << *candidate;
    candidate->addLogMessage(logMessage.str());
    mhMoveResults_.record_rejection("remove");
    objects.insert(objects.begin() + index, candidate);
  }
}

//===========================================================================
std::shared_ptr<Object2D>
ObjectSampler::randomMorph(RNG &rng, const std::shared_ptr<Object2D> &object) {
  if (objectPriors_.empty()) {
    report_error("Add object types to ObjectSampler before calling "
                 "randomMorph.");
  }
  if (objectPriors_.size() == 1) {
    report_error("Cannot morph because there is only one object type.");
  }
  std::shared_ptr<ObjectPrior2D> prior;
  do {
    int index = random_int_mt(rng, 0, objectPriors_.size() - 1);
    prior = objectPriors_[index];
  } while (prior->type() == object->type());

  std::shared_ptr<Object2D> newObject =
      prior->simulate(rng, object->boundingBox());
  newObject->moveTo(object->center());
  // In principle it makes sense to propose a new object roughly the same size
  // as the old object.  However, if the current object is actually too small,
  // then the replacement object will also be too small, and it won't be able to
  // see the edges so as to detect that a different object would actually be a
  // better fit.
  //
  // In an ideal world the random morph would be followed by a call to
  // object->sample_posterior() to give the new object a chance to orient itself
  // before deciding to accept the morph.  If the proposal distribution could be
  // worked out this would be worth trying.

  //  newObject->scaleToArea(object->area());
  return newObject;
}

//===========================================================================
std::shared_ptr<Object2D>
ObjectSampler::newRandomObject(RNG &rng, const BoundingBox &box) {
  if (objectPriors_.empty()) {
    report_error("Add object types to ObjectSampler before calling "
                 "randomObject.");
  }
  Vector intensities(objectPriors_.size());
  double sum = 0;
  for (int i = 0; i < objectPriors_.size(); ++i) {
    double intensity = objectPriors_[i]->intensity();
    intensities[i] = intensity;
    sum += intensity;
  }
  intensities /= sum;
  int index = rmulti_mt(rng, intensities);
  return objectPriors_[index]->simulate(rng, box);
}

double ObjectSampler::overallIntensity() const {
  double ans = 0;
  for (const auto &el : objectPriors_) {
    ans += el->intensity();
  }
  return ans;
}

//===========================================================================
void ObjectSampler::recordObjects(
    const std::vector<std::shared_ptr<Object2D>> &objects) {
  std::vector<std::shared_ptr<Object2D>> element;
  element.reserve(objects.size());
  for (const auto &obj : objects) {
    element.push_back(std::shared_ptr<Object2D>(obj->clone()));
  }
  objectRecord_.emplace_back(std::move(element));
}
//===========================================================================
std::vector<std::shared_ptr<Object2D>>
ObjectSampler::initializeObjectSearch(RNG &rng, const Screen2D &screen,
                                      const BoundingBox &region) {
  std::vector<std::shared_ptr<Object2D>> objects;
  for (double left = region.left(); left <= region.right();
       left += gridSearchSize_) {
    for (double bottom = region.bottom(); bottom <= region.top();
         bottom += gridSearchSize_) {
      double right = std::min(left + gridSearchSize_, region.right());
      double top = std::min(bottom + gridSearchSize_, region.top());
      BoundingBox search_box(left, right, bottom, top);
      objects.push_back(newRandomObject(rng, search_box));
      objects.back()->samplePosterior(rng, screen, search_box, objects,
                                      background_);
    }
  }
  return objects;
}

} // namespace slam
