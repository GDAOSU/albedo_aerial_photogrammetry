// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sys/platform.h"
#include "../sys/ref.h"
#include "../sys/intrinsics.h"
#include "../sys/sysinfo.h"
#include "../sys/atomic.h"
#include "../sys/vector.h"
#include "../sys/string.h"

#include "../math/math.h"
#include "../math/vec2.h"
#include "../math/vec3.h"
#include "../math/vec4.h"
#include "../math/bbox.h"
#include "../math/affinespace.h"

#include "../simd/simd.h"

  /*! Ray structure. */
  struct __aligned(16) Ray
  {
    /*! Default construction does nothing. */
    __forceinline Ray() {}

    /*! Constructs a ray from origin, direction, and ray segment. Near
     *  has to be smaller than far. */
    __forceinline Ray(const embree::Vec3fa& org, 
                      const embree::Vec3fa& dir, 
                      float tnear = embree::zero, 
                      float tfar = embree::inf, 
                      float time = embree::zero, 
                      int mask = -1,
                      unsigned int geomID = RTC_INVALID_GEOMETRY_ID, 
                      unsigned int primID = RTC_INVALID_GEOMETRY_ID)
      : org(org,tnear), dir(dir,time), tfar(tfar), mask(mask), primID(primID), geomID(geomID)
    {
      instID[0] = RTC_INVALID_GEOMETRY_ID;
    }

    /*! Tests if we hit something. */
    __forceinline operator bool() const { return geomID != RTC_INVALID_GEOMETRY_ID; }

  public:
    embree::Vec3ff org;       //!< Ray origin + tnear
    //float tnear;              //!< Start of ray segment
    embree::Vec3ff dir;        //!< Ray direction + tfar
    //float time;               //!< Time of this ray for motion blur.
    float tfar;               //!< End of ray segment
    unsigned int mask;        //!< used to mask out objects during traversal
    unsigned int id;          //!< ray ID
    unsigned int flags;       //!< ray flags

  public:
    embree::Vec3f Ng;         //!< Not normalized geometry normal
    float u;                  //!< Barycentric u coordinate of hit
    float v;                  //!< Barycentric v coordinate of hit
    unsigned int primID;           //!< primitive ID
    unsigned int geomID;           //!< geometry ID
    unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT];           //!< instance ID

    __forceinline float &tnear() { return org.w; };
    __forceinline float &time()  { return dir.w; };
    __forceinline float const &tnear() const { return org.w; };
    __forceinline float const &time()  const { return dir.w; };

  };


__forceinline void init_Ray(Ray &ray,
                            const embree::Vec3fa& org, 
                            const embree::Vec3fa& dir, 
                            float tnear = embree::zero, 
                            float tfar = embree::inf, 
                            float time = embree::zero, 
                            int mask = -1,
                            unsigned int geomID = RTC_INVALID_GEOMETRY_ID, 
                            unsigned int primID = RTC_INVALID_GEOMETRY_ID)
{
  ray = Ray(org,dir,tnear,tfar,time,mask,geomID,primID);
}

typedef Ray Ray1;

__forceinline RTCRayHit* RTCRayHit_(Ray& ray) {
  return (RTCRayHit*)&ray;
}

__forceinline RTCRayHit* RTCRayHit1_(Ray& ray) {
  return (RTCRayHit*)&ray;
}

__forceinline RTCRay* RTCRay_(Ray& ray) {
  return (RTCRay*)&ray;
}

__forceinline RTCHit* RTCHit_(Ray& ray)
{
  RTCHit* hit_ptr = (RTCHit*)&(ray.Ng.x);
  return hit_ptr;
}

__forceinline RTCRay* RTCRay1_(Ray& ray) {
  return (RTCRay*)&ray;
}

  /*! Outputs ray to stream. */ 
  __forceinline embree_ostream operator<<(embree_ostream cout, const Ray& ray) {
    return cout << "{ " << 
      "org = " << ray.org << ", dir = " << ray.dir << ", near = " << ray.tnear() << ", far = " << ray.tfar << ", time = " << ray.time() << ", " <<
      //"instID = " << ray.instID 
      "geomID = " << ray.geomID << ", primID = " << ray.primID <<  ", " << "u = " << ray.u <<  ", v = " << ray.v << ", Ng = " << ray.Ng << " }";
  }

/*! intersection context passed to intersect/occluded calls */
struct IntersectContext
{
  RTCIntersectContext context;
  void* userRayExt;               //!< can be used to pass extended ray data to callbacks
};

__forceinline void InitIntersectionContext(struct IntersectContext* context)
{
  rtcInitIntersectContext(&context->context);
  context->userRayExt = NULL;
}
