#pragma once

#include <cmath>
#include <limits>

using std::sqrt;

constexpr const float infinity = std::numeric_limits<float>::infinity();
constexpr const float pi = 3.141592653589793238462643383279502884197169;

__device__ inline float to_radians(float degrees) {
	return degrees * pi / 180.0f;
}

__device__ inline float clamp(float x, float min, float max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

#include "ray.cuh"
#include "vec3.cuh"
