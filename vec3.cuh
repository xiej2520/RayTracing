#pragma once

#include <cmath>
#include <iostream>

class vec3 {
public:
	float e[3];
	__host__ __device__ constexpr vec3(): e{0,0,0} {}
	__host__ __device__ constexpr vec3(float e0, float e1, float e2): e{e0, e1, e2} {}

	__host__ __device__ constexpr inline float x() const { return e[0]; }
	__host__ __device__ constexpr inline float y() const { return e[1]; }
	__host__ __device__ constexpr inline float z() const { return e[2]; }

	__host__ __device__ constexpr vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ constexpr float operator[](int i) const { return e[i]; }
	__host__ __device__ constexpr float &operator[](int i) { return e[i]; }
	__host__ __device__ constexpr vec3 &operator+=(const vec3 &v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}
	__host__ __device__ constexpr vec3 &operator-=(const vec3 &v) {
		e[0] -= v.e[0];
		e[1] -= v.e[1];
		e[2] -= v.e[2];
		return *this;
	}
	__host__ __device__ constexpr vec3 &operator*=(const float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}
	__host__ __device__ constexpr vec3 &operator*=(const vec3 &v) {
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}
	__host__ __device__ constexpr vec3 &operator/=(const float t) {
		e[0] /= t;
		e[1] /= t;
		e[2] /= t;
		return *this;
	}

	__host__ __device__ float length() const { return sqrt(length_squared()); }
	__host__ __device__ float length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
};

using point3 = vec3;
using color = vec3;

inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
	return vec3(u) += v;
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
	return vec3(u) -= v;
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &u) {
	return vec3(u) *= t;
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
	return vec3(u) *= v;
}

__host__ __device__ inline vec3 operator/(const vec3 &u, float t) {
	return vec3(u) /= t;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
	return vec3(
		u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]
	);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

// random vector in a unit sphere
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
	#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
	vec3 p;
	do {
		p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
	} while (p.length_squared() >= 1.0f);
	return p;
	#undef RANDVEC3
}

__device__ vec3 random_unit_vector(curandState *local_rand_state) {
	return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ vec3 random_in_hemisphere(vec3 normal, curandState *local_rand_state) {
	vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
	if (dot(in_unit_sphere, normal) > 0.0) return in_unit_sphere; // same hemisphere
	else return -in_unit_sphere;
}

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
	#define RANDVEC2 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0)
	vec3 p;
	do {
		p = 2.0f * RANDVEC2 - vec3(1, 1, 0);
	} while (p.length_squared() >= 1.0f);
	return p;
	#undef RANDVEC2
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n) {
	return v - 2.0f * dot(v, n) * n;
}

__device__ vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat) {
	float cos_theta = fminf(dot(-uv, n), 1.0f);	
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}
