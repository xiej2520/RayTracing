#pragma once

#include "ray.cuh"
#include "vec3.cuh"

struct hit_record;

class material {
public:
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
			ray &scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
public:
	color albedo;

	__device__ lambertian(const color &a): albedo(a) {}
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
			ray &scattered, curandState *local_rand_state) const override {
		vec3 scatter_direction = rec.p + rec.normal + random_unit_vector(local_rand_state);
		scattered = ray(rec.p, scatter_direction - rec.p);
		attenuation = albedo;
		return true;
	}
};

class metal : public material {
public:
	color albedo;
	float fuzz;

	__device__ metal(const color &a, float f): albedo(a), fuzz(f < 1 ? f : 1) {}
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
			ray &scattered, curandState *local_rand_state) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
		attenuation = albedo;
		return dot(scattered.direction(), rec.normal) > 0.0f;
	}

};

class dielectric : public material {
	__device__ static float reflectance(float cosine, float ref_index) {
		// Schlick's approximation for reflectance
		float r0 = (1.0f - ref_index) / (1.0f + ref_index);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
	}
public:
	float ir;
	
	__device__ dielectric(float index_of_refraction): ir(index_of_refraction) {}
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
			ray &scattered, curandState *local_rand_state) const override {
		attenuation = color(1, 1, 1);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
		vec3 unit_direction = unit_vector(r_in.direction());
		
		float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
		
		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		vec3 direction;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
			direction = reflect(unit_direction, rec.normal);
		}
		else {
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}
		scattered = ray(rec.p, direction);
		return true;
	}
	
};
