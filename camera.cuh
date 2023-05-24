#pragma once

#include "rtweekend.cuh"

class camera {
private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
public:
	__device__ camera(
			point3 lookfrom,
			point3 lookat,
			vec3 vup,   // up vector
			float vfov, // vertical fov in degrees
			float aspect_ratio,
			float aperture,
			float focus_dist
		) {
		float theta = to_radians(vfov);
		float h = tan(theta / 2);
		float viewport_height = 2.0 * h;
		float viewport_width = aspect_ratio * viewport_height;
		
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		
		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
		
		lens_radius = aperture / 2;
	}
	
	__device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
		vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
		vec3 offset = rd.x() * u + rd.y() * v;
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}
};
