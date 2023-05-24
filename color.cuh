#pragma oncee

#include "vec3.cuh"
#include "rtweekend.cuh"

#include <iostream>

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
	float r = pixel_color.x();
	float g = pixel_color.y();
	float b = pixel_color.z();
	
	float scale = 1.0f / samples_per_pixel;
	r *= scale;
	g *= scale;
	b *= scale;

	// write translated [0, 255] value of each color component.
	out << static_cast<int>(255.999f * clamp(r, 0.0, 0.999)) << ' '
			<< static_cast<int>(255.999f * clamp(g, 0.0, 0.999)) << ' '
			<< static_cast<int>(255.999f * clamp(b, 0.0, 0.999)) << '\n';
}
