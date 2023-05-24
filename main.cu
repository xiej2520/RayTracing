#include <iostream>

#include <curand_kernel.h>

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "rtweekend.cuh"
#include "camera.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "sphere.cuh"

using u8 = uint8_t;
using u32 = uint32_t;

// ncu --metrics smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum .\bin\main.exe out/out.png

/*
nvcc main.cu -o bin/main.exe
nvprof bin/main.exe -o out/out.png
*/

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result != 0) {
		std::cerr << "CUDA error = " << static_cast<u32>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

__device__ float hit_sphere(const point3 &center, float radius, const ray &r) {
	vec3 oc = r.origin() - center;
	float a = r.direction().length_squared();
	float half_b = dot(oc, r.direction());
	float c = oc.length_squared() - radius * radius;
	float discriminant = half_b * half_b - a * c;
	if (discriminant < 0.0f) {
		return -1.0f;
	}
	else {
		return (-half_b - sqrt(discriminant)) / a;
	}
}

constexpr int max_recur_depth = 50;

__device__ color ray_color(const ray &r, hittable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation(1, 1, 1);
	for (int i=0; i<max_recur_depth; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, infinity, rec)) {
			// Uniform scatter
			// vec3 target = rec.p + random_in_hemisphere(rec.normal, local_rand_state);
			ray scattered;
			color attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0, 0, 0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0, 0, 0); // exceed recursion
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	// each thread gets the same seed, a different sequence number, no offset.
	curand_init(723, pixel_index, 0, &rand_state[pixel_index]);
	
}

__global__ void render(vec3 *fb, int max_x, int max_y, int samples_per_pixel,
		camera **cam, hittable **world, curandState *rand_state) {
	// CUDA indexing
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || j >= max_y) return;

	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];

	color pixel_color(0, 0, 0);
	for (int s=0; s<samples_per_pixel; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		pixel_color += ray_color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	pixel_color /= float(samples_per_pixel);
	// gamma 2
	pixel_color[0] = sqrt(pixel_color[0]);
	pixel_color[1] = sqrt(pixel_color[1]);
	pixel_color[2] = sqrt(pixel_color[2]);
	fb[pixel_index] = pixel_color;
}

bool write_to_png(const char *file, vec3 *buf, int w, int h) {
	u8 *png_buf = new u8[3 * w * h * sizeof(u8)];
	int out_index = 0;
	for (int j=h-1; j>=0; j--) {
		for (int i=0; i<w; i++) {
			size_t pixel_index = j * w + i;
			
			float r = buf[pixel_index][0];
			float g = buf[pixel_index][1];
			float b = buf[pixel_index][2];

			png_buf[out_index + 0] = static_cast<u8>(255.99f * r);
			png_buf[out_index + 1] = static_cast<u8>(255.99f * g);
			png_buf[out_index + 2] = static_cast<u8>(255.99f * b);
			out_index += 3;
		}
	}
	
	// 3: RGB, 4: RGBA
	constexpr int comp_channels = 3; // RGB
	int stride = 3 * w;
	if (stbi_write_png(file, w, h, comp_channels, png_buf, stride) != 0) {
		return false;
	}
	return true;
}

/*
__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_cam,
		material **d_mat_ground, material **d_mat_center, material **d_mat_left, material **d_mat_right) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_mat_ground = new lambertian(color(0.8, 0.8, 0.0));
		*d_mat_center = new lambertian(color(0.1, 0.2, 0.5));
		*d_mat_left = new dielectric(1.5);
		*d_mat_right = new metal(color(0.6, 0.1, 0.8), 0.0);
		*(d_list+0) = new sphere({0, -100.5, -1}, 100, *d_mat_ground);
		*(d_list+1) = new sphere({0, 0, -1}, 0.5, *d_mat_center);
		*(d_list+2) = new sphere({-1, 0, -1}, 0.5, *d_mat_left);
		*(d_list+3) = new sphere({-1, 0, -1}, -0.4, *d_mat_left);
		*(d_list+4) = new sphere({1, 0, -1}, 0.5, *d_mat_right);
		*d_world = new hittable_list(d_list, 5);
		float aspect_ratio = 16.0f / 9.0f;
		//*d_cam = new camera({-2, 2, 1}, {0, 0, -1}, {0, 1, 0}, 90, aspect_ratio);
		point3 lookfrom(3, 3, 2), lookat(0, 0, -1);
		vec3 vup(0, 1, 0);
		float dist_to_focus = (lookfrom - lookat).length();
		float aperture = 2.0;
		*d_cam = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
	}
}
*/

/*
__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_cam,
		material **d_mat_ground, material **d_mat_center, material **d_mat_left, material **d_mat_right) {
	delete *d_list;
	delete *(d_list+1);
	delete *(d_list+2);
	delete *(d_list+3);
	delete *(d_list+4);
	delete *d_world;
	delete *d_cam;
	delete *d_mat_ground;
	delete *d_mat_center;
	delete *d_mat_left;
	delete *d_mat_right;
}
*/

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera,
		int image_width, int image_height, curandState *rand_state) {
#define RNG curand_uniform(&local_rand_state)
	curand_init(723, 0, 0, rand_state);
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		d_list[0] = new sphere({0, -1000, -1}, 1000, new lambertian({0.5,0.5,0.5}));
		int i=1;
		for (int a=-11; a<11; a++) {
			for (int b=-11; b<11; b++) {
				float choose_mat = RNG;
				vec3 center(a+RNG, 0.2, b+RNG);
				if (choose_mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2, new lambertian({RNG*RNG, RNG*RNG, RNG*RNG}));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2, new metal({0.5f*(1.0f+RNG), 0.5f*(1.0f+RNG), 0.5f*RNG}, 0));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere({0, 1, 0}, 1.0, new dielectric(1.5));
		d_list[i++] = new sphere({-4, 1, 0}, 1.0, new lambertian({0.25, 0.05, 0.8}));
		d_list[i++] = new sphere({4, 1, 0}, 1.0, new metal({0.7, 0.6, 0.5}, 0.0));
		d_list[i++] = new sphere({-16, 5, -6}, 5.0, new metal({1, 1, 1}, 0.0));
		d_list[i++] = new sphere({-15, 3, 4}, 3.0, new metal({0.8, 0.9, 0.9}, 0.1));
		*rand_state = local_rand_state;
		*d_world = new hittable_list(d_list, 22*22+1+5);
		
		vec3 lookfrom(13, 2, 3), lookat(0, 0, 0);
		float dist_to_focus = 10.0;
		float aperture = 0.1;
		*d_camera = new camera(lookfrom, lookat, {0, 1, 0}, 30, float(image_width)/float(image_height), aperture, dist_to_focus);
	}
#undef RNG
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	for(int i=0; i < 22*22+1+5; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cerr << "File output not specified." << std::endl;
		return -1;
	}
	
	// Image
	const float aspect_ratio = 16.0f / 9.0f;
	const int image_width = 3840;
	const int image_height = static_cast<int>(image_width / aspect_ratio);

	const int samples_per_pixel = 10;
	
	/*
	// World
	hittable **d_list;
	checkCudaErrors(cudaMalloc((void **) &d_list, 5 * sizeof(hittable *)));
	hittable **d_world;
	checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hittable *)));

	// Camera
	camera **d_cam;
	checkCudaErrors(cudaMalloc((void **) &d_cam, sizeof(camera *)));
	
	// Material
	material **d_mat_ground, **d_mat_center, **d_mat_left, **d_mat_right;
	checkCudaErrors(cudaMalloc((void **) &d_mat_ground, sizeof(material *)));
	checkCudaErrors(cudaMalloc((void **) &d_mat_center, sizeof(material *)));
	checkCudaErrors(cudaMalloc((void **) &d_mat_left, sizeof(material *)));
	checkCudaErrors(cudaMalloc((void **) &d_mat_right, sizeof(material *)));

	create_world<<<1, 1>>>(d_list, d_world, d_cam, d_mat_ground, d_mat_center,
			d_mat_left, d_mat_right);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	*/
	curandState *d_rand_world;
	checkCudaErrors(cudaMalloc((void **) &d_rand_world, sizeof(curandState)));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	hittable **d_list;
	checkCudaErrors(cudaMalloc((void **) &d_list, (22*22+1+5) * sizeof(hittable *)));
	material **d_mat_ground;
	checkCudaErrors(cudaMalloc((void **) &d_mat_ground, sizeof(material *)));
	hittable **d_world;
	checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hittable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **) &d_camera, sizeof(camera *)));
	create_world<<<1,1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_world);



	int num_pixels = image_width * image_height;	
	size_t fb_size =  num_pixels * sizeof(vec3);
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **) &fb, fb_size));
	
	int tx = 8;
	int ty = 8;
	
	dim3 blocks(image_width / tx + 1, image_height / ty + 1);
	dim3 threads(tx, ty);

	// Random State
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **) &d_rand_state, num_pixels * sizeof(curandState)));

	render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel,
		d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	write_to_png(argv[1], fb, image_width, image_height);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	/*
	free_world<<<1, 1>>>(d_list, d_world, d_cam, d_mat_ground, d_mat_center,
		d_mat_left, d_mat_right);
	*/
	free_world<<<1, 1>>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));
	
	cudaDeviceReset();
}
