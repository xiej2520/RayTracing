#pragma once

#include "hittable.cuh"

using u32 = uint32_t;

class hittable_list : public hittable {
public:
	u32 list_size;
	hittable **list;

	__device__ hittable_list() {}
	__device__ hittable_list(hittable **list, u32 list_size): list(list), list_size(list_size) {}
	
	__device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const override;

};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
	hit_record temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;
	
	for (int i=0; i<list_size; i++) {
		if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}
