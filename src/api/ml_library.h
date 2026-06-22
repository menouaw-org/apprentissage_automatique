#ifndef PA_ML_LIBRARY_H
#define PA_ML_LIBRARY_H

#include <stdint.h>

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

typedef struct LinearModel LinearModel;

DLLEXPORT int32_t my_add(int32_t a, int32_t b);
DLLEXPORT LinearModel* create_linear_model(float a, float b);
DLLEXPORT float predict_linear_model(const LinearModel* model);
DLLEXPORT void release_linear_model(LinearModel* model);
DLLEXPORT float sum_array(const float* array, int32_t array_length);

#endif