#include "ml_library.h"

#include <stdlib.h>

struct LinearModel {
    float a;
    float b;
};

DLLEXPORT int32_t my_add(int32_t a, int32_t b) {
    return a + b;
}

DLLEXPORT LinearModel* create_linear_model(float a, float b) {
    LinearModel* model = malloc(sizeof(LinearModel));
    if (model == NULL) {
        return NULL;
    }

    model->a = a;
    model->b = b;

    return model;
}

DLLEXPORT float predict_linear_model(const LinearModel* model) {
    if (model == NULL) {
        return 0.0f;
    }

    return model->a + model->b;
}

DLLEXPORT void release_linear_model(LinearModel* model) {
    free(model);
}

DLLEXPORT float sum_array(const float* array, int32_t array_length) {
    if (array == NULL || array_length <= 0) {
        return 0.0f;
    }

    float sum = 0.0f;

    for (int32_t i = 0; i < array_length; i++) {
        sum += array[i];
    }

    return sum;
}