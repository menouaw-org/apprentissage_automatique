#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include <stdint.h>

#define LINEAR_TASK_REGRESSION 0
#define LINEAR_TASK_CLASSIFICATION 1

typedef struct LinearModel LinearModel;

LinearModel* linear_model_create(
    int32_t input_size,
    int32_t output_size,
    int32_t task_type
);

int32_t linear_model_train(
    LinearModel* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
);

int32_t linear_model_predict(
    const LinearModel* model,
    const double* x,
    double* y_pred
);

void linear_model_destroy(LinearModel* model);

#endif