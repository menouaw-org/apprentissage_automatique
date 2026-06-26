#include "ml_library.h"
#include "../linear/linear_model.h"
#include "../mlp/mlp_model.h"

#include <stdlib.h>

DLLEXPORT int32_t my_add(int32_t a, int32_t b) {
    return a + b;
}

DLLEXPORT void* create_linear_model(
    int32_t input_size,
    int32_t output_size,
    int32_t task_type
) {
    return linear_model_create(input_size, output_size, task_type);
}

DLLEXPORT int32_t train_linear_model(
    void* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
) {
    return linear_model_train(
        (LinearModel*) model,
        x,
        y,
        n_samples,
        learning_rate,
        epochs
    );
}

DLLEXPORT int32_t predict_linear_model(
    void* model,
    const double* x,
    double* y_pred
) {
    return linear_model_predict((const LinearModel*) model, x, y_pred);
}

DLLEXPORT int32_t save_linear_model(void* model, const char* path) {
    return linear_model_save((const LinearModel*) model, path);
}

DLLEXPORT void* load_linear_model(const char* path) {
    return linear_model_load(path);
}

DLLEXPORT void destroy_linear_model(void* model) {
    linear_model_destroy((LinearModel*) model);
}

DLLEXPORT void* create_mlp_model(
    int32_t input_size,
    int32_t output_size,
    int32_t hidden_layer_count,
    const int32_t* hidden_layer_sizes,
    int32_t task_type
) {
    return mlp_model_create(
        input_size,
        output_size,
        hidden_layer_count,
        hidden_layer_sizes,
        task_type
    );
}

DLLEXPORT int32_t train_mlp_model(
    void* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
) {
    return mlp_model_train(
        (MlpModel*) model,
        x,
        y,
        n_samples,
        learning_rate,
        epochs
    );
}

DLLEXPORT int32_t predict_mlp_model(
    void* model,
    const double* x,
    double* y_pred
) {
    return mlp_model_predict((MlpModel*) model, x, y_pred);
}

DLLEXPORT int32_t predict_mlp_model_raw(
    void* model,
    const double* x,
    double* y_raw
) {
    return mlp_model_predict_raw((MlpModel*) model, x, y_raw);
}

DLLEXPORT int32_t save_mlp_model(void* model, const char* path) {
    return mlp_model_save((const MlpModel*) model, path);
}

DLLEXPORT void* load_mlp_model(const char* path) {
    return mlp_model_load(path);
}

DLLEXPORT void destroy_mlp_model(void* model) {
    mlp_model_destroy((MlpModel*) model);
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