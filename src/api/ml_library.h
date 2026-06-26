#ifndef PA_ML_LIBRARY_H
#define PA_ML_LIBRARY_H

#include <stdint.h>

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

DLLEXPORT int32_t my_add(int32_t a, int32_t b);

DLLEXPORT void* create_linear_model(
    int32_t input_size,
    int32_t output_size,
    int32_t task_type
);

DLLEXPORT int32_t train_linear_model(
    void* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
);

DLLEXPORT int32_t predict_linear_model(
    void* model,
    const double* x,
    double* y_pred
);

DLLEXPORT int32_t save_linear_model(void* model, const char* path);

DLLEXPORT void* load_linear_model(const char* path);

DLLEXPORT void destroy_linear_model(void* model);

DLLEXPORT void* create_mlp_model(
    int32_t input_size,
    int32_t output_size,
    int32_t hidden_layer_count,
    const int32_t* hidden_layer_sizes,
    int32_t task_type
);

DLLEXPORT int32_t train_mlp_model(
    void* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
);

DLLEXPORT int32_t predict_mlp_model(
    void* model,
    const double* x,
    double* y_pred
);

DLLEXPORT int32_t predict_mlp_model_raw(
    void* model,
    const double* x,
    double* y_raw
);

DLLEXPORT int32_t save_mlp_model(void* model, const char* path);

DLLEXPORT void* load_mlp_model(const char* path);

DLLEXPORT void destroy_mlp_model(void* model);

DLLEXPORT float sum_array(const float* array, int32_t array_length);

#endif