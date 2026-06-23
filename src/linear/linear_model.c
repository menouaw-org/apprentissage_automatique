#include "linear_model.h"

#include <stdlib.h>

struct LinearModel {
    int32_t input_size;
    int32_t output_size;
    int32_t task_type;
    double* weights;
};

static int32_t linear_model_weight_count(const LinearModel* model) {
    if (model == NULL) {
        return 0;
    }

    return (model->input_size + 1) * model->output_size;
}

LinearModel* linear_model_create(
    int32_t input_size,
    int32_t output_size,
    int32_t task_type
) {
    if (input_size <= 0 || output_size <= 0) {
        return NULL;
    }

    if (
        task_type != LINEAR_TASK_REGRESSION &&
        task_type != LINEAR_TASK_CLASSIFICATION
    ) {
        return NULL;
    }

    LinearModel* model = malloc(sizeof(LinearModel));
    if (model == NULL) {
        return NULL;
    }

    model->input_size = input_size;
    model->output_size = output_size;
    model->task_type = task_type;

    int32_t weight_count = linear_model_weight_count(model);
    model->weights = calloc(weight_count, sizeof(double));
    if (model->weights == NULL) {
        free(model);
        return NULL;
    }

    return model;
}

int32_t linear_model_predict(
    const LinearModel* model,
    const double* x,
    double* y_pred
) {
    if (model == NULL || x == NULL || y_pred == NULL) {
        return -1;
    }

    int32_t stride = model->input_size + 1;

    for (int32_t output = 0; output < model->output_size; output++) {
        double sum = 0.0;

        for (int32_t input = 0; input < model->input_size; input++) {
            int32_t weight_index = output * stride + input;
            sum += model->weights[weight_index] * x[input];
        }

        int32_t bias_index = output * stride + model->input_size;
        sum += model->weights[bias_index];

        y_pred[output] = sum;
    }

    return 0;
}

int32_t linear_model_train(
    LinearModel* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
) {
    if (
        model == NULL ||
        x == NULL ||
        y == NULL ||
        n_samples <= 0 ||
        learning_rate <= 0.0 ||
        epochs <= 0
    ) {
        return -1;
    }

    if (model->task_type != LINEAR_TASK_REGRESSION) {
        return -2;
    }

    int32_t stride = model->input_size + 1;

    for (int32_t epoch = 0; epoch < epochs; epoch++) {
        for (int32_t sample = 0; sample < n_samples; sample++) {
            const double* x_sample = x + sample * model->input_size;
            const double* y_sample = y + sample * model->output_size;

            for (int32_t output = 0; output < model->output_size; output++) {
                double prediction = 0.0;

                for (int32_t input = 0; input < model->input_size; input++) {
                    int32_t weight_index = output * stride + input;
                    prediction += model->weights[weight_index] * x_sample[input];
                }

                int32_t bias_index = output * stride + model->input_size;
                prediction += model->weights[bias_index];

                double error = prediction - y_sample[output];

                for (int32_t input = 0; input < model->input_size; input++) {
                    int32_t weight_index = output * stride + input;
                    model->weights[weight_index] -= learning_rate * error * x_sample[input];
                }

                model->weights[bias_index] -= learning_rate * error;
            }
        }
    }

    return 0;
}

void linear_model_destroy(LinearModel* model) {
    if (model == NULL) {
        return;
    }

    free(model->weights);
    free(model);
}