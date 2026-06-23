#include "linear_model.h"

#include <stdlib.h>

struct LinearModel {
    int32_t input_size;
    int32_t output_size;
    int32_t task_type;
    double* weights;
};

static int32_t linear_model_is_supported_task(int32_t task_type) {
    return (
        task_type == LINEAR_TASK_REGRESSION ||
        task_type == LINEAR_TASK_CLASSIFICATION
    );
}

static int32_t linear_model_weight_count_from_dimensions(
    int32_t input_size,
    int32_t output_size
) {
    if (input_size <= 0 || output_size <= 0) {
        return 0;
    }

    return (input_size + 1) * output_size;
}

static int32_t linear_model_weight_count(const LinearModel* model) {
    if (model == NULL) {
        return 0;
    }

    return linear_model_weight_count_from_dimensions(
        model->input_size,
        model->output_size
    );
}

static int32_t linear_model_compute_scores(
    const LinearModel* model,
    const double* x,
    double* scores
) {
    if (model == NULL || x == NULL || scores == NULL) {
        return LINEAR_MODEL_ERROR_INVALID_ARGUMENT;
    }

    int32_t stride = model->input_size + 1;

    for (int32_t output = 0; output < model->output_size; output++) {
        int32_t output_offset = output * stride;
        double sum = 0.0;

        for (int32_t input = 0; input < model->input_size; input++) {
            int32_t weight_index = output_offset + input;
            sum += model->weights[weight_index] * x[input];
        }

        int32_t bias_index = output_offset + model->input_size;
        sum += model->weights[bias_index];

        scores[output] = sum;
    }

    return LINEAR_MODEL_SUCCESS;
}

static double linear_model_binary_class_from_score(double score) {
    if (score >= 0.0) {
        return 1.0;
    }

    return -1.0;
}

static int32_t linear_model_scores_to_classes(
    const LinearModel* model,
    const double* scores,
    double* y_pred
) {
    if (model == NULL || scores == NULL || y_pred == NULL) {
        return LINEAR_MODEL_ERROR_INVALID_ARGUMENT;
    }

    if (model->output_size == 1) {
        y_pred[0] = linear_model_binary_class_from_score(scores[0]);
        return LINEAR_MODEL_SUCCESS;
    }

    int32_t best_output = 0;
    double best_score = scores[0];

    for (int32_t output = 1; output < model->output_size; output++) {
        if (scores[output] > best_score) {
            best_score = scores[output];
            best_output = output;
        }
    }

    for (int32_t output = 0; output < model->output_size; output++) {
        if (output == best_output) {
            y_pred[output] = 1.0;
        } else {
            y_pred[output] = -1.0;
        }
    }

    return LINEAR_MODEL_SUCCESS;
}

LinearModel* linear_model_create(
    int32_t input_size,
    int32_t output_size,
    int32_t task_type
) {
    if (input_size <= 0 || output_size <= 0) {
        return NULL;
    }

    if (!linear_model_is_supported_task(task_type)) {
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
    if (weight_count <= 0) {
        free(model);
        return NULL;
    }

    model->weights = calloc((size_t) weight_count, sizeof(double));
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
        return LINEAR_MODEL_ERROR_INVALID_ARGUMENT;
    }

    int32_t score_status = linear_model_compute_scores(model, x, y_pred);
    if (score_status != LINEAR_MODEL_SUCCESS) {
        return score_status;
    }

    if (model->task_type == LINEAR_TASK_REGRESSION) {
        return LINEAR_MODEL_SUCCESS;
    }

    if (model->task_type == LINEAR_TASK_CLASSIFICATION) {
        return linear_model_scores_to_classes(model, y_pred, y_pred);
    }

    return LINEAR_MODEL_ERROR_UNSUPPORTED_TASK;
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
        return LINEAR_MODEL_ERROR_INVALID_ARGUMENT;
    }

    if (model->task_type != LINEAR_TASK_REGRESSION) {
        return LINEAR_MODEL_ERROR_UNSUPPORTED_TASK;
    }

    int32_t stride = model->input_size + 1;

    for (int32_t epoch = 0; epoch < epochs; epoch++) {
        for (int32_t sample = 0; sample < n_samples; sample++) {
            const double* x_sample = x + sample * model->input_size;
            const double* y_sample = y + sample * model->output_size;

            for (int32_t output = 0; output < model->output_size; output++) {
                int32_t output_offset = output * stride;
                double prediction = 0.0;

                for (int32_t input = 0; input < model->input_size; input++) {
                    int32_t weight_index = output_offset + input;
                    prediction += model->weights[weight_index] * x_sample[input];
                }

                int32_t bias_index = output_offset + model->input_size;
                prediction += model->weights[bias_index];

                double error = prediction - y_sample[output];

                for (int32_t input = 0; input < model->input_size; input++) {
                    int32_t weight_index = output_offset + input;
                    model->weights[weight_index] -= (
                        learning_rate * error * x_sample[input]
                    );
                }

                model->weights[bias_index] -= learning_rate * error;
            }
        }
    }

    return LINEAR_MODEL_SUCCESS;
}

void linear_model_destroy(LinearModel* model) {
    if (model == NULL) {
        return;
    }

    free(model->weights);
    free(model);
}