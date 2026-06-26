#include "mlp_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MLP_MODEL_FILE_VERSION 1

static const char MLP_MODEL_FILE_MAGIC[8] = {
    'P', 'A', 'M', 'L', 'P', '0', '1', '\0'
};

struct MlpModel {
    int32_t input_size;
    int32_t output_size;
    int32_t hidden_layer_count;
    int32_t layer_count;
    int32_t task_type;

    int32_t* layer_sizes;
    double** weights;
    double** biases;
    double** activations;
    double** deltas;
};

static int32_t mlp_model_is_supported_task(int32_t task_type) {
    return task_type == MLP_TASK_CLASSIFICATION;
}

static double mlp_model_activation(double value) {
    return tanh(value);
}

static double mlp_model_activation_derivative_from_activation(
    double activation
) {
    return 1.0 - activation * activation;
}

static double mlp_model_normalize_class_target(double target) {
    if (target > 0.0) {
        return 1.0;
    }

    return -1.0;
}

static double mlp_model_initial_weight(
    int32_t layer,
    int32_t weight_index
) {
    int32_t pattern = ((layer + 1) * 31 + (weight_index + 1) * 17) % 100;
    double centered = ((double) pattern / 100.0) - 0.5;
    double value = centered * 1.0;

    if (value == 0.0) {
        return 0.01;
    }

    return value;
}

static int32_t mlp_model_validate_dimensions(
    int32_t input_size,
    int32_t output_size,
    int32_t hidden_layer_count,
    const int32_t* hidden_layer_sizes,
    int32_t task_type
) {
    if (input_size <= 0 || output_size <= 0 || hidden_layer_count < 0) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    if (!mlp_model_is_supported_task(task_type)) {
        return MLP_MODEL_ERROR_UNSUPPORTED_TASK;
    }

    if (hidden_layer_count > 0 && hidden_layer_sizes == NULL) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    for (int32_t layer = 0; layer < hidden_layer_count; layer++) {
        if (hidden_layer_sizes[layer] <= 0) {
            return MLP_MODEL_ERROR_INVALID_ARGUMENT;
        }
    }

    return MLP_MODEL_SUCCESS;
}

static int32_t mlp_model_previous_layer_size(
    const MlpModel* model,
    int32_t layer
) {
    if (layer == 0) {
        return model->input_size;
    }

    return model->layer_sizes[layer - 1];
}

static void mlp_model_free_layer_arrays(MlpModel* model) {
    if (model == NULL) {
        return;
    }

    if (model->weights != NULL) {
        for (int32_t layer = 0; layer < model->layer_count; layer++) {
            free(model->weights[layer]);
        }
    }

    if (model->biases != NULL) {
        for (int32_t layer = 0; layer < model->layer_count; layer++) {
            free(model->biases[layer]);
        }
    }

    if (model->activations != NULL) {
        for (int32_t layer = 0; layer < model->layer_count; layer++) {
            free(model->activations[layer]);
        }
    }

    if (model->deltas != NULL) {
        for (int32_t layer = 0; layer < model->layer_count; layer++) {
            free(model->deltas[layer]);
        }
    }

    free(model->weights);
    free(model->biases);
    free(model->activations);
    free(model->deltas);

    model->weights = NULL;
    model->biases = NULL;
    model->activations = NULL;
    model->deltas = NULL;
}

static int32_t mlp_model_allocate_layer_arrays(MlpModel* model) {
    model->weights = calloc((size_t) model->layer_count, sizeof(double*));
    model->biases = calloc((size_t) model->layer_count, sizeof(double*));
    model->activations = calloc((size_t) model->layer_count, sizeof(double*));
    model->deltas = calloc((size_t) model->layer_count, sizeof(double*));

    if (
        model->weights == NULL ||
        model->biases == NULL ||
        model->activations == NULL ||
        model->deltas == NULL
    ) {
        return MLP_MODEL_ERROR_ALLOCATION_FAILED;
    }

    for (int32_t layer = 0; layer < model->layer_count; layer++) {
        int32_t previous_size = mlp_model_previous_layer_size(model, layer);
        int32_t current_size = model->layer_sizes[layer];
        int32_t weight_count = previous_size * current_size;

        model->weights[layer] = malloc((size_t) weight_count * sizeof(double));
        model->biases[layer] = calloc((size_t) current_size, sizeof(double));
        model->activations[layer] = calloc(
            (size_t) current_size,
            sizeof(double)
        );
        model->deltas[layer] = calloc((size_t) current_size, sizeof(double));

        if (
            model->weights[layer] == NULL ||
            model->biases[layer] == NULL ||
            model->activations[layer] == NULL ||
            model->deltas[layer] == NULL
        ) {
            return MLP_MODEL_ERROR_ALLOCATION_FAILED;
        }

        for (int32_t weight = 0; weight < weight_count; weight++) {
            model->weights[layer][weight] = mlp_model_initial_weight(
                layer,
                weight
            );
        }
    }

    return MLP_MODEL_SUCCESS;
}

static int32_t mlp_model_forward(
    MlpModel* model,
    const double* x
) {
    if (model == NULL || x == NULL) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    for (int32_t layer = 0; layer < model->layer_count; layer++) {
        int32_t previous_size = mlp_model_previous_layer_size(model, layer);
        int32_t current_size = model->layer_sizes[layer];

        const double* previous_values = x;
        if (layer > 0) {
            previous_values = model->activations[layer - 1];
        }

        for (int32_t neuron = 0; neuron < current_size; neuron++) {
            double sum = model->biases[layer][neuron];

            for (int32_t previous = 0; previous < previous_size; previous++) {
                int32_t weight_index = previous * current_size + neuron;
                sum += (
                    model->weights[layer][weight_index] *
                    previous_values[previous]
                );
            }

            model->activations[layer][neuron] = mlp_model_activation(sum);
        }
    }

    return MLP_MODEL_SUCCESS;
}

static int32_t mlp_model_compute_output_deltas(
    MlpModel* model,
    const double* y_sample
) {
    if (model == NULL || y_sample == NULL) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    int32_t output_layer = model->layer_count - 1;

    for (int32_t output = 0; output < model->output_size; output++) {
        double activation = model->activations[output_layer][output];
        double target = mlp_model_normalize_class_target(y_sample[output]);
        double error = activation - target;

        model->deltas[output_layer][output] = (
            error *
            mlp_model_activation_derivative_from_activation(activation)
        );
    }

    return MLP_MODEL_SUCCESS;
}

static void mlp_model_compute_hidden_deltas(MlpModel* model) {
    int32_t output_layer = model->layer_count - 1;

    for (int32_t layer = output_layer - 1; layer >= 0; layer--) {
        int32_t current_size = model->layer_sizes[layer];
        int32_t next_size = model->layer_sizes[layer + 1];

        for (int32_t neuron = 0; neuron < current_size; neuron++) {
            double propagated_error = 0.0;

            for (int32_t next = 0; next < next_size; next++) {
                int32_t weight_index = neuron * next_size + next;
                propagated_error += (
                    model->weights[layer + 1][weight_index] *
                    model->deltas[layer + 1][next]
                );
            }

            double activation = model->activations[layer][neuron];
            model->deltas[layer][neuron] = (
                propagated_error *
                mlp_model_activation_derivative_from_activation(activation)
            );
        }
    }
}

static void mlp_model_update_parameters(
    MlpModel* model,
    const double* x_sample,
    double learning_rate
) {
    for (int32_t layer = 0; layer < model->layer_count; layer++) {
        int32_t previous_size = mlp_model_previous_layer_size(model, layer);
        int32_t current_size = model->layer_sizes[layer];

        const double* previous_values = x_sample;
        if (layer > 0) {
            previous_values = model->activations[layer - 1];
        }

        for (int32_t neuron = 0; neuron < current_size; neuron++) {
            double delta = model->deltas[layer][neuron];

            for (int32_t previous = 0; previous < previous_size; previous++) {
                int32_t weight_index = previous * current_size + neuron;
                model->weights[layer][weight_index] -= (
                    learning_rate *
                    delta *
                    previous_values[previous]
                );
            }

            model->biases[layer][neuron] -= learning_rate * delta;
        }
    }
}

static int32_t mlp_model_write_layer_parameters(
    const MlpModel* model,
    FILE* file
) {
    for (int32_t layer = 0; layer < model->layer_count; layer++) {
        int32_t previous_size = mlp_model_previous_layer_size(model, layer);
        int32_t current_size = model->layer_sizes[layer];
        int32_t weight_count = previous_size * current_size;

        if (
            fwrite(
                model->weights[layer],
                sizeof(double),
                (size_t) weight_count,
                file
            ) != (size_t) weight_count
        ) {
            return MLP_MODEL_ERROR_INVALID_ARGUMENT;
        }

        if (
            fwrite(
                model->biases[layer],
                sizeof(double),
                (size_t) current_size,
                file
            ) != (size_t) current_size
        ) {
            return MLP_MODEL_ERROR_INVALID_ARGUMENT;
        }
    }

    return MLP_MODEL_SUCCESS;
}

static int32_t mlp_model_read_layer_parameters(
    MlpModel* model,
    FILE* file
) {
    for (int32_t layer = 0; layer < model->layer_count; layer++) {
        int32_t previous_size = mlp_model_previous_layer_size(model, layer);
        int32_t current_size = model->layer_sizes[layer];
        int32_t weight_count = previous_size * current_size;

        if (
            fread(
                model->weights[layer],
                sizeof(double),
                (size_t) weight_count,
                file
            ) != (size_t) weight_count
        ) {
            return MLP_MODEL_ERROR_INVALID_ARGUMENT;
        }

        if (
            fread(
                model->biases[layer],
                sizeof(double),
                (size_t) current_size,
                file
            ) != (size_t) current_size
        ) {
            return MLP_MODEL_ERROR_INVALID_ARGUMENT;
        }
    }

    return MLP_MODEL_SUCCESS;
}

MlpModel* mlp_model_create(
    int32_t input_size,
    int32_t output_size,
    int32_t hidden_layer_count,
    const int32_t* hidden_layer_sizes,
    int32_t task_type
) {
    int32_t validation_status = mlp_model_validate_dimensions(
        input_size,
        output_size,
        hidden_layer_count,
        hidden_layer_sizes,
        task_type
    );

    if (validation_status != MLP_MODEL_SUCCESS) {
        return NULL;
    }

    MlpModel* model = malloc(sizeof(MlpModel));
    if (model == NULL) {
        return NULL;
    }

    model->input_size = input_size;
    model->output_size = output_size;
    model->hidden_layer_count = hidden_layer_count;
    model->layer_count = hidden_layer_count + 1;
    model->task_type = task_type;
    model->layer_sizes = NULL;
    model->weights = NULL;
    model->biases = NULL;
    model->activations = NULL;
    model->deltas = NULL;

    model->layer_sizes = malloc(
        (size_t) model->layer_count * sizeof(int32_t)
    );
    if (model->layer_sizes == NULL) {
        free(model);
        return NULL;
    }

    for (int32_t layer = 0; layer < hidden_layer_count; layer++) {
        model->layer_sizes[layer] = hidden_layer_sizes[layer];
    }
    model->layer_sizes[model->layer_count - 1] = output_size;

    int32_t allocation_status = mlp_model_allocate_layer_arrays(model);
    if (allocation_status != MLP_MODEL_SUCCESS) {
        mlp_model_destroy(model);
        return NULL;
    }

    return model;
}

int32_t mlp_model_train(
    MlpModel* model,
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
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    if (model->task_type != MLP_TASK_CLASSIFICATION) {
        return MLP_MODEL_ERROR_UNSUPPORTED_TASK;
    }

    for (int32_t epoch = 0; epoch < epochs; epoch++) {
        for (int32_t sample = 0; sample < n_samples; sample++) {
            const double* x_sample = x + sample * model->input_size;
            const double* y_sample = y + sample * model->output_size;

            int32_t forward_status = mlp_model_forward(model, x_sample);
            if (forward_status != MLP_MODEL_SUCCESS) {
                return forward_status;
            }

            int32_t delta_status = mlp_model_compute_output_deltas(
                model,
                y_sample
            );
            if (delta_status != MLP_MODEL_SUCCESS) {
                return delta_status;
            }

            mlp_model_compute_hidden_deltas(model);
            mlp_model_update_parameters(model, x_sample, learning_rate);
        }
    }

    return MLP_MODEL_SUCCESS;
}

int32_t mlp_model_predict(
    MlpModel* model,
    const double* x,
    double* y_pred
) {
    if (model == NULL || x == NULL || y_pred == NULL) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    if (model->task_type != MLP_TASK_CLASSIFICATION) {
        return MLP_MODEL_ERROR_UNSUPPORTED_TASK;
    }

    int32_t forward_status = mlp_model_forward(model, x);
    if (forward_status != MLP_MODEL_SUCCESS) {
        return forward_status;
    }

    int32_t output_layer = model->layer_count - 1;

    if (model->output_size == 1) {
        if (model->activations[output_layer][0] >= 0.0) {
            y_pred[0] = 1.0;
        } else {
            y_pred[0] = -1.0;
        }

        return MLP_MODEL_SUCCESS;
    }

    int32_t best_output = 0;
    double best_score = model->activations[output_layer][0];

    for (int32_t output = 1; output < model->output_size; output++) {
        double score = model->activations[output_layer][output];

        if (score > best_score) {
            best_score = score;
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

    return MLP_MODEL_SUCCESS;
}

int32_t mlp_model_predict_raw(
    MlpModel* model,
    const double* x,
    double* y_raw
) {
    if (model == NULL || x == NULL || y_raw == NULL) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    if (model->task_type != MLP_TASK_CLASSIFICATION) {
        return MLP_MODEL_ERROR_UNSUPPORTED_TASK;
    }

    int32_t forward_status = mlp_model_forward(model, x);
    if (forward_status != MLP_MODEL_SUCCESS) {
        return forward_status;
    }

    int32_t output_layer = model->layer_count - 1;

    for (int32_t output = 0; output < model->output_size; output++) {
        y_raw[output] = model->activations[output_layer][output];
    }

    return MLP_MODEL_SUCCESS;
}

int32_t mlp_model_save(const MlpModel* model, const char* path) {
    if (model == NULL || path == NULL) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        return MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    int32_t version = MLP_MODEL_FILE_VERSION;
    int32_t status = MLP_MODEL_SUCCESS;

    if (fwrite(MLP_MODEL_FILE_MAGIC, sizeof(char), 8, file) != 8) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    } else if (fwrite(&version, sizeof(int32_t), 1, file) != 1) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    } else if (fwrite(&model->input_size, sizeof(int32_t), 1, file) != 1) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    } else if (fwrite(&model->output_size, sizeof(int32_t), 1, file) != 1) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    } else if (
        fwrite(&model->hidden_layer_count, sizeof(int32_t), 1, file) != 1
    ) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    } else if (fwrite(&model->task_type, sizeof(int32_t), 1, file) != 1) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    } else if (
        model->hidden_layer_count > 0 &&
        fwrite(
            model->layer_sizes,
            sizeof(int32_t),
            (size_t) model->hidden_layer_count,
            file
        ) != (size_t) model->hidden_layer_count
    ) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    } else {
        status = mlp_model_write_layer_parameters(model, file);
    }

    if (fclose(file) != 0 && status == MLP_MODEL_SUCCESS) {
        status = MLP_MODEL_ERROR_INVALID_ARGUMENT;
    }

    return status;
}

MlpModel* mlp_model_load(const char* path) {
    if (path == NULL) {
        return NULL;
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }

    char magic[8];
    int32_t version = 0;
    int32_t input_size = 0;
    int32_t output_size = 0;
    int32_t hidden_layer_count = 0;
    int32_t task_type = 0;
    int32_t* hidden_layer_sizes = NULL;

    if (fread(magic, sizeof(char), 8, file) != 8) {
        fclose(file);
        return NULL;
    }

    if (memcmp(magic, MLP_MODEL_FILE_MAGIC, 8) != 0) {
        fclose(file);
        return NULL;
    }

    if (fread(&version, sizeof(int32_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    if (version != MLP_MODEL_FILE_VERSION) {
        fclose(file);
        return NULL;
    }

    if (fread(&input_size, sizeof(int32_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    if (fread(&output_size, sizeof(int32_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    if (fread(&hidden_layer_count, sizeof(int32_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    if (fread(&task_type, sizeof(int32_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    if (hidden_layer_count > 0) {
        hidden_layer_sizes = malloc(
            (size_t) hidden_layer_count * sizeof(int32_t)
        );
        if (hidden_layer_sizes == NULL) {
            fclose(file);
            return NULL;
        }

        if (
            fread(
                hidden_layer_sizes,
                sizeof(int32_t),
                (size_t) hidden_layer_count,
                file
            ) != (size_t) hidden_layer_count
        ) {
            free(hidden_layer_sizes);
            fclose(file);
            return NULL;
        }
    }

    if (
        mlp_model_validate_dimensions(
            input_size,
            output_size,
            hidden_layer_count,
            hidden_layer_sizes,
            task_type
        ) != MLP_MODEL_SUCCESS
    ) {
        free(hidden_layer_sizes);
        fclose(file);
        return NULL;
    }

    MlpModel* model = mlp_model_create(
        input_size,
        output_size,
        hidden_layer_count,
        hidden_layer_sizes,
        task_type
    );

    free(hidden_layer_sizes);

    if (model == NULL) {
        fclose(file);
        return NULL;
    }

    if (mlp_model_read_layer_parameters(model, file) != MLP_MODEL_SUCCESS) {
        mlp_model_destroy(model);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return model;
}

void mlp_model_destroy(MlpModel* model) {
    if (model == NULL) {
        return;
    }

    mlp_model_free_layer_arrays(model);
    free(model->layer_sizes);
    free(model);
}