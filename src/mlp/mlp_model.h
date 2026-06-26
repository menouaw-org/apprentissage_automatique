#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <stdint.h>

#define MLP_TASK_CLASSIFICATION 1

#define MLP_MODEL_SUCCESS 0
#define MLP_MODEL_ERROR_INVALID_ARGUMENT -1
#define MLP_MODEL_ERROR_UNSUPPORTED_TASK -2
#define MLP_MODEL_ERROR_ALLOCATION_FAILED -3

/*
 * Conventions de données du MLP:
 *
 * - x est un tableau aplati en ordre ligne, de taille n_samples * input_size.
 * - y est un tableau aplati en ordre ligne, de taille n_samples * output_size.
 * - y_pred doit être alloué par l'appelant avec output_size valeurs.
 * - les entrées, sorties, poids, biais et calculs internes utilisent double.
 * - les biais sont stockés et mis à jour côté C.
 * - les appelants ne doivent pas ajouter de colonne de biais à x.
 * - cette première version est limitée à la classification.
 *
 * Exemple pour une classification à trois sorties:
 * dog    = [ 1.0, -1.0, -1.0]
 * cat    = [-1.0,  1.0, -1.0]
 * others = [-1.0, -1.0,  1.0]
 */

typedef struct MlpModel MlpModel;

MlpModel* mlp_model_create(
    int32_t input_size,
    int32_t output_size,
    int32_t hidden_layer_count,
    const int32_t* hidden_layer_sizes,
    int32_t task_type
);

int32_t mlp_model_train(
    MlpModel* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
);

int32_t mlp_model_predict(
    MlpModel* model,
    const double* x,
    double* y_pred
);

int32_t mlp_model_predict_raw(
    MlpModel* model,
    const double* x,
    double* y_raw
);

int32_t mlp_model_save(const MlpModel* model, const char* path);

MlpModel* mlp_model_load(const char* path);

void mlp_model_destroy(MlpModel* model);

#endif