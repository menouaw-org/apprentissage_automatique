#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include <stdint.h>

#define LINEAR_TASK_REGRESSION 0
#define LINEAR_TASK_CLASSIFICATION 1

#define LINEAR_MODEL_SUCCESS 0
#define LINEAR_MODEL_ERROR_INVALID_ARGUMENT -1
#define LINEAR_MODEL_ERROR_UNSUPPORTED_TASK -2
#define LINEAR_MODEL_ERROR_ALLOCATION_FAILED -3

/*
 * Conventions de données du modèle linéaire:
 *
 * - x est un tableau aplati en ordre ligne, de taille n_samples * input_size.
 * - y est un tableau aplati en ordre ligne, de taille n_samples * output_size.
 * - y_pred doit être alloué par l'appelant avec output_size valeurs.
 * - les entrées, sorties, poids et calculs internes utilisent double.
 * - le biais est stocké en interne comme le dernier poids de chaque sortie.
 * - les appelants ne doivent pas ajouter de colonne de biais à x.
 *
 * Exemple pour input_size = 2 et n_samples = 3:
 * x = [
 *   x00, x01,
 *   x10, x11,
 *   x20, x21
 * ]
 *
 * Exemple pour output_size = 2 et n_samples = 3:
 * y = [
 *   y00, y01,
 *   y10, y11,
 *   y20, y21
 * ]
 */

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