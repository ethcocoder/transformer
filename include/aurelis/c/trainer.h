#ifndef AURELIS_C_TRAINER_H
#define AURELIS_C_TRAINER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a trainer instance */
typedef struct AurelisTrainer AurelisTrainer;

/* Status codes */
typedef enum {
    AURELIS_OK = 0,
    AURELIS_ERR_INVALID_PARAM = -1,
    AURELIS_ERR_MEMORY = -2,
    AURELIS_ERR_CONFIG = -3,
    AURELIS_ERR_DATA = -4,
    AURELIS_ERR_RUNTIME = -5,
} AurelisStatus;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/* Create a trainer from a JSON config string. Returns NULL on error. */
AurelisTrainer* aurelis_trainer_create(const char* config_json);

/* Destroy a trainer and free all resources. */
void aurelis_trainer_destroy(AurelisTrainer* trainer);

/* ------------------------------------------------------------------ */
/*  Data loading                                                      */
/* ------------------------------------------------------------------ */

/* Load text data from a file or directory. Returns status code. */
AurelisStatus aurelis_trainer_load_data(AurelisTrainer* trainer,
                                        const char* path);

/* ------------------------------------------------------------------ */
/*  Training                                                          */
/* ------------------------------------------------------------------ */

/* Run one training epoch. Returns status code. */
AurelisStatus aurelis_trainer_train_epoch(AurelisTrainer* trainer);

/* ------------------------------------------------------------------ */
/*  Query                                                             */
/* ------------------------------------------------------------------ */

/* Get the current average training loss. */
float aurelis_trainer_get_loss(const AurelisTrainer* trainer);

/* Get the current training accuracy (0..1). */
float aurelis_trainer_get_accuracy(const AurelisTrainer* trainer);

/* Get vocabulary size. */
int aurelis_trainer_vocab_size(const AurelisTrainer* trainer);

/* Get total number of parameters. */
size_t aurelis_trainer_num_params(const AurelisTrainer* trainer);

/* ------------------------------------------------------------------ */
/*  Checkpointing                                                     */
/* ------------------------------------------------------------------ */

/* Save a checkpoint to a file path (directory prefix). */
AurelisStatus aurelis_trainer_save(AurelisTrainer* trainer,
                                   const char* path);

/* Load a checkpoint from a file path (directory prefix). */
AurelisStatus aurelis_trainer_load(AurelisTrainer* trainer,
                                   const char* path);

/* ------------------------------------------------------------------ */
/*  Error reporting                                                   */
/* ------------------------------------------------------------------ */

/* Return a human-readable string for the last error. */
const char* aurelis_trainer_last_error(const AurelisTrainer* trainer);

#ifdef __cplusplus
}
#endif

#endif /* AURELIS_C_TRAINER_H */
