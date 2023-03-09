def get_config():
    config = dict(
        rd_lambda = 0.18,
        model_config = dict(
            augmentation_config = [
                "NoAugmentation",
            ],
            encoder_config = [
                "BMSHJEncoder",
            ],
            decoder_config = [
                "BMSHJDecoder",
            ],
        ),
        optimizer_e_config = [
            "Adam",
            dict(
                lr=1e-4,
            ),
        ],
        optimizer_ed_config = [
            "Adam",
            dict(
                lr=1e-4,
            ),
        ],
        optimizer_aux_config = [
            "Adam",
            dict(
                lr=1e-3,
            ),
        ],
        trainer_config = dict(
            model_dir="checkpoints/bmshj_scheduler",
            distributed=False,
            hscore_start=2_000_000,
        ),
    )
    return config
