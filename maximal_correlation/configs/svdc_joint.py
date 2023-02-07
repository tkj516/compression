def get_config():
    config = dict(
        rd_lambda = 0.18,
        hscore_lambda = 1e-3,
        model_config = dict(
            augmentation_config = [
                "AdditiveGaussianNoise",
            ],
            encoder_config = [
                "BMSHJEncoder",
            ],
            decoder_config = [
                "BMSHJDecoder",
            ],
        ),
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
            model_dir="checkpoints/svdc_joint",
            distributed=True,
            batch_size=16,
            joint_optimization=True,
        ),
    )
    return config
