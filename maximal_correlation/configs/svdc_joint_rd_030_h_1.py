def get_config():
    config = dict(
        rd_lambda = 0.30,
        hscore_lambda = 1e-7,
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
            "AdamW",
            dict(
                lr=1e-4,
                weight_decay=0.001,
            ),
        ],
        optimizer_aux_config = [
            "AdamW",
            dict(
                lr=1e-3,
                weight_decay=0.001,
            ),
        ],
        trainer_config = dict(
            model_dir="checkpoints/svdc_joint_rd_030_h_1",
            distributed=False,
            batch_size=8,
            joint_optimization=True,
        ),
    )
    return config
