def get_config():
    config = dict(
        rd_lambda = 0.01,
        model_config = dict(
            augmentation_config = [
                "AdditiveGaussianNoise",
                dict(
                    logvar=0.0,
                )
            ],
            encoder_config = [
                "ELICEncoder",
                dict(
                    in_channels=3,
                    hidden_channels=192,
                    out_channels=256,
                )
            ],
            decoder_config = [
                "ELICDecoder",
                dict(
                    in_channels=256,
                    hidden_channels=192,
                    out_channels=3,
                )
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
            model_dir="checkpoints/svdc_test",
            distributed=False,
        ),
    )
    return config
