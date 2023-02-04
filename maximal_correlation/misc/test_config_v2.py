def get_config():
    config = dict(
        model_config = dict(
            encoder_config=[
                "BMSHJEncoder",
                dict(
                    in_channels=3,
                    hidden_channels=192,
                )
            ],
            decoder_config=[
                "BMSHJDecoder",
                dict(
                    out_channels=3,
                    hidden_channels=192,
                )
            ],
            massager_config=[
                "SimpleMassager",
                dict(
                    channels=192,
                )
            ],
        ),
        vae_loss_config = dict(
            min_logvar=-30.0,
            max_logvar=20.0,
            kld_weight=0.00025,
        ),
        hscore_loss_config = dict(
            feature_dim=192,
        ),
        trainer_config = dict(
            model_dir="checkpoints/test",
        ),
    )
    return config
