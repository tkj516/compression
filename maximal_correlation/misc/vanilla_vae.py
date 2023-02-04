def get_config():
    config = dict(
        model_config = dict(
            encoder_config=[
                "ConvEncoder",
                dict(
                    in_channels=3,
                    hidden_channels=192,
                )
            ],
            decoder_config=[
                "ConvDecoder",
                dict(
                    out_channels=3,
                    hidden_channels=192,
                )
            ],
            massager_config=[
                "NoMassager",
                dict(),
            ],
        ),
        vae_loss_config = dict(
            kld_weight=0.00025,
        ),
        hscore_loss_config = dict(
            feature_dim=192,
        ),
        trainer_config = dict(
            model_dir="checkpoints/vanilla_vae",
        ),
    )
    return config
