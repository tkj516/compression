def get_config():
    config = dict(
        rd_lambda = 0.01,
        hscore_lambda = 1,
        model_config = dict(
            feature_encoder_config = [
                "MNISTEncoder",
                dict(
                    in_channels=1,
                    hidden_channels=16,
                    out_channels=32,
                ),
            ],
            encoder_config = [
                "MNISTEncoder",
                dict(
                    in_channels=1,
                    hidden_channels=16,
                    out_channels=32,
                ),
            ],
            decoder_config = [
                "MNISTDecoder",
                dict(
                    in_channels=64,
                    hidden_channels=32,
                    out_channels=1,
                ),
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
            model_dir="checkpoints/svdc_mnist_v4",
            distributed=False,
            batch_size=64,
            hscore_start=0,
        ),
    )
    return config
