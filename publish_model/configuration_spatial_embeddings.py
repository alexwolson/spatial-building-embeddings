from transformers import PretrainedConfig


class SpatialEmbeddingsConfig(PretrainedConfig):
    model_type = "spatial_embeddings"

    def __init__(
        self,
        backbone_model_name="facebook/dinov2-base",
        input_dim=768,
        hidden_dim=512,
        output_dim=256,
        dropout=0.1,
        num_hidden_layers=1,
        hidden_dim_multiplier=1.0,
        activation="gelu",
        use_residual=True,
        use_layer_norm=True,
        **kwargs,
    ):
        self.backbone_model_name = backbone_model_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim_multiplier = hidden_dim_multiplier
        self.activation = activation
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        super().__init__(**kwargs)
