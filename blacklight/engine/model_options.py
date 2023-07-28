from typing import Optional
from tensorflow import keras


class ModelConfig:
    # def __init__(self, config: Optional[dict] = None):
    #     self.config = config
    #     self.check_config()

    def __init__(self, config: Optional[dict] = None):

        model_options = {
            "layer_information": {
                "problem_type": "classification",
                "input_shape": 4,
                "min_dense_layers": 1,
                "max_dense_layers": 8,
                "min_dense_neurons": 2,
                "max_dense_neurons": 8,
                "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
            },
            "target_layer": (1, "sigmoid"),
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ModelConfig.get_default_metrics(),
            "learning_rate": 0.001,
            "epochs": 1000,
            "batch_size": 32,
            "problem_type": "classification",
            "num_classes": 3, }

        default_config = model_options

        # Update default_config with values from config (if not None)
        if config is not None:
            for key in config.keys():
                if key in default_config and key != "layer_information":
                    default_config[key] = config[key]
                elif key == "layer_information":
                    default_config["layer_information"].update(config["layer_information"])

        # Update default_config based on the problem type
        if default_config.get("problem_type") == "classification":
            if "num_classes" not in default_config:
                raise ValueError("ModelConfig has no num_classes set for problem type Classification.")
            default_config["target_layer"] = (default_config.get("num_classes"), "softmax")
            default_config["loss"] = "categorical_crossentropy"
        elif default_config.get("problem_type") == "binary_classification":
            default_config["target_layer"] = (1, "sigmoid")
            default_config["loss"] = "binary_crossentropy"
            default_config["num_classes"] = 2
        elif default_config.get("problem_type") == "regression":
            default_config["target_layer"] = (1, "linear")
            default_config["loss"] = "mse"

        # Set default values for the remaining attributes, not set by the user
        default_config["verbose"] = default_config.get("verbose", 0)
        default_config["class_weight"] = default_config.get("class_weight", None)
        default_config["validation_data"] = default_config.get("validation_data", None)
        default_config["use_multiprocessing"] = default_config.get("use_multiprocessing", False)
        default_config["early_stopping"] = default_config.get("early_stopping", True)
        default_config["callbacks"] = default_config.get("callbacks", ModelConfig.get_default_callbacks())
        default_config["output_bias"] = default_config.get("output_bias", None)
        default_config["fitness_metric"] = default_config.get("fitness_metric", "auc")

        # Set self.config to the final default_config
        self.config = default_config
        self.check_config()


    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def check_config(self):

        if self.config is None:
            raise ValueError("ModelConfig has no config set.")
        if "target_layer" not in self.config:
            raise ValueError("ModelConfig has no target_layer set.")
        if "loss" not in self.config:
            raise ValueError("ModelConfig has no loss set.")
        if "optimizer" not in self.config:
            raise ValueError("ModelConfig has no optimizer set.")
        if "metrics" not in self.config:
            raise ValueError("ModelConfig has no metrics set.")
        if "learning_rate" not in self.config:
            raise ValueError("ModelConfig has no learning_rate set.")
        if "epochs" not in self.config:
            raise ValueError("ModelConfig has no epochs set.")
        if "batch_size" not in self.config:
            raise ValueError("ModelConfig has no batch_size set.")
        if self.config.get("problem_type") == "classification":
            if "num_classes" not in self.config:
                raise ValueError(
                    "ModelConfig has no num_classes set for problem type Classification.")
            if self.config.get("loss") != "categorical_crossentropy":
                raise ValueError(
                    f"ModelConfig has invalid loss {self.config.get('loss')} for problem type multiclass classification.")
        elif self.config.get("problem_type") == "binary_classification":
            if self.config.get("loss") != "binary_crossentropy":
                raise ValueError(
                    f"ModelConfig has invalid loss {self.config.get('loss')} for problem type binary classification.")

    @staticmethod
    def get_default_metrics():
        return [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

    @staticmethod
    def get_default_callbacks():
        return keras.callbacks.EarlyStopping(
            monitor='auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

    @staticmethod
    def parse_options_to_model_options(config):
        updated_config = config
        # if options = none, set it to the dictonary used in test
        # write documentation explaining what each of the paramters are, and what choices the user has
        # try if the user only implements one parameter, and not the other, what happens?
        # HINT; dont write a bunch of if conditions, use the get function. if it dosent exist set it to the default.

        # implement value errors when important things arent given. when u have layer information, u must have layer shape
        # if updated_config.get("layer_information"):
        #     layer_information = updated_config.get("layer_information")
        #     print(layer_information)
        #     updated_config["input_shape"] = layer_information.get("input_shape")
        #     # Get Dense Layer Information
        #     updated_config["max_dense_layers"] = layer_information.get(
        #         "max_dense_layers")
        #     updated_config["min_dense_layers"] = layer_information.get(
        #         "min_dense_layers")
        #     updated_config["min_dense_neurons"] = layer_information.get(
        #         "min_dense_neurons")
        #     updated_config["max_dense_neurons"] = layer_information.get(
        #         "max_dense_neurons")
        #     updated_config["dense_activation_types"] = layer_information.get(
        #         "dense_activation_types", ["relu", "sigmoid", "tanh", "selu"])
        #     # Get Convolutional Layer Information
        #     updated_config["max_conv_layers"] = layer_information.get(
        #         "max_conv_layers")
        #     updated_config["min_conv_layers"] = layer_information.get(
        #         "min_conv_layers")
        #     # Get Dropout Layer Information
        #     updated_config["max_dropout_layers"] = layer_information.get(
        #         "max_dropout_layers")

        # Determine problem type, which sets the last layer of the model.
        print(updated_config)
        if updated_config.get("problem_type") == "classification":
            # if "num_classes" not in updated_config:
            #     raise ValueError(
            #         "ModelConfig has no num_classes set for problem type Classification.")
            setattr(updated_config, "target_layer", (updated_config.get("num_classes"), "softmax"))
            print(updated_config)

            #setattr(updated_config, "loss", "categorical_crossentropy")
            updated_config["num_classes"] = updated_config.get("num_classes")
        elif updated_config.get("problem_type") == "binary_classification":
            setattr(updated_config, "target_layer", (1, "sigmoid"))
            setattr(updated_config, "loss", "binary_crossentropy")
            setattr(updated_config, "num_classes", 2)
        elif updated_config.get("problem_type") == "regression":
            setattr(updated_config, "target_layer", (1, "linear"))
            setattr(updated_config, "loss", "mse")

        # updated_config["problem_type"] = updated_config.get("problem_type")
        #
        # # Add all the model creation options to the config
        # updated_config["optimizer"] = updated_config.get("optimizer", "adam")
        # updated_config["metrics"] = updated_config.get(
        #     "metrics", ModelConfig.get_default_metrics())
        # updated_config["learning_rate"] = updated_config.get("learning_rate", 0.001)

        # Add all the training options to the config
        setattr(updated_config, "verbose", updated_config.get("verbose", 0))
        setattr(updated_config, "class_weight", updated_config.get("class_weight", None))
        setattr(updated_config, "validation_data", updated_config.get("validation_data", None))
        setattr(updated_config, "use_multiprocessing", updated_config.get("use_multiprocessing", False))
        setattr(updated_config, "early_stopping", updated_config.get("early_stopping", True))
        setattr(updated_config, "callbacks", updated_config.get("callbacks", ModelConfig.get_default_callbacks()))
        setattr(updated_config, "output_bias", updated_config.get("output_bias", None))
        setattr(updated_config, "fitness_metric", updated_config.get("fitness_metric", "auc"))

        return updated_config
