# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: config.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script defines the configuration class for loading and saving
#              configuration settings from a YAML file. It allows for dynamic 
#              attribute creation and provides methods for adding new settings
#              and converting the configuration to a dictionary or YAML format.
# ==================================================================================



import yaml


class Config:
    def __init__(self, config_data: dict):
        """
        Dynamically create attributes from a dictionary.
        """
        for key, value in config_data.items():
            # Rekursiv verschachtelte Dictionaries verarbeiten
            if isinstance(value, dict):
                value = Config(value)  # Erstelle verschachtelte Config-Objekte
            setattr(self, key, value)
        else:
            self.data = config_data

    def __getattr__(self, item):
        """
        Raise an AttributeError if the attribute does not exist.
        :param item:
        :return:
        """
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    @staticmethod
    def from_yaml(yaml_file: str) -> "Config":
        """
        Load a YAML file and create a Config object.
        :param yaml_file:
        :return:
        """
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(data)

    def __repr__(self):
        """
        Return the dictionary representation of the Config object.
        :return:
        """
        return f"Config({self.__dict__})"

    def add_config(self, key, value):
        """
        Add a new configuration setting.
        :param key: The key for the new setting.
        :param value: The value for the new setting.
        """
        if isinstance(value, dict):
            value = Config(value)
        setattr(self, key, value)

    def _to_dict(self):
        """
        Convert the Config object to a dictionary.
        :return:
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value._to_dict()
            else:
                result[key] = value
        return result

    def to_yaml(self, yaml_file: str):
        """
        Save the Config object to a YAML file.
        :param yaml_file:
        :return:
        """
        with open(yaml_file, "w") as f:
            yaml.dump(self._to_dict(), f)
