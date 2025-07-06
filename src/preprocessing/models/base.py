"""
Base model configuration classes for preprocessing pipeline.

This module provides base classes for model configuration and management.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModelConfig(ABC):
    """
    Base class for model configuration.

    This abstract base class defines the interface for model configurations.
    """

    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize BaseModelConfig.

        Args:
            name: Model name
            enabled: Whether the model is enabled
        """
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate model configuration.

        Returns:
            True if configuration is valid
        """
        pass


class ModelRegistry:
    """
    Model registry for managing model configurations.

    This class provides a registry for storing and retrieving model configurations.
    """

    def __init__(self):
        """Initialize ModelRegistry."""
        self.models: Dict[str, BaseModelConfig] = {}
        self.logger = logging.getLogger(__name__)

    def register(self, model: BaseModelConfig) -> None:
        """
        Register a model configuration.

        Args:
            model: Model configuration to register
        """
        self.models[model.name] = model
        self.logger.info(f"Registered model: {model.name}")

    def get(self, name: str) -> Optional[BaseModelConfig]:
        """
        Get a model configuration by name.

        Args:
            name: Model name

        Returns:
            Model configuration or None if not found
        """
        return self.models.get(name)

    def list_models(self) -> List[str]:
        """
        List all registered model names.

        Returns:
            List of model names
        """
        return list(self.models.keys())

    def get_enabled_models(self) -> List[BaseModelConfig]:
        """
        Get all enabled model configurations.

        Returns:
            List of enabled model configurations
        """
        return [model for model in self.models.values() if model.enabled]


class ModelFactory:
    """
    Model factory for creating model instances.

    This class provides factory methods for creating model instances
    from configurations.
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize ModelFactory.

        Args:
            registry: Model registry
        """
        self.registry = registry
        self.logger = logging.getLogger(__name__)

    def create_model(self, name: str, **kwargs) -> Any:
        """
        Create a model instance.

        Args:
            name: Model name
            **kwargs: Additional parameters

        Returns:
            Model instance
        """
        model_config = self.registry.get(name)
        if model_config is None:
            raise ValueError(f"Model not found: {name}")

        if not model_config.enabled:
            raise ValueError(f"Model is disabled: {name}")

        # Validate configuration
        if not model_config.validate():
            raise ValueError(f"Invalid model configuration: {name}")

        # Get parameters
        params = model_config.get_params()
        params.update(kwargs)

        self.logger.info(f"Creating model: {name} with params: {params}")

        # Create model instance (to be implemented by subclasses)
        return self._create_instance(name, params)

    def _create_instance(self, name: str, params: Dict[str, Any]) -> Any:
        """
        Create model instance (to be implemented by subclasses).

        Args:
            name: Model name
            params: Model parameters

        Returns:
            Model instance
        """
        raise NotImplementedError("Subclasses must implement _create_instance")
