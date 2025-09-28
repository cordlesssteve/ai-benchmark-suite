#!/usr/bin/env python3
"""
Enterprise Configuration Management (Sprint 4.0)

Advanced configuration management system for AI Benchmark Suite with:
- Environment-specific configurations (dev, staging, production)
- Secure secrets management and encryption
- Dynamic configuration reloading and validation
- Configuration versioning and rollback capabilities
- Integration with external configuration sources
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from datetime import datetime
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigSource(Enum):
    """Configuration sources"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    CONSUL = "consul"
    KUBERNETES = "kubernetes"


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    name: str
    type: str  # "ollama", "huggingface", "openai", "anthropic"
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "auto"
    max_tokens: int = 512
    temperature: float = 0.2
    timeout: int = 60
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    """Harness-specific configuration"""
    name: str
    enabled: bool = True
    docker_image: Optional[str] = None
    timeout: int = 300
    memory_limit: str = "2g"
    cpu_limit: str = "1.0"
    max_problems: int = 100
    allow_code_execution: bool = True
    safety_checks: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Performance optimization configuration"""
    parallel_execution: bool = True
    max_workers: int = 4
    max_containers: int = 6
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    cache_strategy: str = "conservative"
    memory_optimization: bool = True
    memory_limit_mb: int = 8192
    execution_strategy: str = "concurrent_languages"
    container_cleanup: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    enabled: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000
    alerts_enabled: bool = True
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    retention_days: int = 30
    export_metrics: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_authentication: bool = False
    secret_key: str = "change_in_production"
    encryption_enabled: bool = False
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    rate_limiting: bool = True
    max_requests_per_minute: int = 60
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False


@dataclass
class DatabaseConfig:
    """Database configuration"""
    enabled: bool = True
    type: str = "postgresql"  # postgresql, sqlite, mysql
    host: str = "postgres"
    port: int = 5432
    database: str = "ai_benchmark_suite"
    username: str = "benchmark_user"
    password: str = "benchmark_secure_password_2024"
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    timeout: int = 30


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    type: str = "redis"  # redis, memory, file
    host: str = "redis"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ttl_default: int = 3600
    max_memory: str = "2gb"
    eviction_policy: str = "allkeys-lru"


@dataclass
class EnterpriseConfig:
    """Comprehensive enterprise configuration"""
    environment: Environment
    version: str = "4.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Core configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    harnesses: Dict[str, HarnessConfig] = field(default_factory=dict)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Environment-specific overrides
    env_overrides: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecretManager:
    """Secure secret management with encryption"""

    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv("CONFIG_MASTER_KEY", "default_key_change_in_production")
        self._fernet = self._create_cipher()

    def _create_cipher(self) -> Fernet:
        """Create encryption cipher from master key"""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ai_benchmark_salt',  # Use a fixed salt for consistency
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)

    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value"""
        encrypted = self._fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt value: {e}")

    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted"""
        try:
            base64.urlsafe_b64decode(value.encode())
            return True
        except:
            return False


class ConfigValidator:
    """Configuration validation and schema checking"""

    @staticmethod
    def validate_config(config: EnterpriseConfig) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Environment validation
        if config.environment not in Environment:
            errors.append(f"Invalid environment: {config.environment}")

        # Model validation
        for model_name, model_config in config.models.items():
            if not model_config.name:
                errors.append(f"Model {model_name} missing name")
            if model_config.type not in ["ollama", "huggingface", "openai", "anthropic"]:
                errors.append(f"Model {model_name} has invalid type: {model_config.type}")
            if model_config.max_tokens <= 0:
                errors.append(f"Model {model_name} has invalid max_tokens: {model_config.max_tokens}")

        # Security validation
        if config.environment == Environment.PRODUCTION:
            if config.security.secret_key == "change_in_production":
                errors.append("Production environment must have secure secret_key")
            if not config.security.enable_authentication:
                errors.append("Production environment should enable authentication")

        # Database validation
        if config.database.enabled:
            if not config.database.host:
                errors.append("Database host is required when database is enabled")
            if config.database.port <= 0 or config.database.port > 65535:
                errors.append(f"Invalid database port: {config.database.port}")

        # Optimization validation
        if config.optimization.max_workers <= 0:
            errors.append(f"Invalid max_workers: {config.optimization.max_workers}")
        if config.optimization.max_containers <= 0:
            errors.append(f"Invalid max_containers: {config.optimization.max_containers}")

        return errors

    @staticmethod
    def validate_environment_requirements(config: EnterpriseConfig) -> List[str]:
        """Validate environment-specific requirements"""
        warnings = []

        if config.environment == Environment.PRODUCTION:
            # Production-specific checks
            if not config.security.ssl_enabled:
                warnings.append("Production should enable SSL")
            if config.monitoring.log_level == "DEBUG":
                warnings.append("Production should not use DEBUG log level")
            if not config.monitoring.alerts_enabled:
                warnings.append("Production should enable alerts")

        elif config.environment == Environment.DEVELOPMENT:
            # Development-specific checks
            if config.security.enable_authentication:
                warnings.append("Development may not need authentication")

        return warnings


class EnterpriseConfigManager:
    """
    Enterprise-grade configuration management system.

    Features:
    - Environment-specific configurations
    - Secure secret management
    - Dynamic configuration reloading
    - Configuration validation and schema checking
    - Version control and rollback capabilities
    - Integration with external sources
    """

    def __init__(self, config_dir: Path, environment: Environment = Environment.DEVELOPMENT):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.environment = environment
        self.secret_manager = SecretManager()
        self.validator = ConfigValidator()

        # Configuration state
        self._config: Optional[EnterpriseConfig] = None
        self._config_history: List[EnterpriseConfig] = []
        self._last_modified: float = 0
        self._watchers: List[callable] = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load initial configuration
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from files and environment"""
        config_file = self.config_dir / f"{self.environment.value}.yaml"

        # Start with default configuration
        config = self._create_default_config()

        # Load from file if exists
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config = self._merge_config(config, file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Failed to load config file {config_file}: {e}")

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)

        # Decrypt sensitive values
        config = self._decrypt_config(config)

        # Validate configuration
        errors = self.validator.validate_config(config)
        if errors:
            self.logger.error(f"Configuration validation errors: {errors}")
            raise ValueError(f"Invalid configuration: {errors}")

        warnings = self.validator.validate_environment_requirements(config)
        for warning in warnings:
            self.logger.warning(warning)

        # Store configuration
        self._config = config
        self._config_history.append(config)
        self._last_modified = time.time()

        self.logger.info(f"Configuration loaded for {self.environment.value} environment")

    def _create_default_config(self) -> EnterpriseConfig:
        """Create default configuration"""
        config = EnterpriseConfig(environment=self.environment)

        # Default models
        config.models = {
            "qwen-coder": ModelConfig(
                name="qwen-coder",
                type="ollama",
                endpoint="http://localhost:11434",
                max_tokens=512,
                temperature=0.2
            ),
            "codellama": ModelConfig(
                name="codellama",
                type="ollama",
                endpoint="http://localhost:11434",
                max_tokens=512,
                temperature=0.2
            )
        }

        # Default harnesses
        config.harnesses = {
            "bigcode": HarnessConfig(
                name="bigcode",
                docker_image="python:3.11-slim",
                timeout=300,
                memory_limit="2g",
                max_problems=100
            ),
            "lm-eval": HarnessConfig(
                name="lm-eval",
                docker_image="python:3.11-slim",
                timeout=300,
                memory_limit="2g",
                max_problems=100
            )
        }

        # Environment-specific defaults
        if self.environment == Environment.PRODUCTION:
            config.security.enable_authentication = True
            config.security.ssl_enabled = True
            config.monitoring.alerts_enabled = True
            config.optimization.max_workers = 8
            config.optimization.max_containers = 12
        elif self.environment == Environment.DEVELOPMENT:
            config.monitoring.log_level = "DEBUG"
            config.security.enable_authentication = False

        return config

    def _merge_config(self, base_config: EnterpriseConfig, file_config: Dict[str, Any]) -> EnterpriseConfig:
        """Merge file configuration into base configuration"""
        # Convert file config to dataclass
        config_dict = asdict(base_config)

        # Deep merge
        def deep_merge(base: dict, overlay: dict):
            for key, value in overlay.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(config_dict, file_config)

        # Reconstruct dataclass
        # This is simplified - in production, you'd want more robust deserialization
        return EnterpriseConfig(**config_dict)

    def _apply_env_overrides(self, config: EnterpriseConfig) -> EnterpriseConfig:
        """Apply environment variable overrides"""
        # Define environment variable mappings
        env_mappings = {
            "AI_BENCHMARK_SECRET_KEY": "security.secret_key",
            "AI_BENCHMARK_DB_HOST": "database.host",
            "AI_BENCHMARK_DB_PASSWORD": "database.password",
            "AI_BENCHMARK_REDIS_HOST": "cache.host",
            "AI_BENCHMARK_LOG_LEVEL": "monitoring.log_level",
            "AI_BENCHMARK_MAX_WORKERS": "optimization.max_workers",
            "AI_BENCHMARK_ENABLE_SSL": "security.ssl_enabled",
            "AI_BENCHMARK_ENABLE_AUTH": "security.enable_authentication"
        }

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                self._set_nested_config(config, config_path, value)

        return config

    def _set_nested_config(self, config: EnterpriseConfig, path: str, value: str):
        """Set nested configuration value from dot notation path"""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            current = getattr(current, key)

        # Convert string values to appropriate types
        final_key = keys[-1]
        if hasattr(current, final_key):
            current_value = getattr(current, final_key)
            if isinstance(current_value, bool):
                value = value.lower() in ('true', '1', 'yes')
            elif isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)

        setattr(current, final_key, value)

    def _decrypt_config(self, config: EnterpriseConfig) -> EnterpriseConfig:
        """Decrypt encrypted configuration values"""
        # Decrypt sensitive fields
        sensitive_fields = [
            ("database", "password"),
            ("security", "secret_key"),
            ("cache", "password")
        ]

        for section, field in sensitive_fields:
            section_obj = getattr(config, section)
            if hasattr(section_obj, field):
                value = getattr(section_obj, field)
                if value and self.secret_manager.is_encrypted(value):
                    try:
                        decrypted = self.secret_manager.decrypt_value(value)
                        setattr(section_obj, field, decrypted)
                    except Exception as e:
                        self.logger.warning(f"Failed to decrypt {section}.{field}: {e}")

        # Decrypt model API keys
        for model_config in config.models.values():
            if model_config.api_key and self.secret_manager.is_encrypted(model_config.api_key):
                try:
                    model_config.api_key = self.secret_manager.decrypt_value(model_config.api_key)
                except Exception as e:
                    self.logger.warning(f"Failed to decrypt API key for model {model_config.name}: {e}")

        return config

    def get_config(self) -> EnterpriseConfig:
        """Get current configuration"""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        return self.get_config().models.get(model_name)

    def get_harness_config(self, harness_name: str) -> Optional[HarnessConfig]:
        """Get configuration for specific harness"""
        return self.get_config().harnesses.get(harness_name)

    def save_configuration(self, config: Optional[EnterpriseConfig] = None):
        """Save configuration to file"""
        if config is None:
            config = self.get_config()

        config_file = self.config_dir / f"{self.environment.value}.yaml"

        # Encrypt sensitive values before saving
        config_to_save = self._encrypt_config_for_storage(config)

        # Convert to dict and save
        config_dict = asdict(config_to_save)

        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to {config_file}")

    def _encrypt_config_for_storage(self, config: EnterpriseConfig) -> EnterpriseConfig:
        """Encrypt sensitive values for storage"""
        # Create a copy for modification
        import copy
        config_copy = copy.deepcopy(config)

        # Encrypt sensitive fields if not already encrypted
        sensitive_fields = [
            ("database", "password"),
            ("cache", "password")
        ]

        for section, field in sensitive_fields:
            section_obj = getattr(config_copy, section)
            if hasattr(section_obj, field):
                value = getattr(section_obj, field)
                if value and not self.secret_manager.is_encrypted(value):
                    encrypted = self.secret_manager.encrypt_value(value)
                    setattr(section_obj, field, encrypted)

        # Encrypt model API keys
        for model_config in config_copy.models.values():
            if model_config.api_key and not self.secret_manager.is_encrypted(model_config.api_key):
                model_config.api_key = self.secret_manager.encrypt_value(model_config.api_key)

        return config_copy

    def reload_configuration(self):
        """Reload configuration from sources"""
        self.logger.info("Reloading configuration...")
        old_config = self._config
        self._load_configuration()

        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(old_config, self._config)
            except Exception as e:
                self.logger.error(f"Configuration watcher failed: {e}")

    def add_config_watcher(self, callback: callable):
        """Add configuration change watcher"""
        self._watchers.append(callback)

    def get_config_history(self) -> List[EnterpriseConfig]:
        """Get configuration history"""
        return self._config_history.copy()

    def rollback_config(self, version: int = 1):
        """Rollback to previous configuration version"""
        if len(self._config_history) < version + 1:
            raise ValueError(f"Not enough configuration history for rollback to version {version}")

        previous_config = self._config_history[-(version + 1)]
        self._config = previous_config
        self.save_configuration()

        self.logger.info(f"Configuration rolled back {version} version(s)")

    def export_config_template(self, output_file: Path):
        """Export configuration template for documentation"""
        template_config = self._create_default_config()

        # Add comments/documentation
        config_dict = asdict(template_config)

        with open(output_file, 'w') as f:
            f.write("# AI Benchmark Suite - Configuration Template\n")
            f.write(f"# Environment: {self.environment.value}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration template exported to {output_file}")


# Factory functions
def create_config_manager(config_dir: Path, environment: str = "development") -> EnterpriseConfigManager:
    """Factory function to create configuration manager"""
    env = Environment(environment.lower())
    return EnterpriseConfigManager(config_dir, env)


def load_config_from_env() -> EnterpriseConfigManager:
    """Load configuration manager from environment variables"""
    config_dir = Path(os.getenv("AI_BENCHMARK_CONFIG_DIR", "./config"))
    environment = os.getenv("AI_BENCHMARK_ENVIRONMENT", "development")
    return create_config_manager(config_dir, environment)


# Testing and demonstration
if __name__ == "__main__":
    print("🔧 Enterprise Configuration Manager Demo")
    print("=" * 50)

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)

        # Create configuration manager
        config_manager = create_config_manager(config_dir, "development")

        # Get configuration
        config = config_manager.get_config()
        print(f"Environment: {config.environment.value}")
        print(f"Models configured: {list(config.models.keys())}")
        print(f"Harnesses configured: {list(config.harnesses.keys())}")

        # Test model configuration
        qwen_config = config_manager.get_model_config("qwen-coder")
        if qwen_config:
            print(f"Qwen config: {qwen_config.type}, max_tokens: {qwen_config.max_tokens}")

        # Save configuration
        config_manager.save_configuration()
        print("Configuration saved to file")

        # Export template
        template_file = config_dir / "template.yaml"
        config_manager.export_config_template(template_file)
        print(f"Template exported to {template_file}")

        print("\n✅ Enterprise configuration management demo completed!")
        print("🔧 Demonstrated secure configuration with encryption and validation")