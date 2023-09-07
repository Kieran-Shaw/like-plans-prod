import logging


class Config:
    # Base configuration class. Contains default settings and settings common to all environments.
    DEBUG = False
    TESTING = False
    LOG_LEVEL = logging.INFO  # Default log level


class ProductionConfig(Config):
    # Production specific configuration.
    LOG_LEVEL = logging.WARNING  # Production tends to have a higher log level


class DevelopmentConfig(Config):
    # Development environment specific configuration
    LOG_LEVEL = logging.DEBUG  # Debug level logging in development
