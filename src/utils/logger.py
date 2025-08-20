"""
Logging utilities for the lettuce disease classification project.
Provides consistent logging across all modules.
"""

import logging
import os
from typing import Optional
from datetime import datetime


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None,
                  console_output: bool = True,
                  log_format: Optional[str] = None) -> None:
    """
    Setup logging configuration for the entire project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, no file logging)
        console_output: Whether to output logs to console
        log_format: Custom log format string
    """
    
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Progress logger for long-running operations."""
    
    def __init__(self, 
                 logger: logging.Logger,
                 total_items: int,
                 operation_name: str = "Processing",
                 log_interval: int = 100):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance to use
            total_items: Total number of items to process
            operation_name: Name of the operation being performed
            log_interval: How often to log progress (every N items)
        """
        self.logger = logger
        self.total_items = total_items
        self.operation_name = operation_name
        self.log_interval = log_interval
        self.processed = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """
        Update progress and log if necessary.
        
        Args:
            increment: Number of items processed in this update
        """
        self.processed += increment
        
        if self.processed % self.log_interval == 0 or self.processed == self.total_items:
            elapsed = datetime.now() - self.start_time
            percentage = (self.processed / self.total_items) * 100
            rate = self.processed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            
            self.logger.info(
                f"{self.operation_name}: {self.processed}/{self.total_items} "
                f"({percentage:.1f}%) - {rate:.1f} items/sec"
            )
    
    def finish(self):
        """Log completion message."""
        elapsed = datetime.now() - self.start_time
        rate = self.total_items / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        self.logger.info(
            f"{self.operation_name} completed: {self.total_items} items "
            f"in {elapsed.total_seconds():.1f}s ({rate:.1f} items/sec)"
        )


class ModelTrainingLogger:
    """Specialized logger for model training progress."""
    
    def __init__(self, logger: logging.Logger, model_name: str):
        """
        Initialize model training logger.
        
        Args:
            logger: Logger instance to use
            model_name: Name of the model being trained
        """
        self.logger = logger
        self.model_name = model_name
        self.epoch_start_time = None
        self.training_start_time = None
    
    def start_training(self, total_epochs: int):
        """Log training start."""
        self.training_start_time = datetime.now()
        self.logger.info(f"🚀 Starting training for {self.model_name}")
        self.logger.info(f"📊 Total epochs: {total_epochs}")
        self.logger.info("=" * 60)
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        self.logger.info(f"📈 Epoch {epoch}/{total_epochs} - {self.model_name}")
    
    def log_batch_progress(self, batch_idx: int, total_batches: int, 
                          loss: float, accuracy: float = None):
        """Log batch progress (only for important milestones)."""
        if batch_idx % max(1, total_batches // 10) == 0:  # Log 10 times per epoch
            progress = (batch_idx / total_batches) * 100
            msg = f"  Batch {batch_idx}/{total_batches} ({progress:.1f}%) - Loss: {loss:.4f}"
            if accuracy is not None:
                msg += f", Acc: {accuracy:.4f}"
            self.logger.info(msg)
    
    def end_epoch(self, epoch: int, train_loss: float, train_acc: float = None,
                  val_loss: float = None, val_acc: float = None):
        """Log epoch completion."""
        if self.epoch_start_time:
            epoch_time = datetime.now() - self.epoch_start_time
            
            msg = f"✅ Epoch {epoch} completed in {epoch_time.total_seconds():.1f}s"
            msg += f" - Train Loss: {train_loss:.4f}"
            
            if train_acc is not None:
                msg += f", Train Acc: {train_acc:.4f}"
            if val_loss is not None:
                msg += f", Val Loss: {val_loss:.4f}"
            if val_acc is not None:
                msg += f", Val Acc: {val_acc:.4f}"
            
            self.logger.info(msg)
    
    def end_training(self, final_metrics: dict = None):
        """Log training completion."""
        if self.training_start_time:
            total_time = datetime.now() - self.training_start_time
            self.logger.info("=" * 60)
            self.logger.info(f"🎉 Training completed for {self.model_name}")
            self.logger.info(f"⏱️ Total training time: {total_time}")
            
            if final_metrics:
                self.logger.info("📊 Final Metrics:")
                for metric, value in final_metrics.items():
                    self.logger.info(f"  {metric}: {value}")


class ExperimentLogger:
    """Logger for tracking experiments and results."""
    
    def __init__(self, logger: logging.Logger, experiment_name: str):
        """
        Initialize experiment logger.
        
        Args:
            logger: Logger instance to use
            experiment_name: Name of the experiment
        """
        self.logger = logger
        self.experiment_name = experiment_name
        self.start_time = None
        self.results = {}
    
    def start_experiment(self, description: str = None):
        """Log experiment start."""
        self.start_time = datetime.now()
        self.logger.info("🔬 " + "=" * 50)
        self.logger.info(f"🔬 EXPERIMENT: {self.experiment_name}")
        if description:
            self.logger.info(f"📝 Description: {description}")
        self.logger.info(f"🕐 Start time: {self.start_time}")
        self.logger.info("🔬 " + "=" * 50)
    
    def log_config(self, config_dict: dict):
        """Log experiment configuration."""
        self.logger.info("⚙️ Experiment Configuration:")
        for key, value in config_dict.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_phase(self, phase_name: str, details: str = None):
        """Log experiment phase."""
        self.logger.info(f"📋 Phase: {phase_name}")
        if details:
            self.logger.info(f"   Details: {details}")
    
    def log_result(self, metric_name: str, value: float, details: str = None):
        """Log experiment result."""
        self.results[metric_name] = value
        msg = f"📊 {metric_name}: {value}"
        if details:
            msg += f" ({details})"
        self.logger.info(msg)
    
    def log_comparison(self, results_dict: dict, title: str = "Comparison"):
        """Log comparison of multiple results."""
        self.logger.info(f"📈 {title}:")
        for name, value in results_dict.items():
            self.logger.info(f"  {name:25} | {value}")
    
    def end_experiment(self, success: bool = True, summary: str = None):
        """Log experiment completion."""
        if self.start_time:
            total_time = datetime.now() - self.start_time
            
            status = "✅ COMPLETED" if success else "❌ FAILED"
            self.logger.info("🔬 " + "=" * 50)
            self.logger.info(f"🔬 EXPERIMENT {status}: {self.experiment_name}")
            self.logger.info(f"⏱️ Total time: {total_time}")
            
            if self.results:
                self.logger.info("📊 Final Results:")
                for metric, value in self.results.items():
                    self.logger.info(f"  {metric}: {value}")
            
            if summary:
                self.logger.info(f"📝 Summary: {summary}")
            
            self.logger.info("🔬 " + "=" * 50)


def log_system_info(logger: logging.Logger):
    """Log system information."""
    import platform
    import psutil
    import torch
    
    logger.info("💻 System Information:")
    logger.info(f"  OS: {platform.system()} {platform.release()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU Cores: {psutil.cpu_count()}")
    logger.info(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name()}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
    else:
        logger.info("  GPU: Not available")
    
    logger.info(f"  PyTorch: {torch.__version__}")


def create_session_logger(session_name: str, 
                         log_dir: str = "results/logs",
                         log_level: str = "INFO") -> logging.Logger:
    """
    Create a logger for a specific session.
    
    Args:
        session_name: Name of the session
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{session_name}_{timestamp}.log")
    
    # Setup logging
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=True
    )
    
    # Get logger
    logger = get_logger(session_name)
    
    # Log session start
    logger.info("🌿 " + "=" * 60)
    logger.info(f"🌿 LETTUCE DISEASE CLASSIFICATION - {session_name.upper()}")
    logger.info(f"🌿 Session started at: {datetime.now()}")
    logger.info(f"🌿 Log file: {log_file}")
    logger.info("🌿 " + "=" * 60)
    
    # Log system info
    log_system_info(logger)
    
    return logger


# Utility decorators for logging
def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"⏱️ Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = end_time - start_time
                logger.info(f"✅ {func.__name__} completed in {duration.total_seconds():.2f}s")
                return result
            
            except Exception as e:
                end_time = datetime.now()
                duration = end_time - start_time
                logger.error(f"❌ {func.__name__} failed after {duration.total_seconds():.2f}s: {e}")
                raise
            
        return wrapper
    return decorator


def log_function_call(logger: logging.Logger, log_args: bool = False):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if log_args:
                logger.debug(f"🔧 Calling {func.__name__} with args={args}, kwargs={kwargs}")
            else:
                logger.debug(f"🔧 Calling {func.__name__}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Context manager for logging operations
class LoggedOperation:
    """Context manager for logging long operations."""
    
    def __init__(self, logger: logging.Logger, operation_name: str, log_level: str = "INFO"):
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = getattr(logging, log_level.upper())
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.log_level, f"🔄 Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.log(self.log_level, 
                          f"✅ {self.operation_name} completed in {duration.total_seconds():.2f}s")
        else:
            self.logger.error(
                f"❌ {self.operation_name} failed after {duration.total_seconds():.2f}s: {exc_val}"
            )
        
        return False  # Don't suppress exceptions