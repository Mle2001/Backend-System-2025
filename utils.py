"""
utils.py - Utility Functions và Helpers cho VideoRAG System
Chứa các helper functions được sử dụng xuyên suốt hệ thống
"""

import os
import sys
import time
import json
import hashlib
import logging
import traceback
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Iterator
from functools import wraps
from contextlib import contextmanager
import numpy as np
from dataclasses import dataclass
import psutil
import pickle

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter với colors cho terminal output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to levelname
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    enable_colors: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (optional)
        log_dir: Directory cho log files
        enable_colors: Enable colored output
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger('VideoRAG')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if enable_colors:
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        if not log_file.endswith('.log'):
            log_file += '.log'
        log_path = os.path.join(log_dir, log_file)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to file: {log_path}")
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(f'VideoRAG.{name}' if name else 'VideoRAG')

# ============================================================================
# VIDEO UTILITIES
# ============================================================================

def validate_video_format(video_path: str, supported_formats: List[str] = None) -> bool:
    """
    Validate video file format
    
    Args:
        video_path: Đường dẫn đến video file
        supported_formats: List các format được hỗ trợ
        
    Returns:
        True nếu format được hỗ trợ
    """
    if supported_formats is None:
        supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v']
    
    if not os.path.exists(video_path):
        return False
    
    file_ext = Path(video_path).suffix.lower().lstrip('.')
    return file_ext in [fmt.lower() for fmt in supported_formats]

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video information using ffprobe/opencv
    
    Args:
        video_path: Đường dẫn đến video file
        
    Returns:
        Dict chứa video info (duration, fps, resolution, etc.)
    """
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'size_bytes': os.path.getsize(video_path),
            'format': Path(video_path).suffix.lower().lstrip('.')
        }
        
    except Exception as e:
        get_logger('utils').error(f"Error getting video info: {e}")
        return {'path': video_path, 'error': str(e)}

def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp string to seconds
    
    Args:
        timestamp: Format "HH:MM:SS" hoặc "MM:SS" hoặc seconds
        
    Returns:
        Seconds as float
    """
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    
    try:
        parts = timestamp.split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:  # Just seconds
            return float(timestamp)
    except (ValueError, AttributeError):
        return 0.0

def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to timestamp string
    
    Args:
        seconds: Seconds as float
        
    Returns:
        Timestamp string in format "HH:MM:SS"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{secs:06.3f}"

def generate_video_id(video_path: str) -> str:
    """
    Generate unique ID cho video dựa trên path và metadata
    
    Args:
        video_path: Đường dẫn đến video
        
    Returns:
        Unique video ID
    """
    # Combine path, size, và modified time để tạo unique hash
    try:
        stat = os.stat(video_path)
        content = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        # Fallback to path-only hash
        return hashlib.md5(video_path.encode()).hexdigest()[:16]

# ============================================================================
# SIMILARITY CALCULATION UTILITIES
# ============================================================================

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate cosine similarity giữa 2 vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    try:
        # Normalize vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vector1, vector2) / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    
    except Exception as e:
        get_logger('utils').error(f"Error calculating cosine similarity: {e}")
        return 0.0

def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate Euclidean distance giữa 2 vectors"""
    try:
        return float(np.linalg.norm(vector1 - vector2))
    except Exception as e:
        get_logger('utils').error(f"Error calculating euclidean distance: {e}")
        return float('inf')

def calculate_batch_similarities(
    query_vector: np.ndarray, 
    candidate_vectors: List[np.ndarray],
    metric: str = "cosine"
) -> List[float]:
    """
    Calculate similarities batch-wise cho performance
    
    Args:
        query_vector: Query vector
        candidate_vectors: List of candidate vectors
        metric: Similarity metric ("cosine" hoặc "euclidean")
        
    Returns:
        List of similarity scores
    """
    try:
        if metric == "cosine":
            similarities = [cosine_similarity(query_vector, vec) for vec in candidate_vectors]
        elif metric == "euclidean":
            distances = [euclidean_distance(query_vector, vec) for vec in candidate_vectors]
            # Convert distances to similarities (lower distance = higher similarity)
            max_dist = max(distances) if distances else 1.0
            similarities = [1.0 - (dist / max_dist) for dist in distances]
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return similarities
    
    except Exception as e:
        get_logger('utils').error(f"Error calculating batch similarities: {e}")
        return [0.0] * len(candidate_vectors)

# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

def batch_processor(
    items: List[Any], 
    batch_size: int,
    process_func: Callable[[List[Any]], Any],
    max_workers: Optional[int] = None,
    show_progress: bool = True
) -> List[Any]:
    """
    Process items trong batches với optional multiprocessing
    
    Args:
        items: List of items to process
        batch_size: Size của mỗi batch
        process_func: Function để process mỗi batch
        max_workers: Number of workers cho multiprocessing
        show_progress: Show progress bar
        
    Returns:
        List of processed results
    """
    logger = get_logger('batch_processor')
    
    if not items:
        return []
    
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    results = []
    
    logger.info(f"Processing {len(items)} items in {len(batches)} batches (size: {batch_size})")
    
    try:
        if max_workers and max_workers > 1:
            # Multiprocessing
            with multiprocessing.Pool(max_workers) as pool:
                if show_progress:
                    from tqdm import tqdm
                    results = list(tqdm(pool.imap(process_func, batches), total=len(batches)))
                else:
                    results = pool.map(process_func, batches)
        else:
            # Sequential processing
            if show_progress:
                from tqdm import tqdm
                batches = tqdm(batches, desc="Processing batches")
            
            for batch in batches:
                result = process_func(batch)
                results.append(result)
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise
    
    # Flatten results if needed
    flat_results = []
    for result in results:
        if isinstance(result, list):
            flat_results.extend(result)
        else:
            flat_results.append(result)
    
    logger.info(f"Batch processing completed: {len(flat_results)} results")
    return flat_results

def chunk_list(items: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """
    Chia list thành chunks
    
    Args:
        items: List to chunk
        chunk_size: Size của mỗi chunk
        
    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

# ============================================================================
# ERROR HANDLING UTILITIES
# ============================================================================

def safe_execute(func: Callable, *args, default=None, log_errors: bool = True, **kwargs):
    """
    Execute function safely với error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default value nếu function fails
        log_errors: Log errors to logger
        **kwargs: Function keyword arguments
        
    Returns:
        Function result hoặc default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = get_logger('safe_execute')
            logger.error(f"Error executing {func.__name__}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        return default

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Decorator để retry function khi fail
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay giữa retries (seconds)
        backoff_factor: Multiplier cho delay mỗi retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('retry')
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {current_delay:.1f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
        return wrapper
    return decorator

@contextmanager
def error_context(context_name: str, reraise: bool = True):
    """
    Context manager cho error handling với context info
    
    Args:
        context_name: Name của context cho logging
        reraise: Whether to reraise exception
    """
    logger = get_logger('error_context')
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {context_name}: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        if reraise:
            raise

# ============================================================================
# PERFORMANCE MONITORING UTILITIES
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Container cho performance metrics"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    gpu_memory: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PerformanceMonitor:
    """Monitor performance của functions và processes"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.logger = get_logger('performance')
    
    @contextmanager
    def monitor(self, operation_name: str = "operation"):
        """Context manager để monitor performance"""
        # Get initial measurements
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        
        # GPU memory (if available)
        start_gpu_memory = None
        try:
            import torch
            if torch.cuda.is_available():
                start_gpu_memory = torch.cuda.memory_allocated(0) / 1024 / 1024  # MB
        except ImportError:
            pass
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            end_gpu_memory = None
            if start_gpu_memory is not None:
                try:
                    import torch
                    end_gpu_memory = torch.cuda.memory_allocated(0) / 1024 / 1024  # MB
                except ImportError:
                    pass
            
            # Create metrics
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=(start_cpu + end_cpu) / 2,
                gpu_memory=end_gpu_memory - start_gpu_memory if end_gpu_memory else None
            )
            
            self.metrics_history.append(metrics)
            
            # Log metrics
            self.logger.info(f"📊 {operation_name} completed:")
            self.logger.info(f"  ⏱️ Time: {metrics.execution_time:.2f}s")
            self.logger.info(f"  🧠 Memory: {metrics.memory_usage:+.1f}MB")
            self.logger.info(f"  🖥️ CPU: {metrics.cpu_usage:.1f}%")
            if metrics.gpu_memory:
                self.logger.info(f"  🎮 GPU Memory: {metrics.gpu_memory:+.1f}MB")
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> PerformanceMetrics:
        """Get average metrics từ history"""
        if not self.metrics_history:
            return PerformanceMetrics(0, 0, 0)
        
        metrics_subset = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        avg_time = sum(m.execution_time for m in metrics_subset) / len(metrics_subset)
        avg_memory = sum(m.memory_usage for m in metrics_subset) / len(metrics_subset)
        avg_cpu = sum(m.cpu_usage for m in metrics_subset) / len(metrics_subset)
        
        gpu_metrics = [m.gpu_memory for m in metrics_subset if m.gpu_memory is not None]
        avg_gpu = sum(gpu_metrics) / len(gpu_metrics) if gpu_metrics else None
        
        return PerformanceMetrics(avg_time, avg_memory, avg_cpu, avg_gpu)

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def measure_time(func):
    """Decorator để measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with performance_monitor.monitor(f"{func.__name__}"):
            return func(*args, **kwargs)
    return wrapper

# ============================================================================
# CACHING UTILITIES
# ============================================================================

def generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key từ function arguments"""
    content = f"{args}_{sorted(kwargs.items())}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate file hash cho caching"""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        get_logger('utils').error(f"Error calculating file hash: {e}")
        return ""

class DiskCache:
    """Simple disk-based cache"""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.logger = get_logger('cache')
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Any:
        """Get item từ cache"""
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Error loading from cache: {e}")
        return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set item trong cache"""
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Check cache size
            self._cleanup_if_needed()
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
            return False
    
    def _cleanup_if_needed(self):
        """Clean up cache nếu exceed max size"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            max_size_bytes = self.max_size_mb * 1024 * 1024
            
            if total_size > max_size_bytes:
                # Delete oldest files
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob("*.pkl")]
                files.sort(key=lambda x: x[1])  # Sort by modification time
                
                for file_path, _ in files:
                    file_path.unlink()
                    total_size -= file_path.stat().st_size
                    if total_size <= max_size_bytes * 0.8:  # Clean to 80% of max
                        break
                
                self.logger.info(f"🧹 Cleaned cache, new size: {total_size/1024/1024:.1f}MB")
        
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")

# Global cache instance
disk_cache = DiskCache()

def cached(cache_key_func: Optional[Callable] = None, ttl_seconds: Optional[int] = None):
    """
    Decorator để cache function results
    
    Args:
        cache_key_func: Function để generate cache key
        ttl_seconds: Time to live cho cached results
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = disk_cache.get(cache_key)
            if cached_result is not None:
                # Check TTL if specified
                if ttl_seconds:
                    cache_time = cached_result.get('timestamp', 0)
                    if time.time() - cache_time > ttl_seconds:
                        cached_result = None
                    else:
                        return cached_result['data']
                else:
                    return cached_result
            
            # Execute function và cache result
            result = func(*args, **kwargs)
            
            if ttl_seconds:
                cache_data = {
                    'data': result,
                    'timestamp': time.time()
                }
                disk_cache.set(cache_key, cache_data)
            else:
                disk_cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator

# ============================================================================
# SYSTEM RESOURCE UTILITIES
# ============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    try:
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        # GPU info
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = {
                    'gpu_count': gpu_count,
                    'gpu_names': [torch.cuda.get_device_name(i) for i in range(gpu_count)],
                    'gpu_memory_gb': [torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                                    for i in range(gpu_count)]
                }
        except ImportError:
            pass
        
        return {
            'cpu_count': cpu_count,
            'cpu_usage_percent': cpu_percent,
            'memory_total_gb': round(memory_gb, 1),
            'memory_available_gb': round(memory_available_gb, 1),
            'memory_usage_percent': memory.percent,
            'disk_total_gb': round(disk_total_gb, 1),
            'disk_free_gb': round(disk_free_gb, 1),
            'disk_usage_percent': disk.percent,
            **gpu_info
        }
        
    except Exception as e:
        get_logger('utils').error(f"Error getting system info: {e}")
        return {}

def check_resource_availability(
    min_memory_gb: float = 4.0,
    min_disk_gb: float = 10.0,
    require_gpu: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check system resource availability
    
    Args:
        min_memory_gb: Minimum required memory (GB)
        min_disk_gb: Minimum required disk space (GB)
        require_gpu: Whether GPU is required
        
    Returns:
        (is_available, list_of_issues)
    """
    issues = []
    
    try:
        system_info = get_system_info()
        
        # Check memory
        if system_info.get('memory_available_gb', 0) < min_memory_gb:
            issues.append(f"Insufficient memory: {system_info.get('memory_available_gb', 0):.1f}GB < {min_memory_gb}GB required")
        
        # Check disk space
        if system_info.get('disk_free_gb', 0) < min_disk_gb:
            issues.append(f"Insufficient disk space: {system_info.get('disk_free_gb', 0):.1f}GB < {min_disk_gb}GB required")
        
        # Check GPU
        if require_gpu and system_info.get('gpu_count', 0) == 0:
            issues.append("GPU required but not available")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Error checking resources: {e}")
        return False, issues

# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text cho processing"""
    import re
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters (keep alphanumeric, space, basic punctuation)
    text = re.sub(r'[^\w\s\.,!?\-]', '', text)
    
    return text

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords từ text using simple TF-IDF approach"""
    import re
    from collections import Counter
    
    if not text:
        return []
    
    # Simple stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stopwords và count
    filtered_words = [w for w in words if w not in stopwords]
    word_counts = Counter(filtered_words)
    
    # Return top keywords
    return [word for word, count in word_counts.most_common(max_keywords)]

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump object to JSON string"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default

# ============================================================================
# THREADING UTILITIES
# ============================================================================

class ThreadSafeDict:
    """Thread-safe dictionary implementation"""
    
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
    
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key, value):
        with self._lock:
            self._dict[key] = value
    
    def delete(self, key):
        with self._lock:
            self._dict.pop(key, None)
    
    def keys(self):
        with self._lock:
            return list(self._dict.keys())
    
    def items(self):
        with self._lock:
            return list(self._dict.items())
    
    def __len__(self):
        with self._lock:
            return len(self._dict)

class WorkerPool:
    """Simple worker pool cho parallel processing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() + 4)
        self.logger = get_logger('worker_pool')
    
    def map(self, func: Callable, items: List[Any], show_progress: bool = True) -> List[Any]:
        """Map function over items using worker pool"""
        if not items:
            return []
        
        try:
            with multiprocessing.Pool(self.max_workers) as pool:
                if show_progress:
                    from tqdm import tqdm
                    results = list(tqdm(pool.imap(func, items), total=len(items)))
                else:
                    results = pool.map(func, items)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in worker pool: {e}")
            # Fallback to sequential processing
            if show_progress:
                from tqdm import tqdm
                items = tqdm(items, desc="Sequential fallback")
            
            return [func(item) for item in items]

# ============================================================================
# TESTING UTILITIES
# ============================================================================

def run_utils_tests():
    """Run comprehensive tests cho utility functions"""
    logger = get_logger('tests')
    logger.info("🧪 Running utility function tests...")
    
    # Test timestamp conversion
    assert timestamp_to_seconds("01:30:45") == 5445.0
    assert timestamp_to_seconds("30:45") == 1845.0
    assert seconds_to_timestamp(5445.0).startswith("01:30:45")
    logger.info("✅ Timestamp conversion tests passed")
    
    # Test similarity calculations
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    vec3 = np.array([1, 0, 0])
    
    assert cosine_similarity(vec1, vec3) == 1.0
    assert cosine_similarity(vec1, vec2) == 0.0
    logger.info("✅ Similarity calculation tests passed")
    
    # Test caching
    @cached()
    def test_function(x):
        return x * 2
    
    result1 = test_function(5)
    result2 = test_function(5)  # Should be cached
    assert result1 == result2 == 10
    logger.info("✅ Caching tests passed")
    
    # Test performance monitoring
    with performance_monitor.monitor("test_operation"):
        time.sleep(0.1)
    
    assert len(performance_monitor.metrics_history) > 0
    logger.info("✅ Performance monitoring tests passed")
    
    # Test error handling
    @retry_on_failure(max_retries=2, delay=0.1)
    def failing_function():
        raise ValueError("Test error")
    
    try:
        failing_function()
        assert False, "Should have raised exception"
    except ValueError:
        pass  # Expected
    
    logger.info("✅ Error handling tests passed")
    
    logger.info("🎉 All utility function tests passed!")

def print_system_summary():
    """Print comprehensive system summary"""
    logger = get_logger('system')
    
    print("\n" + "="*60)
    print("🖥️ System Information Summary")
    print("="*60)
    
    system_info = get_system_info()
    
    # CPU info
    print(f"\n🧠 CPU:")
    print(f"  Cores: {system_info.get('cpu_count', 'Unknown')}")
    print(f"  Usage: {system_info.get('cpu_usage_percent', 0):.1f}%")
    
    # Memory info
    print(f"\n💾 Memory:")
    print(f"  Total: {system_info.get('memory_total_gb', 0):.1f} GB")
    print(f"  Available: {system_info.get('memory_available_gb', 0):.1f} GB")
    print(f"  Usage: {system_info.get('memory_usage_percent', 0):.1f}%")
    
    # Disk info
    print(f"\n💿 Disk:")
    print(f"  Total: {system_info.get('disk_total_gb', 0):.1f} GB")
    print(f"  Free: {system_info.get('disk_free_gb', 0):.1f} GB")
    print(f"  Usage: {system_info.get('disk_usage_percent', 0):.1f}%")
    
    # GPU info
    if system_info.get('gpu_count', 0) > 0:
        print(f"\n🎮 GPU:")
        print(f"  Count: {system_info.get('gpu_count', 0)}")
        for i, (name, memory) in enumerate(zip(
            system_info.get('gpu_names', []), 
            system_info.get('gpu_memory_gb', [])
        )):
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    else:
        print(f"\n🎮 GPU: Not available")
    
    # Resource check
    print(f"\n🔍 Resource Check:")
    is_available, issues = check_resource_availability(
        min_memory_gb=4.0,
        min_disk_gb=10.0,
        require_gpu=False
    )
    
    if is_available:
        print("  ✅ System meets minimum requirements")
    else:
        print("  ❌ System issues detected:")
        for issue in issues:
            print(f"    - {issue}")
    
    print("="*60 + "\n")

# ============================================================================
# CONFIGURATION VALIDATION HELPERS
# ============================================================================

def validate_paths(paths: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Validate paths trong configuration
    
    Args:
        paths: Dictionary of path names và values
        
    Returns:
        (all_valid, list_of_issues)
    """
    issues = []
    
    for name, path in paths.items():
        try:
            path_obj = Path(path)
            
            # Check if parent directory exists hoặc có thể create
            if not path_obj.parent.exists():
                try:
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    issues.append(f"Cannot create directory for {name}: {path_obj.parent}")
            
            # Check write permissions
            if path_obj.exists() and not os.access(path, os.W_OK):
                issues.append(f"No write permission for {name}: {path}")
                
        except Exception as e:
            issues.append(f"Invalid path for {name}: {path} ({e})")
    
    return len(issues) == 0, issues

def validate_model_files(model_configs: Dict) -> Tuple[bool, List[str]]:
    """
    Validate model files existence và accessibility
    
    Args:
        model_configs: Dictionary of model configurations
        
    Returns:
        (all_valid, list_of_issues)
    """
    issues = []
    
    for model_name, config in model_configs.items():
        if not config.enabled:
            continue
            
        model_path = config.model_path
        
        # Skip API-based models
        if model_path.startswith(('gpt-', 'text-embedding-')):
            continue
        
        # Check local model files
        if os.path.exists(model_path):
            if not os.access(model_path, os.R_OK):
                issues.append(f"No read permission for {model_name}: {model_path}")
        else:
            # For HuggingFace models, just log warning
            get_logger('validation').warning(f"Model {model_name} will be downloaded: {model_path}")
    
    return len(issues) == 0, issues

# ============================================================================
# CLEANUP UTILITIES
# ============================================================================

def cleanup_temp_files(temp_dir: str = "./temp", max_age_hours: int = 24):
    """
    Clean up temporary files older than specified age
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age of files to keep (hours)
    """
    logger = get_logger('cleanup')
    
    try:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        cleaned_size = 0
        
        for file_path in temp_path.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_size = file_path.stat().st_size
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        cleaned_size += file_size
                    except PermissionError:
                        logger.warning(f"Could not delete {file_path}: Permission denied")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned {cleaned_count} files ({cleaned_size/1024/1024:.1f} MB)")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def cleanup_cache(cache_dir: str = "./cache", keep_recent_hours: int = 168):  # 1 week
    """
    Clean up cache files older than specified time
    
    Args:
        cache_dir: Cache directory path
        keep_recent_hours: Hours to keep recent cache files
    """
    logger = get_logger('cleanup')
    
    try:
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return
        
        current_time = time.time()
        max_age_seconds = keep_recent_hours * 3600
        cleaned_count = 0
        
        for cache_file in cache_path.glob('*.pkl'):
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    cache_file.unlink()
                    cleaned_count += 1
                except PermissionError:
                    logger.warning(f"Could not delete cache file {cache_file}")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned {cleaned_count} cache files")
            
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")

# ============================================================================
# EXAMPLE USAGE VÀ TESTING
# ============================================================================

# Example usage
if __name__ == "__main__":
    print("🧪 Testing VideoRAG Utility Functions")
    print("="*50)
    
    # Setup logging
    logger = setup_logging("INFO", "utils_test.log", enable_colors=True)
    logger.info("Starting utility function tests...")
    
    # Print system summary
    print_system_summary()
    
    # Run comprehensive tests
    try:
        run_utils_tests()
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Test performance monitoring
    print("\n📊 Performance Monitoring Demo:")
    with performance_monitor.monitor("demo_operation"):
        # Simulate some work
        time.sleep(0.5)
        dummy_data = [i**2 for i in range(1000)]
        result = sum(dummy_data)
        logger.info(f"Computed result: {result}")
    
    # Show average metrics
    avg_metrics = performance_monitor.get_average_metrics()
    print(f"Average execution time: {avg_metrics.execution_time:.3f}s")
    
    # Test caching
    print("\n💾 Cache Demo:")
    
    @cached(ttl_seconds=60)
    def expensive_computation(n):
        logger.info(f"Computing expensive operation for n={n}")
        time.sleep(0.1)  # Simulate expensive operation
        return n ** 3
    
    # First call - should compute
    start_time = time.time()
    result1 = expensive_computation(10)
    time1 = time.time() - start_time
    
    # Second call - should use cache
    start_time = time.time()
    result2 = expensive_computation(10)
    time2 = time.time() - start_time
    
    logger.info(f"First call: {time1:.3f}s, Second call: {time2:.3f}s")
    logger.info(f"Speedup: {time1/time2:.1f}x")
    
    # Test batch processing
    print("\n⚡ Batch Processing Demo:")
    
    def square_numbers(batch):
        return [x**2 for x in batch]
    
    numbers = list(range(100))
    
    with performance_monitor.monitor("batch_processing"):
        squared = batch_processor(
            items=numbers,
            batch_size=10,
            process_func=square_numbers,
            max_workers=2,
            show_progress=True
        )
    
    logger.info(f"Processed {len(squared)} numbers")
    
    # Test error handling
    print("\n🛡️ Error Handling Demo:")
    
    @retry_on_failure(max_retries=3, delay=0.1)
    def unreliable_function(success_rate=0.3):
        import random
        if random.random() < success_rate:
            return "Success!"
        else:
            raise RuntimeError("Random failure")
    
    try:
        with error_context("demo_error_handling"):
            result = safe_execute(unreliable_function, success_rate=0.8, default="Failed")
            logger.info(f"Function result: {result}")
    except Exception as e:
        logger.error(f"Function failed: {e}")
    
    # Test video utilities
    print("\n🎬 Video Utilities Demo:")
    
    # Test timestamp conversion
    timestamps = ["01:30:45", "30:45", "45", 3645.0]
    for ts in timestamps:
        seconds = timestamp_to_seconds(ts)
        converted_back = seconds_to_timestamp(seconds)
        logger.info(f"{ts} → {seconds}s → {converted_back}")
    
    # Test similarity calculations
    print("\n📊 Similarity Calculation Demo:")
    
    # Create some test vectors
    vec_a = np.random.rand(100)
    vec_b = np.random.rand(100)
    vec_c = vec_a + 0.1 * np.random.rand(100)  # Similar to vec_a
    
    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)
    
    logger.info(f"Similarity A-B (random): {sim_ab:.3f}")
    logger.info(f"Similarity A-C (similar): {sim_ac:.3f}")
    
    # Test text processing
    print("\n📝 Text Processing Demo:")
    
    sample_text = """
    This is a sample text for testing keyword extraction.
    The text contains various words and should extract meaningful keywords.
    Testing, extraction, keywords, meaningful, sample are important words.
    """
    
    normalized = normalize_text(sample_text)
    keywords = extract_keywords(normalized, max_keywords=5)
    
    logger.info(f"Original text length: {len(sample_text)}")
    logger.info(f"Normalized text length: {len(normalized)}")
    logger.info(f"Extracted keywords: {keywords}")
    
    # Cleanup demo
    print("\n🧹 Cleanup Demo:")
    cleanup_temp_files("./temp", max_age_hours=0.1)  # Clean files older than 6 minutes
    cleanup_cache("./cache", keep_recent_hours=1)     # Clean cache older than 1 hour
    
    print("\n🎉 VideoRAG Utility Functions Demo Completed!")
    print("="*50)