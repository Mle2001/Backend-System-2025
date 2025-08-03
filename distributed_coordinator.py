"""
distributed_coordinator.py - Distributed Processing Coordinator cho VideoRAG System
Quản lý distributed processing của videos trên nhiều máy tính/worker nodes
"""

import os
import time
import json
import socket
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import pickle
import hashlib

import requests
from flask import Flask, request, jsonify
import psutil

from config import Config, DistributedConfig, get_config
from utils import (
    get_logger, safe_execute, measure_time, performance_monitor,
    ThreadSafeDict, retry_on_failure, error_context
)
from video_processor import VideoProcessor, ProcessedVideo
from knowledge_builder import KnowledgeBuilder, KnowledgeGraph
from model_manager import ModelManager


# ==================== ENUMS AND DATA STRUCTURES ====================

class TaskStatus(Enum):
    """Status của processing tasks"""
    PENDING = "pending"
    ASSIGNED = "assigned"  
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkerStatus(Enum):
    """Status của worker nodes"""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"

@dataclass
class ProcessingTask:
    """Represent một video processing task"""
    task_id: str
    video_path: str
    video_id: str
    assigned_worker: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_time: float = field(default_factory=time.time)
    assigned_time: Optional[float] = None
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    result: Optional[ProcessedVideo] = None
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1  # Higher number = higher priority

@dataclass 
class WorkerNode:
    """Represent một worker node"""
    worker_id: str
    host: str
    port: int
    status: WorkerStatus = WorkerStatus.OFFLINE
    current_task: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_processing_time: float = 0.0
    system_info: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClusterStatus:
    """Status của toàn bộ cluster"""
    total_workers: int = 0
    available_workers: int = 0
    busy_workers: int = 0
    offline_workers: int = 0
    total_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cluster_utilization: float = 0.0
    last_updated: float = field(default_factory=time.time)


# ==================== MAIN DISTRIBUTED COORDINATOR CLASS ====================

class DistributedCoordinator:
    """
    Distributed Processing Coordinator cho VideoRAG System
    Quản lý distributed processing của videos trên multiple worker nodes
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DistributedCoordinator
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.dist_config = self.config.distributed
        self.logger = get_logger('distributed_coordinator')
        
        # Core components
        self._init_storage()
        self._init_communication()
        self._init_monitoring()
        self._init_task_management()
        
        # Control flags
        self.is_running = False
        self.shutdown_requested = False
        
        self.logger.info(f"🌐 DistributedCoordinator initialized")
        self.logger.info(f"📡 Coordinator: {self.dist_config.coordinator_host}:{self.dist_config.coordinator_port}")
        
    def _init_storage(self):
        """Initialize storage systems"""
        # Worker và task storage
        self.workers: ThreadSafeDict = ThreadSafeDict()
        self.tasks: ThreadSafeDict = ThreadSafeDict()
        self.completed_results: ThreadSafeDict = ThreadSafeDict()
        
        # Task queues
        self.pending_tasks = Queue()
        self.result_queue = Queue()
        
        # Synchronization
        self._coordinator_lock = threading.RLock()
        self._task_lock = threading.RLock()
        
    def _init_communication(self):
        """Initialize communication components"""
        # Flask app cho coordinator API
        self.coordinator_app = Flask(__name__)
        self.coordinator_app.logger.disabled = True  # Disable Flask logging
        
        # Setup routes
        self._setup_coordinator_routes()
        
        # HTTP session cho worker communication
        self.session = requests.Session()
        self.session.timeout = 30
        
    def _init_monitoring(self):
        """Initialize monitoring components"""
        self.cluster_status = ClusterStatus()
        self.monitoring_threads = []
        
    def _init_task_management(self):
        """Initialize task management"""
        self.task_assignment_thread = None
        self.result_collection_thread = None
        self.health_monitoring_thread = None
        
    def _setup_coordinator_routes(self):
        """Setup Flask routes cho coordinator API"""
        
        @self.coordinator_app.route('/api/worker/register', methods=['POST'])
        def register_worker():
            """Register một worker node"""
            try:
                worker_data = request.json
                worker_id = worker_data.get('worker_id')
                host = worker_data.get('host')
                port = worker_data.get('port')
                capabilities = worker_data.get('capabilities', {})
                
                if not all([worker_id, host, port]):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Create worker node
                worker = WorkerNode(
                    worker_id=worker_id,
                    host=host,
                    port=port,
                    status=WorkerStatus.AVAILABLE,
                    capabilities=capabilities
                )
                
                self.workers.set(worker_id, worker)
                self.logger.info(f"✅ Worker registered: {worker_id} at {host}:{port}")
                
                return jsonify({'status': 'registered', 'worker_id': worker_id})
                
            except Exception as e:
                self.logger.error(f"❌ Error registering worker: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.coordinator_app.route('/api/worker/heartbeat', methods=['POST'])
        def worker_heartbeat():
            """Handle worker heartbeat"""
            try:
                data = request.json
                worker_id = data.get('worker_id')
                status = data.get('status')
                current_task = data.get('current_task')
                system_info = data.get('system_info', {})
                
                if not worker_id:
                    return jsonify({'error': 'Missing worker_id'}), 400
                
                worker = self.workers.get(worker_id)
                if not worker:
                    return jsonify({'error': 'Worker not registered'}), 404
                
                # Update worker status
                worker.last_heartbeat = time.time()
                worker.last_seen = time.time()
                if status:
                    worker.status = WorkerStatus(status)
                if current_task is not None:
                    worker.current_task = current_task
                worker.system_info = system_info
                
                self.workers.set(worker_id, worker)
                
                return jsonify({'status': 'ok'})
                
            except Exception as e:
                self.logger.error(f"❌ Error handling heartbeat: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.coordinator_app.route('/api/task/result', methods=['POST'])
        def submit_task_result():
            """Handle task result submission từ worker"""
            try:
                data = request.json
                task_id = data.get('task_id')
                worker_id = data.get('worker_id')
                success = data.get('success', False)
                result_data = data.get('result')
                error_message = data.get('error_message', '')
                
                if not all([task_id, worker_id]):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Get task
                task = self.tasks.get(task_id)
                if not task:
                    return jsonify({'error': 'Task not found'}), 404
                
                # Update task
                task.completed_time = time.time()
                
                if success and result_data:
                    # Deserialize result
                    try:
                        # Assuming result_data is serialized ProcessedVideo
                        task.result = self._deserialize_processed_video(result_data)
                        task.status = TaskStatus.COMPLETED
                        self.completed_results.set(task_id, task.result)
                        
                        self.logger.info(f"✅ Task completed: {task_id} by {worker_id}")
                        
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error_message = f"Result deserialization error: {e}"
                        self.logger.error(f"❌ Error deserializing result for {task_id}: {e}")
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = error_message
                    self.logger.error(f"❌ Task failed: {task_id} - {error_message}")
                
                # Update worker
                worker = self.workers.get(worker_id)
                if worker:
                    worker.current_task = None
                    worker.status = WorkerStatus.AVAILABLE
                    if success:
                        worker.total_tasks_completed += 1
                    else:
                        worker.total_tasks_failed += 1
                    
                    # Update average processing time
                    if task.started_time:
                        processing_time = task.completed_time - task.started_time
                        if worker.total_tasks_completed > 0:
                            worker.average_processing_time = (
                                (worker.average_processing_time * (worker.total_tasks_completed - 1) + processing_time) /
                                worker.total_tasks_completed
                            )
                        else:
                            worker.average_processing_time = processing_time
                    
                    self.workers.set(worker_id, worker)
                
                self.tasks.set(task_id, task)
                
                # Add to result queue
                self.result_queue.put(task)
                
                return jsonify({'status': 'received'})
                
            except Exception as e:
                self.logger.error(f"❌ Error handling task result: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.coordinator_app.route('/api/cluster/status', methods=['GET'])
        def get_cluster_status():
            """Get cluster status"""
            try:
                self._update_cluster_status()
                return jsonify(asdict(self.cluster_status))
            except Exception as e:
                return jsonify({'error': str(e)}), 500

# ==================== COORDINATOR CONTROL METHODS ====================
    
    def start_coordinator(self) -> bool:
        """
        Start distributed coordinator
        
        Returns:
            True if started successfully
        """
        if self.is_running:
            self.logger.warning("⚠️ Coordinator already running")
            return True
        
        try:
            self.logger.info("🚀 Starting distributed coordinator...")
            
            # Start Flask coordinator server
            self._start_coordinator_server()
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            # Start task management threads
            self._start_task_management_threads()
            
            # Discovery existing workers
            self._discover_workers()
            
            self.is_running = True
            self.shutdown_requested = False
            
            self.logger.info("✅ Distributed coordinator started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting coordinator: {e}")
            return False
    
    def stop_coordinator(self):
        """Stop distributed coordinator"""
        if not self.is_running:
            return
        
        self.logger.info("🛑 Stopping distributed coordinator...")
        
        # Set shutdown flag
        self.shutdown_requested = True
        
        # Stop all threads
        self._stop_all_threads()
        
        # Cancel pending tasks
        self._cancel_pending_tasks()
        
        # Notify workers of shutdown
        self._notify_workers_shutdown()
        
        self.is_running = False
        self.logger.info("✅ Distributed coordinator stopped")
    
    def _start_coordinator_server(self):
        """Start Flask coordinator server trong separate thread"""
        def run_server():
            try:
                self.coordinator_app.run(
                    host=self.dist_config.coordinator_host,
                    port=self.dist_config.coordinator_port,
                    debug=False,
                    use_reloader=False
                )
            except Exception as e:
                self.logger.error(f"❌ Coordinator server error: {e}")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        self.logger.info(f"📡 Coordinator server started on {self.dist_config.coordinator_host}:{self.dist_config.coordinator_port}")
    
    def _start_monitoring_threads(self):
        """Start monitoring threads"""
        # Health monitoring thread
        self.health_monitoring_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.health_monitoring_thread.start()
        
        # Cluster status update thread
        cluster_status_thread = threading.Thread(
            target=self._cluster_status_loop,
            daemon=True
        )
        cluster_status_thread.start()
        
        self.monitoring_threads = [self.health_monitoring_thread, cluster_status_thread]
        
    def _start_task_management_threads(self):
        """Start task management threads"""
        # Task assignment thread
        self.task_assignment_thread = threading.Thread(
            target=self._task_assignment_loop,
            daemon=True
        )
        self.task_assignment_thread.start()
        
        # Result collection thread
        self.result_collection_thread = threading.Thread(
            target=self._result_collection_loop,
            daemon=True
        )
        self.result_collection_thread.start()
    
    def _stop_all_threads(self):
        """Stop all background threads"""
        # Wait for threads to finish (with timeout)
        threads_to_wait = [
            self.health_monitoring_thread,
            self.task_assignment_thread,
            self.result_collection_thread
        ] + self.monitoring_threads
        
        for thread in threads_to_wait:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)


    # ==================== TASK DISTRIBUTION METHODS ====================
    
    @measure_time
    def distribute_video_tasks(
        self,
        video_paths: List[str],
        processing_options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Distribute video processing tasks to workers
        
        Args:
            video_paths: List of video file paths
            processing_options: Optional processing parameters
            
        Returns:
            List of task IDs
        """
        self.logger.info(f"📋 Distributing {len(video_paths)} video tasks")
        
        if not self.is_running:
            raise RuntimeError("Coordinator not running")
        
        task_ids = []
        
        try:
            with performance_monitor.monitor("distribute_video_tasks"):
                for video_path in video_paths:
                    # Create task
                    task_id = self._create_processing_task(video_path, processing_options)
                    task_ids.append(task_id)
                    
                    # Add to pending queue
                    self.pending_tasks.put(task_id)
                
                self.logger.info(f"✅ Created {len(task_ids)} tasks for distribution")
                
        except Exception as e:
            self.logger.error(f"❌ Error distributing tasks: {e}")
            raise
        
        return task_ids
    
    def _create_processing_task(
        self,
        video_path: str,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create processing task"""
        from utils import generate_video_id
        
        video_id = generate_video_id(video_path)
        task_id = f"task_{int(time.time())}_{hashlib.md5(video_path.encode()).hexdigest()[:8]}"
        
        task = ProcessingTask(
            task_id=task_id,
            video_path=video_path,
            video_id=video_id,
            priority=processing_options.get('priority', 1) if processing_options else 1
        )
        
        self.tasks.set(task_id, task)
        
        self.logger.debug(f"📝 Created task: {task_id} for {os.path.basename(video_path)}")
        return task_id
    
    def _task_assignment_loop(self):
        """Background thread cho task assignment"""
        self.logger.info("🔄 Task assignment loop started")
        
        while not self.shutdown_requested:
            try:
                # Get available workers
                available_workers = self._get_available_workers()
                
                if not available_workers:
                    time.sleep(5)  # Wait before checking again
                    continue
                
                # Assign tasks to workers
                assigned_count = 0
                for worker in available_workers:
                    try:
                        # Get next pending task
                        task_id = self.pending_tasks.get(timeout=1.0)
                        
                        # Assign task to worker
                        success = self._assign_task_to_worker(task_id, worker.worker_id)
                        
                        if success:
                            assigned_count += 1
                        else:
                            # Put task back in queue
                            self.pending_tasks.put(task_id)
                        
                    except Empty:
                        break  # No more pending tasks
                    except Exception as e:
                        self.logger.error(f"❌ Error assigning task: {e}")
                
                if assigned_count > 0:
                    self.logger.info(f"📤 Assigned {assigned_count} tasks")
                
                time.sleep(2)  # Brief pause before next assignment cycle
                
            except Exception as e:
                self.logger.error(f"❌ Error in task assignment loop: {e}")
                time.sleep(5)
        
        self.logger.info("🛑 Task assignment loop stopped")
    
    def _get_available_workers(self) -> List[WorkerNode]:
        """Get list of available workers"""
        available_workers = []
        
        for worker_id in self.workers.keys():
            worker = self.workers.get(worker_id)
            if worker and worker.status == WorkerStatus.AVAILABLE:
                available_workers.append(worker)
        
        # Sort by performance (average processing time, ascending)
        available_workers.sort(key=lambda w: w.average_processing_time)
        
        return available_workers
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _assign_task_to_worker(self, task_id: str, worker_id: str) -> bool:
        """Assign task to specific worker"""
        try:
            task = self.tasks.get(task_id)
            worker = self.workers.get(worker_id)
            
            if not task or not worker:
                return False
            
            # Send task to worker
            worker_url = f"http://{worker.host}:{worker.port}/api/task/assign"
            
            task_data = {
                'task_id': task_id,
                'video_path': task.video_path,
                'video_id': task.video_id,
                'processing_options': {}
            }
            
            response = self.session.post(worker_url, json=task_data, timeout=30)
            
            if response.status_code == 200:
                # Update task và worker status
                task.assigned_worker = worker_id
                task.status = TaskStatus.ASSIGNED
                task.assigned_time = time.time()
                
                worker.current_task = task_id
                worker.status = WorkerStatus.BUSY
                
                self.tasks.set(task_id, task)
                self.workers.set(worker_id, worker)
                
                self.logger.debug(f"📤 Task {task_id} assigned to {worker_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to assign task {task_id} to {worker_id}: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error assigning task {task_id} to {worker_id}: {e}")
            return False


    # ==================== RESULT COLLECTION METHODS ====================
    
    def collect_processing_results(self, timeout_seconds: Optional[int] = None) -> List[ProcessedVideo]:
        """
        Collect processing results từ workers
        
        Args:
            timeout_seconds: Timeout for waiting for results
            
        Returns:
            List of ProcessedVideo results
        """
        results = []
        start_time = time.time()
        
        self.logger.info("📥 Collecting processing results...")
        
        try:
            while True:
                # Check timeout
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    break
                
                # Check if all tasks completed
                pending_count = self.pending_tasks.qsize()
                processing_count = self._count_tasks_by_status(TaskStatus.PROCESSING)
                assigned_count = self._count_tasks_by_status(TaskStatus.ASSIGNED)
                
                if pending_count == 0 and processing_count == 0 and assigned_count == 0:
                    break
                
                # Collect available results
                try:
                    task = self.result_queue.get(timeout=5.0)
                    if task.result:
                        results.append(task.result)
                        self.logger.info(f"📥 Collected result for {task.task_id}")
                except Empty:
                    continue
                
        except Exception as e:
            self.logger.error(f"❌ Error collecting results: {e}")
        
        self.logger.info(f"✅ Collected {len(results)} results")
        return results
    
    def _result_collection_loop(self):
        """Background thread cho result collection"""
        self.logger.info("🔄 Result collection loop started")
        
        while not self.shutdown_requested:
            try:
                # Process results from queue
                try:
                    task = self.result_queue.get(timeout=5.0)
                    
                    # Process completed task
                    self._process_completed_task(task)
                    
                except Empty:
                    continue
                
            except Exception as e:
                self.logger.error(f"❌ Error in result collection loop: {e}")
                time.sleep(5)
        
        self.logger.info("🛑 Result collection loop stopped")
    
    def _process_completed_task(self, task: ProcessingTask):
        """Process completed task"""
        try:
            if task.status == TaskStatus.COMPLETED:
                self.logger.info(f"✅ Task completed: {task.task_id}")
                
                # Store result
                if task.result:
                    self.completed_results.set(task.task_id, task.result)
                
            elif task.status == TaskStatus.FAILED:
                self.logger.error(f"❌ Task failed: {task.task_id} - {task.error_message}")
                
                # Handle retry logic
                if task.retry_count < task.max_retries:
                    self._retry_failed_task(task)
                else:
                    self.logger.error(f"💀 Task exhausted retries: {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing completed task: {e}")
    
    def _retry_failed_task(self, task: ProcessingTask):
        """Retry failed task"""
        try:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.assigned_worker = None
            task.assigned_time = None
            task.started_time = None
            task.error_message = ""
            
            self.tasks.set(task.task_id, task)
            self.pending_tasks.put(task.task_id)
            
            self.logger.info(f"🔄 Retrying task: {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
            
        except Exception as e:
            self.logger.error(f"❌ Error retrying task: {e}")


    # ==================== WORKER MONITORING METHODS ====================
    
    def monitor_worker_status(self):
        """Monitor worker status và health"""
        self.logger.info("👀 Starting worker monitoring...")
        
        for worker_id in self.workers.keys():
            worker = self.workers.get(worker_id)
            if worker:
                self._check_worker_health(worker)
    
    def _health_monitoring_loop(self):
        """Background thread cho health monitoring"""
        self.logger.info("🔄 Health monitoring loop started")
        
        while not self.shutdown_requested:
            try:
                self.monitor_worker_status()
                time.sleep(self.dist_config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in health monitoring: {e}")
                time.sleep(30)
        
        self.logger.info("🛑 Health monitoring loop stopped")
    
    def _check_worker_health(self, worker: WorkerNode):
        """Check health của một worker"""
        try:
            current_time = time.time()
            time_since_heartbeat = current_time - worker.last_heartbeat
            
            # Check if worker missed heartbeat
            if time_since_heartbeat > self.dist_config.heartbeat_interval * 2:
                if worker.status != WorkerStatus.OFFLINE:
                    self.logger.warning(f"⚠️ Worker {worker.worker_id} missed heartbeat ({time_since_heartbeat:.1f}s)")
                    worker.status = WorkerStatus.OFFLINE
                    self.workers.set(worker.worker_id, worker)
                    
                    # Handle offline worker
                    self._handle_offline_worker(worker)
            
            # Ping worker directly
            elif worker.status != WorkerStatus.OFFLINE:
                try:
                    ping_url = f"http://{worker.host}:{worker.port}/api/worker/ping"
                    response = self.session.get(ping_url, timeout=10)
                    
                    if response.status_code == 200:
                        worker.last_seen = current_time
                        if worker.status == WorkerStatus.OFFLINE:
                            worker.status = WorkerStatus.AVAILABLE
                            self.logger.info(f"✅ Worker {worker.worker_id} back online")
                    else:
                        raise Exception(f"HTTP {response.status_code}")
                        
                except Exception as e:
                    if worker.status != WorkerStatus.OFFLINE:
                        self.logger.warning(f"⚠️ Cannot ping worker {worker.worker_id}: {e}")
                        worker.status = WorkerStatus.ERROR
                    
                self.workers.set(worker.worker_id, worker)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking worker health: {e}")
    
    def _handle_offline_worker(self, worker: WorkerNode):
        """Handle offline worker"""
        try:
            # Reassign current task if any
            if worker.current_task:
                task = self.tasks.get(worker.current_task)
                if task and task.status in [TaskStatus.ASSIGNED, TaskStatus.PROCESSING]:
                    self.logger.info(f"🔄 Reassigning task {task.task_id} from offline worker {worker.worker_id}")
                    
                    # Reset task status
                    task.assigned_worker = None
                    task.status = TaskStatus.PENDING
                    task.assigned_time = None
                    task.started_time = None
                    
                    self.tasks.set(task.task_id, task)
                    self.pending_tasks.put(task.task_id)
                
                # Clear worker's current task
                worker.current_task = None
                self.workers.set(worker.worker_id, worker)
                
        except Exception as e:
            self.logger.error(f"❌ Error handling offline worker: {e}")


    # ==================== FAILURE HANDLING METHODS ====================
    
    def handle_node_failures(self):
        """Handle node failures và recovery"""
        self.logger.info("🛡️ Checking for node failures...")
        
        try:
            failed_workers = []
            current_time = time.time()
            
            # Identify failed workers
            for worker_id in self.workers.keys():
                worker = self.workers.get(worker_id)
                if not worker:
                    continue
                
                # Consider worker failed if offline for too long
                time_offline = current_time - worker.last_seen
                if (worker.status == WorkerStatus.OFFLINE and 
                    time_offline > self.dist_config.heartbeat_interval * 5):
                    failed_workers.append(worker)
            
            # Handle failed workers
            for worker in failed_workers:
                self._handle_failed_worker(worker)
            
            if failed_workers:
                self.logger.warning(f"⚠️ Handled {len(failed_workers)} failed workers")
                
        except Exception as e:
            self.logger.error(f"❌ Error handling node failures: {e}")
    
    def _handle_failed_worker(self, worker: WorkerNode):
        """Handle một failed worker"""
        try:
            self.logger.warning(f"💀 Handling failed worker: {worker.worker_id}")
            
            # Reassign any assigned/processing tasks
            for task_id in self.tasks.keys():
                task = self.tasks.get(task_id)
                if task and task.assigned_worker == worker.worker_id:
                    if task.status in [TaskStatus.ASSIGNED, TaskStatus.PROCESSING]:
                        self.logger.info(f"🔄 Reassigning task {task_id} from failed worker")
                        
                        # Reset task
                        task.assigned_worker = None
                        task.status = TaskStatus.PENDING
                        task.retry_count += 1
                        
                        if task.retry_count <= task.max_retries:
                            self.tasks.set(task_id, task)
                            self.pending_tasks.put(task_id)
                        else:
                            task.status = TaskStatus.FAILED
                            task.error_message = f"Worker {worker.worker_id} failed"
                            self.tasks.set(task_id, task)
            
            # Mark worker as permanently failed (có thể remove sau)
            worker.status = WorkerStatus.OFFLINE
            worker.current_task = None
            self.workers.set(worker.worker_id, worker)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling failed worker: {e}")


    # ==================== DATABASE SYNCHRONIZATION METHODS ====================
    
    @measure_time
    def synchronize_databases(self):
        """Synchronize databases across distributed nodes"""
        self.logger.info("🔄 Starting database synchronization...")
        
        try:
            with performance_monitor.monitor("database_synchronization"):
                # Collect all processed results
                all_results = []
                
                # Get results from completed tasks
                for task_id in self.tasks.keys():
                    task = self.tasks.get(task_id)
                    if task and task.status == TaskStatus.COMPLETED and task.result:
                        all_results.append(task.result)
                
                # Get results from completed_results storage
                for result_id in self.completed_results.keys():
                    result = self.completed_results.get(result_id)
                    if result:
                        all_results.append(result)
                
                self.logger.info(f"📊 Synchronizing {len(all_results)} processed videos")
                
                # Merge knowledge graphs if available
                if all_results:
                    merged_graph = self.merge_distributed_knowledge_graphs(all_results)
                    
                    # Store merged graph
                    if merged_graph:
                        self._store_merged_graph(merged_graph)
                        self.logger.info("✅ Knowledge graph merged and stored")
                
                # Synchronize metadata với workers
                self._sync_metadata_with_workers()
                
        except Exception as e:
            self.logger.error(f"❌ Error synchronizing databases: {e}")
    
    def merge_distributed_knowledge_graphs(
        self, 
        processed_videos: List[ProcessedVideo]
    ) -> Optional[KnowledgeGraph]:
        """
        Merge knowledge graphs từ distributed processing
        
        Args:
            processed_videos: List of ProcessedVideo objects
            
        Returns:
            Merged KnowledgeGraph hoặc None
        """
        try:
            self.logger.info(f"🔗 Merging knowledge graphs từ {len(processed_videos)} videos")
            
            # Initialize knowledge builder
            knowledge_builder = KnowledgeBuilder(self.config)
            
            # Build individual knowledge graphs
            individual_graphs = []
            
            for processed_video in processed_videos:
                if processed_video.success and processed_video.chunks:
                    # Build knowledge graph cho video này
                    graph = knowledge_builder.build_knowledge_graph(
                        [processed_video],
                        graph_id=f"graph_{processed_video.video_id}",
                        enable_cross_video_linking=False
                    )
                    
                    if graph and graph.entities:
                        individual_graphs.append(graph)
            
            if not individual_graphs:
                self.logger.warning("⚠️ No valid knowledge graphs to merge")
                return None
            
            # Merge all graphs
            merged_graph = knowledge_builder.merge_knowledge_graphs(individual_graphs)
            
            self.logger.info(f"✅ Merged knowledge graph: {len(merged_graph.entities)} entities, {len(merged_graph.relations)} relations")
            
            return merged_graph
            
        except Exception as e:
            self.logger.error(f"❌ Error merging knowledge graphs: {e}")
            return None
    
    def _store_merged_graph(self, knowledge_graph: KnowledgeGraph):
        """Store merged knowledge graph"""
        try:
            # Save to local storage
            graph_file = Path(self.config.paths.get('data_dir', './data')) / 'merged_knowledge_graph.pkl'
            
            knowledge_builder = KnowledgeBuilder(self.config)
            success = knowledge_builder.save_knowledge_graph(knowledge_graph, str(graph_file))
            
            if success:
                self.logger.info(f"💾 Merged graph saved to {graph_file}")
            else:
                self.logger.error("❌ Failed to save merged graph")
                
        except Exception as e:
            self.logger.error(f"❌ Error storing merged graph: {e}")
    
    def _sync_metadata_with_workers(self):
        """Sync metadata với active workers"""
        try:
            # Collect metadata
            metadata = {
                'cluster_status': asdict(self.cluster_status),
                'total_processed_videos': len(self.completed_results.keys()),
                'sync_timestamp': time.time()
            }
            
            # Send to all active workers
            active_workers = [
                worker for worker in self.workers.keys() 
                if self.workers.get(worker) and self.workers.get(worker).status != WorkerStatus.OFFLINE
            ]
            
            for worker_id in active_workers:
                worker = self.workers.get(worker_id)
                if worker:
                    try:
                        sync_url = f"http://{worker.host}:{worker.port}/api/sync/metadata"
                        response = self.session.post(sync_url, json=metadata, timeout=30)
                        
                        if response.status_code == 200:
                            self.logger.debug(f"✅ Synced metadata với {worker_id}")
                        else:
                            self.logger.warning(f"⚠️ Failed to sync metadata với {worker_id}")
                            
                    except Exception as e:
                        self.logger.debug(f"❌ Error syncing với worker {worker_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing metadata with workers: {e}")


    # ==================== UTILITY AND HELPER METHODS ====================
    
    def _discover_workers(self):
        """Discover existing workers trong network"""
        try:
            self.logger.info("🔍 Discovering existing workers...")
            
            # Check configured worker nodes
            if self.dist_config.worker_nodes:
                for worker_node in self.dist_config.worker_nodes:
                    self._try_connect_worker(worker_node)
            
            # Broadcast discovery message (simple implementation)
            self._broadcast_discovery()
            
        except Exception as e:
            self.logger.error(f"❌ Error discovering workers: {e}")
    
    def _try_connect_worker(self, worker_address: str):
        """Try to connect to a worker node"""
        try:
            # Parse address
            if ':' in worker_address:
                host, port = worker_address.split(':')
                port = int(port)
            else:
                host = worker_address
                port = 8889  # Default worker port
            
            # Try to ping worker
            ping_url = f"http://{host}:{port}/api/worker/ping"
            response = self.session.get(ping_url, timeout=10)
            
            if response.status_code == 200:
                worker_info = response.json()
                worker_id = worker_info.get('worker_id')
                
                if worker_id and worker_id not in self.workers.keys():
                    # Register discovered worker
                    worker = WorkerNode(
                        worker_id=worker_id,
                        host=host,
                        port=port,
                        status=WorkerStatus.AVAILABLE,
                        capabilities=worker_info.get('capabilities', {})
                    )
                    
                    self.workers.set(worker_id, worker)
                    self.logger.info(f"🔍 Discovered worker: {worker_id} at {host}:{port}")
                    
        except Exception as e:
            self.logger.debug(f"🔍 Cannot connect to worker {worker_address}: {e}")
    
    def _broadcast_discovery(self):
        """Broadcast discovery message to local network"""
        try:
            # Simple UDP broadcast implementation
            import socket
            
            discovery_message = {
                'type': 'coordinator_discovery',
                'coordinator_host': self.dist_config.coordinator_host,
                'coordinator_port': self.dist_config.coordinator_port,
                'timestamp': time.time()
            }
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            message_bytes = json.dumps(discovery_message).encode('utf-8')
            sock.sendto(message_bytes, ('<broadcast>', 8890))  # Broadcast port
            
            sock.close()
            
            self.logger.debug("📡 Broadcasted discovery message")
            
        except Exception as e:
            self.logger.debug(f"📡 Error broadcasting discovery: {e}")
    
    def _notify_workers_shutdown(self):
        """Notify workers về coordinator shutdown"""
        try:
            shutdown_message = {
                'type': 'coordinator_shutdown',
                'timestamp': time.time()
            }
            
            active_workers = [
                worker for worker in self.workers.keys() 
                if self.workers.get(worker) and self.workers.get(worker).status != WorkerStatus.OFFLINE
            ]
            
            for worker_id in active_workers:
                worker = self.workers.get(worker_id)
                if worker:
                    try:
                        shutdown_url = f"http://{worker.host}:{worker.port}/api/coordinator/shutdown"
                        self.session.post(shutdown_url, json=shutdown_message, timeout=10)
                        
                    except Exception as e:
                        self.logger.debug(f"Error notifying worker {worker_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error notifying workers of shutdown: {e}")
    
    def _cancel_pending_tasks(self):
        """Cancel all pending tasks"""
        try:
            cancelled_count = 0
            
            # Cancel pending tasks in queue
            while not self.pending_tasks.empty():
                try:
                    task_id = self.pending_tasks.get_nowait()
                    task = self.tasks.get(task_id)
                    if task:
                        task.status = TaskStatus.CANCELLED
                        self.tasks.set(task_id, task)
                        cancelled_count += 1
                except Empty:
                    break
            
            # Cancel assigned but not started tasks
            for task_id in self.tasks.keys():
                task = self.tasks.get(task_id)
                if task and task.status == TaskStatus.ASSIGNED:
                    task.status = TaskStatus.CANCELLED
                    self.tasks.set(task_id, task)
                    cancelled_count += 1
            
            if cancelled_count > 0:
                self.logger.info(f"🚫 Cancelled {cancelled_count} pending tasks")
                
        except Exception as e:
            self.logger.error(f"❌ Error cancelling pending tasks: {e}")


    # ==================== CLUSTER STATUS METHODS ====================
    
    def _cluster_status_loop(self):
        """Background thread cho cluster status updates"""
        self.logger.info("🔄 Cluster status loop started")
        
        while not self.shutdown_requested:
            try:
                self._update_cluster_status()
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in cluster status loop: {e}")
                time.sleep(60)
        
        self.logger.info("🛑 Cluster status loop stopped")
    
    def _update_cluster_status(self):
        """Update cluster status"""
        try:
            # Count workers by status
            worker_counts = {status: 0 for status in WorkerStatus}
            
            for worker_id in self.workers.keys():
                worker = self.workers.get(worker_id)
                if worker:
                    worker_counts[worker.status] += 1
            
            # Count tasks by status
            task_counts = {status: 0 for status in TaskStatus}
            
            for task_id in self.tasks.keys():
                task = self.tasks.get(task_id)
                if task:
                    task_counts[task.status] += 1
            
            # Calculate utilization
            total_workers = sum(worker_counts.values())
            busy_workers = worker_counts[WorkerStatus.BUSY]
            utilization = (busy_workers / total_workers * 100) if total_workers > 0 else 0
            
            # Update cluster status
            self.cluster_status = ClusterStatus(
                total_workers=total_workers,
                available_workers=worker_counts[WorkerStatus.AVAILABLE],
                busy_workers=busy_workers,
                offline_workers=worker_counts[WorkerStatus.OFFLINE],
                total_tasks=sum(task_counts.values()),
                pending_tasks=task_counts[TaskStatus.PENDING] + self.pending_tasks.qsize(),
                processing_tasks=task_counts[TaskStatus.PROCESSING],
                completed_tasks=task_counts[TaskStatus.COMPLETED],
                failed_tasks=task_counts[TaskStatus.FAILED],
                cluster_utilization=utilization,
                last_updated=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error updating cluster status: {e}")
    
    def _count_tasks_by_status(self, status: TaskStatus) -> int:
        """Count tasks by status"""
        count = 0
        for task_id in self.tasks.keys():
            task = self.tasks.get(task_id)
            if task and task.status == status:
                count += 1
        return count
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get comprehensive cluster summary"""
        self._update_cluster_status()
        
        # Worker details
        worker_details = []
        for worker_id in self.workers.keys():
            worker = self.workers.get(worker_id)
            if worker:
                worker_details.append({
                    'worker_id': worker.worker_id,
                    'host': worker.host,
                    'port': worker.port,
                    'status': worker.status.value,
                    'current_task': worker.current_task,
                    'tasks_completed': worker.total_tasks_completed,
                    'tasks_failed': worker.total_tasks_failed,
                    'avg_processing_time': worker.average_processing_time,
                    'last_seen': worker.last_seen
                })
        
        # Task summary
        task_summary = {}
        for task_id in self.tasks.keys():
            task = self.tasks.get(task_id)
            if task:
                status_key = task.status.value
                if status_key not in task_summary:
                    task_summary[status_key] = 0
                task_summary[status_key] += 1
        
        return {
            'cluster_status': asdict(self.cluster_status),
            'workers': worker_details,
            'task_summary': task_summary,
            'coordinator_info': {
                'host': self.dist_config.coordinator_host,
                'port': self.dist_config.coordinator_port,
                'uptime': time.time() - getattr(self, '_start_time', time.time()),
                'is_running': self.is_running
            }
        }


    # ==================== SERIALIZATION METHODS ====================
    
    def _serialize_processed_video(self, processed_video: ProcessedVideo) -> Dict[str, Any]:
        """Serialize ProcessedVideo for transmission"""
        try:
            # Convert to dict, handling special cases
            video_dict = asdict(processed_video)
            
            # Handle numpy arrays in embeddings
            for chunk in video_dict.get('chunks', []):
                if chunk.get('chunk_embedding') is not None:
                    chunk['chunk_embedding'] = chunk['chunk_embedding'].tolist()
                
                for frame in chunk.get('frames', []):
                    if frame.get('embedding') is not None:
                        frame['embedding'] = frame['embedding'].tolist()
                    
                    # Remove PIL Image objects
                    if 'image' in frame:
                        del frame['image']
            
            return video_dict
            
        except Exception as e:
            self.logger.error(f"❌ Error serializing ProcessedVideo: {e}")
            return {}
    
    def _deserialize_processed_video(self, video_dict: Dict[str, Any]) -> ProcessedVideo:
        """Deserialize ProcessedVideo from dict"""
        try:
            # Convert numpy arrays back
            for chunk in video_dict.get('chunks', []):
                if chunk.get('chunk_embedding') is not None:
                    chunk['chunk_embedding'] = np.array(chunk['chunk_embedding'])
                
                for frame in chunk.get('frames', []):
                    if frame.get('embedding') is not None:
                        frame['embedding'] = np.array(frame['embedding'])
            
            # Reconstruct ProcessedVideo
            # Note: This is simplified - in practice you'd need to properly reconstruct
            # all nested dataclass objects
            return ProcessedVideo(**video_dict)
            
        except Exception as e:
            self.logger.error(f"❌ Error deserializing ProcessedVideo: {e}")
            return None


    # ==================== CLEANUP AND RESOURCE MANAGEMENT ====================
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            tasks_to_remove = []
            
            for task_id in self.tasks.keys():
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # Check if task is old and completed/failed
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task_age = current_time - (task.completed_time or task.created_time)
                    
                    if task_age > max_age_seconds:
                        tasks_to_remove.append(task_id)
            
            # Remove old tasks
            for task_id in tasks_to_remove:
                self.tasks.delete(task_id)
                self.completed_results.delete(task_id)
            
            if tasks_to_remove:
                self.logger.info(f"🧹 Cleaned up {len(tasks_to_remove)} old tasks")
                
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up tasks: {e}")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        try:
            process = psutil.Process()
            
            return {
                'memory_mb': process.memory_info().rss / (1024**2),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'num_workers': len(self.workers.keys()),
                'num_tasks': len(self.tasks.keys()),
                'pending_queue_size': self.pending_tasks.qsize(),
                'result_queue_size': self.result_queue.qsize()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting resource usage: {e}")
            return {}
    
    def __del__(self):
        """Cleanup when coordinator is destroyed"""
        try:
            if self.is_running:
                self.stop_coordinator()
        except:
            pass


# ==================== GLOBAL INSTANCE MANAGEMENT ====================

# Global distributed coordinator instance
_global_distributed_coordinator = None

def get_distributed_coordinator(config: Optional[Config] = None) -> DistributedCoordinator:
    """
    Get global distributed coordinator instance (singleton pattern)
    
    Args:
        config: Configuration object (chỉ sử dụng lần đầu)
        
    Returns:
        DistributedCoordinator instance
    """
    global _global_distributed_coordinator
    if _global_distributed_coordinator is None:
        _global_distributed_coordinator = DistributedCoordinator(config)
    return _global_distributed_coordinator

def reset_distributed_coordinator():
    """Reset global distributed coordinator instance"""
    global _global_distributed_coordinator
    if _global_distributed_coordinator:
        if _global_distributed_coordinator.is_running:
            _global_distributed_coordinator.stop_coordinator()
        del _global_distributed_coordinator
    _global_distributed_coordinator = None


# ==================== WORKER API CLIENT METHODS ====================

class WorkerAPIClient:
    """Client để communicate với worker nodes"""
    
    def __init__(self, worker_host: str, worker_port: int):
        self.worker_host = worker_host
        self.worker_port = worker_port
        self.base_url = f"http://{worker_host}:{worker_port}"
        self.session = requests.Session()
        self.session.timeout = 30
        
    def ping(self) -> bool:
        """Ping worker để check availability"""
        try:
            response = self.session.get(f"{self.base_url}/api/worker/ping")
            return response.status_code == 200
        except:
            return False
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get worker status"""
        try:
            response = self.session.get(f"{self.base_url}/api/worker/status")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def assign_task(self, task_data: Dict[str, Any]) -> bool:
        """Assign task to worker"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/task/assign",
                json=task_data
            )
            return response.status_code == 200
        except:
            return False
