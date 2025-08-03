"""
video_rag_manager.py - Main Wrapper Class cho VideoRAG System
Entry point chính cho user, wrap tất cả functionality
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Local imports
from config import Config, get_config
from utils import (
    get_logger, measure_time, performance_monitor,
    validate_video_format, get_video_info, generate_video_id,
    safe_execute, retry_on_failure, seconds_to_timestamp
)
from model_manager import get_model_manager, ModelManager
from video_processor import get_video_processor, VideoProcessor, ProcessedVideo
from knowledge_builder import get_knowledge_builder, KnowledgeBuilder, KnowledgeGraph, Entity
from database_manager import get_database_manager, DatabaseManager
from retrieval_engine import get_retrieval_engine, RetrievalEngine
from chat_interface import get_chat_interface, ChatInterface
from distributed_coordinator import get_distributed_coordinator, DistributedCoordinator


# ==================== DATA CLASSES ====================

@dataclass
class VideoEntry:
    """Represent một video trong system"""
    video_id: str
    video_path: str
    video_info: Dict[str, Any]
    processed_video: Optional[ProcessedVideo] = None
    knowledge_graph_id: Optional[str] = None
    processing_status: str = "pending"  # pending, processing, completed, failed
    processing_time: float = 0.0
    error_message: str = ""
    added_time: float = 0.0
    last_accessed: float = 0.0


@dataclass 
class QueryResult:
    """Result từ video query"""
    success: bool
    query: str
    results: List[Dict[str, Any]] = None
    total_results: int = 0
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class SystemStats:
    """System statistics"""
    total_videos: int
    processed_videos: int
    failed_videos: int
    total_frames: int
    total_entities: int
    total_relations: int
    total_knowledge_graphs: int
    system_memory_mb: float
    gpu_memory_mb: Optional[float]
    disk_usage_mb: float

# ==================== MAIN CLASS DEFINITION ====================

class VideoRAGManager:
    """
    Main Wrapper Class cho VideoRAG System
    Entry point chính cho user interaction
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        auto_load_models: bool = True,
        enable_distributed: bool = False
    ):
        """
        Initialize VideoRAG Manager
        
        Args:
            config_path: Path to configuration file
            auto_load_models: Automatically load AI models
            enable_distributed: Enable distributed processing
        """
        
        # Initialize configuration
        self.config = get_config(config_path) if config_path else get_config()
        self.logger = get_logger('video_rag_manager')
        
        # Initialize components
        self._init_components(auto_load_models)
        
        # Video management
        self.videos: Dict[str, VideoEntry] = {}
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
        
        # Threading và locks
        self._init_threading()
        
        # Distributed processing
        self.enable_distributed = enable_distributed
        self.distributed_coordinator = None
        if enable_distributed:
            self.distributed_coordinator = get_distributed_coordinator(self.config)
        
        # System state
        self.is_initialized = False
        self.startup_time = time.time()
        
        self.logger.info("🎯 VideoRAG Manager initialized")
        self.logger.info(f"📊 Configuration: {len(self.config.models)} models, distributed: {enable_distributed}")
        
        # Initialize system
        self._initialize_system()


    # ==================== INITIALIZATION METHODS ====================
    
    def _init_components(self, auto_load_models: bool):
        """Initialize all system components"""
        
        try:
            # Core components
            self.model_manager = get_model_manager(self.config)
            self.video_processor = get_video_processor(self.config)
            self.knowledge_builder = get_knowledge_builder(self.config)
            
            # Database và retrieval (placeholder for now)
            self.database_manager = None  # get_database_manager(self.config)
            self.retrieval_engine = None  # get_retrieval_engine(self.config)
            self.chat_interface = None    # get_chat_interface(self.config)
            
            self.logger.info("✅ Core components initialized")
            
            # Auto-load models if requested
            if auto_load_models:
                self._load_essential_models()
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def _init_threading(self):
        """Initialize threading components"""
        self._video_lock = threading.RLock()
        self._kg_lock = threading.RLock()
        self._processing_lock = threading.RLock()
        self._active_tasks = {}
        self._task_counter = 0
    
    def _load_essential_models(self):
        """Load essential models for system operation"""
        
        essential_models = ['text_embedding', 'gpt4o_mini']
        optional_models = ['minicpm_v', 'whisper', 'imagebind']
        
        self.logger.info("🤖 Loading essential models...")
        
        # Load essential models
        for model_name in essential_models:
            if self.config.is_model_enabled(model_name):
                success = self.model_manager.load_model(model_name)
                if not success:
                    self.logger.error(f"❌ Failed to load essential model: {model_name}")
                    raise Exception(f"Essential model {model_name} failed to load")
        
        # Load optional models (best effort)
        for model_name in optional_models:
            if self.config.is_model_enabled(model_name):
                success = self.model_manager.load_model(model_name)
                if success:
                    self.logger.info(f"✅ Loaded optional model: {model_name}")
                else:
                    self.logger.warning(f"⚠️ Failed to load optional model: {model_name}")
    
    def _initialize_system(self):
        """Initialize system directories và validate setup"""
        
        try:
            # Create necessary directories
            for path_name, path_value in self.config.paths.items():
                Path(path_value).mkdir(parents=True, exist_ok=True)
            
            # Validate system requirements
            self._validate_system_requirements()
            
            # Initialize databases
            if self.database_manager:
                self.database_manager.initialize()
            
            self.is_initialized = True
            self.logger.info("✅ System initialization completed")
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def _validate_system_requirements(self):
        """Validate system requirements"""
        
        from utils import check_resource_availability
        
        # Check basic resources
        is_available, issues = check_resource_availability(
            min_memory_gb=4.0,
            min_disk_gb=10.0,
            require_gpu=False
        )
        
        if not is_available:
            self.logger.warning("⚠️ System resource issues detected:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        # Validate configuration
        if not self.config.validate_config():
            raise Exception("Configuration validation failed")
        
# ==================== VIDEO MANAGEMENT METHODS ====================
    
    @measure_time
    def add_videos(
        self,
        video_paths: Union[str, List[str]],
        distributed: bool = False,
        enable_detailed_analysis: bool = True,
        process_immediately: bool = True
    ) -> Dict[str, bool]:
        """
        Add videos to the system
        
        Args:
            video_paths: Single video path hoặc list of paths
            distributed: Use distributed processing
            enable_detailed_analysis: Enable detailed frame analysis
            process_immediately: Process videos immediately hoặc queue for later
            
        Returns:
            Dict mapping video paths to success status
        """
        
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        
        self.logger.info(f"📥 Adding {len(video_paths)} videos to system")
        
        results = {}
        valid_videos = []
        
        # Validate videos first
        for video_path in video_paths:
            if not os.path.exists(video_path):
                self.logger.error(f"❌ Video not found: {video_path}")
                results[video_path] = False
                continue
            
            if not validate_video_format(video_path, self.config.processing.supported_formats):
                self.logger.error(f"❌ Invalid video format: {video_path}")
                results[video_path] = False
                continue
            
            # Check if video already exists
            video_id = generate_video_id(video_path)
            if video_id in self.videos:
                self.logger.warning(f"⚠️ Video already exists: {video_path}")
                results[video_path] = True
                continue
            
            valid_videos.append(video_path)
        
        # Add valid videos to registry
        for video_path in valid_videos:
            video_id = generate_video_id(video_path)
            video_info = get_video_info(video_path)
            
            video_entry = VideoEntry(
                video_id=video_id,
                video_path=video_path,
                video_info=video_info,
                processing_status="pending",
                added_time=time.time()
            )
            
            with self._video_lock:
                self.videos[video_id] = video_entry
            
            results[video_path] = True
            self.logger.info(f"✅ Added video: {os.path.basename(video_path)} (ID: {video_id})")
        
        # Process videos if requested
        if process_immediately and valid_videos:
            if distributed and self.enable_distributed:
                processing_results = self._process_videos_distributed(
                    valid_videos, enable_detailed_analysis
                )
            else:
                processing_results = self._process_videos_local(
                    valid_videos, enable_detailed_analysis
                )
            
            # Update results with processing status
            for video_path, success in processing_results.items():
                if video_path in results:
                    results[video_path] = results[video_path] and success
        
        successful_adds = sum(1 for success in results.values() if success)
        self.logger.info(f"📊 Video addition completed: {successful_adds}/{len(video_paths)} successful")
        
        return results
    
    def _process_videos_local(
        self,
        video_paths: List[str],
        enable_detailed_analysis: bool
    ) -> Dict[str, bool]:
        """Process videos locally"""
        
        results = {}
        
        self.logger.info(f"🔄 Processing {len(video_paths)} videos locally")
        
        for video_path in video_paths:
            video_id = generate_video_id(video_path)
            
            try:
                # Update status
                with self._video_lock:
                    if video_id in self.videos:
                        self.videos[video_id].processing_status = "processing"
                
                # Process video
                processed_video = self.video_processor.process_video(
                    video_path,
                    enable_detailed_analysis=enable_detailed_analysis,
                    save_frames=False,
                    parallel_processing=True
                )
                
                if processed_video.success:
                    # Build knowledge graph
                    kg_result = self._build_knowledge_graph_for_video(processed_video)
                    
                    # Update video entry
                    with self._video_lock:
                        if video_id in self.videos:
                            self.videos[video_id].processed_video = processed_video
                            self.videos[video_id].knowledge_graph_id = kg_result.graph_id if kg_result else None
                            self.videos[video_id].processing_status = "completed"
                            self.videos[video_id].processing_time = processed_video.processing_time
                    
                    # Store in database
                    if self.database_manager:
                        self.database_manager.store_video_data(processed_video, kg_result)
                    
                    results[video_path] = True
                    self.logger.info(f"✅ Video processed successfully: {os.path.basename(video_path)}")
                
                else:
                    # Update failure status
                    with self._video_lock:
                        if video_id in self.videos:
                            self.videos[video_id].processing_status = "failed"
                            self.videos[video_id].error_message = processed_video.error_message
                    
                    results[video_path] = False
                    self.logger.error(f"❌ Video processing failed: {video_path} - {processed_video.error_message}")
            
            except Exception as e:
                # Update failure status
                with self._video_lock:
                    if video_id in self.videos:
                        self.videos[video_id].processing_status = "failed"
                        self.videos[video_id].error_message = str(e)
                
                results[video_path] = False
                self.logger.error(f"❌ Error processing video {video_path}: {e}")
        
        return results
    
    def _process_videos_distributed(
        self,
        video_paths: List[str],
        enable_detailed_analysis: bool
    ) -> Dict[str, bool]:
        """Process videos using distributed coordinator"""
        
        # Placeholder for distributed processing
        self.logger.info(f"🌐 Distributed processing not yet implemented, falling back to local")
        return self._process_videos_local(video_paths, enable_detailed_analysis)
    
    def _build_knowledge_graph_for_video(self, processed_video: ProcessedVideo) -> Optional[KnowledgeGraph]:
        """Build knowledge graph for processed video"""
        
        try:
            kg = self.knowledge_builder.build_knowledge_graph(
                [processed_video],
                graph_id=f"kg_{processed_video.video_id}",
                enable_cross_video_linking=False
            )
            
            with self._kg_lock:
                self.knowledge_graphs[kg.graph_id] = kg
            
            self.logger.debug(f"✅ Knowledge graph built: {kg.graph_id}")
            return kg
            
        except Exception as e:
            self.logger.error(f"❌ Error building knowledge graph: {e}")
            return None
    
    def remove_video(self, video_id: str, cleanup_files: bool = True) -> bool:
        """
        Remove video from system
        
        Args:
            video_id: Video ID to remove
            cleanup_files: Whether to cleanup temporary files
            
        Returns:
            True if successfully removed
        """
        
        try:
            with self._video_lock:
                if video_id not in self.videos:
                    self.logger.warning(f"⚠️ Video not found: {video_id}")
                    return False
                
                video_entry = self.videos[video_id]
                
                # Remove from knowledge graphs
                if video_entry.knowledge_graph_id:
                    with self._kg_lock:
                        if video_entry.knowledge_graph_id in self.knowledge_graphs:
                            del self.knowledge_graphs[video_entry.knowledge_graph_id]
                
                # Remove from database
                if self.database_manager:
                    self.database_manager.remove_video_data(video_id)
                
                # Cleanup temporary files
                if cleanup_files:
                    self._cleanup_video_files(video_id)
                
                # Remove from videos registry
                del self.videos[video_id]
                
                self.logger.info(f"✅ Video removed: {video_id}")
                return True
        
        except Exception as e:
            self.logger.error(f"❌ Error removing video {video_id}: {e}")
            return False
    
    def _cleanup_video_files(self, video_id: str):
        """Cleanup temporary files for a video"""
        
        try:
            temp_dir = Path(self.config.paths.get('temp_dir', './temp'))
            
            # Clean frames
            frames_dir = temp_dir / "frames" / video_id
            if frames_dir.exists():
                import shutil
                shutil.rmtree(frames_dir, ignore_errors=True)
            
            # Clean audio files
            audio_dir = temp_dir / "audio" / video_id
            if audio_dir.exists():
                import shutil
                shutil.rmtree(audio_dir, ignore_errors=True)
            
            self.logger.debug(f"🧹 Cleaned temporary files for video: {video_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning video files: {e}")

# ==================== QUERY METHODS ====================
    
    @measure_time
    def query_videos(
        self,
        query: str,
        return_type: str = "clips",
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        enable_cross_modal: bool = True
    ) -> QueryResult:
        """
        Query videos for relevant content
        
        Args:
            query: Search query
            return_type: "clips", "frames", "videos", "entities"
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            enable_cross_modal: Enable cross-modal retrieval
            
        Returns:
            QueryResult object với matching content
        """
        
        start_time = time.time()
        
        self.logger.info(f"🔍 Querying videos: '{query}' (type: {return_type}, top_k: {top_k})")
        
        try:
            if not self.is_initialized:
                raise Exception("System not initialized")
            
            # Check if we have processed videos
            processed_videos = [v for v in self.videos.values() if v.processing_status == "completed"]
            if not processed_videos:
                return QueryResult(
                    success=False,
                    query=query,
                    error_message="No processed videos available for querying"
                )
            
            # Use retrieval engine for advanced querying
            if self.retrieval_engine:
                results = self.retrieval_engine.query(
                    query=query,
                    return_type=return_type,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    enable_cross_modal=enable_cross_modal
                )
            else:
                # Fallback to simple implementation
                results = self._simple_query_implementation(
                    query, return_type, top_k, similarity_threshold, enable_cross_modal
                )
            
            processing_time = time.time() - start_time
            
            return QueryResult(
                success=True,
                query=query,
                results=results,
                total_results=len(results),
                processing_time=processing_time,
                metadata={
                    'return_type': return_type,
                    'top_k': top_k,
                    'similarity_threshold': similarity_threshold,
                    'processed_videos': len(processed_videos)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Query failed: {e}")
            
            return QueryResult(
                success=False,
                query=query,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _simple_query_implementation(
        self,
        query: str,
        return_type: str,
        top_k: int,
        similarity_threshold: float,
        enable_cross_modal: bool
    ) -> List[Dict[str, Any]]:
        """Simple query implementation (fallback for full retrieval engine)"""
        
        results = []
        
        try:
            # Generate query embedding
            query_result = self.model_manager.inference_text_embedding([query])
            if not query_result.success:
                raise Exception(f"Failed to generate query embedding: {query_result.error_message}")
            
            query_embedding = query_result.result[0]
            
            # Search through processed videos
            for video_entry in self.videos.values():
                if video_entry.processing_status != "completed" or not video_entry.processed_video:
                    continue
                
                processed_video = video_entry.processed_video
                
                # Search through chunks based on return type
                for chunk in processed_video.chunks:
                    
                    if return_type == "clips":
                        results.extend(self._extract_clips_from_chunk(chunk, video_entry, query))
                    
                    elif return_type == "frames":
                        results.extend(self._extract_frames_from_chunk(chunk, video_entry, query))
                    
                    elif return_type == "entities":
                        results.extend(self._extract_entities_from_video(video_entry, query))
            
            # Sort by confidence and return top_k
            results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Error in simple query implementation: {e}")
            return []
    
    def _extract_clips_from_chunk(self, chunk, video_entry, query):
        """Extract clip results from chunk"""
        results = []
        
        if chunk.unified_description:
            # Simple text similarity for now
            if any(word.lower() in chunk.unified_description.lower() 
                  for word in query.split()):
                
                result = {
                    'type': 'clip',
                    'video_id': video_entry.video_id,
                    'video_path': video_entry.video_path,
                    'chunk_id': chunk.chunk_id,
                    'start_time': chunk.start_time,
                    'end_time': chunk.end_time,
                    'start_timestamp': seconds_to_timestamp(chunk.start_time),
                    'end_timestamp': seconds_to_timestamp(chunk.end_time),
                    'description': chunk.unified_description,
                    'confidence': 0.8,  # Placeholder
                    'frames_count': len(chunk.frames)
                }
                
                if chunk.audio_segment and chunk.audio_segment.transcript:
                    result['transcript'] = chunk.audio_segment.transcript
                
                results.append(result)
        
        return results
    
    def _extract_frames_from_chunk(self, chunk, video_entry, query):
        """Extract frame results from chunk"""
        results = []
        
        for frame in chunk.frames:
            if frame.caption:
                if any(word.lower() in frame.caption.lower() 
                      for word in query.split()):
                    
                    result = {
                        'type': 'frame',
                        'video_id': video_entry.video_id,
                        'video_path': video_entry.video_path,
                        'frame_id': frame.frame_id,
                        'timestamp': frame.timestamp,
                        'timestamp_str': seconds_to_timestamp(frame.timestamp),
                        'caption': frame.caption,
                        'confidence': frame.confidence,
                        'frame_index': frame.frame_index,
                        'image_path': frame.image_path
                    }
                    results.append(result)
        
        return results
    
    def _extract_entities_from_video(self, video_entry, query):
        """Extract entity results from video's knowledge graph"""
        results = []
        
        kg_id = video_entry.knowledge_graph_id
        if kg_id and kg_id in self.knowledge_graphs:
            kg = self.knowledge_graphs[kg_id]
            
            # Simple entity search
            for entity in kg.entities.values():
                if (query.lower() in entity.name.lower() or 
                    query.lower() in entity.description.lower()):
                    
                    result = {
                        'type': 'entity',
                        'video_id': video_entry.video_id,
                        'video_path': video_entry.video_path,
                        'entity_id': entity.entity_id,
                        'entity_name': entity.name,
                        'entity_type': entity.entity_type,
                        'description': entity.description,
                        'confidence': entity.confidence,
                        'timestamps': entity.timestamps
                    }
                    results.append(result)
        
        return results
    
    def query_by_image(
        self,
        image_path: str,
        return_type: str = "frames",
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> QueryResult:
        """
        Query videos using image similarity
        
        Args:
            image_path: Path to query image
            return_type: "frames", "clips", "videos"
            top_k: Number of top results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            QueryResult object
        """
        
        start_time = time.time()
        
        try:
            if not os.path.exists(image_path):
                raise Exception(f"Image not found: {image_path}")
            
            # Generate image embedding using ImageBind or similar
            image_result = self.model_manager.inference_imagebind_image([image_path])
            if not image_result.success:
                raise Exception(f"Failed to generate image embedding: {image_result.error_message}")
            
            query_embedding = image_result.result[0]
            
            # Use retrieval engine for similarity search
            if self.retrieval_engine:
                results = self.retrieval_engine.query_by_embedding(
                    embedding=query_embedding,
                    modality="image",
                    return_type=return_type,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
            else:
                # Simple fallback implementation
                results = self._simple_image_query(query_embedding, return_type, top_k)
            
            processing_time = time.time() - start_time
            
            return QueryResult(
                success=True,
                query=f"Image query: {os.path.basename(image_path)}",
                results=results,
                total_results=len(results),
                processing_time=processing_time,
                metadata={
                    'query_type': 'image',
                    'image_path': image_path,
                    'return_type': return_type
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Image query failed: {e}")
            
            return QueryResult(
                success=False,
                query=f"Image query: {os.path.basename(image_path) if os.path.exists(image_path) else 'invalid'}",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _simple_image_query(self, query_embedding, return_type, top_k):
        """Simple image similarity search implementation"""
        
        results = []
        
        # This would need proper embedding comparison logic
        # Placeholder implementation
        for video_entry in self.videos.values():
            if video_entry.processing_status == "completed" and video_entry.processed_video:
                for chunk in video_entry.processed_video.chunks:
                    for frame in chunk.frames:
                        # Placeholder similarity calculation
                        similarity = 0.5  # Would use actual embedding comparison
                        
                        if similarity > 0.3:  # Threshold
                            result = {
                                'type': 'frame',
                                'video_id': video_entry.video_id,
                                'video_path': video_entry.video_path,
                                'frame_id': frame.frame_id,
                                'timestamp': frame.timestamp,
                                'similarity': similarity,
                                'confidence': similarity
                            }
                            results.append(result)
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
# ==================== CHAT INTERFACE METHODS ====================
    
    def chat_with_frames(
        self,
        frame_ids: Union[str, List[str]],
        user_message: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Chat with LLM về frame content
        
        Args:
            frame_ids: Single frame ID hoặc list of frame IDs
            user_message: User's message/question
            conversation_context: Previous conversation context
            
        Returns:
            Dict với LLM response và metadata
        """
        
        if isinstance(frame_ids, str):
            frame_ids = [frame_ids]
        
        self.logger.info(f"💬 Chat with {len(frame_ids)} frames: '{user_message}'")
        
        try:
            # Collect frame information
            frame_data = self._collect_frame_data(frame_ids)
            
            if not frame_data:
                return {
                    'success': False,
                    'error_message': 'No frames found with the provided IDs'
                }
            
            # Use chat interface if available
            if self.chat_interface:
                result = self.chat_interface.chat_with_frames(
                    frame_data=frame_data,
                    user_message=user_message,
                    conversation_context=conversation_context
                )
            else:
                # Fallback to direct model inference
                result = self._direct_frame_chat(frame_data, user_message, conversation_context)
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Error in chat with frames: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def chat_with_video(
        self,
        video_id: str,
        user_message: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        include_transcript: bool = True,
        include_entities: bool = True
    ) -> Dict[str, Any]:
        """
        Chat với LLM về toàn bộ video content
        
        Args:
            video_id: Video ID to chat about
            user_message: User's message/question
            conversation_context: Previous conversation context
            include_transcript: Include audio transcript
            include_entities: Include extracted entities
            
        Returns:
            Dict với LLM response và metadata
        """
        
        self.logger.info(f"💬 Chat with video {video_id}: '{user_message}'")
        
        try:
            # Get video information
            with self._video_lock:
                if video_id not in self.videos:
                    return {
                        'success': False,
                        'error_message': f'Video not found: {video_id}'
                    }
                
                video_entry = self.videos[video_id]
                
                if video_entry.processing_status != "completed":
                    return {
                        'success': False,
                        'error_message': f'Video not processed yet: {video_entry.processing_status}'
                    }
            
            # Collect video content
            video_content = self._collect_video_content(
                video_entry, include_transcript, include_entities
            )
            
            # Use chat interface if available
            if self.chat_interface:
                result = self.chat_interface.chat_with_video(
                    video_content=video_content,
                    user_message=user_message,
                    conversation_context=conversation_context
                )
            else:
                # Fallback to direct model inference
                result = self._direct_video_chat(video_content, user_message, conversation_context)
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Error in chat with video: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def chat_with_entities(
        self,
        entity_ids: Union[str, List[str]],
        user_message: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Chat với LLM về specific entities
        
        Args:
            entity_ids: Single entity ID hoặc list of entity IDs
            user_message: User's message/question
            conversation_context: Previous conversation context
            
        Returns:
            Dict với LLM response và metadata
        """
        
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        
        self.logger.info(f"💬 Chat with {len(entity_ids)} entities: '{user_message}'")
        
        try:
            # Collect entity information
            entity_data = self._collect_entity_data(entity_ids)
            
            if not entity_data:
                return {
                    'success': False,
                    'error_message': 'No entities found with the provided IDs'
                }
            
            # Use chat interface if available
            if self.chat_interface:
                result = self.chat_interface.chat_with_entities(
                    entity_data=entity_data,
                    user_message=user_message,
                    conversation_context=conversation_context
                )
            else:
                # Fallback to direct model inference
                result = self._direct_entity_chat(entity_data, user_message, conversation_context)
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Error in chat with entities: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _collect_frame_data(self, frame_ids: List[str]) -> List[Dict[str, Any]]:
        """Collect frame information from frame IDs"""
        
        frame_data = []
        
        for video_entry in self.videos.values():
            if video_entry.processing_status != "completed" or not video_entry.processed_video:
                continue
            
            for chunk in video_entry.processed_video.chunks:
                for frame in chunk.frames:
                    if frame.frame_id in frame_ids:
                        frame_info = {
                            'frame_id': frame.frame_id,
                            'timestamp': frame.timestamp,
                            'timestamp_str': seconds_to_timestamp(frame.timestamp),
                            'caption': frame.caption,
                            'confidence': frame.confidence,
                            'video_path': video_entry.video_path,
                            'video_id': video_entry.video_id,
                            'image_path': frame.image_path
                        }
                        frame_data.append(frame_info)
        
        return frame_data
    
    def _collect_video_content(
        self, 
        video_entry: VideoEntry, 
        include_transcript: bool, 
        include_entities: bool
    ) -> Dict[str, Any]:
        """Collect comprehensive video content"""
        
        content = {
            'video_id': video_entry.video_id,
            'video_path': video_entry.video_path,
            'video_info': video_entry.video_info,
            'chunks': [],
            'transcript': "",
            'entities': [],
            'summary': ""
        }
        
        if not video_entry.processed_video:
            return content
        
        # Collect chunk information
        for chunk in video_entry.processed_video.chunks:
            chunk_info = {
                'chunk_id': chunk.chunk_id,
                'start_time': chunk.start_time,
                'end_time': chunk.end_time,
                'start_timestamp': seconds_to_timestamp(chunk.start_time),
                'end_timestamp': seconds_to_timestamp(chunk.end_time),
                'description': chunk.unified_description,
                'frame_count': len(chunk.frames)
            }
            
            if include_transcript and chunk.audio_segment and chunk.audio_segment.transcript:
                chunk_info['transcript'] = chunk.audio_segment.transcript
                content['transcript'] += f"[{chunk_info['start_timestamp']}-{chunk_info['end_timestamp']}] {chunk.audio_segment.transcript}\n"
            
            content['chunks'].append(chunk_info)
        
        # Collect entities from knowledge graph
        if include_entities and video_entry.knowledge_graph_id:
            kg_id = video_entry.knowledge_graph_id
            if kg_id in self.knowledge_graphs:
                kg = self.knowledge_graphs[kg_id]
                
                for entity in kg.entities.values():
                    entity_info = {
                        'entity_id': entity.entity_id,
                        'name': entity.name,
                        'entity_type': entity.entity_type,
                        'description': entity.description,
                        'confidence': entity.confidence,
                        'timestamps': entity.timestamps
                    }
                    content['entities'].append(entity_info)
        
        # Generate summary if available
        if video_entry.processed_video.summary:
            content['summary'] = video_entry.processed_video.summary
        
        return content
    
    def _collect_entity_data(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Collect entity information from entity IDs"""
        
        entity_data = []
        
        for kg in self.knowledge_graphs.values():
            for entity in kg.entities.values():
                if entity.entity_id in entity_ids:
                    entity_info = {
                        'entity_id': entity.entity_id,
                        'name': entity.name,
                        'entity_type': entity.entity_type,
                        'description': entity.description,
                        'confidence': entity.confidence,
                        'timestamps': entity.timestamps,
                        'knowledge_graph_id': kg.graph_id
                    }
                    
                    # Add related entities
                    related_entities = []
                    for relation in kg.relations.values():
                        if relation.source_entity_id == entity.entity_id:
                            if relation.target_entity_id in kg.entities:
                                target_entity = kg.entities[relation.target_entity_id]
                                related_entities.append({
                                    'entity_name': target_entity.name,
                                    'relation_type': relation.relation_type,
                                    'confidence': relation.confidence
                                })
                    
                    entity_info['related_entities'] = related_entities
                    entity_data.append(entity_info)
        
        return entity_data
    
    def _direct_frame_chat(
        self, 
        frame_data: List[Dict[str, Any]], 
        user_message: str, 
        conversation_context: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """Direct chat with frames using model inference"""
        
        try:
            # Prepare conversation messages
            messages = []
            
            # Add conversation context if provided
            if conversation_context:
                messages.extend(conversation_context)
            
            # Prepare frame context
            frame_context = "Frame Information:\n"
            for i, frame in enumerate(frame_data, 1):
                frame_context += f"{i}. Frame {frame['frame_id']} at {frame['timestamp_str']}:\n"
                frame_context += f"   Caption: {frame['caption']}\n"
                frame_context += f"   Source: {os.path.basename(frame['video_path'])}\n\n"
            
            # Add user message with frame context
            user_content = f"{frame_context}\nUser Question: {user_message}\n\nPlease answer based on the frame information provided above."
            
            messages.append({
                'role': 'user',
                'content': user_content
            })
            
            # Get LLM response
            result = self.model_manager.inference_gpt4o_mini(
                messages,
                max_tokens=1000,
                temperature=0.1
            )
            
            if result.success:
                return {
                    'success': True,
                    'response': result.result,
                    'frame_count': len(frame_data),
                    'frames_used': [f['frame_id'] for f in frame_data],
                    'execution_time': result.execution_time
                }
            else:
                return {
                    'success': False,
                    'error_message': result.error_message
                }
        
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _direct_video_chat(
        self, 
        video_content: Dict[str, Any], 
        user_message: str, 
        conversation_context: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """Direct chat with video using model inference"""
        
        try:
            # Prepare conversation messages
            messages = []
            
            # Add conversation context if provided
            if conversation_context:
                messages.extend(conversation_context)
            
            # Prepare video context
            video_context = f"Video Information:\n"
            video_context += f"Video: {os.path.basename(video_content['video_path'])}\n"
            video_context += f"Duration: {video_content['video_info'].get('duration', 'Unknown')} seconds\n\n"
            
            # Add summary if available
            if video_content['summary']:
                video_context += f"Video Summary:\n{video_content['summary']}\n\n"
            
            # Add chunk descriptions
            if video_content['chunks']:
                video_context += "Video Segments:\n"
                for chunk in video_content['chunks']:
                    video_context += f"[{chunk['start_timestamp']}-{chunk['end_timestamp']}] {chunk['description']}\n"
                    if 'transcript' in chunk and chunk['transcript']:
                        video_context += f"   Transcript: {chunk['transcript'][:100]}...\n"
                video_context += "\n"
            
            # Add entities if available
            if video_content['entities']:
                video_context += "Detected Entities:\n"
                for entity in video_content['entities'][:10]:  # Limit to 10 entities
                    video_context += f"- {entity['name']} ({entity['entity_type']}): {entity['description']}\n"
                video_context += "\n"
            
            # Add user message with video context
            user_content = f"{video_context}\nUser Question: {user_message}\n\nPlease answer based on the video information provided above."
            
            messages.append({
                'role': 'user',
                'content': user_content
            })
            
            # Get LLM response
            result = self.model_manager.inference_gpt4o_mini(
                messages,
                max_tokens=1500,
                temperature=0.1
            )
            
            if result.success:
                return {
                    'success': True,
                    'response': result.result,
                    'video_id': video_content['video_id'],
                    'chunks_count': len(video_content['chunks']),
                    'entities_count': len(video_content['entities']),
                    'execution_time': result.execution_time
                }
            else:
                return {
                    'success': False,
                    'error_message': result.error_message
                }
        
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _direct_entity_chat(
        self, 
        entity_data: List[Dict[str, Any]], 
        user_message: str, 
        conversation_context: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """Direct chat with entities using model inference"""
        
        try:
            # Prepare conversation messages
            messages = []
            
            # Add conversation context if provided
            if conversation_context:
                messages.extend(conversation_context)
            
            # Prepare entity context
            entity_context = "Entity Information:\n"
            for i, entity in enumerate(entity_data, 1):
                entity_context += f"{i}. {entity['name']} ({entity['entity_type']}):\n"
                entity_context += f"   Description: {entity['description']}\n"
                entity_context += f"   Confidence: {entity['confidence']:.2f}\n"
                
                if entity['timestamps']:
                    timestamps_str = [seconds_to_timestamp(ts) for ts in entity['timestamps'][:3]]
                    entity_context += f"   Appears at: {', '.join(timestamps_str)}\n"
                
                if entity['related_entities']:
                    related = [f"{rel['entity_name']} ({rel['relation_type']})" 
                              for rel in entity['related_entities'][:3]]
                    entity_context += f"   Related to: {', '.join(related)}\n"
                
                entity_context += "\n"
            
            # Add user message with entity context
            user_content = f"{entity_context}\nUser Question: {user_message}\n\nPlease answer based on the entity information provided above."
            
            messages.append({
                'role': 'user',
                'content': user_content
            })
            
            # Get LLM response
            result = self.model_manager.inference_gpt4o_mini(
                messages,
                max_tokens=1000,
                temperature=0.1
            )
            
            if result.success:
                return {
                    'success': True,
                    'response': result.result,
                    'entity_count': len(entity_data),
                    'entities_used': [e['entity_id'] for e in entity_data],
                    'execution_time': result.execution_time
                }
            else:
                return {
                    'success': False,
                    'error_message': result.error_message
                }
        
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }
        
# ==================== SYSTEM STATUS AND MANAGEMENT ====================
    
    def get_system_stats(self) -> SystemStats:
        """Get comprehensive system statistics"""
        
        try:
            # Video statistics
            total_videos = len(self.videos)
            processed_videos = len([v for v in self.videos.values() if v.processing_status == "completed"])
            failed_videos = len([v for v in self.videos.values() if v.processing_status == "failed"])
            
            # Frame statistics
            total_frames = 0
            for video_entry in self.videos.values():
                if video_entry.processed_video:
                    total_frames += video_entry.processed_video.total_frames
            
            # Knowledge graph statistics
            total_entities = 0
            total_relations = 0
            for kg in self.knowledge_graphs.values():
                total_entities += len(kg.entities)
                total_relations += len(kg.relations)
            
            # System resource usage
            import psutil
            process = psutil.Process()
            system_memory_mb = process.memory_info().rss / (1024**2)
            
            # GPU memory
            gpu_memory_mb = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024**2)
            except ImportError:
                pass
            
            # Disk usage (temp directory)
            disk_usage_mb = 0
            temp_dir = Path(self.config.paths.get('temp_dir', './temp'))
            if temp_dir.exists():
                disk_usage_mb = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file()) / (1024**2)
            
            return SystemStats(
                total_videos=total_videos,
                processed_videos=processed_videos,
                failed_videos=failed_videos,
                total_frames=total_frames,
                total_entities=total_entities,
                total_relations=total_relations,
                total_knowledge_graphs=len(self.knowledge_graphs),
                system_memory_mb=system_memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                disk_usage_mb=disk_usage_mb
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system stats: {e}")
            return SystemStats(0, 0, 0, 0, 0, 0, 0, 0.0, None, 0.0)
    
    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific video"""
        
        with self._video_lock:
            if video_id not in self.videos:
                return None
            
            video_entry = self.videos[video_id]
            
            info = {
                'video_id': video_entry.video_id,
                'video_path': video_entry.video_path,
                'video_info': video_entry.video_info,
                'processing_status': video_entry.processing_status,
                'processing_time': video_entry.processing_time,
                'added_time': video_entry.added_time,
                'last_accessed': video_entry.last_accessed,
                'error_message': video_entry.error_message
            }
            
            if video_entry.processed_video:
                info.update({
                    'total_frames': video_entry.processed_video.total_frames,
                    'total_chunks': len(video_entry.processed_video.chunks),
                    'total_duration': video_entry.processed_video.total_duration
                })
            
            if video_entry.knowledge_graph_id and video_entry.knowledge_graph_id in self.knowledge_graphs:
                kg = self.knowledge_graphs[video_entry.knowledge_graph_id]
                info.update({
                    'entities_count': len(kg.entities),
                    'relations_count': len(kg.relations)
                })
            
            # Update last accessed time
            video_entry.last_accessed = time.time()
            
            return info
    
    def list_videos(
        self,
        status_filter: Optional[str] = None,
        sort_by: str = "added_time",
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all videos in the system
        
        Args:
            status_filter: Filter by processing status
            sort_by: Sort by field ("added_time", "processing_time", "duration")
            reverse: Sort in reverse order
            
        Returns:
            List of video information dicts
        """
        
        videos_list = []
        
        with self._video_lock:
            for video_entry in self.videos.values():
                
                # Apply status filter
                if status_filter and video_entry.processing_status != status_filter:
                    continue
                
                video_info = {
                    'video_id': video_entry.video_id,
                    'video_path': video_entry.video_path,
                    'filename': os.path.basename(video_entry.video_path),
                    'processing_status': video_entry.processing_status,
                    'processing_time': video_entry.processing_time,
                    'added_time': video_entry.added_time,
                    'duration': video_entry.video_info.get('duration', 0),
                    'size_mb': video_entry.video_info.get('size_bytes', 0) / (1024**2),
                    'resolution': video_entry.video_info.get('resolution', 'Unknown')
                }
                
                if video_entry.processed_video:
                    video_info.update({
                        'total_frames': video_entry.processed_video.total_frames,
                        'chunks_count': len(video_entry.processed_video.chunks)
                    })
                
                videos_list.append(video_info)
        
        # Sort videos
        sort_key = lambda x: x.get(sort_by, 0)
        videos_list.sort(key=sort_key, reverse=reverse)
        
        return videos_list
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all loaded models"""
        
        try:
            model_status = {}
            
            if self.model_manager:
                for model_name in self.config.models.keys():
                    status = self.model_manager.get_model_status(model_name)
                    model_status[model_name] = {
                        'loaded': status.get('loaded', False),
                        'memory_usage': status.get('memory_usage', 0),
                        'last_used': status.get('last_used', 0),
                        'error_count': status.get('error_count', 0)
                    }
            
            return model_status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting model status: {e}")
            return {}
    
    def start_distributed_processing(self) -> bool:
        """Start distributed processing coordinator"""
        
        if not self.enable_distributed:
            self.logger.warning("⚠️ Distributed processing not enabled")
            return False
        
        try:
            self.logger.info("🌐 Starting distributed processing coordinator...")
            
            if self.distributed_coordinator:
                return self.distributed_coordinator.start()
            else:
                self.logger.warning("⚠️ Distributed coordinator not available")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error starting distributed processing: {e}")
            return False
    
    def stop_distributed_processing(self) -> bool:
        """Stop distributed processing coordinator"""
        
        try:
            if self.distributed_coordinator:
                success = self.distributed_coordinator.stop()
                self.logger.info("🛑 Distributed processing coordinator stopped")
                return success
            else:
                self.logger.warning("⚠️ Distributed coordinator not running")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping distributed processing: {e}")
            return False
    
    def get_processing_queue_status(self) -> Dict[str, Any]:
        """Get status of video processing queue"""
        
        try:
            queue_status = {
                'pending_videos': 0,
                'processing_videos': 0,
                'completed_videos': 0,
                'failed_videos': 0,
                'queue_size': 0,
                'active_tasks': len(self._active_tasks),
                'estimated_completion_time': 0
            }
            
            with self._video_lock:
                for video_entry in self.videos.values():
                    status = video_entry.processing_status
                    if status == "pending":
                        queue_status['pending_videos'] += 1
                    elif status == "processing":
                        queue_status['processing_videos'] += 1
                    elif status == "completed":
                        queue_status['completed_videos'] += 1
                    elif status == "failed":
                        queue_status['failed_videos'] += 1
            
            queue_status['queue_size'] = queue_status['pending_videos'] + queue_status['processing_videos']
            
            # Estimate completion time based on average processing time
            if queue_status['queue_size'] > 0 and queue_status['completed_videos'] > 0:
                completed_videos = [v for v in self.videos.values() if v.processing_status == "completed"]
                if completed_videos:
                    avg_processing_time = sum(v.processing_time for v in completed_videos) / len(completed_videos)
                    queue_status['estimated_completion_time'] = avg_processing_time * queue_status['queue_size']
            
            return queue_status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting queue status: {e}")
            return {}
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge graphs"""
        
        try:
            stats = {
                'total_graphs': len(self.knowledge_graphs),
                'total_entities': 0,
                'total_relations': 0,
                'entity_types': {},
                'relation_types': {},
                'average_entities_per_graph': 0,
                'average_relations_per_graph': 0
            }
            
            with self._kg_lock:
                for kg in self.knowledge_graphs.values():
                    stats['total_entities'] += len(kg.entities)
                    stats['total_relations'] += len(kg.relations)
                    
                    # Count entity types
                    for entity in kg.entities.values():
                        entity_type = entity.entity_type
                        stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
                    
                    # Count relation types
                    for relation in kg.relations.values():
                        relation_type = relation.relation_type
                        stats['relation_types'][relation_type] = stats['relation_types'].get(relation_type, 0) + 1
                
                # Calculate averages
                if stats['total_graphs'] > 0:
                    stats['average_entities_per_graph'] = stats['total_entities'] / stats['total_graphs']
                    stats['average_relations_per_graph'] = stats['total_relations'] / stats['total_graphs']
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting knowledge graph stats: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        
        try:
            metrics = {
                'uptime_seconds': time.time() - self.startup_time,
                'total_processing_time': 0,
                'average_processing_time_per_video': 0,
                'successful_processing_rate': 0,
                'queries_per_hour': 0,  # Would need query tracking
                'memory_efficiency': 0,
                'throughput_videos_per_hour': 0
            }
            
            # Calculate processing metrics
            completed_videos = [v for v in self.videos.values() if v.processing_status == "completed"]
            failed_videos = [v for v in self.videos.values() if v.processing_status == "failed"]
            
            if completed_videos:
                metrics['total_processing_time'] = sum(v.processing_time for v in completed_videos)
                metrics['average_processing_time_per_video'] = metrics['total_processing_time'] / len(completed_videos)
            
            # Calculate success rate
            total_processed = len(completed_videos) + len(failed_videos)
            if total_processed > 0:
                metrics['successful_processing_rate'] = len(completed_videos) / total_processed
            
            # Calculate throughput
            if metrics['uptime_seconds'] > 0:
                videos_per_second = len(completed_videos) / metrics['uptime_seconds']
                metrics['throughput_videos_per_hour'] = videos_per_second * 3600
            
            # Memory efficiency (processed frames per MB)
            current_stats = self.get_system_stats()
            if current_stats.system_memory_mb > 0 and current_stats.total_frames > 0:
                metrics['memory_efficiency'] = current_stats.total_frames / current_stats.system_memory_mb
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error getting performance metrics: {e}")
            return {}
        
# ==================== HEALTH CHECK METHODS ====================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive system health check
        
        Returns:
            Dict with health status of all components
        """
        
        health_status = {
            'overall_status': 'healthy',
            'timestamp': time.time(),
            'components': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check initialization
            if not self.is_initialized:
                health_status['overall_status'] = 'unhealthy'
                health_status['issues'].append('System not properly initialized')
            
            # Check model manager
            if self.model_manager:
                model_health = self.model_manager.health_check()
                health_status['components']['model_manager'] = {
                    'status': 'healthy' if all(model_health.values()) else 'degraded',
                    'models': model_health
                }
                
                failed_models = [name for name, status in model_health.items() if not status]
                if failed_models:
                    health_status['issues'].append(f"Failed models: {failed_models}")
            
            # Check video processor
            health_status['components']['video_processor'] = {
                'status': 'healthy' if self.video_processor else 'failed'
            }
            
            # Check knowledge builder
            health_status['components']['knowledge_builder'] = {
                'status': 'healthy' if self.knowledge_builder else 'failed'
            }
            
            # Check database manager
            if self.database_manager:
                db_health = self.database_manager.health_check()
                health_status['components']['database_manager'] = {
                    'status': 'healthy' if db_health.get('connected', False) else 'degraded',
                    'connection_status': db_health.get('connected', False),
                    'query_performance': db_health.get('avg_query_time', 0)
                }
            else:
                health_status['components']['database_manager'] = {
                    'status': 'not_available'
                }
            
            # Check retrieval engine
            if self.retrieval_engine:
                retrieval_health = self.retrieval_engine.health_check()
                health_status['components']['retrieval_engine'] = {
                    'status': 'healthy' if retrieval_health.get('index_ready', False) else 'degraded',
                    'index_status': retrieval_health.get('index_ready', False),
                    'search_performance': retrieval_health.get('avg_search_time', 0)
                }
            else:
                health_status['components']['retrieval_engine'] = {
                    'status': 'not_available'
                }
            
            # Check distributed coordinator
            if self.distributed_coordinator:
                dist_health = self.distributed_coordinator.health_check()
                health_status['components']['distributed_coordinator'] = {
                    'status': 'healthy' if dist_health.get('running', False) else 'degraded',
                    'worker_count': dist_health.get('worker_count', 0),
                    'queue_size': dist_health.get('queue_size', 0)
                }
            else:
                health_status['components']['distributed_coordinator'] = {
                    'status': 'not_available' if not self.enable_distributed else 'disabled'
                }
            
            # Check system resources
            from utils import check_resource_availability
            resources_ok, resource_issues = check_resource_availability()
            health_status['components']['system_resources'] = {
                'status': 'healthy' if resources_ok else 'degraded',
                'issues': resource_issues
            }
            
            if resource_issues:
                health_status['issues'].extend(resource_issues)
            
            # Check video processing status
            if self.videos:
                failed_videos = len([v for v in self.videos.values() if v.processing_status == "failed"])
                total_videos = len(self.videos)
                
                if failed_videos > 0:
                    failure_rate = failed_videos / total_videos
                    if failure_rate > 0.5:
                        health_status['issues'].append(f"High video processing failure rate: {failure_rate:.1%}")
            
            # Generate recommendations
            self._generate_health_recommendations(health_status)
            
            # Final overall status
            if len(health_status['issues']) == 0:
                health_status['overall_status'] = 'healthy'
            elif len(health_status['issues']) < 3:
                health_status['overall_status'] = 'degraded'
            else:
                health_status['overall_status'] = 'unhealthy'
            
            return health_status
            
        except Exception as e:
            return [f'Error generating recommendations: {str(e)}']
    
    def _get_directory_size(self, directory: Path) -> float:
        """Get directory size in MB"""
        
        try:
            if not directory.exists():
                return 0.0
            
            total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            return total_size / (1024**2)
            
        except Exception:
            return 0.0


    # ==================== CLEANUP AND MAINTENANCE ====================
    
    def cleanup_system(
        self,
        remove_failed_videos: bool = False,
        cleanup_temp_files: bool = True,
        cleanup_old_cache: bool = True,
        max_age_hours: int = 24
    ) -> Dict[str, int]:
        """
        Cleanup system resources
        
        Args:
            remove_failed_videos: Remove videos with failed processing status
            cleanup_temp_files: Clean temporary files
            cleanup_old_cache: Clean old cache files
            max_age_hours: Maximum age for cleanup
            
        Returns:
            Dict with cleanup statistics
        """
        
        cleanup_stats = {
            'removed_videos': 0,
            'cleaned_temp_files': 0,
            'cleaned_cache_files': 0,
            'freed_memory_mb': 0
        }
        
        try:
            # Remove failed videos
            if remove_failed_videos:
                failed_video_ids = []
                with self._video_lock:
                    for video_id, video_entry in self.videos.items():
                        if video_entry.processing_status == "failed":
                            failed_video_ids.append(video_id)
                
                for video_id in failed_video_ids:
                    if self.remove_video(video_id, cleanup_files=True):
                        cleanup_stats['removed_videos'] += 1
            
            # Cleanup temporary files
            if cleanup_temp_files:
                temp_dir = Path(self.config.paths.get('temp_dir', './temp'))
                if temp_dir.exists():
                    from utils import cleanup_temp_files
                    cleanup_temp_files(str(temp_dir), max_age_hours)
                    cleanup_stats['cleaned_temp_files'] = 1
            
            # Cleanup cache
            if cleanup_old_cache:
                cache_dir = Path(self.config.paths.get('data_dir', './data')) / 'cache'
                if cache_dir.exists():
                    from utils import cleanup_cache
                    cleanup_cache(str(cache_dir), max_age_hours)
                    cleanup_stats['cleaned_cache_files'] = 1
            
            # Memory cleanup
            import gc
            freed_objects = gc.collect()
            cleanup_stats['freed_memory_mb'] = freed_objects * 0.001  # Rough estimate
            
            if self._check_torch_available():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.logger.info(f"🧹 System cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error during system cleanup: {e}")
            return cleanup_stats
    
    def optimize_memory(self) -> Dict[str, float]:
        """Optimize memory usage"""
        
        memory_stats = {'before_mb': 0, 'after_mb': 0, 'saved_mb': 0}
        
        try:
            import psutil
            process = psutil.Process()
            
            # Memory before optimization
            memory_stats['before_mb'] = process.memory_info().rss / (1024**2)
            
            # Unload unused models
            if hasattr(self, 'model_manager'):
                self.model_manager.cleanup_models(unused_threshold_minutes=30)
            
            # Clear caches
            if self.database_manager:
                self.database_manager.clear_cache()
            
            if self.retrieval_engine:
                self.retrieval_engine.clear_cache()
            
            # Garbage collection
            import gc
            gc.collect()
            
            # GPU memory cleanup
            if self._check_torch_available():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Memory after optimization
            memory_stats['after_mb'] = process.memory_info().rss / (1024**2)
            memory_stats['saved_mb'] = memory_stats['before_mb'] - memory_stats['after_mb']
            
            self.logger.info(f"🚀 Memory optimized: {memory_stats['saved_mb']:.1f}MB saved")
            return memory_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing memory: {e}")
            return memory_stats
    
    def _check_torch_available(self) -> bool:
        """Check if PyTorch is available"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def maintenance_routine(self, deep_clean: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive maintenance routine
        
        Args:
            deep_clean: Perform deep cleaning operations
            
        Returns:
            Maintenance results
        """
        
        self.logger.info("🔧 Starting maintenance routine...")
        
        maintenance_results = {
            'start_time': time.time(),
            'operations': {},
            'overall_success': True
        }
        
        try:
            # Health check
            health_status = self.health_check()
            maintenance_results['operations']['health_check'] = {
                'success': True,
                'status': health_status['overall_status'],
                'issues_count': len(health_status['issues'])
            }
            
            # Memory optimization
            memory_stats = self.optimize_memory()
            maintenance_results['operations']['memory_optimization'] = {
                'success': True,
                'memory_saved_mb': memory_stats['saved_mb']
            }
            
            # Cleanup system
            cleanup_stats = self.cleanup_system(
                remove_failed_videos=deep_clean,
                cleanup_temp_files=True,
                cleanup_old_cache=deep_clean,
                max_age_hours=24 if not deep_clean else 1
            )
            maintenance_results['operations']['cleanup'] = {
                'success': True,
                'stats': cleanup_stats
            }
            
            # Update video access times
            self._update_video_access_times()
            maintenance_results['operations']['access_time_update'] = {'success': True}
            
            # Optimize knowledge graphs
            if deep_clean:
                kg_optimization = self._optimize_knowledge_graphs()
                maintenance_results['operations']['kg_optimization'] = kg_optimization
            
            # Database maintenance
            if self.database_manager and deep_clean:
                db_maintenance = self.database_manager.run_maintenance()
                maintenance_results['operations']['database_maintenance'] = db_maintenance
            
            # Model health check and reload if necessary
            if self.model_manager:
                model_maintenance = self._model_maintenance()
                maintenance_results['operations']['model_maintenance'] = model_maintenance
            
            maintenance_results['end_time'] = time.time()
            maintenance_results['duration_seconds'] = maintenance_results['end_time'] - maintenance_results['start_time']
            
            self.logger.info(f"✅ Maintenance routine completed in {maintenance_results['duration_seconds']:.1f}s")
            return maintenance_results
            
        except Exception as e:
            maintenance_results['overall_success'] = False
            maintenance_results['error'] = str(e)
            self.logger.error(f"❌ Maintenance routine failed: {e}")
            return maintenance_results
    
    def _update_video_access_times(self):
        """Update video access times for LRU management"""
        
        current_time = time.time()
        with self._video_lock:
            for video_entry in self.videos.values():
                if video_entry.last_accessed == 0:
                    video_entry.last_accessed = current_time
    
    def _optimize_knowledge_graphs(self) -> Dict[str, Any]:
        """Optimize knowledge graphs by removing redundant entities/relations"""
        
        try:
            optimization_stats = {
                'success': True,
                'graphs_processed': 0,
                'entities_removed': 0,
                'relations_removed': 0
            }
            
            with self._kg_lock:
                for kg_id, kg in self.knowledge_graphs.items():
                    if self.knowledge_builder:
                        stats = self.knowledge_builder.optimize_knowledge_graph(kg)
                        optimization_stats['graphs_processed'] += 1
                        optimization_stats['entities_removed'] += stats.get('entities_removed', 0)
                        optimization_stats['relations_removed'] += stats.get('relations_removed', 0)
            
            return optimization_stats
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _model_maintenance(self) -> Dict[str, Any]:
        """Perform model maintenance operations"""
        
        try:
            maintenance_stats = {
                'success': True,
                'models_checked': 0,
                'models_reloaded': 0,
                'errors_cleared': 0
            }
            
            for model_name in self.config.models.keys():
                maintenance_stats['models_checked'] += 1
                
                # Check model health
                model_status = self.model_manager.get_model_status(model_name)
                
                # Reload if high error rate
                if model_status.get('error_count', 0) > 50:
                    self.logger.info(f"🔄 Reloading model with high error rate: {model_name}")
                    if self.model_manager.reload_model(model_name):
                        maintenance_stats['models_reloaded'] += 1
                        maintenance_stats['errors_cleared'] += model_status.get('error_count', 0)
            
            return maintenance_stats
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            self.logger.error(f"❌ Error during health check: {e}")
            return {
                'overall_status': 'error',
                'timestamp': time.time(),
                'error': str(e),
                'components': {},
                'issues': [f'Health check failed: {e}'],
                'recommendations': ['Check system logs and restart if necessary']
            }
    
    def _generate_health_recommendations(self, health_status: Dict[str, Any]):
        """Generate health recommendations based on issues"""
        
        if health_status['issues']:
            health_status['overall_status'] = 'degraded'
            
            if 'System not properly initialized' in str(health_status['issues']):
                health_status['recommendations'].append('Restart the VideoRAG Manager')
            
            if 'Failed models' in str(health_status['issues']):
                health_status['recommendations'].append('Check model configurations and API keys')
            
            if 'memory' in str(health_status['issues']).lower():
                health_status['recommendations'].append('Run memory optimization or add more RAM')
            
            if 'disk' in str(health_status['issues']).lower():
                health_status['recommendations'].append('Clean up temporary files or add more disk space')
            
            if 'High video processing failure rate' in str(health_status['issues']):
                health_status['recommendations'].append('Check video formats and processing settings')
            
            if 'database' in str(health_status['issues']).lower():
                health_status['recommendations'].append('Check database connection and configuration')
    
    def diagnose_system(self) -> Dict[str, Any]:
        """
        Advanced system diagnostics
        
        Returns:
            Detailed diagnostic information
        """
        
        diagnostics = {
            'timestamp': time.time(),
            'system_info': {},
            'component_details': {},
            'performance_analysis': {},
            'resource_usage': {},
            'recommendations': []
        }
        
        try:
            # System information
            import platform
            import psutil
            
            diagnostics['system_info'] = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage': {
                    'total_gb': psutil.disk_usage('/').total / (1024**3),
                    'free_gb': psutil.disk_usage('/').free / (1024**3)
                }
            }
            
            # GPU information
            try:
                import torch
                if torch.cuda.is_available():
                    diagnostics['system_info']['gpu_count'] = torch.cuda.device_count()
                    diagnostics['system_info']['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except ImportError:
                diagnostics['system_info']['gpu_available'] = False
            
            # Component diagnostics
            diagnostics['component_details']['model_manager'] = self._diagnose_model_manager()
            diagnostics['component_details']['video_processor'] = self._diagnose_video_processor()
            diagnostics['component_details']['knowledge_builder'] = self._diagnose_knowledge_builder()
            
            if self.database_manager:
                diagnostics['component_details']['database_manager'] = self.database_manager.get_diagnostics()
            
            if self.retrieval_engine:
                diagnostics['component_details']['retrieval_engine'] = self.retrieval_engine.get_diagnostics()
            
            # Performance analysis
            diagnostics['performance_analysis'] = self.get_performance_metrics()
            
            # Resource usage analysis
            diagnostics['resource_usage'] = self._analyze_resource_usage()
            
            # Generate advanced recommendations
            diagnostics['recommendations'] = self._generate_advanced_recommendations(diagnostics)
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"❌ Error during system diagnostics: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e),
                'recommendations': ['Run basic health check instead']
            }
    
    def _diagnose_model_manager(self) -> Dict[str, Any]:
        """Diagnose model manager component"""
        
        try:
            if not self.model_manager:
                return {'status': 'not_available'}
            
            diagnostics = {
                'loaded_models': {},
                'memory_usage': {},
                'performance_metrics': {},
                'error_rates': {}
            }
            
            for model_name in self.config.models.keys():
                model_status = self.model_manager.get_model_status(model_name)
                diagnostics['loaded_models'][model_name] = model_status.get('loaded', False)
                diagnostics['memory_usage'][model_name] = model_status.get('memory_usage', 0)
                diagnostics['error_rates'][model_name] = model_status.get('error_count', 0)
            
            return diagnostics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _diagnose_video_processor(self) -> Dict[str, Any]:
        """Diagnose video processor component"""
        
        try:
            if not self.video_processor:
                return {'status': 'not_available'}
            
            return {
                'temp_dir_size_mb': self._get_directory_size(Path(self.config.paths.get('temp_dir', './temp'))),
                'processing_capabilities': self.video_processor.get_capabilities(),
                'supported_formats': self.config.processing.supported_formats,
                'processing_settings': asdict(self.config.processing)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _diagnose_knowledge_builder(self) -> Dict[str, Any]:
        """Diagnose knowledge builder component"""
        
        try:
            if not self.knowledge_builder:
                return {'status': 'not_available'}
            
            return {
                'active_graphs': len(self.knowledge_graphs),
                'builder_settings': self.knowledge_builder.get_settings(),
                'entity_extraction_stats': self.knowledge_builder.get_extraction_stats()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze current resource usage"""
        
        try:
            import psutil
            
            # Process-specific resources
            process = psutil.Process()
            
            usage = {
                'cpu_percent': process.cpu_percent(interval=1),
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / (1024**2),
                'open_files': len(process.open_files()),
                'threads': process.num_threads()
            }
            
            # System-wide resources
            usage['system_cpu_percent'] = psutil.cpu_percent(interval=1)
            usage['system_memory_percent'] = psutil.virtual_memory().percent
            usage['system_disk_percent'] = psutil.disk_usage('/').percent
            
            # Temporary directory usage
            temp_dir = Path(self.config.paths.get('temp_dir', './temp'))
            if temp_dir.exists():
                usage['temp_dir_mb'] = self._get_directory_size(temp_dir)
            
            return usage
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_advanced_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """Generate advanced recommendations based on diagnostics"""
        
        recommendations = []
        
        try:
            # Memory recommendations
            if diagnostics['resource_usage'].get('memory_percent', 0) > 80:
                recommendations.append('High memory usage detected - consider optimizing memory or adding more RAM')
            
            # CPU recommendations
            if diagnostics['resource_usage'].get('cpu_percent', 0) > 90:
                recommendations.append('High CPU usage detected - consider reducing concurrent processing')
            
            # Disk recommendations
            if diagnostics['resource_usage'].get('system_disk_percent', 0) > 90:
                recommendations.append('Low disk space - clean up temporary files and old data')
            
            # Model recommendations
            model_details = diagnostics['component_details'].get('model_manager', {})
            if model_details.get('error_rates'):
                high_error_models = [name for name, rate in model_details['error_rates'].items() if rate > 10]
                if high_error_models:
                    recommendations.append(f'High error rates in models: {high_error_models} - check configurations')
            
            # Performance recommendations
            perf = diagnostics.get('performance_analysis', {})
            if perf.get('successful_processing_rate', 1) < 0.8:
                recommendations.append('Low processing success rate - check video formats and model configurations')
            
            if perf.get('average_processing_time_per_video', 0) > 300:  # 5 minutes
                recommendations.append('High processing time per video - consider optimizing settings or hardware')
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating advanced recommendations: {e}")
            return [f'Error generating recommendations: {str(e)}']
        
# ==================== EXPORT/IMPORT METHODS ====================
    
    def export_system_data(
        self,
        export_path: str,
        include_videos: bool = True,
        include_knowledge_graphs: bool = True,
        include_metadata: bool = True,
        include_models: bool = False,
        compress_export: bool = True
    ) -> bool:
        """
        Export system data to file
        
        Args:
            export_path: Path to export file
            include_videos: Include video data
            include_knowledge_graphs: Include knowledge graphs
            include_metadata: Include system metadata
            include_models: Include model configurations
            compress_export: Compress the export file
            
        Returns:
            True if export successful
        """
        
        try:
            self.logger.info(f"📤 Starting system data export to: {export_path}")
            
            export_data = {
                'export_time': time.time(),
                'system_version': '1.0.0',
                'export_config': {
                    'include_videos': include_videos,
                    'include_knowledge_graphs': include_knowledge_graphs,
                    'include_metadata': include_metadata,
                    'include_models': include_models
                },
                'config_summary': {
                    'models': list(self.config.models.keys()),
                    'processing': asdict(self.config.processing),
                    'retrieval': asdict(self.config.retrieval)
                }
            }
            
            # Export videos
            if include_videos:
                export_data['videos'] = {}
                with self._video_lock:
                    for video_id, video_entry in self.videos.items():
                        video_data = {
                            'video_path': video_entry.video_path,
                            'video_info': video_entry.video_info,
                            'processing_status': video_entry.processing_status,
                            'processing_time': video_entry.processing_time,
                            'added_time': video_entry.added_time,
                            'knowledge_graph_id': video_entry.knowledge_graph_id,
                            'error_message': video_entry.error_message
                        }
                        
                        # Include processed video summary
                        if video_entry.processed_video:
                            video_data['processed_summary'] = {
                                'total_frames': video_entry.processed_video.total_frames,
                                'total_chunks': len(video_entry.processed_video.chunks),
                                'total_duration': video_entry.processed_video.total_duration,
                                'summary': getattr(video_entry.processed_video, 'summary', '')
                            }
                        
                        export_data['videos'][video_id] = video_data
                
                self.logger.info(f"📹 Exported {len(export_data['videos'])} videos")
            
            # Export knowledge graphs
            if include_knowledge_graphs:
                export_data['knowledge_graphs'] = {}
                with self._kg_lock:
                    for kg_id, kg in self.knowledge_graphs.items():
                        kg_data = {
                            'graph_id': kg.graph_id,
                            'creation_time': getattr(kg, 'creation_time', time.time()),
                            'entities': {},
                            'relations': {},
                            'statistics': {
                                'entity_count': len(kg.entities),
                                'relation_count': len(kg.relations)
                            }
                        }
                        
                        # Export entities
                        for entity_id, entity in kg.entities.items():
                            kg_data['entities'][entity_id] = {
                                'entity_id': entity.entity_id,
                                'name': entity.name,
                                'entity_type': entity.entity_type,
                                'description': entity.description,
                                'confidence': entity.confidence,
                                'timestamps': entity.timestamps
                            }
                        
                        # Export relations
                        for relation_id, relation in kg.relations.items():
                            kg_data['relations'][relation_id] = {
                                'relation_id': relation.relation_id,
                                'source_entity_id': relation.source_entity_id,
                                'target_entity_id': relation.target_entity_id,
                                'relation_type': relation.relation_type,
                                'confidence': relation.confidence,
                                'description': getattr(relation, 'description', '')
                            }
                        
                        export_data['knowledge_graphs'][kg_id] = kg_data
                
                self.logger.info(f"🧠 Exported {len(export_data['knowledge_graphs'])} knowledge graphs")
            
            # Export metadata
            if include_metadata:
                stats = self.get_system_stats()
                performance_metrics = self.get_performance_metrics()
                
                export_data['metadata'] = {
                    'system_stats': asdict(stats),
                    'performance_metrics': performance_metrics,
                    'uptime_seconds': time.time() - self.startup_time,
                    'export_timestamp': time.time()
                }
                
                self.logger.info("📊 Exported system metadata")
            
            # Export model configurations
            if include_models:
                export_data['models'] = {}
                for model_name in self.config.models.keys():
                    model_status = self.model_manager.get_model_status(model_name)
                    export_data['models'][model_name] = {
                        'config': self.config.models[model_name],
                        'status': model_status,
                        'loaded': model_status.get('loaded', False)
                    }
                
                self.logger.info(f"🤖 Exported {len(export_data['models'])} model configurations")
            
            # Save to file
            import json
            
            if compress_export and export_path.endswith('.json'):
                export_path = export_path.replace('.json', '.json.gz')
            
            if export_path.endswith('.gz'):
                import gzip
                with gzip.open(export_path, 'wt', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ System data exported successfully to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting system data: {e}")
            return False
    
    def import_system_data(
        self, 
        import_path: str, 
        merge_mode: bool = True,
        selective_import: Optional[Dict[str, bool]] = None
    ) -> bool:
        """
        Import system data from file
        
        Args:
            import_path: Path to import file
            merge_mode: Merge with existing data (vs replace)
            selective_import: Dict specifying what to import
            
        Returns:
            True if import successful
        """
        
        try:
            if not os.path.exists(import_path):
                raise Exception(f"Import file not found: {import_path}")
            
            self.logger.info(f"📥 Starting system data import from: {import_path}")
            
            # Load data
            import json
            
            if import_path.endswith('.gz'):
                import gzip
                with gzip.open(import_path, 'rt', encoding='utf-8') as f:
                    import_data = json.load(f)
            else:
                with open(import_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
            
            # Validate import data
            if 'export_time' not in import_data:
                raise Exception("Invalid import file format")
            
            self.logger.info(f"📋 Import file created at: {time.ctime(import_data['export_time'])}")
            
            # Set default selective import
            if selective_import is None:
                selective_import = {
                    'videos': True,
                    'knowledge_graphs': True,
                    'metadata': False,
                    'models': False
                }
            
            import_stats = {
                'videos_imported': 0,
                'knowledge_graphs_imported': 0,
                'videos_skipped': 0,
                'errors': []
            }
            
            # Import videos
            if selective_import.get('videos', True) and 'videos' in import_data:
                for video_id, video_data in import_data['videos'].items():
                    try:
                        video_path = video_data['video_path']
                        
                        # Check if video file still exists
                        if not os.path.exists(video_path):
                            self.logger.warning(f"⚠️ Video file not found during import: {video_path}")
                            import_stats['videos_skipped'] += 1
                            continue
                        
                        # Skip if already exists in merge mode
                        if merge_mode and video_id in self.videos:
                            import_stats['videos_skipped'] += 1
                            continue
                        
                        # Create video entry
                        video_entry = VideoEntry(
                            video_id=video_id,
                            video_path=video_path,
                            video_info=video_data.get('video_info', {}),
                            processing_status=video_data.get('processing_status', 'pending'),
                            processing_time=video_data.get('processing_time', 0),
                            added_time=video_data.get('added_time', time.time()),
                            knowledge_graph_id=video_data.get('knowledge_graph_id'),
                            error_message=video_data.get('error_message', '')
                        )
                        
                        with self._video_lock:
                            self.videos[video_id] = video_entry
                        
                        import_stats['videos_imported'] += 1
                        
                    except Exception as e:
                        import_stats['errors'].append(f"Error importing video {video_id}: {str(e)}")
                
                self.logger.info(f"📹 Imported {import_stats['videos_imported']} videos, skipped {import_stats['videos_skipped']}")
            
            # Import knowledge graphs
            if selective_import.get('knowledge_graphs', True) and 'knowledge_graphs' in import_data:
                for kg_id, kg_data in import_data['knowledge_graphs'].items():
                    try:
                        # Skip if already exists in merge mode
                        if merge_mode and kg_id in self.knowledge_graphs:
                            continue
                        
                        # Reconstruct knowledge graph
                        kg = self._reconstruct_knowledge_graph(kg_data)
                        
                        with self._kg_lock:
                            self.knowledge_graphs[kg_id] = kg
                        
                        import_stats['knowledge_graphs_imported'] += 1
                        
                    except Exception as e:
                        import_stats['errors'].append(f"Error importing knowledge graph {kg_id}: {str(e)}")
                
                self.logger.info(f"🧠 Imported {import_stats['knowledge_graphs_imported']} knowledge graphs")
            
            # Import model configurations (if requested)
            if selective_import.get('models', False) and 'models' in import_data:
                self._import_model_configurations(import_data['models'])
            
            # Log any errors
            if import_stats['errors']:
                self.logger.warning(f"⚠️ Import completed with {len(import_stats['errors'])} errors:")
                for error in import_stats['errors'][:5]:  # Show first 5 errors
                    self.logger.warning(f"  - {error}")
            
            self.logger.info(f"✅ System data imported successfully from: {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error importing system data: {e}")
            return False
    
    def _reconstruct_knowledge_graph(self, kg_data: Dict[str, Any]) -> KnowledgeGraph:
        """Reconstruct knowledge graph from export data"""
        
        # Create new knowledge graph
        kg = KnowledgeGraph(
            graph_id=kg_data['graph_id'],
            entities={},
            relations={}
        )
        
        # Reconstruct entities
        for entity_id, entity_data in kg_data['entities'].items():
            entity = Entity(
                entity_id=entity_data['entity_id'],
                name=entity_data['name'],
                entity_type=entity_data['entity_type'],
                description=entity_data['description'],
                confidence=entity_data['confidence'],
                timestamps=entity_data['timestamps']
            )
            kg.entities[entity_id] = entity
        
        # Reconstruct relations
        for relation_id, relation_data in kg_data['relations'].items():
            # Assuming Relation class exists with similar structure
            from knowledge_builder import Relation
            relation = Relation(
                relation_id=relation_data['relation_id'],
                source_entity_id=relation_data['source_entity_id'],
                target_entity_id=relation_data['target_entity_id'],
                relation_type=relation_data['relation_type'],
                confidence=relation_data['confidence'],
                description=relation_data.get('description', '')
            )
            kg.relations[relation_id] = relation
        
        return kg
    
    def _import_model_configurations(self, models_data: Dict[str, Any]):
        """Import model configurations"""
        
        try:
            for model_name, model_data in models_data.items():
                if model_name in self.config.models:
                    # Update model configuration if needed
                    imported_config = model_data.get('config', {})
                    if imported_config != self.config.models[model_name]:
                        self.logger.info(f"🔄 Updating configuration for model: {model_name}")
                        # Update config and potentially reload model
                        self.config.models[model_name].update(imported_config)
            
        except Exception as e:
            self.logger.error(f"❌ Error importing model configurations: {e}")
    
    def backup_system(self, backup_dir: str = "./backups") -> str:
        """
        Create a full system backup
        
        Args:
            backup_dir: Directory to store backups
            
        Returns:
            Path to backup file
        """
        
        try:
            # Create backup directory
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"videorag_backup_{timestamp}.json.gz"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Export everything
            success = self.export_system_data(
                export_path=backup_path,
                include_videos=True,
                include_knowledge_graphs=True,
                include_metadata=True,
                include_models=True,
                compress_export=True
            )
            
            if success:
                self.logger.info(f"💾 System backup created: {backup_path}")
                return backup_path
            else:
                raise Exception("Backup export failed")
                
        except Exception as e:
            self.logger.error(f"❌ Error creating system backup: {e}")
            return ""
    
    def restore_from_backup(self, backup_path: str, confirm_restore: bool = False) -> bool:
        """
        Restore system from backup
        
        Args:
            backup_path: Path to backup file
            confirm_restore: Confirmation flag for safety
            
        Returns:
            True if restore successful
        """
        
        if not confirm_restore:
            self.logger.warning("⚠️ Restore operation requires confirmation flag")
            return False
        
        try:
            self.logger.info(f"🔄 Starting system restore from: {backup_path}")
            
            # Clear current data (with backup)
            temp_backup = self.backup_system("./temp_restore_backup")
            
            # Clear current system state
            with self._video_lock:
                self.videos.clear()
            
            with self._kg_lock:
                self.knowledge_graphs.clear()
            
            # Import from backup
            success = self.import_system_data(
                import_path=backup_path,
                merge_mode=False,  # Replace mode for restore
                selective_import={
                    'videos': True,
                    'knowledge_graphs': True,
                    'metadata': True,
                    'models': True
                }
            )
            
            if success:
                self.logger.info(f"✅ System restored successfully from: {backup_path}")
                
                # Clean up temporary backup
                if temp_backup and os.path.exists(temp_backup):
                    os.remove(temp_backup)
                
                return True
            else:
                # Restore failed, recover from temp backup
                self.logger.error("❌ Restore failed, recovering from temporary backup...")
                self.import_system_data(temp_backup, merge_mode=False)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error during system restore: {e}")
            return False


    # ==================== CONTEXT MANAGERS ====================
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit với cleanup"""
        try:
            self.logger.info("🔄 VideoRAG Manager context exit - performing cleanup...")
            
            # Quick cleanup
            self.cleanup_system(cleanup_temp_files=True, max_age_hours=1)
            
            # Stop distributed processing if running
            if self.distributed_coordinator:
                self.stop_distributed_processing()
            
            # Unload models to free memory
            if self.model_manager:
                self.model_manager.unload_all_models()
            
        except Exception as e:
            self.logger.error(f"❌ Error during context manager cleanup: {e}")
    
    def __del__(self):
        """Destructor với cleanup"""
        try:
            if hasattr(self, 'logger'):
                self.logger.info("🔄 VideoRAG Manager destructor cleanup...")
            
            # Quick cleanup
            if hasattr(self, 'video_processor'):
                self.video_processor.cleanup_temp_files(max_age_hours=1)
            
            # GPU memory cleanup
            if self._check_torch_available():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except:
            pass


# ==================== GLOBAL INSTANCE MANAGEMENT ====================

# Global video RAG manager instance
_global_video_rag_manager = None

def get_video_rag_manager(
    config_path: Optional[str] = None,
    auto_load_models: bool = True,
    enable_distributed: bool = False
) -> VideoRAGManager:
    """
    Get global VideoRAG manager instance (singleton pattern)
    
    Args:
        config_path: Path to configuration file (only used first time)
        auto_load_models: Auto load models (only used first time)
        enable_distributed: Enable distributed processing (only used first time)
        
    Returns:
        VideoRAGManager instance
    """
    global _global_video_rag_manager
    if _global_video_rag_manager is None:
        _global_video_rag_manager = VideoRAGManager(
            config_path=config_path,
            auto_load_models=auto_load_models,
            enable_distributed=enable_distributed
        )
    return _global_video_rag_manager

def reset_video_rag_manager():
    """Reset global VideoRAG manager instance"""
    global _global_video_rag_manager
    if _global_video_rag_manager:
        del _global_video_rag_manager
    _global_video_rag_manager = None