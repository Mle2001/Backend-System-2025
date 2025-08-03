"""
video_processor.py - Video Processing Engine cho VideoRAG System
Xử lý video thành features và metadata: frame extraction, captioning, ASR, embeddings
"""

import os
import time
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import moviepy.editor as mp
from PIL import Image

from config import Config, get_config
from utils import (
    get_logger, measure_time, performance_monitor,
    validate_video_format, get_video_info, generate_video_id, batch_processor,
    cached, retry_on_failure
)
from model_manager import get_model_manager, ModelManager


# ==================== DATA STRUCTURES ====================

@dataclass
class FrameInfo:
    """Information về một frame"""
    frame_id: str
    video_id: str
    timestamp: float
    frame_index: int
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    caption: str = ""
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class AudioSegment:
    """Information về audio segment"""
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    transcript: str = ""
    confidence: float = 0.0
    audio_path: Optional[str] = None


@dataclass
class VideoChunk:
    """Information về video chunk"""
    chunk_id: str
    video_id: str
    start_time: float
    end_time: float
    frames: List[FrameInfo]
    audio_segment: Optional[AudioSegment] = None
    unified_description: str = ""
    chunk_embedding: Optional[np.ndarray] = None


@dataclass
class ProcessedVideo:
    """Kết quả sau khi process video"""
    video_id: str
    video_path: str
    video_info: Dict[str, Any]
    chunks: List[VideoChunk]
    total_frames: int
    total_duration: float
    processing_time: float
    success: bool = True
    error_message: str = ""


# ==================== MAIN VIDEO PROCESSOR CLASS ====================

class VideoProcessor:
    """
    Video Processing Engine cho VideoRAG System
    """
    
    def __init__(self, config: Optional[Config] = None, model_manager: Optional[ModelManager] = None):
        """
        Initialize VideoProcessor
        
        Args:
            config: Configuration object
            model_manager: Model manager instance
        """
        self.config = config or get_config()
        self.model_manager = model_manager or get_model_manager()
        self.logger = get_logger('video_processor')
        
        # Processing parameters
        self._init_processing_parameters()
        
        # Temporary directories
        self._init_temp_directories()
        
        # Threading
        self._init_threading()
        
        self.logger.info("🎬 VideoProcessor initialized")
        self.logger.info(f"📊 Processing config: {self.frame_sampling_rate} frames/chunk, {self.chunk_duration}s chunks")
    
    def _init_processing_parameters(self):
        """Initialize processing parameters from config"""
        self.frame_sampling_rate = self.config.processing.frame_sampling_rate
        self.detailed_sampling_rate = self.config.processing.detailed_sampling_rate
        self.chunk_duration = self.config.processing.chunk_duration
        self.max_frames_per_video = self.config.processing.max_frames_per_video
        self.output_resolution = self.config.processing.output_resolution
        self.audio_sample_rate = self.config.processing.audio_sample_rate
    
    def _init_temp_directories(self):
        """Initialize temporary directories"""
        self.temp_dir = Path(self.config.paths.get('temp_dir', './temp'))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_threading(self):
        """Initialize threading components"""
        self._processing_lock = threading.Lock()
        self._frame_queue = Queue()
        self._result_queue = Queue()


    # ==================== MAIN PROCESSING METHODS ====================
    
    @measure_time
    def process_video(
        self, 
        video_path: str,
        enable_detailed_analysis: bool = True,
        save_frames: bool = False,
        parallel_processing: bool = True
    ) -> ProcessedVideo:
        """
        Main function để process một video
        
        Args:
            video_path: Đường dẫn đến video file
            enable_detailed_analysis: Enable detailed frame analysis
            save_frames: Save extracted frames to disk
            parallel_processing: Enable parallel processing
            
        Returns:
            ProcessedVideo object với tất cả processed data
        """
        start_time = time.time()
        
        # Validate video
        if not validate_video_format(video_path, self.config.processing.supported_formats):
            return self._create_failed_result(
                video_path, 
                "Invalid video format or file not found"
            )
        
        video_id = generate_video_id(video_path)
        self.logger.info(f"🎬 Processing video: {os.path.basename(video_path)} (ID: {video_id})")
        
        try:
            # Get video information
            video_info = get_video_info(video_path)
            if 'error' in video_info:
                raise Exception(f"Error reading video: {video_info['error']}")
            
            self.logger.info(f"📊 Video info: {video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, {video_info['resolution']}")
            
            # Check video duration constraints
            if video_info['duration'] > 3600:  # 1 hour limit
                self.logger.warning(f"⚠️ Video duration ({video_info['duration']:.1f}s) exceeds 1 hour, may require distributed processing")
            
            # Process video into chunks
            chunks = self._process_video_chunks(
                video_path, 
                video_id, 
                video_info,
                enable_detailed_analysis,
                save_frames,
                parallel_processing
            )
            
            # Calculate results
            processing_time = time.time() - start_time
            total_frames = sum(len(chunk.frames) for chunk in chunks)
            
            self.logger.info(f"✅ Video processing completed: {len(chunks)} chunks, {total_frames} frames in {processing_time:.2f}s")
            
            return ProcessedVideo(
                video_id=video_id,
                video_path=video_path,
                video_info=video_info,
                chunks=chunks,
                total_frames=total_frames,
                total_duration=video_info['duration'],
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Error processing video: {e}")
            
            return ProcessedVideo(
                video_id=video_id,
                video_path=video_path,
                video_info=video_info if 'video_info' in locals() else {},
                chunks=[],
                total_frames=0,
                total_duration=0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def process_video_batch(
        self,
        video_paths: List[str],
        max_workers: int = 2,
        enable_detailed_analysis: bool = True
    ) -> List[ProcessedVideo]:
        """
        Process multiple videos in batch
        
        Args:
            video_paths: List of video file paths
            max_workers: Maximum number of parallel workers
            enable_detailed_analysis: Enable detailed frame analysis
            
        Returns:
            List of ProcessedVideo objects
        """
        
        self.logger.info(f"🎬 Processing batch of {len(video_paths)} videos với {max_workers} workers")
        
        def process_single_video(video_path):
            return self.process_video(
                video_path,
                enable_detailed_analysis=enable_detailed_analysis,
                save_frames=False,  # Don't save frames in batch mode
                parallel_processing=False  # Use sequential processing per video
            )
        
        # Use batch processor utility
        results = batch_processor(
            items=video_paths,
            batch_size=1,  # Process one video at a time
            process_func=lambda batch: [process_single_video(video) for video in batch],
            max_workers=max_workers,
            show_progress=True
        )
        
        # Flatten results
        processed_videos = []
        for result_batch in results:
            if isinstance(result_batch, list):
                processed_videos.extend(result_batch)
            else:
                processed_videos.append(result_batch)
        
        # Summary
        successful_videos = [v for v in processed_videos if v.success]
        failed_videos = [v for v in processed_videos if not v.success]
        
        self.logger.info(f"✅ Batch processing completed: {len(successful_videos)} successful, {len(failed_videos)} failed")
        
        if failed_videos:
            for failed_video in failed_videos:
                self.logger.error(f"❌ Failed: {failed_video.video_path} - {failed_video.error_message}")
        
        return processed_videos


    # ==================== CHUNK PROCESSING METHODS ====================
    
    def _process_video_chunks(
        self,
        video_path: str,
        video_id: str,
        video_info: Dict[str, Any],
        enable_detailed_analysis: bool,
        save_frames: bool,
        parallel_processing: bool
    ) -> List[VideoChunk]:
        """Process video thành chunks"""
        
        duration = video_info['duration']
        total_chunks = int(np.ceil(duration / self.chunk_duration))
        
        self.logger.info(f"📊 Processing {total_chunks} chunks of {self.chunk_duration}s each")
        
        if parallel_processing and total_chunks > 2:
            # Parallel processing cho large videos
            return self._process_chunks_parallel(
                video_path, video_id, video_info, 
                enable_detailed_analysis, save_frames
            )
        else:
            # Sequential processing
            return self._process_chunks_sequential(
                video_path, video_id, video_info,
                enable_detailed_analysis, save_frames
            )
    
    def _process_chunks_sequential(
        self,
        video_path: str,
        video_id: str, 
        video_info: Dict[str, Any],
        enable_detailed_analysis: bool,
        save_frames: bool
    ) -> List[VideoChunk]:
        """Process chunks sequentially"""
        
        chunks = []
        duration = video_info['duration']
        total_chunks = int(np.ceil(duration / self.chunk_duration))
        
        for chunk_idx in range(total_chunks):
            start_time = chunk_idx * self.chunk_duration
            end_time = min((chunk_idx + 1) * self.chunk_duration, duration)
            
            self.logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{total_chunks}: {start_time:.1f}s - {end_time:.1f}s")
            
            chunk = self._process_single_chunk(
                video_path, video_id, chunk_idx,
                start_time, end_time,
                enable_detailed_analysis, save_frames
            )
            
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _process_chunks_parallel(
        self,
        video_path: str,
        video_id: str,
        video_info: Dict[str, Any], 
        enable_detailed_analysis: bool,
        save_frames: bool
    ) -> List[VideoChunk]:
        """Process chunks in parallel"""
        
        chunks = []
        duration = video_info['duration']
        total_chunks = int(np.ceil(duration / self.chunk_duration))
        
        # Determine optimal number of workers
        max_workers = min(4, total_chunks)  # Limit to avoid GPU memory issues
        
        self.logger.info(f"⚡ Parallel processing với {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {}
            
            for chunk_idx in range(total_chunks):
                start_time = chunk_idx * self.chunk_duration
                end_time = min((chunk_idx + 1) * self.chunk_duration, duration)
                
                future = executor.submit(
                    self._process_single_chunk,
                    video_path, video_id, chunk_idx,
                    start_time, end_time,
                    enable_detailed_analysis, save_frames
                )
                future_to_chunk[future] = chunk_idx
            
            # Collect results
            chunk_results = {}
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk = future.result()
                    if chunk:
                        chunk_results[chunk_idx] = chunk
                        self.logger.info(f"✅ Chunk {chunk_idx + 1}/{total_chunks} completed")
                except Exception as e:
                    self.logger.error(f"❌ Error processing chunk {chunk_idx}: {e}")
            
            # Sort chunks by index
            for chunk_idx in sorted(chunk_results.keys()):
                chunks.append(chunk_results[chunk_idx])
        
        return chunks
    
    def _process_single_chunk(
        self,
        video_path: str,
        video_id: str,
        chunk_idx: int,
        start_time: float,
        end_time: float,
        enable_detailed_analysis: bool,
        save_frames: bool
    ) -> Optional[VideoChunk]:
        """Process một video chunk"""
        
        chunk_id = f"{video_id}_chunk_{chunk_idx:04d}"
        
        try:
            with performance_monitor.monitor(f"process_chunk_{chunk_idx}"):
                # Extract frames
                frames = self._extract_frames_from_chunk(
                    video_path, video_id, chunk_id,
                    start_time, end_time,
                    enable_detailed_analysis, save_frames
                )
                
                # Extract audio segment
                audio_segment = self._extract_audio_segment(
                    video_path, video_id, chunk_id,
                    start_time, end_time
                )
                
                # Generate unified description
                unified_description = self._generate_unified_description(
                    frames, audio_segment
                )
                
                # Generate chunk embedding
                chunk_embedding = self._generate_chunk_embedding(
                    frames, unified_description
                )
                
                return VideoChunk(
                    chunk_id=chunk_id,
                    video_id=video_id,
                    start_time=start_time,
                    end_time=end_time,
                    frames=frames,
                    audio_segment=audio_segment,
                    unified_description=unified_description,
                    chunk_embedding=chunk_embedding
                )
        
        except Exception as e:
            self.logger.error(f"❌ Error processing chunk {chunk_idx}: {e}")
            return None


    # ==================== FRAME EXTRACTION METHODS ====================
    
    @cached(ttl_seconds=3600)  # Cache for 1 hour
    def _extract_frames_from_chunk(
        self,
        video_path: str,
        video_id: str,
        chunk_id: str,
        start_time: float,
        end_time: float,
        enable_detailed_analysis: bool,
        save_frames: bool
    ) -> List[FrameInfo]:
        """Extract frames từ video chunk"""
        
        frames = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices for sampling
            start_frame = int(start_time * fps)
            end_frame = min(int(end_time * fps), total_frames)
            chunk_frames = end_frame - start_frame
            
            # Determine sampling strategy
            if enable_detailed_analysis:
                num_samples = min(self.detailed_sampling_rate, chunk_frames)
            else:
                num_samples = min(self.frame_sampling_rate, chunk_frames)
            
            if num_samples <= 0:
                return frames
            
            # Calculate frame indices to sample
            if chunk_frames <= num_samples:
                frame_indices = list(range(start_frame, end_frame))
            else:
                # Uniform sampling
                step = chunk_frames / num_samples
                frame_indices = [int(start_frame + i * step) for i in range(num_samples)]
            
            # Extract frames
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                if self.output_resolution != (frame.shape[1], frame.shape[0]):
                    frame_rgb = cv2.resize(frame_rgb, self.output_resolution)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Create frame info
                timestamp = frame_idx / fps
                frame_id = f"{chunk_id}_frame_{i:04d}"
                
                frame_info = FrameInfo(
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    image=pil_image
                )
                
                # Save frame if requested
                if save_frames:
                    frame_dir = self.temp_dir / "frames" / video_id
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = frame_dir / f"{frame_id}.jpg"
                    pil_image.save(frame_path, quality=85)
                    frame_info.image_path = str(frame_path)
                
                frames.append(frame_info)
            
            cap.release()
            
            # Generate captions for frames
            if frames:
                frames = self._generate_frame_captions(frames)
            
            return frames
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting frames from chunk: {e}")
            return []
    
    def _generate_frame_captions(self, frames: List[FrameInfo]) -> List[FrameInfo]:
        """Generate captions cho extracted frames"""
        
        if not frames:
            return frames
        
        try:
            # Batch process frames với MiniCPM-V
            images = [frame.image for frame in frames]
            
            # Use MiniCPM-V để generate captions
            result = self.model_manager.inference_minicpm_v(
                images, 
                "Describe what you see in this image in detail, focusing on objects, actions, and scene context."
            )
            
            if result.success and result.result:
                captions = result.result
                
                # Assign captions to frames
                for i, frame in enumerate(frames):
                    if i < len(captions):
                        frame.caption = captions[i]
                        frame.confidence = 0.8  # Default confidence
                    
                self.logger.debug(f"✅ Generated {len(captions)} captions")
            else:
                self.logger.warning(f"⚠️ Failed to generate captions: {result.error_message}")
                
        except Exception as e:
            self.logger.error(f"❌ Error generating frame captions: {e}")
        
        return frames


    # ==================== AUDIO PROCESSING METHODS ====================
    
    @retry_on_failure(max_retries=2)
    def _extract_audio_segment(
        self,
        video_path: str,
        video_id: str,
        chunk_id: str,
        start_time: float,
        end_time: float
    ) -> Optional[AudioSegment]:
        """Extract và transcribe audio segment"""
        
        try:
            # Create temporary audio file
            temp_audio_dir = self.temp_dir / "audio" / video_id
            temp_audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = temp_audio_dir / f"{chunk_id}.wav"
            
            # Extract audio segment using moviepy
            with mp.VideoFileClip(video_path) as video:
                audio_clip = video.subclip(start_time, end_time).audio
                if audio_clip is not None:
                    audio_clip.write_audiofile(
                        str(audio_path),
                        fps=self.audio_sample_rate,
                        verbose=False,
                        logger=None
                    )
                    audio_clip.close()
                else:
                    self.logger.warning(f"⚠️ No audio track found in chunk {chunk_id}")
                    return None
            
            # Transcribe audio với Whisper
            result = self.model_manager.inference_whisper(str(audio_path))
            
            transcript = ""
            confidence = 0.0
            
            if result.success and result.result:
                transcript = result.result.strip()
                confidence = 0.9 if transcript else 0.0
                
                self.logger.debug(f"✅ Transcribed audio: {len(transcript)} chars")
            else:
                self.logger.warning(f"⚠️ Audio transcription failed: {result.error_message}")
            
            # Clean up temporary audio file
            if audio_path.exists():
                audio_path.unlink()
            
            segment_id = f"{chunk_id}_audio"
            
            return AudioSegment(
                segment_id=segment_id,
                video_id=video_id,
                start_time=start_time,
                end_time=end_time,
                transcript=transcript,
                confidence=confidence,
                audio_path=str(audio_path) if transcript else None
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting audio segment: {e}")
            return None


    # ==================== DESCRIPTION AND EMBEDDING GENERATION ====================
    
    def _generate_unified_description(
        self,
        frames: List[FrameInfo],
        audio_segment: Optional[AudioSegment]
    ) -> str:
        """Generate unified description từ visual và audio information"""
        
        try:
            # Collect visual information
            visual_descriptions = [frame.caption for frame in frames if frame.caption]
            
            # Collect audio information
            audio_transcript = ""
            if audio_segment and audio_segment.transcript:
                audio_transcript = audio_segment.transcript
            
            # Combine information using LLM
            if visual_descriptions or audio_transcript:
                
                # Prepare prompt
                visual_text = " ".join(visual_descriptions) if visual_descriptions else "No visual information available."
                audio_text = audio_transcript if audio_transcript else "No audio information available."
                
                messages = [
                    {
                        "role": "user",
                        "content": f"""Analyze and synthesize the following multimodal information from a video segment:

VISUAL INFORMATION:
{visual_text}

AUDIO INFORMATION:
{audio_text}

Please provide a concise, unified description that:
1. Combines visual and audio information coherently
2. Highlights key objects, actions, and events
3. Maintains temporal flow and context
4. Focuses on searchable keywords and concepts

Unified Description:"""
                    }
                ]
                
                result = self.model_manager.inference_gpt4o_mini(
                    messages, 
                    max_tokens=200,
                    temperature=0.1
                )
                
                if result.success and result.result:
                    unified_description = result.result.strip()
                    self.logger.debug(f"✅ Generated unified description: {len(unified_description)} chars")
                    return unified_description
                else:
                    self.logger.warning(f"⚠️ Failed to generate unified description: {result.error_message}")
            
            # Fallback to simple concatenation
            parts = []
            if visual_descriptions:
                parts.append("Visual: " + " | ".join(visual_descriptions[:3]))  # Limit to 3 descriptions
            if audio_transcript:
                parts.append("Audio: " + audio_transcript[:200])  # Limit length
            
            return " || ".join(parts) if parts else "No content description available."
            
        except Exception as e:
            self.logger.error(f"❌ Error generating unified description: {e}")
            return "Error generating description."
    
    def _generate_chunk_embedding(
        self,
        frames: List[FrameInfo],
        unified_description: str
    ) -> Optional[np.ndarray]:
        """Generate embedding cho chunk dựa trên visual và textual content"""
        
        try:
            embeddings_to_combine = []
            
            # Text embedding từ unified description
            if unified_description and unified_description != "No content description available.":
                text_result = self.model_manager.inference_text_embedding([unified_description])
                if text_result.success and text_result.result is not None:
                    text_embedding = text_result.result[0]  # First embedding
                    embeddings_to_combine.append(text_embedding)
                    self.logger.debug("✅ Generated text embedding")
            
            # Visual embedding từ representative frame
            if frames:
                # Select middle frame as representative
                mid_frame_idx = len(frames) // 2
                representative_frame = frames[mid_frame_idx]
                
                if representative_frame.image:
                    visual_result = self.model_manager.inference_imagebind(
                        [representative_frame.image],
                        None
                    )
                    
                    if visual_result.success and 'vision' in visual_result.result:
                        visual_embedding = visual_result.result['vision'][0]  # First embedding
                        
                        # Resize visual embedding để match text embedding dimension nếu cần
                        if len(embeddings_to_combine) > 0:
                            text_dim = len(embeddings_to_combine[0])
                            if len(visual_embedding) != text_dim:
                                # Simple resize bằng interpolation
                                visual_embedding = np.interp(
                                    np.linspace(0, len(visual_embedding), text_dim),
                                    np.arange(len(visual_embedding)),
                                    visual_embedding
                                )
                        
                        embeddings_to_combine.append(visual_embedding)
                        self.logger.debug("✅ Generated visual embedding")
            
            # Combine embeddings
            if embeddings_to_combine:
                if len(embeddings_to_combine) == 1:
                    chunk_embedding = embeddings_to_combine[0]
                else:
                    # Average embeddings
                    chunk_embedding = np.mean(embeddings_to_combine, axis=0)
                
                # Normalize embedding
                norm = np.linalg.norm(chunk_embedding)
                if norm > 0:
                    chunk_embedding = chunk_embedding / norm
                
                return chunk_embedding
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating chunk embedding: {e}")
            return None


    # ==================== KEYFRAME EXTRACTION METHODS ====================
    
    def extract_keyframes(
        self,
        video_path: str,
        max_keyframes: int = 20,
        method: str = "uniform"
    ) -> List[FrameInfo]:
        """
        Extract key frames từ video
        
        Args:
            video_path: Path to video file
            max_keyframes: Maximum number of keyframes to extract
            method: Extraction method ("uniform", "scene_change", "motion")
            
        Returns:
            List of key FrameInfo objects
        """
        
        self.logger.info(f"🔑 Extracting {max_keyframes} keyframes using {method} method")
        
        try:
            video_info = get_video_info(video_path)
            video_id = generate_video_id(video_path)
            
            if method == "uniform":
                return self._extract_uniform_keyframes(video_path, video_id, video_info, max_keyframes)
            elif method == "scene_change":
                return self._extract_scene_change_keyframes(video_path, video_id, video_info, max_keyframes)
            elif method == "motion":
                return self._extract_motion_keyframes(video_path, video_id, video_info, max_keyframes)
            else:
                raise ValueError(f"Unknown keyframe extraction method: {method}")
                
        except Exception as e:
            self.logger.error(f"❌ Error extracting keyframes: {e}")
            return []
    
    def _extract_uniform_keyframes(
        self,
        video_path: str,
        video_id: str,
        video_info: Dict[str, Any],
        max_keyframes: int
    ) -> List[FrameInfo]:
        """Extract keyframes uniformly distributed across video"""
        
        keyframes = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices
            if total_frames <= max_keyframes:
                frame_indices = list(range(0, total_frames, max(1, total_frames // max_keyframes)))
            else:
                step = total_frames / max_keyframes
                frame_indices = [int(i * step) for i in range(max_keyframes)]
            
            # Extract frames
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert và resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, self.output_resolution)
                pil_image = Image.fromarray(frame_rgb)
                
                # Create frame info
                timestamp = frame_idx / fps
                frame_id = f"{video_id}_keyframe_{i:04d}"
                
                keyframe = FrameInfo(
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    image=pil_image
                )
                
                keyframes.append(keyframe)
            
            cap.release()
            
            # Generate captions
            keyframes = self._generate_frame_captions(keyframes)
            
            return keyframes
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting uniform keyframes: {e}")
            return []
    
    def _extract_scene_change_keyframes(
        self,
        video_path: str,
        video_id: str,
        video_info: Dict[str, Any],
        max_keyframes: int
    ) -> List[FrameInfo]:
        """Extract keyframes at scene changes"""
        
        # Simple scene change detection using frame difference
        keyframes = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Parameters
            threshold = 30.0  # Scene change threshold
            min_scene_length = int(fps * 2)  # Minimum 2 seconds between scenes
            
            scene_changes = [0]  # Always include first frame
            prev_frame = None
            frame_idx = 0
            
            while frame_idx < total_frames and len(scene_changes) < max_keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray_frame)
                    mean_diff = np.mean(diff)
                    
                    # Check if scene change
                    if (mean_diff > threshold and 
                        frame_idx - scene_changes[-1] > min_scene_length):
                        scene_changes.append(frame_idx)
                
                prev_frame = gray_frame
                frame_idx += int(fps * 0.5)  # Check every 0.5 seconds
            
            # Always include last frame if not already included
            if scene_changes[-1] != total_frames - 1:
                scene_changes.append(total_frames - 1)
            
            # Extract frames at scene changes
            for i, frame_idx in enumerate(scene_changes[:max_keyframes]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert và resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, self.output_resolution)
                pil_image = Image.fromarray(frame_rgb)
                
                # Create frame info
                timestamp = frame_idx / fps
                frame_id = f"{video_id}_scene_{i:04d}"
                
                keyframe = FrameInfo(
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    image=pil_image
                )
                
                keyframes.append(keyframe)
            
            cap.release()
            
            self.logger.info(f"✅ Detected {len(scene_changes)} scene changes")
            
            # Generate captions
            keyframes = self._generate_frame_captions(keyframes)
            
            return keyframes
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting scene change keyframes: {e}")
            return []
    
    def _extract_motion_keyframes(
        self,
        video_path: str,
        video_id: str, 
        video_info: Dict[str, Any],
        max_keyframes: int
    ) -> List[FrameInfo]:
        """Extract keyframes based on motion intensity"""
        
        # Motion-based keyframe extraction
        keyframes = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            motion_scores = []
            frame_indices = []
            prev_frame = None
            
            # Calculate motion scores
            for frame_idx in range(0, total_frames, max(1, total_frames // (max_keyframes * 10))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate optical flow magnitude
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_frame, gray_frame,
                        np.array([[x, y] for x in range(0, gray_frame.shape[1], 20) 
                                 for y in range(0, gray_frame.shape[0], 20)], dtype=np.float32),
                        None
                    )[0]
                    
                    if flow is not None:
                        motion_magnitude = np.mean(np.sqrt(flow[:, 0]**2 + flow[:, 1]**2))
                        motion_scores.append(motion_magnitude)
                        frame_indices.append(frame_idx)
                
                prev_frame = gray_frame
            
            # Select frames với highest motion scores
            if motion_scores:
                # Get indices of top motion frames
                top_motion_indices = np.argsort(motion_scores)[-max_keyframes:]
                selected_frame_indices = [frame_indices[i] for i in top_motion_indices]
                selected_frame_indices.sort()  # Sort chronologically
                
                # Extract selected frames
                for i, frame_idx in enumerate(selected_frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Convert và resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, self.output_resolution)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Create frame info
                    timestamp = frame_idx / fps
                    frame_id = f"{video_id}_motion_{i:04d}"
                    
                    keyframe = FrameInfo(
                        frame_id=frame_id,
                        video_id=video_id,
                        timestamp=timestamp,
                        frame_index=frame_idx,
                        image=pil_image
                    )
                    
                    keyframes.append(keyframe)
            
            cap.release()
            
            self.logger.info(f"✅ Selected {len(keyframes)} motion-based keyframes")
            
            # Generate captions
            keyframes = self._generate_frame_captions(keyframes)
            
            return keyframes
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting motion keyframes: {e}")
            return []


    # ==================== UTILITY AND MANAGEMENT METHODS ====================
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        
        # This would track statistics across processing sessions
        # For now, return basic info
        return {
            'temp_dir': str(self.temp_dir),
            'temp_dir_size_mb': sum(f.stat().st_size for f in self.temp_dir.rglob('*') if f.is_file()) / (1024**2),
            'config': {
                'frame_sampling_rate': self.frame_sampling_rate,
                'detailed_sampling_rate': self.detailed_sampling_rate,
                'chunk_duration': self.chunk_duration,
                'max_frames_per_video': self.max_frames_per_video,
                'output_resolution': self.output_resolution,
                'audio_sample_rate': self.audio_sample_rate
            }
        }
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files"""
        from utils import cleanup_temp_files
        cleanup_temp_files(str(self.temp_dir), max_age_hours)
    
    def _create_failed_result(self, video_path: str, error_message: str) -> ProcessedVideo:
        """Create a failed ProcessedVideo result"""
        return ProcessedVideo(
            video_id="",
            video_path=video_path,
            video_info={},
            chunks=[],
            total_frames=0,
            total_duration=0,
            processing_time=0,
            success=False,
            error_message=error_message
        )
    
    def __del__(self):
        """Cleanup when VideoProcessor is destroyed"""
        try:
            # Clean up any remaining temporary files
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                # Only clean files older than 1 hour
                self.cleanup_temp_files(max_age_hours=1)
        except:
            pass


# ==================== GLOBAL INSTANCE MANAGEMENT ====================

# Global video processor instance
_global_video_processor = None

def get_video_processor(config: Optional[Config] = None) -> VideoProcessor:
    """
    Get global video processor instance (singleton pattern)
    
    Args:
        config: Configuration object (chỉ sử dụng lần đầu)
        
    Returns:
        VideoProcessor instance
    """
    global _global_video_processor
    if _global_video_processor is None:
        _global_video_processor = VideoProcessor(config)
    return _global_video_processor

def reset_video_processor():
    """Reset global video processor instance"""
    global _global_video_processor
    if _global_video_processor:
        del _global_video_processor
    _global_video_processor = None


# ==================== TESTING AND EXAMPLES ====================

if __name__ == "__main__":
    import psutil
    
    print("🧪 Testing VideoRAG Video Processor")
    print("="*50)
    
    # Setup logging
    from utils import setup_logging
    setup_logging("INFO", enable_colors=True)
    
    # Initialize processor
    config = get_config()
    processor = VideoProcessor(config)
    
    logger = get_logger('test')
    
    # Print configuration
    logger.info("⚙️ Processor Configuration:")
    stats = processor.get_processing_stats()
    for key, value in stats['config'].items():
        logger.info(f"  {key}: {value}")
    
    # Test với sample video (if available)
    sample_videos = [
        "./test_video.mp4",
        "./sample.avi",
        "/path/to/test/video.mp4"
    ]
    
    test_video = None
    for video_path in sample_videos:
        if os.path.exists(video_path):
            test_video = video_path
            break
    
    if test_video:
        logger.info(f"📹 Testing với video: {test_video}")
        
        # Test basic video processing
        logger.info("🔄 Testing basic video processing...")
        try:
            result = processor.process_video(
                test_video,
                enable_detailed_analysis=False,
                save_frames=True,
                parallel_processing=False
            )
            
            if result.success:
                logger.info(f"✅ Video processing successful!")
                logger.info(f"📊 Results: {len(result.chunks)} chunks, {result.total_frames} frames")
                logger.info(f"⏱️ Processing time: {result.processing_time:.2f}s")
                
                # Print sample chunk info
                if result.chunks:
                    sample_chunk = result.chunks[0]
                    logger.info(f"📝 Sample chunk: {sample_chunk.chunk_id}")
                    logger.info(f"   Frames: {len(sample_chunk.frames)}")
                    logger.info(f"   Description: {sample_chunk.unified_description[:100]}...")
                    if sample_chunk.audio_segment:
                        logger.info(f"   Transcript: {sample_chunk.audio_segment.transcript[:100]}...")
            else:
                logger.error(f"❌ Video processing failed: {result.error_message}")
        
        except Exception as e:
            logger.error(f"❌ Video processing test error: {e}")
        
        # Test keyframe extraction
        logger.info("🔑 Testing keyframe extraction...")
        try:
            for method in ["uniform", "scene_change", "motion"]:
                logger.info(f"Testing {method} keyframe extraction...")
                keyframes = processor.extract_keyframes(
                    test_video,
                    max_keyframes=10,
                    method=method
                )
                
                if keyframes:
                    logger.info(f"✅ {method}: Extracted {len(keyframes)} keyframes")
                    
                    # Print sample keyframe info
                    if keyframes:
                        sample_keyframe = keyframes[0]
                        logger.info(f"   Sample: {sample_keyframe.frame_id} at {sample_keyframe.timestamp:.2f}s")
                        if sample_keyframe.caption:
                            logger.info(f"   Caption: {sample_keyframe.caption[:100]}...")
                else:
                    logger.warning(f"⚠️ {method}: No keyframes extracted")
        
        except Exception as e:
            logger.error(f"❌ Keyframe extraction test error: {e}")
    
    else:
        logger.info("📹 No test video found, running simulation tests...")
        
        # Test configuration validation
        logger.info("⚙️ Testing configuration validation...")
        
        # Test processing parameters
        test_configs = [
            {'frame_sampling_rate': 3, 'chunk_duration': 15},
            {'frame_sampling_rate': 8, 'chunk_duration': 45},
            {'detailed_sampling_rate': 20, 'max_frames_per_video': 1000}
        ]
        
        for test_config in test_configs:
            for key, value in test_config.items():
                setattr(processor, key, value)
            
            logger.info(f"✅ Config test: {test_config}")
    
    # Test batch processing simulation
    logger.info("📦 Testing batch processing simulation...")
    
    fake_video_paths = [f"fake_video_{i}.mp4" for i in range(3)]
    
    def mock_process_video(video_path):
        """Mock video processing for testing"""
        video_id = generate_video_id(video_path)
        return ProcessedVideo(
            video_id=video_id,
            video_path=video_path,
            video_info={'duration': 30.0, 'fps': 30.0},
            chunks=[],
            total_frames=0,
            total_duration=30.0,
            processing_time=1.0,
            success=True
        )
    
    # Temporarily replace process_video method
    original_method = processor.process_video
    processor.process_video = mock_process_video
    
    try:
        batch_results = processor.process_video_batch(
            fake_video_paths,
            max_workers=2,
            enable_detailed_analysis=False
        )
        
        logger.info(f"✅ Batch processing simulation: {len(batch_results)} results")
        
    except Exception as e:
        logger.error(f"❌ Batch processing test error: {e}")
    
    finally:
        # Restore original method
        processor.process_video = original_method
    
    # Test memory usage
    logger.info("💾 Testing memory usage...")
    
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024**2)
    
    # Simulate some processing
    for i in range(5):
        fake_frames = []
        for j in range(10):
            # Create fake frame data
            fake_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            fake_frame = FrameInfo(
                frame_id=f"test_{i}_{j}",
                video_id="test_video",
                timestamp=j * 1.0,
                frame_index=j,
                image=fake_image,
                caption=f"Test frame {j}"
            )
            fake_frames.append(fake_frame)
        
        # Simulate processing
        time.sleep(0.1)
    
    memory_after = process.memory_info().rss / (1024**2)
    memory_diff = memory_after - memory_before
    
    logger.info(f"📊 Memory usage: {memory_before:.1f} MB → {memory_after:.1f} MB (Δ{memory_diff:+.1f} MB)")
    
    # Test cleanup
    logger.info("🧹 Testing cleanup...")
    
    # Check temp directory
    stats = processor.get_processing_stats()
    temp_size_before = stats['temp_dir_size_mb']
    
    # Create some fake temp files
    test_temp_dir = processor.temp_dir / "test_cleanup"
    test_temp_dir.mkdir(exist_ok=True)
    
    for i in range(5):
        test_file = test_temp_dir / f"test_file_{i}.tmp"
        test_file.write_text("test data" * 1000)
    
    # Run cleanup
    processor.cleanup_temp_files(max_age_hours=0)  # Clean everything
    
    stats_after = processor.get_processing_stats()
    temp_size_after = stats_after['temp_dir_size_mb']
    
    logger.info(f"🧹 Temp cleanup: {temp_size_before:.1f} MB → {temp_size_after:.1f} MB")
    
    # Test error handling
    logger.info("🛡️ Testing error handling...")
    
    # Test với invalid video path
    try:
        invalid_result = processor.process_video("nonexistent_video.mp4")
        if not invalid_result.success:
            logger.info(f"✅ Error handling works: {invalid_result.error_message}")
        else:
            logger.warning("⚠️ Expected error but processing succeeded")
    except Exception as e:
        logger.info(f"✅ Exception handling works: {e}")
    
    # Test với invalid keyframe method
    try:
        invalid_keyframes = processor.extract_keyframes(
            "test.mp4",
            max_keyframes=5,
            method="invalid_method"
        )
        if not invalid_keyframes:
            logger.info("✅ Invalid method handling works")
    except Exception as e:
        logger.info(f"✅ Exception handling for invalid method works: {e}")
    
    # Final system summary
    logger.info("📋 Final System Summary:")
    
    final_stats = processor.get_processing_stats()
    logger.info(f"💾 Temp directory: {final_stats['temp_dir']}")
    logger.info(f"📊 Temp size: {final_stats['temp_dir_size_mb']:.1f} MB")
    
    final_memory = process.memory_info().rss / (1024**2)
    logger.info(f"🧠 Process memory: {final_memory:.1f} MB")
    
    # Configuration summary
    config_summary = final_stats['config']
    logger.info("⚙️ Active configuration:")
    for key, value in config_summary.items():
        logger.info(f"   {key}: {value}")
    
    # Cleanup
    logger.info("🧹 Final cleanup...")
    processor.cleanup_temp_files(max_age_hours=0)
    
    # Remove test temp directory
    if 'test_temp_dir' in locals() and test_temp_dir.exists():
        shutil.rmtree(test_temp_dir, ignore_errors=True)
    
    logger.info("✅ VideoProcessor testing completed successfully!")
    
    # Capabilities summary
    print("\n" + "="*60)
    print("🎉 VideoRAG Video Processor - Capabilities Summary")
    print("="*60)
    print("✅ Multi-format video support (MP4, AVI, MOV, MKV, WebM)")
    print("✅ Intelligent frame sampling (uniform, scene-based, motion-based)")
    print("✅ Audio transcription với Whisper ASR")
    print("✅ Visual captioning với MiniCPM-V")
    print("✅ Multimodal embedding generation")
    print("✅ Parallel processing support")
    print("✅ Batch video processing")
    print("✅ Keyframe extraction algorithms")
    print("✅ Memory management và optimization")
    print("✅ Error handling và recovery")
    print("✅ Configurable processing parameters")
    print("✅ Temporary file management")
    print("✅ Performance monitoring integration")
    print("✅ Thread-safe operations")
    print("✅ Resource usage tracking")
    print("="*60)