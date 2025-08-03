"""
chat_interface.py - LLM Chat Interface cho VideoRAG System
Tương tác với LLM về frame content, multi-frame reasoning, conversation management
"""

import os
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np
from PIL import Image

from config import Config, get_config
from utils import (
    get_logger, measure_time, performance_monitor,
    safe_execute, cached, retry_on_failure, normalize_text,
    safe_json_loads, safe_json_dumps
)
from model_manager import get_model_manager, ModelManager, InferenceResult
from video_processor import FrameInfo, VideoChunk, ProcessedVideo


# ==================== DATA STRUCTURES ====================

@dataclass
class ChatMessage:
    """Represent một message trong conversation"""
    message_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_references: List[str] = field(default_factory=list)  # Frame IDs referenced
    confidence: float = 1.0


@dataclass
class ConversationContext:
    """Context cho conversation session"""
    conversation_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    active_frames: List[FrameInfo] = field(default_factory=list)
    active_videos: List[str] = field(default_factory=list)  # Video IDs
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameAnalysisResult:
    """Result từ frame analysis"""
    frame_id: str
    analysis_type: str  # "description", "question_answer", "comparison", etc.
    result: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response từ chat interface"""
    success: bool
    response_text: str
    conversation_id: str
    message_id: str = ""
    frame_analyses: List[FrameAnalysisResult] = field(default_factory=list)
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== MAIN CHAT INTERFACE CLASS ====================

class ChatInterface:
    """
    LLM Chat Interface cho VideoRAG System
    """
    
    def __init__(self, config: Optional[Config] = None, model_manager: Optional[ModelManager] = None):
        """
        Initialize ChatInterface
        
        Args:
            config: Configuration object
            model_manager: Model manager instance
        """
        self.config = config or get_config()
        self.model_manager = model_manager or get_model_manager()
        self.logger = get_logger('chat_interface')
        
        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        self._conversation_lock = threading.RLock()
        
        # Analysis cache
        self.analysis_cache: Dict[str, FrameAnalysisResult] = {}
        self._cache_lock = threading.RLock()
        
        # Configuration
        self._init_chat_parameters()
        
        self.logger.info("💬 ChatInterface initialized")
        self.logger.info(f"📊 Max conversation length: {self.max_conversation_length}")
    
    def _init_chat_parameters(self):
        """Initialize chat parameters"""
        self.max_conversation_length = 50  # Maximum messages to keep
        self.context_window_tokens = 8000  # Approximate token limit
        self.default_temperature = 0.3
        self.max_response_tokens = 1000
        
        # Prompt templates
        self._init_prompt_templates()
    
    def _init_prompt_templates(self):
        """Initialize prompt templates"""
        self.prompt_templates = {
            'system_prompt': """You are an AI assistant specialized in analyzing video content. You can see and understand images from video frames, and you help users analyze, discuss, and extract insights from visual content.

Your capabilities:
- Analyze individual frames in detail
- Compare multiple frames
- Track changes over time in video sequences
- Answer questions about visual content
- Provide detailed descriptions of scenes, objects, and actions
- Identify relationships between different visual elements

Guidelines:
- Be precise and factual in your descriptions
- When uncertain, express your confidence level
- Reference specific visual elements you observe
- Maintain context from previous frames when relevant
- Ask clarifying questions if needed""",
            
            'frame_analysis': """Analyze the following frame from a video:

Frame Information:
- Frame ID: {frame_id}
- Timestamp: {timestamp}
- Video: {video_id}

Previous Context: {context}

User Question: {user_question}

Please provide a detailed analysis addressing the user's question. Focus on:
1. Direct visual observations
2. Relevant details that answer the question
3. Any notable objects, actions, or scene elements
4. Confidence in your observations""",
            
            'multi_frame_comparison': """Compare and analyze the following frames from a video sequence:

{frame_details}

Context from conversation: {conversation_context}

User Question: {user_question}

Please analyze these frames considering:
1. Changes between frames over time
2. Consistent elements across frames
3. Actions or movements occurring
4. Relationships between different elements
5. Overall narrative or sequence flow""",
            
            'conversation_summary': """Summarize the following conversation about video frames:

{conversation_history}

Provide a concise summary focusing on:
1. Key insights discovered
2. Main topics discussed
3. Important frame references
4. Unresolved questions or areas for further exploration"""
        }


    # ==================== MAIN CHAT METHODS ====================
    
    def start_conversation(
        self,
        frames: List[FrameInfo],
        conversation_id: Optional[str] = None,
        initial_context: Optional[str] = None
    ) -> str:
        """
        Start a new conversation session với frames
        
        Args:
            frames: List of frames để discuss
            conversation_id: Custom conversation ID
            initial_context: Initial context for conversation
            
        Returns:
            Conversation ID
        """
        
        if not conversation_id:
            conversation_id = f"chat_{int(time.time())}_{len(self.conversations):04d}"
        
        with self._conversation_lock:
            # Create conversation context
            video_ids = list(set(frame.video_id for frame in frames))
            
            context = ConversationContext(
                conversation_id=conversation_id,
                active_frames=frames,
                active_videos=video_ids,
                metadata={
                    'initial_context': initial_context,
                    'frame_count': len(frames),
                    'video_count': len(video_ids)
                }
            )
            
            # Add system message
            system_message = ChatMessage(
                message_id=f"{conversation_id}_sys_{int(time.time())}",
                role="system",
                content=self.prompt_templates['system_prompt'],
                metadata={'type': 'system_init'}
            )
            
            context.messages.append(system_message)
            
            # Add initial context if provided
            if initial_context:
                context_message = ChatMessage(
                    message_id=f"{conversation_id}_ctx_{int(time.time())}",
                    role="system", 
                    content=f"Initial context: {initial_context}",
                    metadata={'type': 'context'}
                )
                context.messages.append(context_message)
            
            self.conversations[conversation_id] = context
        
        self.logger.info(f"💬 Started conversation {conversation_id} với {len(frames)} frames")
        return conversation_id
    
    @measure_time
    def chat_with_frames(
        self,
        conversation_id: str,
        user_message: str,
        specific_frame_ids: Optional[List[str]] = None,
        analysis_type: str = "general"
    ) -> ChatResponse:
        """
        Chat với LLM về frame content
        
        Args:
            conversation_id: ID của conversation
            user_message: User's message/question
            specific_frame_ids: Specific frames to focus on
            analysis_type: Type of analysis ("general", "detailed", "comparison")
            
        Returns:
            ChatResponse với LLM's response
        """
        start_time = time.time()
        
        try:
            # Get conversation context
            if conversation_id not in self.conversations:
                return ChatResponse(
                    success=False,
                    response_text="",
                    conversation_id=conversation_id,
                    error_message="Conversation not found",
                    processing_time=time.time() - start_time
                )
            
            with self._conversation_lock:
                context = self.conversations[conversation_id]
                
                # Determine frames to analyze
                target_frames = self._select_target_frames(
                    context, specific_frame_ids
                )
                
                if not target_frames:
                    return ChatResponse(
                        success=False,
                        response_text="",
                        conversation_id=conversation_id,
                        error_message="No frames available for analysis",
                        processing_time=time.time() - start_time
                    )
                
                # Add user message to conversation
                user_msg = ChatMessage(
                    message_id=f"{conversation_id}_user_{int(time.time())}",
                    role="user",
                    content=user_message,
                    frame_references=[f.frame_id for f in target_frames]
                )
                context.messages.append(user_msg)
                
                # Generate response based on analysis type
                if analysis_type == "comparison" and len(target_frames) > 1:
                    response = self._generate_multi_frame_response(
                        context, target_frames, user_message
                    )
                else:
                    response = self._generate_single_frame_response(
                        context, target_frames, user_message, analysis_type
                    )
                
                if response.success:
                    # Add assistant message to conversation
                    assistant_msg = ChatMessage(
                        message_id=response.message_id,
                        role="assistant",
                        content=response.response_text,
                        frame_references=[f.frame_id for f in target_frames],
                        confidence=0.9
                    )
                    context.messages.append(assistant_msg)
                    
                    # Update context
                    context.last_updated = time.time()
                    
                    # Manage conversation length
                    self._manage_conversation_length(context)
                
                response.processing_time = time.time() - start_time
                return response
                
        except Exception as e:
            self.logger.error(f"❌ Error in chat_with_frames: {e}")
            return ChatResponse(
                success=False,
                response_text="",
                conversation_id=conversation_id,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def analyze_frame_content(
        self,
        frame: FrameInfo,
        analysis_prompt: str,
        use_cache: bool = True
    ) -> FrameAnalysisResult:
        """
        Analyze specific frame content
        
        Args:
            frame: Frame to analyze
            analysis_prompt: Specific analysis prompt
            use_cache: Whether to use cached results
            
        Returns:
            FrameAnalysisResult
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{frame.frame_id}_{hash(analysis_prompt)}"
        
        if use_cache:
            with self._cache_lock:
                if cache_key in self.analysis_cache:
                    cached_result = self.analysis_cache[cache_key]
                    self.logger.debug(f"📋 Using cached analysis for {frame.frame_id}")
                    return cached_result
        
        try:
            # Prepare frame description
            frame_description = self._prepare_frame_description(frame)
            
            # Create analysis prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing video frames. Provide detailed, accurate descriptions of what you observe."
                },
                {
                    "role": "user",
                    "content": f"""Analyze this video frame:

Frame Information:
{frame_description}

Analysis Request: {analysis_prompt}

Please provide a detailed analysis focusing on the specific request while noting any other relevant visual elements."""
                }
            ]
            
            # Get LLM response
            result = self.model_manager.inference_gpt4o_mini(
                messages,
                max_tokens=self.max_response_tokens,
                temperature=self.default_temperature
            )
            
            processing_time = time.time() - start_time
            
            if result.success:
                analysis_result = FrameAnalysisResult(
                    frame_id=frame.frame_id,
                    analysis_type="content_analysis",
                    result=result.result,
                    confidence=0.9,
                    processing_time=processing_time,
                    metadata={
                        'prompt': analysis_prompt,
                        'frame_timestamp': frame.timestamp
                    }
                )
                
                # Cache result
                if use_cache:
                    with self._cache_lock:
                        self.analysis_cache[cache_key] = analysis_result
                
                return analysis_result
            else:
                return FrameAnalysisResult(
                    frame_id=frame.frame_id,
                    analysis_type="content_analysis",
                    result=f"Analysis failed: {result.error_message}",
                    confidence=0.0,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Error analyzing frame content: {e}")
            return FrameAnalysisResult(
                frame_id=frame.frame_id,
                analysis_type="content_analysis",
                result=f"Error: {str(e)}",
                confidence=0.0,
                processing_time=processing_time
            )


    # ==================== FRAME SELECTION AND ANALYSIS METHODS ====================
    
    def _select_target_frames(
        self,
        context: ConversationContext,
        specific_frame_ids: Optional[List[str]] = None
    ) -> List[FrameInfo]:
        """Select target frames for analysis"""
        
        if specific_frame_ids:
            # Filter by specific IDs
            target_frames = [
                frame for frame in context.active_frames
                if frame.frame_id in specific_frame_ids
            ]
        else:
            # Use all active frames (limited)
            target_frames = context.active_frames[:10]  # Limit to prevent overload
        
        return target_frames
    
    def _generate_single_frame_response(
        self,
        context: ConversationContext,
        frames: List[FrameInfo],
        user_message: str,
        analysis_type: str
    ) -> ChatResponse:
        """Generate response for single frame or general analysis"""
        
        try:
            # Prepare conversation history
            conversation_history = self._prepare_conversation_history(context)
            
            # Prepare frame information
            frame_details = []
            for frame in frames:
                frame_info = self._prepare_frame_description(frame)
                frame_details.append(frame_info)
            
            # Create comprehensive prompt
            frame_context = "\n\n".join(frame_details)
            
            messages = [
                {
                    "role": "system",
                    "content": self.prompt_templates['system_prompt']
                }
            ]
            
            # Add conversation history (limited)
            recent_messages = conversation_history[-6:]  # Last 6 messages
            for msg in recent_messages:
                if msg.role != "system":
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Add current analysis request
            analysis_prompt = f"""Current Frame(s):
{frame_context}

User Question: {user_message}

Please analyze the frame(s) and provide a helpful response. Consider:
1. Direct visual observations relevant to the question
2. Context from our previous conversation
3. Specific details that address the user's inquiry
4. Any notable patterns or interesting elements"""

            messages.append({
                "role": "user",
                "content": analysis_prompt
            })
            
            # Generate response
            result = self.model_manager.inference_gpt4o_mini(
                messages,
                max_tokens=self.max_response_tokens,
                temperature=self.default_temperature
            )
            
            if result.success:
                message_id = f"{context.conversation_id}_asst_{int(time.time())}"
                
                return ChatResponse(
                    success=True,
                    response_text=result.result,
                    conversation_id=context.conversation_id,
                    message_id=message_id,
                    metadata={
                        'analysis_type': analysis_type,
                        'frame_count': len(frames)
                    }
                )
            else:
                return ChatResponse(
                    success=False,
                    response_text="",
                    conversation_id=context.conversation_id,
                    error_message=result.error_message
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error generating single frame response: {e}")
            return ChatResponse(
                success=False,
                response_text="",
                conversation_id=context.conversation_id,
                error_message=str(e)
            )
    
    def _generate_multi_frame_response(
        self,
        context: ConversationContext,
        frames: List[FrameInfo],
        user_message: str
    ) -> ChatResponse:
        """Generate response for multi-frame comparison"""
        
        try:
            # Prepare frame details for comparison
            frame_details = []
            for i, frame in enumerate(frames):
                frame_info = f"""Frame {i+1}:
- ID: {frame.frame_id}
- Timestamp: {frame.timestamp:.2f}s
- Video: {frame.video_id}
- Description: {frame.caption or 'No caption available'}"""
                frame_details.append(frame_info)
            
            frame_details_text = "\n\n".join(frame_details)
            
            # Prepare conversation context
            conversation_history = self._prepare_conversation_history(context)
            recent_context = self._summarize_recent_context(conversation_history[-10:])
            
            # Use multi-frame comparison template
            prompt = self.prompt_templates['multi_frame_comparison'].format(
                frame_details=frame_details_text,
                conversation_context=recent_context,
                user_question=user_message
            )
            
            messages = [
                {
                    "role": "system", 
                    "content": self.prompt_templates['system_prompt']
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Generate response
            result = self.model_manager.inference_gpt4o_mini(
                messages,
                max_tokens=self.max_response_tokens,
                temperature=self.default_temperature
            )
            
            if result.success:
                message_id = f"{context.conversation_id}_asst_{int(time.time())}"
                
                return ChatResponse(
                    success=True,
                    response_text=result.result,
                    conversation_id=context.conversation_id,
                    message_id=message_id,
                    metadata={
                        'analysis_type': 'multi_frame_comparison',
                        'frame_count': len(frames),
                        'temporal_span': max(f.timestamp for f in frames) - min(f.timestamp for f in frames)
                    }
                )
            else:
                return ChatResponse(
                    success=False,
                    response_text="",
                    conversation_id=context.conversation_id,
                    error_message=result.error_message
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error generating multi-frame response: {e}")
            return ChatResponse(
                success=False,
                response_text="",
                conversation_id=context.conversation_id,
                error_message=str(e)
            )


    # ==================== HELPER METHODS ====================
    
    def _prepare_frame_description(self, frame: FrameInfo) -> str:
        """Prepare detailed frame description for LLM"""
        
        description_parts = [
            f"Frame ID: {frame.frame_id}",
            f"Video: {frame.video_id}",
            f"Timestamp: {frame.timestamp:.2f}s"
        ]
        
        if frame.caption:
            description_parts.append(f"Visual Content: {frame.caption}")
        else:
            description_parts.append("Visual Content: No caption available")
        
        if hasattr(frame, 'confidence') and frame.confidence:
            description_parts.append(f"Analysis Confidence: {frame.confidence:.2f}")
        
        return "\n".join(description_parts)
    
    def _prepare_conversation_history(self, context: ConversationContext) -> List[ChatMessage]:
        """Prepare conversation history for context"""
        
        # Filter out system messages for history
        user_assistant_messages = [
            msg for msg in context.messages 
            if msg.role in ["user", "assistant"]
        ]
        
        return user_assistant_messages
    
    def _summarize_recent_context(self, messages: List[ChatMessage]) -> str:
        """Summarize recent conversation context"""
        
        if not messages:
            return "No previous conversation context."
        
        context_parts = []
        for msg in messages[-5:]:  # Last 5 messages
            if msg.role == "user":
                context_parts.append(f"User asked: {msg.content[:100]}...")
            elif msg.role == "assistant":
                context_parts.append(f"Assistant responded about: {msg.content[:100]}...")
        
        return " | ".join(context_parts)
    
    def _manage_conversation_length(self, context: ConversationContext):
        """Manage conversation length to stay within limits"""
        
        if len(context.messages) > self.max_conversation_length:
            # Keep system messages and recent messages
            system_messages = [msg for msg in context.messages if msg.role == "system"]
            recent_messages = [msg for msg in context.messages if msg.role != "system"][-self.max_conversation_length//2:]
            
            context.messages = system_messages + recent_messages
            self.logger.debug(f"🔄 Trimmed conversation {context.conversation_id} to {len(context.messages)} messages")


    # ==================== MULTI-FRAME REASONING METHODS ====================
    
    def multi_frame_reasoning(
        self,
        frames: List[FrameInfo],
        reasoning_query: str,
        reasoning_type: str = "temporal"
    ) -> FrameAnalysisResult:
        """
        Perform multi-frame reasoning
        
        Args:
            frames: List of frames to analyze
            reasoning_query: Query for reasoning task
            reasoning_type: Type of reasoning ("temporal", "spatial", "causal")
            
        Returns:
            FrameAnalysisResult với reasoning conclusions
        """
        start_time = time.time()
        
        try:
            if not frames:
                raise ValueError("No frames provided for reasoning")
            
            # Sort frames by timestamp for temporal reasoning
            if reasoning_type == "temporal":
                frames = sorted(frames, key=lambda f: f.timestamp)
            
            # Prepare frame sequence description
            frame_sequence = []
            for i, frame in enumerate(frames):
                frame_desc = f"""Frame {i+1} (t={frame.timestamp:.2f}s):
{frame.caption or 'No description available'}"""
                frame_sequence.append(frame_desc)
            
            sequence_text = "\n\n".join(frame_sequence)
            
            # Create reasoning prompt based on type
            reasoning_prompts = {
                "temporal": f"""Analyze the temporal progression in this video sequence:

{sequence_text}

Question: {reasoning_query}

Please analyze:
1. How the scene changes over time
2. Sequence of events or actions
3. Temporal relationships between elements
4. Any patterns or trends observed
5. Conclusions about what happens in this sequence""",

                "spatial": f"""Analyze the spatial relationships in these video frames:

{sequence_text}

Question: {reasoning_query}

Please analyze:
1. Spatial arrangement of objects/people
2. Movement patterns and trajectories  
3. Scene layout and composition changes
4. Relationships between different spatial elements
5. How spatial context affects the narrative""",

                "causal": f"""Analyze causal relationships in this video sequence:

{sequence_text}

Question: {reasoning_query}

Please analyze:
1. Cause-and-effect relationships between events
2. Actions that lead to observable outcomes
3. Dependencies between different elements
4. Logical flow of events
5. Causal conclusions about what's happening"""
            }
            
            prompt = reasoning_prompts.get(reasoning_type, reasoning_prompts["temporal"])
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at video analysis and reasoning. Provide detailed, logical analysis of video sequences."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Generate reasoning response
            result = self.model_manager.inference_gpt4o_mini(
                messages,
                max_tokens=self.max_response_tokens,
                temperature=0.2  # Lower temperature for reasoning
            )
            
            processing_time = time.time() - start_time
            
            if result.success:
                return FrameAnalysisResult(
                    frame_id=f"sequence_{frames[0].frame_id}_to_{frames[-1].frame_id}",
                    analysis_type=f"multi_frame_{reasoning_type}",
                    result=result.result,
                    confidence=0.85,
                    processing_time=processing_time,
                    metadata={
                        'reasoning_type': reasoning_type,
                        'frame_count': len(frames),
                        'temporal_span': frames[-1].timestamp - frames[0].timestamp,
                        'query': reasoning_query
                    }
                )
            else:
                return FrameAnalysisResult(
                    frame_id="reasoning_failed",
                    analysis_type=f"multi_frame_{reasoning_type}",
                    result=f"Reasoning failed: {result.error_message}",
                    confidence=0.0,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Error in multi-frame reasoning: {e}")
            return FrameAnalysisResult(
                frame_id="reasoning_error",
                analysis_type=f"multi_frame_{reasoning_type}",
                result=f"Error: {str(e)}",
                confidence=0.0,
                processing_time=processing_time
            )
    
    def generate_frame_descriptions(
        self,
        frames: List[FrameInfo],
        description_style: str = "detailed"
    ) -> List[str]:
        """
        Generate descriptions for multiple frames
        
        Args:
            frames: List of frames
            description_style: Style of description ("brief", "detailed", "technical")
            
        Returns:
            List of generated descriptions
        """
        
        descriptions = []
        
        try:
            # Process frames in batches to optimize API calls
            batch_size = 5
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Prepare batch prompt
                frame_prompts = []
                for j, frame in enumerate(batch_frames):
                    existing_caption = frame.caption or "No existing description"
                    
                    if description_style == "brief":
                        prompt = f"Frame {j+1}: Provide a brief 1-2 sentence description of this video frame. Existing caption: {existing_caption}"
                    elif description_style == "technical":
                        prompt = f"Frame {j+1}: Provide a technical analysis including objects, composition, lighting, and scene elements. Existing caption: {existing_caption}"
                    else:  # detailed
                        prompt = f"Frame {j+1}: Provide a detailed description of this video frame including objects, actions, scene context, and notable details. Existing caption: {existing_caption}"
                    
                    frame_prompts.append(prompt)
                
                batch_prompt = "\n\n".join(frame_prompts)
                batch_prompt += f"\n\nPlease provide {description_style} descriptions for each frame, numbered 1-{len(batch_frames)}:"
                
                messages = [
                    {
                        "role": "system",
                        "content": f"You are an expert at describing video content. Provide {description_style} descriptions as requested."
                    },
                    {
                        "role": "user",
                        "content": batch_prompt
                    }
                ]
                
                # Get batch response
                result = self.model_manager.inference_gpt4o_mini(
                    messages,
                    max_tokens=self.max_response_tokens,
                    temperature=0.3
                )
                
                if result.success:
                    # Parse batch response
                    batch_descriptions = self._parse_batch_descriptions(result.result, len(batch_frames))
                    descriptions.extend(batch_descriptions)
                else:
                    # Fallback descriptions
                    for frame in batch_frames:
                        descriptions.append(f"Error generating description: {result.error_message}")
                
                # Small delay between batches
                time.sleep(0.1)
            
            return descriptions
            
        except Exception as e:
            self.logger.error(f"❌ Error generating frame descriptions: {e}")
            return [f"Error: {str(e)}" for _ in frames]
    
    def _parse_batch_descriptions(self, response_text: str, expected_count: int) -> List[str]:
        """Parse batch description response"""
        
        descriptions = []
        
        try:
            # Split by numbered lines
            lines = response_text.strip().split('\n')
            current_description = []
            
            for line in lines:
                line = line.strip()
                
                # Check if line starts with number
                if line and (line[0].isdigit() and (': ' in line or '. ' in line)):
                    # Save previous description
                    if current_description:
                        descriptions.append(' '.join(current_description))
                    
                    # Start new description
                    if ': ' in line:
                        current_description = [line.split(': ', 1)[1]]
                    else:
                        current_description = [line.split('. ', 1)[1]]
                else:
                    # Continue current description
                    if line:
                        current_description.append(line)
            
            # Add last description
            if current_description:
                descriptions.append(' '.join(current_description))
            
            # Ensure we have the right number of descriptions
            while len(descriptions) < expected_count:
                descriptions.append("Description parsing failed")
            
            return descriptions[:expected_count]
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing batch descriptions: {e}")
            return ["Parsing error" for _ in range(expected_count)]


    # ==================== CONVERSATION MANAGEMENT METHODS ====================
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context"""
        with self._conversation_lock:
            return self.conversations.get(conversation_id)
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all active conversations"""
        
        with self._conversation_lock:
            conversation_list = []
            
            for conv_id, context in self.conversations.items():
                conversation_list.append({
                    'conversation_id': conv_id,
                    'creation_time': context.creation_time,
                    'last_updated': context.last_updated,
                    'message_count': len(context.messages),
                    'frame_count': len(context.active_frames),
                    'video_count': len(context.active_videos)
                })
            
            # Sort by last updated
            conversation_list.sort(key=lambda x: x['last_updated'], reverse=True)
            return conversation_list
    
    def end_conversation(self, conversation_id: str) -> bool:
        """
        End a conversation and clean up resources
        
        Args:
            conversation_id: ID của conversation cần kết thúc
            
        Returns:
            True nếu conversation được kết thúc thành công, False nếu không
        """
        
        try:
            with self._conversation_lock:
                # Kiểm tra conversation có tồn tại không
                if conversation_id not in self.conversations:
                    self.logger.warning(f"⚠️ Conversation {conversation_id} not found")
                    return False
                
                # Lấy context của conversation
                context = self.conversations[conversation_id]
                
                # Log thông tin trước khi kết thúc
                message_count = len(context.messages)
                frame_count = len(context.active_frames)
                duration_hours = (time.time() - context.creation_time) / 3600
                
                self.logger.info(f"🏁 Ending conversation {conversation_id}")
                self.logger.info(f"   📊 Stats: {message_count} messages, {frame_count} frames, {duration_hours:.2f}h duration")
                
                # Save conversation summary nếu conversation đủ dài
                if message_count >= 3:  # Chỉ save summary cho conversations có ít nhất 3 messages
                    try:
                        self._save_conversation_summary(context)
                        self.logger.debug(f"💾 Saved summary for conversation {conversation_id}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to save summary for {conversation_id}: {e}")
                        # Không return False vì summary failure không nên block conversation ending
                
                # Clean up conversation-specific cache entries
                self._cleanup_conversation_cache(conversation_id)
                
                # Remove conversation khỏi active conversations
                del self.conversations[conversation_id]
                
                # Log successful completion
                self.logger.info(f"✅ Successfully ended conversation {conversation_id}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error ending conversation {conversation_id}: {e}")
            self.logger.debug(f"Full error traceback:", exc_info=True)
            return False

    def _cleanup_conversation_cache(self, conversation_id: str):
        """
        Clean up cache entries related to a specific conversation
        
        Args:
            conversation_id: ID của conversation
        """
        
        try:
            with self._cache_lock:
                # Find và remove cache entries liên quan đến conversation này
                cache_keys_to_remove = []
                
                for cache_key, analysis_result in self.analysis_cache.items():
                    # Check nếu analysis result có liên quan đến conversation
                    if hasattr(analysis_result, 'metadata') and analysis_result.metadata:
                        if analysis_result.metadata.get('conversation_id') == conversation_id:
                            cache_keys_to_remove.append(cache_key)
                    
                    # Check frame IDs trong cache key
                    if conversation_id in cache_key:
                        cache_keys_to_remove.append(cache_key)
                
                # Remove identified cache entries
                for key in cache_keys_to_remove:
                    del self.analysis_cache[key]
                
                if cache_keys_to_remove:
                    self.logger.debug(f"🧹 Cleaned {len(cache_keys_to_remove)} cache entries for conversation {conversation_id}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error cleaning conversation cache: {e}")
            # Don't raise exception vì cache cleanup failure không critical
    
    def __del__(self):
        """Cleanup when ChatInterface is destroyed"""
        try:
            # Save summaries of all active conversations
            with self._conversation_lock:
                for context in self.conversations.values():
                    self._save_conversation_summary(context)
            
            # Clear caches
            with self._cache_lock:
                self.analysis_cache.clear()
                
        except:
            pass


# ==================== GLOBAL INSTANCE MANAGEMENT ====================

# Global chat interface instance
_global_chat_interface = None

def get_chat_interface(config: Optional[Config] = None) -> ChatInterface:
    """
    Get global chat interface instance (singleton pattern)
    
    Args:
        config: Configuration object (chỉ sử dụng lần đầu)
        
    Returns:
        ChatInterface instance
    """
    global _global_chat_interface
    if _global_chat_interface is None:
        _global_chat_interface = ChatInterface(config)
    return _global_chat_interface

def reset_chat_interface():
    """Reset global chat interface instance"""
    global _global_chat_interface
    if _global_chat_interface:
        del _global_chat_interface
    _global_chat_interface = None