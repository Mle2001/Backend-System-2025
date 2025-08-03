"""
retrieval_engine.py - Hybrid Retrieval System cho VideoRAG System
Truy xuất clips/frames liên quan dựa trên query sử dụng textual semantic matching và visual content matching
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from config import Config, get_config
from utils import (
    get_logger, measure_time, performance_monitor, cosine_similarity,
    normalize_text, extract_keywords, safe_execute, retry_on_failure
)
from database_manager import DatabaseManager, get_database_manager, QueryResult
from model_manager import ModelManager, get_model_manager, InferenceResult
from knowledge_builder import KnowledgeBuilder, get_knowledge_builder, Entity, Relation
from video_processor import ProcessedVideo, VideoChunk, FrameInfo


# ==================== DATA STRUCTURES ====================

@dataclass
class QueryContext:
    """Context information cho query processing"""
    original_query: str
    processed_query: str
    query_type: str  # "text", "visual", "hybrid", "entity", "temporal"
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    temporal_constraints: Optional[Dict[str, Any]] = None
    video_filters: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.6
    max_results: int = 10


@dataclass
class RetrievalResult:
    """Result từ retrieval operation"""
    result_id: str
    result_type: str  # "frame", "chunk", "entity", "video_segment"
    content: Any  # FrameInfo, VideoChunk, Entity, etc.
    relevance_score: float
    text_score: float = 0.0
    visual_score: float = 0.0
    entity_score: float = 0.0
    temporal_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class RetrievalResponse:
    """Complete response từ retrieval engine"""
    success: bool
    query_context: QueryContext
    results: List[RetrievalResult] = field(default_factory=list)
    total_found: int = 0
    processing_time: float = 0.0
    retrieval_strategy: str = ""
    error_message: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)


# ==================== MAIN RETRIEVAL ENGINE CLASS ====================

class RetrievalEngine:
    """
    Hybrid Retrieval System cho VideoRAG
    Kết hợp textual semantic matching, visual content matching, và entity-based retrieval
    """
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        database_manager: Optional[DatabaseManager] = None,
        model_manager: Optional[ModelManager] = None,
        knowledge_builder: Optional[KnowledgeBuilder] = None
    ):
        """Initialize RetrievalEngine"""
        self.config = config or get_config()
        self.db_manager = database_manager or get_database_manager()
        self.model_manager = model_manager or get_model_manager()
        self.knowledge_builder = knowledge_builder or get_knowledge_builder()
        self.logger = get_logger('retrieval_engine')
        
        self._init_retrieval_parameters()
        self._init_query_cache()
        self._init_threading()
        
        self.logger.info("🔍 RetrievalEngine initialized")
        self.logger.info(f"📊 Config: Text weight={self.text_weight}, Visual weight={self.visual_weight}")
    
    def _init_retrieval_parameters(self):
        """Initialize retrieval parameters từ config"""
        retrieval_config = self.config.retrieval
        
        self.similarity_threshold = retrieval_config.similarity_threshold
        self.top_k_text = retrieval_config.top_k_text
        self.top_k_visual = retrieval_config.top_k_visual
        self.top_k_final = retrieval_config.top_k_final
        self.relevance_filter_threshold = retrieval_config.relevance_filter_threshold
        
        # Scoring weights
        self.text_weight = 0.4
        self.visual_weight = 0.3
        self.entity_weight = 0.2
        self.temporal_weight = 0.1
        
        # Enable cross-modal và temporal filtering
        self.enable_cross_modal = retrieval_config.enable_cross_modal
        self.enable_temporal_filtering = retrieval_config.enable_temporal_filtering
    
    def _init_query_cache(self):
        """Initialize query result caching"""
        self.query_cache: Dict[str, Tuple[RetrievalResponse, float]] = {}
        self.cache_max_size = 100
        self.cache_ttl = 3600  # 1 hour
        self._cache_lock = threading.RLock()
    
    def _init_threading(self):
        """Initialize threading components"""
        self._retrieval_lock = threading.RLock()


    # ==================== MAIN RETRIEVAL METHODS ====================
    
    @measure_time
    def retrieve_video_clips(
        self,
        query: str,
        video_id: Optional[str] = None,
        retrieval_strategy: str = "hybrid",
        max_results: int = None,
        include_entities: bool = True,
        include_temporal_context: bool = True
    ) -> RetrievalResponse:
        """Main method để retrieve video clips liên quan đến query"""
        start_time = time.time()
        
        if max_results is None:
            max_results = self.top_k_final
        
        self.logger.info(f"🔍 Retrieving clips for query: '{query[:50]}...' using {retrieval_strategy}")
        
        try:
            with performance_monitor.monitor(f"retrieve_clips_{retrieval_strategy}"):
                
                # Step 1: Process user query
                query_context = self.process_user_query(
                    query, video_id, max_results, include_entities, include_temporal_context
                )
                
                # Check cache
                cache_key = self._generate_cache_key(query_context, retrieval_strategy)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    self.logger.info(f"📂 Retrieved from cache: {len(cached_response.results)} results")
                    return cached_response
                
                # Step 2: Execute retrieval strategy
                if retrieval_strategy == "text":
                    results = self._textual_semantic_matching(query_context)
                elif retrieval_strategy == "visual":
                    results = self._visual_content_matching(query_context)
                elif retrieval_strategy == "entity":
                    results = self._entity_based_retrieval(query_context)
                elif retrieval_strategy == "hybrid":
                    results = self._hybrid_retrieval(query_context)
                else:
                    raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")
                
                # Step 3: Post-process và rank results
                final_results = self._post_process_results(results, query_context)
                
                # Step 4: Create response
                processing_time = time.time() - start_time
                
                response = RetrievalResponse(
                    success=True,
                    query_context=query_context,
                    results=final_results[:max_results],
                    total_found=len(results),
                    processing_time=processing_time,
                    retrieval_strategy=retrieval_strategy,
                    debug_info={
                        'raw_results_count': len(results),
                        'filtered_results_count': len(final_results),
                        'cache_hit': False
                    }
                )
                
                # Cache response
                self._cache_response(cache_key, response)
                
                self.logger.info(f"✅ Retrieved {len(final_results)} clips in {processing_time:.2f}s")
                return response
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Error in retrieval: {e}")
            
            return RetrievalResponse(
                success=False,
                query_context=QueryContext(query, query, "error"),
                processing_time=processing_time,
                retrieval_strategy=retrieval_strategy,
                error_message=str(e)
            )

    def retrieve_by_timerange(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        content_types: List[str] = None
    ) -> RetrievalResponse:
        """Retrieve content from specific time range in video"""
        start_query_time = time.time()
        
        if content_types is None:
            content_types = ['frames', 'audio', 'entities']
        
        try:
            query_context = QueryContext(
                original_query=f"timerange:{video_id}:{start_time}-{end_time}",
                processed_query=f"video content from {start_time}s to {end_time}s",
                query_type="temporal",
                video_filters=[video_id],
                temporal_constraints={
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                }
            )
            
            results = []
            
            # Retrieve frames in time range
            if 'frames' in content_types and self.db_manager.metadata_db:
                results.extend(self._retrieve_frames_in_timerange(video_id, start_time, end_time))
            
            # Retrieve audio segments in time range
            if 'audio' in content_types and self.db_manager.metadata_db:
                results.extend(self._retrieve_audio_in_timerange(video_id, start_time, end_time))
            
            # Retrieve entities in time range
            if 'entities' in content_types and self.db_manager.graph_db:
                results.extend(self._retrieve_entities_in_timerange(video_id, start_time, end_time))
            
            # Sort by timestamp for temporal queries
            results.sort(key=lambda x: x.metadata.get('timestamp', 0))
            
            processing_time = time.time() - start_query_time
            
            return RetrievalResponse(
                success=True,
                query_context=query_context,
                results=results,
                total_found=len(results),
                processing_time=processing_time,
                retrieval_strategy="temporal_range",
                debug_info={
                    'time_range': f"{start_time}s - {end_time}s",
                    'duration': end_time - start_time,
                    'content_types': content_types
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_query_time
            return RetrievalResponse(
                success=False,
                query_context=QueryContext("", "", "temporal"),
                processing_time=processing_time,
                error_message=str(e)
            )

    def retrieve_by_entity(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        include_related: bool = True,
        max_results: int = 10
    ) -> RetrievalResponse:
        """Retrieve content related to a specific entity"""
        start_time = time.time()
        
        try:
            # Create query context for entity
            query_context = QueryContext(
                original_query=f"entity:{entity_name}",
                processed_query=entity_name,
                query_type="entity",
                entities=[entity_name],
                max_results=max_results
            )
            
            # Find entity in knowledge graph
            entities = []
            if self.db_manager.graph_db:
                entities = self.db_manager.graph_db.query_entities(
                    name_pattern=entity_name,
                    entity_type=entity_type
                )
            
            if not entities:
                return RetrievalResponse(
                    success=False,
                    query_context=query_context,
                    error_message=f"Entity '{entity_name}' not found"
                )
            
            target_entity = entities[0]  # Use first match
            results = []
            
            # Get all video content where entity appears
            for video_id in target_entity.source_videos:
                for video_id_ts, start_time_ts, end_time_ts in target_entity.timestamps:
                    if video_id_ts == video_id:
                        result = RetrievalResult(
                            result_id=f"{target_entity.entity_id}_{start_time_ts}",
                            result_type='entity_segment',
                            content={
                                'entity': target_entity,
                                'video_id': video_id,
                                'start_time': start_time_ts,
                                'end_time': end_time_ts,
                                'duration': end_time_ts - start_time_ts
                            },
                            relevance_score=target_entity.confidence,
                            entity_score=target_entity.confidence,
                            metadata={
                                'entity_name': target_entity.name,
                                'entity_type': target_entity.entity_type,
                                'video_id': video_id,
                                'timestamp_range': (start_time_ts, end_time_ts)
                            },
                            explanation=f"Entity {target_entity.name} appears in video"
                        )
                        results.append(result)
            
            # Include related entities if requested
            if include_related:
                related_entities = self._find_related_entities(target_entity)
                
                for related_entity, relation_strength in related_entities[:5]:
                    for video_id in related_entity.source_videos:
                        for video_id_ts, start_time_ts, end_time_ts in related_entity.timestamps:
                            if video_id_ts == video_id:
                                result = RetrievalResult(
                                    result_id=f"{related_entity.entity_id}_{start_time_ts}",
                                    result_type='related_entity_segment',
                                    content={
                                        'entity': related_entity,
                                        'video_id': video_id,
                                        'start_time': start_time_ts,
                                        'end_time': end_time_ts,
                                        'relation_to_target': f"Related to {target_entity.name}"
                                    },
                                    relevance_score=related_entity.confidence * relation_strength * 0.8,
                                    entity_score=related_entity.confidence * relation_strength,
                                    metadata={
                                        'entity_name': related_entity.name,
                                        'entity_type': related_entity.entity_type,
                                        'video_id': video_id,
                                        'relation_strength': relation_strength,
                                        'primary_entity': target_entity.name
                                    },
                                    explanation=f"Related entity {related_entity.name} (relation: {relation_strength:.2f})"
                                )
                                results.append(result)
            
            # Sort và limit results
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            results = results[:max_results]
            
            processing_time = time.time() - start_time
            
            return RetrievalResponse(
                success=True,
                query_context=query_context,
                results=results,
                total_found=len(results),
                processing_time=processing_time,
                retrieval_strategy="entity_focused",
                debug_info={
                    'target_entity': target_entity.name,
                    'related_entities_found': len(related_entities) if include_related else 0
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return RetrievalResponse(
                success=False,
                query_context=QueryContext(entity_name, entity_name, "entity"),
                processing_time=processing_time,
                error_message=str(e)
            )

    def retrieve_similar_to_content(
        self,
        reference_content: Union[FrameInfo, VideoChunk, Entity],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        exclude_same_video: bool = False
    ) -> RetrievalResponse:
        """Retrieve content similar to reference content"""
        start_time = time.time()
        
        try:
            # Determine reference type và get embedding
            ref_embedding = None
            ref_video_id = None
            query_text = ""
            
            if isinstance(reference_content, FrameInfo):
                ref_embedding = reference_content.embedding
                ref_video_id = reference_content.video_id
                query_text = f"similar to frame: {reference_content.caption}"
            elif isinstance(reference_content, VideoChunk):
                ref_embedding = reference_content.chunk_embedding
                ref_video_id = reference_content.video_id
                query_text = f"similar to chunk: {reference_content.unified_description}"
            elif hasattr(reference_content, 'embedding'):
                ref_embedding = reference_content.embedding
                if hasattr(reference_content, 'source_videos'):
                    ref_video_id = list(reference_content.source_videos)[0] if reference_content.source_videos else None
                query_text = f"similar to entity: {getattr(reference_content, 'name', 'unknown')}"
            
            if ref_embedding is None:
                return RetrievalResponse(
                    success=False,
                    query_context=QueryContext(query_text, query_text, "similarity"),
                    error_message="Reference content has no embedding"
                )
            
            query_context = QueryContext(
                original_query=query_text,
                processed_query=query_text,
                query_type="similarity",
                confidence_threshold=similarity_threshold,
                max_results=max_results
            )
            
            results = []
            
            # Search similar embeddings
            if self.db_manager.vector_db:
                filters = {}
                if exclude_same_video and ref_video_id:
                    # This would require implementing a "not equal" filter in vector DB
                    pass
                
                similar_embeddings = self.db_manager.vector_db.search_similar(
                    ref_embedding,
                    top_k=max_results * 2,  # Get more for filtering
                    threshold=similarity_threshold,
                    filters=filters
                )
                
                for embedding_entry, similarity in similar_embeddings:
                    # Skip same content
                    if (embedding_entry.source_id == getattr(reference_content, 'frame_id', None) or
                        embedding_entry.source_id == getattr(reference_content, 'chunk_id', None) or
                        embedding_entry.source_id == getattr(reference_content, 'entity_id', None)):
                        continue
                    
                    # Skip same video if requested
                    if exclude_same_video and embedding_entry.video_id == ref_video_id:
                        continue
                    
                    result = RetrievalResult(
                        result_id=embedding_entry.source_id,
                        result_type=f'similar_{embedding_entry.source_type}',
                        content=embedding_entry,
                        relevance_score=similarity,
                        visual_score=similarity,
                        metadata=embedding_entry.metadata,
                        explanation=f"Similar {embedding_entry.source_type} (similarity: {similarity:.3f})"
                    )
                    results.append(result)
            
            # Sort by similarity
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            results = results[:max_results]
            
            processing_time = time.time() - start_time
            
            return RetrievalResponse(
                success=True,
                query_context=query_context,
                results=results,
                total_found=len(results),
                processing_time=processing_time,
                retrieval_strategy="similarity_based",
                debug_info={
                    'reference_type': type(reference_content).__name__,
                    'similarity_threshold': similarity_threshold,
                    'exclude_same_video': exclude_same_video
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return RetrievalResponse(
                success=False,
                query_context=QueryContext("", "", "similarity"),
                processing_time=processing_time,
                error_message=str(e)
            )


    # ==================== QUERY PROCESSING METHODS ====================
    
    def process_user_query(
        self,
        query: str,
        video_id: Optional[str] = None,
        max_results: int = 10,
        include_entities: bool = True,
        include_temporal_context: bool = True
    ) -> QueryContext:
        """Process và analyze user query để extract semantic information"""
        try:
            # Normalize query text
            processed_query = normalize_text(query)
            
            # Extract keywords
            keywords = extract_keywords(processed_query, max_keywords=10)
            
            # Determine query type
            query_type = self._determine_query_type(processed_query, keywords)
            
            # Extract entities if enabled
            entities = []
            if include_entities:
                entities = self._extract_query_entities(processed_query)
            
            # Extract temporal constraints
            temporal_constraints = None
            if include_temporal_context:
                temporal_constraints = self._extract_temporal_constraints(processed_query)
            
            # Video filters
            video_filters = [video_id] if video_id else []
            
            query_context = QueryContext(
                original_query=query,
                processed_query=processed_query,
                query_type=query_type,
                keywords=keywords,
                entities=entities,
                temporal_constraints=temporal_constraints,
                video_filters=video_filters,
                confidence_threshold=self.relevance_filter_threshold,
                max_results=max_results
            )
            
            self.logger.debug(f"📝 Processed query: type={query_type}, keywords={len(keywords)}, entities={len(entities)}")
            return query_context
            
        except Exception as e:
            self.logger.error(f"❌ Error processing query: {e}")
            return QueryContext(query, query, "error")
    
    def _determine_query_type(self, query: str, keywords: List[str]) -> str:
        """Determine type of query based on content"""
        query_lower = query.lower()
        
        # Visual indicators
        visual_keywords = ['see', 'show', 'visual', 'image', 'frame', 'appearance', 'looks', 'color']
        if any(keyword in query_lower for keyword in visual_keywords):
            return "visual"
        
        # Entity indicators  
        entity_keywords = ['who', 'person', 'people', 'character', 'actor', 'object', 'thing']
        if any(keyword in query_lower for keyword in entity_keywords):
            return "entity"
        
        # Temporal indicators
        temporal_keywords = ['when', 'time', 'before', 'after', 'during', 'start', 'end', 'beginning']
        if any(keyword in query_lower for keyword in temporal_keywords):
            return "temporal"
        
        # Default to hybrid for complex queries
        if len(keywords) > 3:
            return "hybrid"
        
        return "text"
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities từ query using knowledge graph"""
        entities = []
        
        try:
            # Get entities from knowledge graph matching query terms
            if self.db_manager.graph_db:
                # Simple keyword matching against entity names
                query_words = query.lower().split()
                
                for graph in self.db_manager.graph_db.knowledge_graphs.values():
                    for entity in graph.entities.values():
                        entity_name_lower = entity.name.lower()
                        
                        # Check if any query word matches entity name
                        for word in query_words:
                            if len(word) > 2 and word in entity_name_lower:
                                entities.append(entity.name)
                                break
                        
                        # Check aliases
                        for alias in entity.aliases:
                            alias_lower = alias.lower()
                            for word in query_words:
                                if len(word) > 2 and word in alias_lower:
                                    entities.append(entity.name)
                                    break
            
            return list(set(entities))  # Remove duplicates
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting entities: {e}")
            return []
    
    def _extract_temporal_constraints(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal constraints từ query"""
        constraints = {}
        
        try:
            query_lower = query.lower()
            
            # Simple temporal pattern matching
            if 'beginning' in query_lower or 'start' in query_lower:
                constraints['time_preference'] = 'early'
            elif 'end' in query_lower or 'ending' in query_lower:
                constraints['time_preference'] = 'late'
            elif 'middle' in query_lower:
                constraints['time_preference'] = 'middle'
            
            # Duration indicators
            if 'long' in query_lower:
                constraints['duration_preference'] = 'long'
            elif 'short' in query_lower or 'brief' in query_lower:
                constraints['duration_preference'] = 'short'
            
            return constraints if constraints else None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting temporal constraints: {e}")
            return None


    # ==================== RETRIEVAL STRATEGY IMPLEMENTATIONS ====================
    
    def _textual_semantic_matching(self, query_context: QueryContext) -> List[RetrievalResult]:
        """Textual semantic matching using knowledge graph và metadata"""
        results = []
        
        try:
            # Search frames by caption
            if self.db_manager.metadata_db:
                caption_results = self.db_manager.metadata_db.search_frames_by_caption(
                    query_context.processed_query,
                    video_id=query_context.video_filters[0] if query_context.video_filters else None,
                    limit=self.top_k_text
                )
                
                for frame_data in caption_results:
                    # Calculate text relevance score
                    text_score = self._calculate_text_relevance(
                        query_context.processed_query,
                        frame_data['caption'],
                        query_context.keywords
                    )
                    
                    if text_score >= self.similarity_threshold:
                        result = RetrievalResult(
                            result_id=frame_data['frame_id'],
                            result_type='frame',
                            content=frame_data,
                            relevance_score=text_score,
                            text_score=text_score,
                            metadata={
                                'video_filename': frame_data.get('video_filename', ''),
                                'timestamp': frame_data.get('timestamp', 0),
                                'confidence': frame_data.get('confidence', 0)
                            },
                            explanation=f"Caption match: {frame_data['caption'][:50]}..."
                        )
                        results.append(result)
            
            # Search audio by transcript
            if self.db_manager.metadata_db:
                audio_results = self.db_manager.metadata_db.search_audio_by_transcript(
                    query_context.processed_query,
                    video_id=query_context.video_filters[0] if query_context.video_filters else None,
                    limit=self.top_k_text
                )
                
                for audio_data in audio_results:
                    text_score = self._calculate_text_relevance(
                        query_context.processed_query,
                        audio_data['transcript'],
                        query_context.keywords
                    )
                    
                    if text_score >= self.similarity_threshold:
                        result = RetrievalResult(
                            result_id=audio_data['segment_id'],
                            result_type='audio_segment',
                            content=audio_data,
                            relevance_score=text_score,
                            text_score=text_score,
                            metadata={
                                'video_filename': audio_data.get('video_filename', ''),
                                'start_time': audio_data.get('start_time', 0),
                                'end_time': audio_data.get('end_time', 0),
                                'confidence': audio_data.get('confidence', 0)
                            },
                            explanation=f"Transcript match: {audio_data['transcript'][:50]}..."
                        )
                        results.append(result)
            
            # Search using knowledge graph entities
            if query_context.entities and self.db_manager.graph_db:
                entity_results = self.db_manager.graph_db.query_entities(
                    name_pattern=query_context.processed_query,
                    video_id=query_context.video_filters[0] if query_context.video_filters else None
                )
                
                for entity in entity_results:
                    entity_score = self._calculate_entity_relevance(query_context, entity)
                    
                    if entity_score >= self.similarity_threshold:
                        result = RetrievalResult(
                            result_id=entity.entity_id,
                            result_type='entity',
                            content=entity,
                            relevance_score=entity_score,
                            text_score=entity_score,
                            metadata={
                                'entity_type': entity.entity_type,
                                'source_videos': list(entity.source_videos),
                                'confidence': entity.confidence
                            },
                            explanation=f"Entity match: {entity.name} ({entity.entity_type})"
                        )
                        results.append(result)
            
            self.logger.debug(f"📝 Text retrieval: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in textual semantic matching: {e}")
            return []
    
    def _visual_content_matching(self, query_context: QueryContext) -> List[RetrievalResult]:
        """Visual content matching using embeddings"""
        results = []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query_context.processed_query)
            
            if query_embedding is None:
                self.logger.warning("⚠️ Could not generate query embedding for visual search")
                return []
            
            # Search similar visual embeddings
            if self.db_manager.vector_db:
                filters = {}
                if query_context.video_filters:
                    filters['video_id'] = query_context.video_filters[0]
                
                # Search frame embeddings
                frame_search = self.db_manager.vector_db.search_similar(
                    query_embedding,
                    top_k=self.top_k_visual,
                    threshold=self.similarity_threshold,
                    filters={**filters, 'source_type': 'frame'}
                )
                
                for embedding_entry, similarity in frame_search:
                    result = RetrievalResult(
                        result_id=embedding_entry.source_id,
                        result_type='frame_embedding',
                        content=embedding_entry,
                        relevance_score=similarity,
                        visual_score=similarity,
                        metadata=embedding_entry.metadata,
                        explanation=f"Visual similarity: {similarity:.3f}"
                    )
                    results.append(result)
                
                # Search chunk embeddings  
                chunk_search = self.db_manager.vector_db.search_similar(
                    query_embedding,
                    top_k=self.top_k_visual,
                    threshold=self.similarity_threshold,
                    filters={**filters, 'source_type': 'chunk'}
                )
                
                for embedding_entry, similarity in chunk_search:
                    result = RetrievalResult(
                        result_id=embedding_entry.source_id,
                        result_type='chunk_embedding',
                        content=embedding_entry,
                        relevance_score=similarity,
                        visual_score=similarity,
                        metadata=embedding_entry.metadata,
                        explanation=f"Chunk visual similarity: {similarity:.3f}"
                    )
                    results.append(result)
            
            self.logger.debug(f"👁️ Visual retrieval: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in visual content matching: {e}")
            return []
    
    def _entity_based_retrieval(self, query_context: QueryContext) -> List[RetrievalResult]:
        """Entity-based retrieval using knowledge graph"""
        results = []
        
        try:
            if not self.db_manager.graph_db:
                return []
            
            # Search entities by name pattern
            entities = self.db_manager.graph_db.query_entities(
                name_pattern=query_context.processed_query,
                video_id=query_context.video_filters[0] if query_context.video_filters else None
            )
            
            for entity in entities:
                entity_score = self._calculate_entity_relevance(query_context, entity)
                
                if entity_score >= self.similarity_threshold:
                    # Find related entities through relations
                    related_entities = self._find_related_entities(entity)
                    
                    result = RetrievalResult(
                        result_id=entity.entity_id,
                        result_type='entity',
                        content=entity,
                        relevance_score=entity_score,
                        entity_score=entity_score,
                        metadata={
                            'entity_type': entity.entity_type,
                            'source_videos': list(entity.source_videos),
                            'related_entities': len(related_entities),
                            'timestamps': entity.timestamps
                        },
                        explanation=f"Entity: {entity.name} ({entity.entity_type})"
                    )
                    results.append(result)
                    
                    # Add related entities with lower scores
                    for related_entity, relation_strength in related_entities[:3]:  # Top 3
                        related_score = entity_score * relation_strength * 0.8
                        
                        if related_score >= self.similarity_threshold:
                            related_result = RetrievalResult(
                                result_id=related_entity.entity_id,
                                result_type='related_entity',
                                content=related_entity,
                                relevance_score=related_score,
                                entity_score=related_score,
                                metadata={
                                    'entity_type': related_entity.entity_type,
                                    'relation_to_query': f"Related to {entity.name}",
                                    'relation_strength': relation_strength
                                },
                                explanation=f"Related to {entity.name}: {related_entity.name}"
                            )
                            results.append(related_result)
            
            # Search entities by embeddings if available
            if query_context.entities and self.db_manager.vector_db:
                query_embedding = self._generate_query_embedding(query_context.processed_query)
                
                if query_embedding is not None:
                    entity_embeddings = self.db_manager.vector_db.search_similar(
                        query_embedding,
                        top_k=self.top_k_visual,
                        threshold=self.similarity_threshold,
                        filters={'source_type': 'entity'}
                    )
                    
                    for embedding_entry, similarity in entity_embeddings:
                        # Avoid duplicates from name-based search
                        if not any(r.result_id == embedding_entry.source_id for r in results):
                            result = RetrievalResult(
                                result_id=embedding_entry.source_id,
                                result_type='entity_embedding',
                                content=embedding_entry,
                                relevance_score=similarity,
                                entity_score=similarity,
                                metadata=embedding_entry.metadata,
                                explanation=f"Entity embedding similarity: {similarity:.3f}"
                            )
                            results.append(result)
            
            self.logger.debug(f"🏷️ Entity retrieval: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in entity-based retrieval: {e}")
            return []
    
    def _hybrid_retrieval(self, query_context: QueryContext) -> List[RetrievalResult]:
        """Hybrid retrieval combining all strategies"""
        try:
            all_results = []
            
            # Run all retrieval strategies
            text_results = self._textual_semantic_matching(query_context)
            visual_results = self._visual_content_matching(query_context)
            entity_results = self._entity_based_retrieval(query_context)
            
            # Combine results
            all_results.extend(text_results)
            all_results.extend(visual_results) 
            all_results.extend(entity_results)
            
            # Deduplicate and merge scores for same content
            merged_results = self._merge_duplicate_results(all_results)
            
            # Apply temporal scoring if enabled
            if self.enable_temporal_filtering and query_context.temporal_constraints:
                merged_results = self._apply_temporal_scoring(merged_results, query_context)
            
            self.logger.debug(f"🔄 Hybrid retrieval: {len(merged_results)} merged results")
            return merged_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid retrieval: {e}")
            return []


    # ==================== TIMERANGE RETRIEVAL HELPERS ====================
    
    def _retrieve_frames_in_timerange(self, video_id: str, start_time: float, end_time: float) -> List[RetrievalResult]:
        """Retrieve frames in time range"""
        results = []
        
        try:
            with self.db_manager.metadata_db._lock:
                import sqlite3
                with sqlite3.connect(self.db_manager.metadata_db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT frame_id, timestamp, caption, confidence, image_path
                        FROM frames 
                        WHERE video_id = ? AND timestamp BETWEEN ? AND ?
                        ORDER BY timestamp
                    """, (video_id, start_time, end_time))
                    
                    frame_rows = cursor.fetchall()
                    
                    for frame_row in frame_rows:
                        result = RetrievalResult(
                            result_id=frame_row[0],
                            result_type='timerange_frame',
                            content={
                                'frame_id': frame_row[0],
                                'timestamp': frame_row[1],
                                'caption': frame_row[2],
                                'confidence': frame_row[3],
                                'image_path': frame_row[4],
                                'video_id': video_id
                            },
                            relevance_score=1.0,  # Perfect temporal match
                            temporal_score=1.0,
                            metadata={
                                'timestamp': frame_row[1],
                                'in_timerange': True
                            },
                            explanation=f"Frame at {frame_row[1]:.1f}s"
                        )
                        results.append(result)
        
        except Exception as e:
            self.logger.error(f"❌ Error retrieving frames: {e}")
        
        return results
    
    def _retrieve_audio_in_timerange(self, video_id: str, start_time: float, end_time: float) -> List[RetrievalResult]:
        """Retrieve audio segments in time range"""
        results = []
        
        try:
            with self.db_manager.metadata_db._lock:
                import sqlite3
                with sqlite3.connect(self.db_manager.metadata_db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT segment_id, start_time, end_time, transcript, confidence
                        FROM audio_segments 
                        WHERE video_id = ? AND (
                            (start_time BETWEEN ? AND ?) OR 
                            (end_time BETWEEN ? AND ?) OR
                            (start_time <= ? AND end_time >= ?)
                        )
                        ORDER BY start_time
                    """, (video_id, start_time, end_time, start_time, end_time, start_time, end_time))
                    
                    audio_rows = cursor.fetchall()
                    
                    for audio_row in audio_rows:
                        result = RetrievalResult(
                            result_id=audio_row[0],
                            result_type='timerange_audio',
                            content={
                                'segment_id': audio_row[0],
                                'start_time': audio_row[1],
                                'end_time': audio_row[2],
                                'transcript': audio_row[3],
                                'confidence': audio_row[4],
                                'video_id': video_id
                            },
                            relevance_score=1.0,
                            temporal_score=1.0,
                            metadata={
                                'time_overlap': True,
                                'segment_duration': audio_row[2] - audio_row[1]
                            },
                            explanation=f"Audio segment {audio_row[1]:.1f}s-{audio_row[2]:.1f}s"
                        )
                        results.append(result)
        
        except Exception as e:
            self.logger.error(f"❌ Error retrieving audio: {e}")
        
        return results
    
    def _retrieve_entities_in_timerange(self, video_id: str, start_time: float, end_time: float) -> List[RetrievalResult]:
        """Retrieve entities in time range"""
        results = []
        
        try:
            if self.db_manager.graph_db:
                for graph in self.db_manager.graph_db.knowledge_graphs.values():
                    for entity in graph.entities.values():
                        if video_id in entity.source_videos:
                            # Check if entity appears in time range
                            for entity_video_id, entity_start, entity_end in entity.timestamps:
                                if entity_video_id == video_id:
                                    # Check temporal overlap
                                    if not (entity_end < start_time or entity_start > end_time):
                                        overlap_start = max(start_time, entity_start)
                                        overlap_end = min(end_time, entity_end)
                                        overlap_duration = overlap_end - overlap_start
                                        
                                        result = RetrievalResult(
                                            result_id=f"{entity.entity_id}_{entity_start}",
                                            result_type='timerange_entity',
                                            content={
                                                'entity': entity,
                                                'video_id': video_id,
                                                'entity_start_time': entity_start,
                                                'entity_end_time': entity_end,
                                                'overlap_start': overlap_start,
                                                'overlap_end': overlap_end,
                                                'overlap_duration': overlap_duration
                                            },
                                            relevance_score=entity.confidence,
                                            entity_score=entity.confidence,
                                            temporal_score=overlap_duration / (end_time - start_time),
                                            metadata={
                                                'entity_name': entity.name,
                                                'entity_type': entity.entity_type,
                                                'temporal_overlap': overlap_duration
                                            },
                                            explanation=f"Entity {entity.name} overlaps {overlap_duration:.1f}s"
                                        )
                                        results.append(result)
        
        except Exception as e:
            self.logger.error(f"❌ Error retrieving entities: {e}")
        
        return results


    # ==================== HELPER METHODS ====================
    
    def _calculate_text_relevance(self, query: str, text: str, keywords: List[str]) -> float:
        """Calculate text relevance score"""
        if not text:
            return 0.0
        
        try:
            query_lower = query.lower()
            text_lower = text.lower()
            
            # Direct substring match
            substring_score = 0.0
            if query_lower in text_lower:
                substring_score = 0.8
            
            # Keyword matching
            keyword_score = 0.0
            if keywords:
                matched_keywords = 0
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        matched_keywords += 1
                keyword_score = (matched_keywords / len(keywords)) * 0.6
            
            # Word overlap scoring
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            
            if query_words and text_words:
                overlap = len(query_words.intersection(text_words))
                union = len(query_words.union(text_words))
                overlap_score = (overlap / union) * 0.4 if union > 0 else 0.0
            else:
                overlap_score = 0.0
            
            # Combine scores
            total_score = max(substring_score, keyword_score + overlap_score)
            
            return min(total_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating text relevance: {e}")
            return 0.0
    
    def _calculate_entity_relevance(self, query_context: QueryContext, entity: Entity) -> float:
        """Calculate entity relevance score"""
        try:
            score = 0.0
            
            # Name matching
            if query_context.processed_query.lower() in entity.name.lower():
                score += 0.8
            
            # Keyword matching in description
            if entity.description:
                desc_score = self._calculate_text_relevance(
                    query_context.processed_query,
                    entity.description,
                    query_context.keywords
                )
                score += desc_score * 0.5
            
            # Alias matching
            for alias in entity.aliases:
                if query_context.processed_query.lower() in alias.lower():
                    score += 0.6
                    break
            
            # Entity type relevance
            if query_context.query_type == "entity":
                score += 0.2
            
            # Confidence boost
            score += entity.confidence * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating entity relevance: {e}")
            return 0.0
    
    def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for query text"""
        try:
            result = self.model_manager.inference_text_embedding([query])
            
            if result.success and result.result is not None:
                return result.result[0]  # First embedding
            else:
                self.logger.warning(f"⚠️ Failed to generate query embedding: {result.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error generating query embedding: {e}")
            return None
    
    def _find_related_entities(self, entity: Entity) -> List[Tuple[Entity, float]]:
        """Find entities related to given entity through relations"""
        related = []
        
        try:
            if not self.db_manager.graph_db:
                return related
            
            # Find all relations involving this entity
            for graph in self.db_manager.graph_db.knowledge_graphs.values():
                for relation in graph.relations.values():
                    related_entity_id = None
                    relation_strength = relation.confidence
                    
                    if relation.source_entity == entity.entity_id:
                        related_entity_id = relation.target_entity
                    elif relation.target_entity == entity.entity_id:
                        related_entity_id = relation.source_entity
                    
                    if related_entity_id and related_entity_id in graph.entities:
                        related_entity = graph.entities[related_entity_id]
                        related.append((related_entity, relation_strength))
            
            # Sort by relation strength
            related.sort(key=lambda x: x[1], reverse=True)
            
            return related[:10]  # Top 10 related entities
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error finding related entities: {e}")
            return []
    
    def _merge_duplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Merge duplicate results and combine scores"""
        merged = {}
        
        try:
            for result in results:
                key = f"{result.result_type}_{result.result_id}"
                
                if key in merged:
                    # Merge scores
                    existing = merged[key]
                    existing.text_score = max(existing.text_score, result.text_score)
                    existing.visual_score = max(existing.visual_score, result.visual_score)
                    existing.entity_score = max(existing.entity_score, result.entity_score)
                    
                    # Recalculate relevance score
                    existing.relevance_score = (
                        existing.text_score * self.text_weight +
                        existing.visual_score * self.visual_weight +
                        existing.entity_score * self.entity_weight
                    )
                    
                    # Combine explanations
                    if result.explanation not in existing.explanation:
                        existing.explanation += f" | {result.explanation}"
                else:
                    merged[key] = result
            
            return list(merged.values())
            
        except Exception as e:
            self.logger.error(f"❌ Error merging duplicate results: {e}")
            return results
    
    def _apply_temporal_scoring(
        self, 
        results: List[RetrievalResult], 
        query_context: QueryContext
    ) -> List[RetrievalResult]:
        """Apply temporal scoring based on constraints"""
        try:
            if not query_context.temporal_constraints:
                return results
            
            constraints = query_context.temporal_constraints
            
            for result in results:
                temporal_score = 0.0
                
                # Get timestamp from result
                timestamp = None
                if hasattr(result.content, 'timestamp'):
                    timestamp = result.content.timestamp
                elif 'timestamp' in result.metadata:
                    timestamp = result.metadata['timestamp']
                elif hasattr(result.content, 'start_time'):
                    timestamp = result.content.start_time
                
                if timestamp is not None:
                    # Apply time preference scoring
                    time_pref = constraints.get('time_preference')
                    if time_pref == 'early' and timestamp < 30:  # First 30 seconds
                        temporal_score += 0.3
                    elif time_pref == 'late' and timestamp > 120:  # After 2 minutes  
                        temporal_score += 0.3
                    elif time_pref == 'middle' and 30 <= timestamp <= 120:
                        temporal_score += 0.3
                
                # Apply duration preference if available
                duration_pref = constraints.get('duration_preference')
                if duration_pref and hasattr(result.content, 'end_time'):
                    duration = result.content.end_time - result.content.start_time
                    if duration_pref == 'short' and duration < 10:
                        temporal_score += 0.2
                    elif duration_pref == 'long' and duration > 30:
                        temporal_score += 0.2
                
                # Update scores
                result.temporal_score = temporal_score
                result.relevance_score += temporal_score * self.temporal_weight
            
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error applying temporal scoring: {e}")
            return results
    
    def _post_process_results(
        self, 
        results: List[RetrievalResult], 
        query_context: QueryContext
    ) -> List[RetrievalResult]:
        """Post-process và rank final results"""
        try:
            # Filter by confidence threshold
            filtered_results = [
                r for r in results 
                if r.relevance_score >= query_context.confidence_threshold
            ]
            
            # Sort by relevance score
            filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply diversity filtering to avoid too many similar results
            diverse_results = self._apply_diversity_filtering(filtered_results)
            
            # Enhance results với additional metadata
            enhanced_results = self._enhance_results_metadata(diverse_results, query_context)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"❌ Error post-processing results: {e}")
            return results
    
    def _apply_diversity_filtering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply diversity filtering để avoid redundant results"""
        try:
            if len(results) <= 5:
                return results
            
            diverse_results = []
            used_videos = set()
            
            for result in results:
                # Get video ID from result
                video_id = self._get_video_id(result)
                
                # Limit results per video
                if video_id:
                    video_count = sum(1 for r in diverse_results if self._get_video_id(r) == video_id)
                    if video_count >= 3:  # Max 3 results per video
                        continue
                
                diverse_results.append(result)
                
                if len(diverse_results) >= self.top_k_final * 2:
                    break
            
            return diverse_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error applying diversity filtering: {e}")
            return results
    
    def _enhance_results_metadata(
        self, 
        results: List[RetrievalResult], 
        query_context: QueryContext
    ) -> List[RetrievalResult]:
        """Enhance results với additional metadata"""
        try:
            for result in results:
                # Add query matching information
                result.metadata['query_keywords_matched'] = [
                    kw for kw in query_context.keywords
                    if self._keyword_matches_result(kw, result)
                ]
                
                # Add ranking information
                result.metadata['rank'] = len([r for r in results if r.relevance_score > result.relevance_score]) + 1
                
                # Add score breakdown
                result.metadata['score_breakdown'] = {
                    'text': result.text_score,
                    'visual': result.visual_score,
                    'entity': result.entity_score,
                    'temporal': result.temporal_score,
                    'total': result.relevance_score
                }
            
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error enhancing results metadata: {e}")
            return results
    
    def _get_video_id(self, result: RetrievalResult) -> Optional[str]:
        """Extract video ID from result"""
        if hasattr(result.content, 'video_id'):
            return result.content.video_id
        elif 'video_id' in result.metadata:
            return result.metadata['video_id']
        elif isinstance(result.content, dict) and 'video_id' in result.content:
            return result.content['video_id']
        return None
    
    def _keyword_matches_result(self, keyword: str, result: RetrievalResult) -> bool:
        """Check if keyword matches result content"""
        keyword_lower = keyword.lower()
        
        # Check in explanation
        if keyword_lower in result.explanation.lower():
            return True
        
        # Check in content description
        if hasattr(result.content, 'caption') and result.content.caption:
            if keyword_lower in result.content.caption.lower():
                return True
        
        if hasattr(result.content, 'transcript') and result.content.transcript:
            if keyword_lower in result.content.transcript.lower():
                return True
        
        if hasattr(result.content, 'description') and result.content.description:
            if keyword_lower in result.content.description.lower():
                return True
        
        return False


    # ==================== CACHING METHODS ====================
    
    def _generate_cache_key(self, query_context: QueryContext, strategy: str) -> str:
        """Generate cache key for query"""
        import hashlib
        
        key_components = [
            query_context.processed_query,
            strategy,
            str(query_context.video_filters),
            str(query_context.max_results),
            str(query_context.confidence_threshold)
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _get_cached_response(self, cache_key: str) -> Optional[RetrievalResponse]:
        """Get cached response if available và valid"""
        with self._cache_lock:
            if cache_key in self.query_cache:
                cached_response, cache_time = self.query_cache[cache_key]
                
                # Check TTL
                if time.time() - cache_time < self.cache_ttl:
                    return cached_response
                else:
                    # Remove expired cache
                    del self.query_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: RetrievalResponse):
        """Cache response với TTL"""
        with self._cache_lock:
            # Clean old cache entries if needed
            if len(self.query_cache) >= self.cache_max_size:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.query_cache.items(),
                    key=lambda x: x[1][1]  # Sort by cache time
                )
                
                # Remove oldest 25%
                to_remove = len(sorted_cache) // 4
                for i in range(to_remove):
                    del self.query_cache[sorted_cache[i][0]]
            
            self.query_cache[cache_key] = (response, time.time())


    # ==================== RELEVANCE FILTERING METHODS ====================
    
    @retry_on_failure(max_retries=2)
    def filter_relevance(
        self,
        results: List[RetrievalResult],
        query_context: QueryContext,
        use_llm: bool = True
    ) -> List[RetrievalResult]:
        """Filter results based on relevance using LLM"""
        if not results or not use_llm:
            return results
        
        try:
            # Prepare results for LLM evaluation
            results_for_evaluation = []
            
            for i, result in enumerate(results[:10]):  # Limit to top 10 for LLM
                content_description = ""
                
                if result.result_type == 'frame':
                    content_description = result.content.get('caption', '') if isinstance(result.content, dict) else getattr(result.content, 'caption', '')
                elif result.result_type == 'audio_segment':
                    content_description = result.content.get('transcript', '') if isinstance(result.content, dict) else getattr(result.content, 'transcript', '')
                elif result.result_type == 'entity':
                    entity = result.content.get('entity', result.content) if isinstance(result.content, dict) else result.content
                    content_description = f"{getattr(entity, 'name', '')}: {getattr(entity, 'description', '')}"
                elif 'embedding' in result.result_type:
                    content_description = str(result.content.metadata)
                
                results_for_evaluation.append({
                    'index': i,
                    'type': result.result_type,
                    'description': content_description,
                    'score': result.relevance_score,
                    'explanation': result.explanation
                })
            
            # Create LLM prompt for relevance filtering
            prompt = f"""Evaluate the relevance of these video search results for the query: "{query_context.original_query}"

Query context:
- Processed query: {query_context.processed_query}
- Query type: {query_context.query_type}
- Keywords: {query_context.keywords}

Results to evaluate:
"""
            
            for i, res_eval in enumerate(results_for_evaluation):
                prompt += f"""
{i+1}. Type: {res_eval['type']}
   Description: {res_eval['description'][:200]}
   Current score: {res_eval['score']:.3f}
   Explanation: {res_eval['explanation']}
"""
            
            prompt += """
For each result, determine if it's relevant (1) or not relevant (0) to the query.
Consider semantic meaning, context, and user intent.

Respond with only a JSON array of 1s and 0s, one for each result.
Example: [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]

Relevance scores:"""
            
            # Get LLM evaluation
            llm_result = self.model_manager.inference_gpt4o_mini(
                [{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            if llm_result.success and llm_result.result:
                try:
                    # Parse LLM response
                    import json
                    response_text = llm_result.result.strip()
                    
                    # Extract JSON array from response
                    if '[' in response_text and ']' in response_text:
                        start_idx = response_text.find('[')
                        end_idx = response_text.rfind(']') + 1
                        json_text = response_text[start_idx:end_idx]
                        relevance_scores = json.loads(json_text)
                        
                        # Filter results based on LLM evaluation
                        filtered_results = []
                        for i, (result, is_relevant) in enumerate(zip(results, relevance_scores)):
                            if is_relevant == 1:
                                result.metadata['llm_filtered'] = True
                                filtered_results.append(result)
                            elif i >= len(relevance_scores):
                                # Include remaining results if LLM didn't evaluate them
                                filtered_results.append(result)
                        
                        # Add remaining results that weren't evaluated
                        if len(results) > len(relevance_scores):
                            filtered_results.extend(results[len(relevance_scores):])
                        
                        self.logger.debug(f"🤖 LLM filtered {len(results)} → {len(filtered_results)} results")
                        return filtered_results
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error parsing LLM relevance response: {e}")
            
            # Fallback to score-based filtering
            threshold = query_context.confidence_threshold
            filtered_results = [r for r in results if r.relevance_score >= threshold]
            
            return filtered_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in relevance filtering: {e}")
            return results


    # ==================== STATISTICS AND MONITORING ====================
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics"""
        try:
            stats = {
                'cache_stats': {
                    'cache_size': len(self.query_cache),
                    'cache_max_size': self.cache_max_size,
                    'cache_ttl': self.cache_ttl
                },
                'config': {
                    'similarity_threshold': self.similarity_threshold,
                    'top_k_text': self.top_k_text,
                    'top_k_visual': self.top_k_visual,
                    'top_k_final': self.top_k_final,
                    'text_weight': self.text_weight,
                    'visual_weight': self.visual_weight,
                    'entity_weight': self.entity_weight,
                    'temporal_weight': self.temporal_weight
                },
                'capabilities': {
                    'database_manager': self.db_manager is not None,
                    'model_manager': self.model_manager is not None,
                    'knowledge_builder': self.knowledge_builder is not None,
                    'cross_modal_enabled': self.enable_cross_modal,
                    'temporal_filtering_enabled': self.enable_temporal_filtering
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting retrieval stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear query cache"""
        with self._cache_lock:
            self.query_cache.clear()
            self.logger.info("🧹 Query cache cleared")


# ==================== GLOBAL INSTANCE MANAGEMENT ====================

# Global retrieval engine instance
_global_retrieval_engine = None

def get_retrieval_engine(
    config: Optional[Config] = None,
    database_manager: Optional[DatabaseManager] = None,
    model_manager: Optional[ModelManager] = None,
    knowledge_builder: Optional[KnowledgeBuilder] = None
) -> RetrievalEngine:
    """Get global retrieval engine instance (singleton pattern)"""
    global _global_retrieval_engine
    if _global_retrieval_engine is None:
        _global_retrieval_engine = RetrievalEngine(config, database_manager, model_manager, knowledge_builder)
    return _global_retrieval_engine

def reset_retrieval_engine():
    """Reset global retrieval engine instance"""
    global _global_retrieval_engine
    if _global_retrieval_engine:
        del _global_retrieval_engine
    _global_retrieval_engine = None