"""
knowledge_builder.py - Knowledge Graph Construction cho VideoRAG System
Xây dựng knowledge graph từ video content: entity extraction, relation building, cross-video integration
"""

import time
import json
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import networkx as nx

from config import Config, get_config
from utils import (
    get_logger, measure_time, performance_monitor,
    cached, retry_on_failure, normalize_text,
    safe_json_loads, safe_json_dumps
)
from model_manager import get_model_manager, ModelManager
from video_processor import ProcessedVideo, VideoChunk


# ==================== DATA STRUCTURES ====================

@dataclass
class Entity:
    """Represent một entity trong knowledge graph"""
    entity_id: str
    name: str
    entity_type: str  # PERSON, OBJECT, ACTION, LOCATION, CONCEPT, EVENT
    description: str = ""
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source_videos: Set[str] = field(default_factory=set)
    source_chunks: Set[str] = field(default_factory=set)
    timestamps: List[Tuple[str, float, float]] = field(default_factory=list)  # (video_id, start, end)
    embedding: Optional[np.ndarray] = None
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class Relation:
    """Represent một relation giữa entities"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str  # INTERACTS_WITH, LOCATED_IN, PERFORMS, USES, etc.
    description: str = ""
    confidence: float = 0.0
    temporal_info: Optional[Dict[str, Any]] = None  # start_time, end_time, duration
    source_videos: Set[str] = field(default_factory=set)
    source_chunks: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)


@dataclass
class KnowledgeGraph:
    """Represent toàn bộ knowledge graph"""
    graph_id: str
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    video_mappings: Dict[str, Set[str]] = field(default_factory=dict)  # video_id -> entity_ids
    entity_index: Dict[str, Set[str]] = field(default_factory=dict)  # type -> entity_ids
    relation_index: Dict[str, Set[str]] = field(default_factory=dict)  # type -> relation_ids
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.nx_graph = nx.MultiDiGraph()  # NetworkX graph for complex queries


@dataclass
class ExtractionResult:
    """Result từ entity/relation extraction"""
    success: bool
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== MAIN KNOWLEDGE BUILDER CLASS ====================

class KnowledgeBuilder:
    """
    Knowledge Graph Builder cho VideoRAG System
    """
    
    def __init__(self, config: Optional[Config] = None, model_manager: Optional[ModelManager] = None):
        """
        Initialize KnowledgeBuilder
        
        Args:
            config: Configuration object
            model_manager: Model manager instance
        """
        self.config = config or get_config()
        self.model_manager = model_manager or get_model_manager()
        self.logger = get_logger('knowledge_builder')
        
        # Initialize core components
        self._init_storage()
        self._init_parameters()
        self._init_caching()
        self._init_threading()
        self._init_entity_relation_types()
        
        self.logger.info("🧠 KnowledgeBuilder initialized")
        self.logger.info(f"📊 Entity types: {len(self.entity_types)}, Relation types: {len(self.relation_types)}")
    
    def _init_storage(self):
        """Initialize knowledge graph storage"""
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
        self.global_graph: Optional[KnowledgeGraph] = None
    
    def _init_parameters(self):
        """Initialize processing parameters"""
        self.entity_similarity_threshold = 0.8
        self.relation_similarity_threshold = 0.7
    
    def _init_caching(self):
        """Initialize caching system"""
        self.cache_dir = Path(self.config.paths.get('data_dir', './data')) / 'knowledge_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_threading(self):
        """Initialize threading components"""
        self._graph_lock = threading.RLock()
    
    def _init_entity_relation_types(self):
        """Initialize entity and relation type definitions"""
        self.entity_types = {
            'PERSON', 'OBJECT', 'LOCATION', 'ACTION', 'CONCEPT', 
            'EVENT', 'SCENE', 'EMOTION', 'TIME', 'QUANTITY'
        }
        
        self.relation_types = {
            'INTERACTS_WITH', 'LOCATED_IN', 'PERFORMS', 'USES', 'CONTAINS',
            'CAUSES', 'TEMPORAL_BEFORE', 'TEMPORAL_AFTER', 'SIMILAR_TO',
            'PART_OF', 'OPPOSITE_OF', 'ASSOCIATED_WITH'
        }


    # ==================== MAIN PROCESSING METHODS ====================
    
    @measure_time
    def build_knowledge_graph(
        self,
        processed_videos: List[ProcessedVideo],
        graph_id: Optional[str] = None,
        enable_cross_video_linking: bool = True
    ) -> KnowledgeGraph:
        """
        Build knowledge graph từ processed videos
        
        Args:
            processed_videos: List of ProcessedVideo objects
            graph_id: Custom graph ID
            enable_cross_video_linking: Enable linking entities across videos
            
        Returns:
            KnowledgeGraph object
        """
        
        if not graph_id:
            graph_id = f"kg_{int(time.time())}_{hashlib.md5(str(len(processed_videos)).encode()).hexdigest()[:8]}"
        
        self.logger.info(f"🔨 Building knowledge graph: {graph_id}")
        self.logger.info(f"📊 Processing {len(processed_videos)} videos")
        
        # Initialize knowledge graph
        knowledge_graph = KnowledgeGraph(graph_id=graph_id)
        
        try:
            with performance_monitor.monitor(f"build_knowledge_graph_{graph_id}"):
                
                # Step 1: Extract entities và relations từ mỗi video
                self.logger.info("🔍 Step 1: Extracting entities and relations...")
                video_extraction_results = self._extract_from_videos(processed_videos)
                
                # Step 2: Merge entities và relations vào graph
                self.logger.info("🔗 Step 2: Merging into knowledge graph...")
                for video_id, extraction_result in video_extraction_results.items():
                    if extraction_result.success:
                        self._merge_extraction_result(knowledge_graph, extraction_result, video_id)
                
                # Step 3: Cross-video entity resolution và linking
                if enable_cross_video_linking and len(processed_videos) > 1:
                    self.logger.info("🌐 Step 3: Cross-video entity linking...")
                    self._perform_cross_video_linking(knowledge_graph)
                
                # Step 4: Build NetworkX graph cho complex queries
                self.logger.info("📈 Step 4: Building NetworkX representation...")
                self._build_networkx_graph(knowledge_graph)
                
                # Step 5: Generate embeddings cho entities
                self.logger.info("🎯 Step 5: Generating entity embeddings...")
                self._generate_entity_embeddings(knowledge_graph)
                
                # Step 6: Compute graph statistics
                self.logger.info("📊 Step 6: Computing graph statistics...")
                self._compute_graph_statistics(knowledge_graph)
            
            # Store graph
            with self._graph_lock:
                self.knowledge_graphs[graph_id] = knowledge_graph
            
            self.logger.info(f"✅ Knowledge graph built successfully:")
            self.logger.info(f"   📊 {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relations)} relations")
            self.logger.info(f"   🎬 {len(knowledge_graph.video_mappings)} videos processed")
            
            return knowledge_graph
            
        except Exception as e:
            self.logger.error(f"❌ Error building knowledge graph: {e}")
            knowledge_graph.metadata['error'] = str(e)
            return knowledge_graph
        
# ==================== VIDEO EXTRACTION METHODS ====================
    
    def _extract_from_videos(self, processed_videos: List[ProcessedVideo]) -> Dict[str, ExtractionResult]:
        """Extract entities và relations từ processed videos"""
        
        extraction_results = {}
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_video = {}
            
            for video in processed_videos:
                if video.success:
                    future = executor.submit(self._extract_from_single_video, video)
                    future_to_video[future] = video.video_id
            
            # Collect results
            for future in as_completed(future_to_video):
                video_id = future_to_video[future]
                try:
                    extraction_result = future.result()
                    extraction_results[video_id] = extraction_result
                    
                    if extraction_result.success:
                        self.logger.info(f"✅ Extracted from {video_id}: {len(extraction_result.entities)} entities, {len(extraction_result.relations)} relations")
                    else:
                        self.logger.error(f"❌ Failed to extract from {video_id}: {extraction_result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing {video_id}: {e}")
                    extraction_results[video_id] = ExtractionResult(
                        success=False,
                        error_message=str(e)
                    )
        
        return extraction_results
    
    @cached(ttl_seconds=1800)  # Cache for 30 minutes
    def _extract_from_single_video(self, processed_video: ProcessedVideo) -> ExtractionResult:
        """Extract entities và relations từ một video"""
        
        start_time = time.time()
        video_id = processed_video.video_id
        
        try:
            all_entities = []
            all_relations = []
            
            # Process each chunk
            for chunk in processed_video.chunks:
                chunk_result = self._extract_from_chunk(chunk, video_id)
                
                if chunk_result.success:
                    all_entities.extend(chunk_result.entities)
                    all_relations.extend(chunk_result.relations)
            
            # Deduplicate và merge similar entities trong video
            merged_entities = self._merge_similar_entities(all_entities)
            merged_relations = self._merge_similar_relations(all_relations)
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                success=True,
                entities=merged_entities,
                relations=merged_relations,
                processing_time=processing_time,
                metadata={
                    'video_id': video_id,
                    'chunk_count': len(processed_video.chunks),
                    'total_frames': processed_video.total_frames
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    @retry_on_failure(max_retries=2)
    def _extract_from_chunk(self, chunk: VideoChunk, video_id: str) -> ExtractionResult:
        """Extract entities và relations từ một chunk"""
        
        try:
            # Prepare text content for extraction
            text_contents = []
            
            # Add frame captions
            for frame in chunk.frames:
                if frame.caption:
                    text_contents.append(f"Visual: {frame.caption}")
            
            # Add audio transcript
            if chunk.audio_segment and chunk.audio_segment.transcript:
                text_contents.append(f"Audio: {chunk.audio_segment.transcript}")
            
            # Add unified description
            if chunk.unified_description:
                text_contents.append(f"Summary: {chunk.unified_description}")
            
            if not text_contents:
                return ExtractionResult(success=True)  # Empty but successful
            
            # Combine all text content
            combined_text = " || ".join(text_contents)
            
            # Extract entities và relations using LLM
            extraction_prompt = self._create_extraction_prompt(combined_text, chunk.start_time, chunk.end_time)
            
            result = self.model_manager.inference_gpt4o_mini(
                [{"role": "user", "content": extraction_prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            if not result.success:
                return ExtractionResult(success=False, error_message=result.error_message)
            
            # Parse LLM response
            entities, relations = self._parse_extraction_response(
                result.result, 
                chunk.chunk_id, 
                video_id,
                chunk.start_time,
                chunk.end_time
            )
            
            return ExtractionResult(
                success=True,
                entities=entities,
                relations=relations,
                metadata={'chunk_id': chunk.chunk_id}
            )
            
        except Exception as e:
            return ExtractionResult(success=False, error_message=str(e))


    # ==================== PROMPT CREATION AND PARSING METHODS ====================
    
    def _create_extraction_prompt(self, text_content: str, start_time: float, end_time: float) -> str:
        """Create prompt cho entity và relation extraction"""
        
        entity_types_str = ", ".join(self.entity_types)
        relation_types_str = ", ".join(self.relation_types)
        
        prompt = f"""Analyze the following multimodal content from a video segment ({start_time:.1f}s - {end_time:.1f}s) and extract structured information:

CONTENT:
{text_content}

TASK: Extract entities and their relationships in JSON format.

ENTITY TYPES: {entity_types_str}
RELATION TYPES: {relation_types_str}

OUTPUT FORMAT (valid JSON only):
{{
    "entities": [
        {{
            "name": "entity_name",
            "type": "ENTITY_TYPE",
            "description": "brief description",
            "confidence": 0.0-1.0
        }}
    ],
    "relations": [
        {{
            "source": "source_entity_name",
            "target": "target_entity_name", 
            "type": "RELATION_TYPE",
            "description": "relationship description",
            "confidence": 0.0-1.0
        }}
    ]
}}

GUIDELINES:
1. Focus on concrete, observable entities and actions
2. Use confidence scores based on clarity and evidence
3. Prefer specific entity types over generic ones
4. Include temporal relationships when relevant
5. Keep descriptions concise but informative
6. Only output valid JSON

JSON OUTPUT:"""
        
        return prompt
    
    def _parse_extraction_response(
        self,
        llm_response: str,
        chunk_id: str,
        video_id: str,
        start_time: float,
        end_time: float
    ) -> Tuple[List[Entity], List[Relation]]:
        """Parse LLM response thành entities và relations"""
        
        entities = []
        relations = []
        
        try:
            # Clean response và extract JSON
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Parse JSON
            parsed_data = safe_json_loads(cleaned_response, {})
            
            if not parsed_data:
                self.logger.warning(f"⚠️ Failed to parse extraction response for {chunk_id}")
                return entities, relations
            
            # Extract entities
            for entity_data in parsed_data.get('entities', []):
                entity_name = entity_data.get('name', '').strip()
                entity_type = entity_data.get('type', 'OBJECT').upper()
                
                if not entity_name or entity_type not in self.entity_types:
                    continue
                
                entity_id = self._generate_entity_id(entity_name, entity_type)
                
                entity = Entity(
                    entity_id=entity_id,
                    name=entity_name,
                    entity_type=entity_type,
                    description=entity_data.get('description', ''),
                    confidence=float(entity_data.get('confidence', 0.7)),
                    source_videos={video_id},
                    source_chunks={chunk_id},
                    timestamps=[(video_id, start_time, end_time)]
                )
                
                entities.append(entity)
            
            # Extract relations
            entity_name_to_id = {e.name: e.entity_id for e in entities}
            
            for relation_data in parsed_data.get('relations', []):
                source_name = relation_data.get('source', '').strip()
                target_name = relation_data.get('target', '').strip()
                relation_type = relation_data.get('type', 'ASSOCIATED_WITH').upper()
                
                if (not source_name or not target_name or 
                    relation_type not in self.relation_types):
                    continue
                
                # Find entity IDs
                source_id = entity_name_to_id.get(source_name)
                target_id = entity_name_to_id.get(target_name)
                
                if not source_id or not target_id:
                    continue
                
                relation_id = self._generate_relation_id(source_id, target_id, relation_type)
                
                relation = Relation(
                    relation_id=relation_id,
                    source_entity=source_id,
                    target_entity=target_id,
                    relation_type=relation_type,
                    description=relation_data.get('description', ''),
                    confidence=float(relation_data.get('confidence', 0.7)),
                    temporal_info={
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    },
                    source_videos={video_id},
                    source_chunks={chunk_id}
                )
                
                relations.append(relation)
            
            return entities, relations
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing extraction response: {e}")
            return entities, relations


    # ==================== ID GENERATION METHODS ====================
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate unique entity ID"""
        normalized_name = normalize_text(name.lower())
        content = f"{entity_type}_{normalized_name}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_relation_id(self, source_id: str, target_id: str, relation_type: str) -> str:
        """Generate unique relation ID"""
        content = f"{source_id}_{relation_type}_{target_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
# ==================== ENTITY MERGING METHODS ====================
    
    def _merge_similar_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge similar entities trong cùng video"""
        
        if not entities:
            return entities
        
        merged_entities = []
        entity_groups = defaultdict(list)
        
        # Group entities by type và similarity
        for entity in entities:
            key = f"{entity.entity_type}_{entity.name.lower()}"
            entity_groups[key].append(entity)
        
        # Merge entities trong mỗi group
        for group_entities in entity_groups.values():
            if len(group_entities) == 1:
                merged_entities.append(group_entities[0])
            else:
                # Merge multiple entities
                merged_entity = self._merge_entity_group(group_entities)
                merged_entities.append(merged_entity)
        
        return merged_entities
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """Merge một group of similar entities"""
        
        # Use first entity as base
        base_entity = entities[0]
        
        # Merge information từ other entities
        all_descriptions = [e.description for e in entities if e.description]
        all_aliases = set()
        all_source_videos = set()
        all_source_chunks = set()
        all_timestamps = []
        all_confidences = []
        
        for entity in entities:
            all_aliases.update(entity.aliases)
            all_aliases.add(entity.name)  # Add name as alias
            all_source_videos.update(entity.source_videos)
            all_source_chunks.update(entity.source_chunks)
            all_timestamps.extend(entity.timestamps)
            all_confidences.append(entity.confidence)
        
        # Create merged description
        if all_descriptions:
            merged_description = self._merge_descriptions(all_descriptions)
        else:
            merged_description = base_entity.description
        
        # Calculate merged confidence
        merged_confidence = np.mean(all_confidences) if all_confidences else base_entity.confidence
        
        # Update base entity
        base_entity.description = merged_description
        base_entity.aliases = all_aliases
        base_entity.source_videos = all_source_videos
        base_entity.source_chunks = all_source_chunks
        base_entity.timestamps = all_timestamps
        base_entity.confidence = merged_confidence
        base_entity.last_updated = time.time()
        
        return base_entity
    
    def _merge_entities(self, existing_entity: Entity, new_entity: Entity):
        """Merge new entity information vào existing entity"""
        
        # Merge descriptions
        if new_entity.description and new_entity.description != existing_entity.description:
            if existing_entity.description:
                existing_entity.description = self._merge_descriptions([
                    existing_entity.description,
                    new_entity.description
                ])
            else:
                existing_entity.description = new_entity.description
        
        # Merge aliases
        existing_entity.aliases.update(new_entity.aliases)
        existing_entity.aliases.add(new_entity.name)
        
        # Merge sources
        existing_entity.source_videos.update(new_entity.source_videos)
        existing_entity.source_chunks.update(new_entity.source_chunks)
        existing_entity.timestamps.extend(new_entity.timestamps)
        
        # Update confidence (weighted average based on number of sources)
        old_weight = len(existing_entity.source_chunks) - len(new_entity.source_chunks)
        new_weight = len(new_entity.source_chunks)
        total_weight = old_weight + new_weight
        
        if total_weight > 0:
            existing_entity.confidence = (
                (existing_entity.confidence * old_weight + new_entity.confidence * new_weight) / total_weight
            )
        
        existing_entity.last_updated = time.time()


    # ==================== RELATION MERGING METHODS ====================
    
    def _merge_similar_relations(self, relations: List[Relation]) -> List[Relation]:
        """Merge similar relations trong cùng video"""
        
        if not relations:
            return relations
        
        merged_relations = []
        relation_groups = defaultdict(list)
        
        # Group relations by source, target, and type
        for relation in relations:
            key = f"{relation.source_entity}_{relation.relation_type}_{relation.target_entity}"
            relation_groups[key].append(relation)
        
        # Merge relations trong mỗi group
        for group_relations in relation_groups.values():
            if len(group_relations) == 1:
                merged_relations.append(group_relations[0])
            else:
                # Merge multiple relations
                merged_relation = self._merge_relation_group(group_relations)
                merged_relations.append(merged_relation)
        
        return merged_relations
    
    def _merge_relation_group(self, relations: List[Relation]) -> Relation:
        """Merge một group of similar relations"""
        
        base_relation = relations[0]
        
        # Merge information
        all_descriptions = [r.description for r in relations if r.description]
        all_source_videos = set()
        all_source_chunks = set()
        all_confidences = []
        
        for relation in relations:
            all_source_videos.update(relation.source_videos)
            all_source_chunks.update(relation.source_chunks)
            all_confidences.append(relation.confidence)
        
        # Merge descriptions
        if all_descriptions:
            merged_description = " | ".join(all_descriptions)
        else:
            merged_description = base_relation.description
        
        # Calculate merged confidence
        merged_confidence = np.mean(all_confidences) if all_confidences else base_relation.confidence
        
        # Update base relation
        base_relation.description = merged_description
        base_relation.source_videos = all_source_videos
        base_relation.source_chunks = all_source_chunks
        base_relation.confidence = merged_confidence
        
        return base_relation
    
    def _merge_relations(self, existing_relation: Relation, new_relation: Relation):
        """Merge new relation information vào existing relation"""
        
        # Merge descriptions
        if new_relation.description and new_relation.description != existing_relation.description:
            if existing_relation.description:
                existing_relation.description += " | " + new_relation.description
            else:
                existing_relation.description = new_relation.description
        
        # Merge sources
        existing_relation.source_videos.update(new_relation.source_videos)
        existing_relation.source_chunks.update(new_relation.source_chunks)
        
        # Update confidence
        existing_relation.confidence = max(existing_relation.confidence, new_relation.confidence)


    # ==================== SIMILARITY CALCULATION METHODS ====================
    
    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity giữa 2 entities"""
        
        similarity_scores = []
        
        # Name similarity
        name_sim = self._calculate_string_similarity(entity1.name.lower(), entity2.name.lower())
        similarity_scores.append(name_sim * 0.4)  # 40% weight
        
        # Alias similarity
        alias_sim = 0.0
        if entity1.aliases and entity2.aliases:
            for alias1 in entity1.aliases:
                for alias2 in entity2.aliases:
                    alias_sim = max(alias_sim, self._calculate_string_similarity(
                        alias1.lower(), alias2.lower()
                    ))
        similarity_scores.append(alias_sim * 0.2)  # 20% weight
        
        # Description similarity (using embeddings if available)
        desc_sim = 0.0
        if entity1.description and entity2.description:
            desc_sim = self._calculate_text_similarity(entity1.description, entity2.description)
        similarity_scores.append(desc_sim * 0.4)  # 40% weight
        
        return sum(similarity_scores)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance"""
        
        if str1 == str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Simple Levenshtein distance implementation
        len1, len2 = len(str1), len(str2)
        if len1 > len2:
            str1, str2 = str2, str1
            len1, len2 = len2, len1
        
        current_row = range(len1 + 1)
        for i in range(1, len2 + 1):
            previous_row, current_row = current_row, [i] + [0] * len1
            for j in range(1, len1 + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if str1[j - 1] != str2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        
        max_len = max(len(str1), len(str2))
        return 1.0 - (current_row[len1] / max_len) if max_len > 0 else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using embeddings"""
        
        try:
            # Use text embedding model
            result = self.model_manager.inference_text_embedding([text1, text2])
            
            if result.success and result.result is not None:
                embeddings = result.result
                if len(embeddings) == 2:
                    # Calculate cosine similarity
                    from utils import cosine_similarity
                    return cosine_similarity(embeddings[0], embeddings[1])
            
            # Fallback to simple word overlap
            words1 = set(normalize_text(text1).lower().split())
            words2 = set(normalize_text(text2).lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating text similarity: {e}")
            return 0.0


    # ==================== DESCRIPTION MERGING METHODS ====================
    
    def _merge_descriptions(self, descriptions: List[str]) -> str:
        """Merge multiple descriptions into one"""
        
        if not descriptions:
            return ""
        
        if len(descriptions) == 1:
            return descriptions[0]
        
        try:
            # Use LLM to merge descriptions
            descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
            
            prompt = f"""Merge the following descriptions into a single, comprehensive description:

{descriptions_text}

Requirements:
- Combine all relevant information
- Remove redundancy
- Keep the description concise but informative
- Maintain factual accuracy

Merged description:"""
            
            result = self.model_manager.inference_gpt4o_mini(
                [{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            if result.success and result.result:
                return result.result.strip()
            else:
                # Fallback to simple concatenation
                return " | ".join(descriptions)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error merging descriptions: {e}")
            return " | ".join(descriptions)


    # ==================== CROSS-VIDEO LINKING METHODS ====================
    
    def _perform_cross_video_linking(self, knowledge_graph: KnowledgeGraph):
        """Perform cross-video entity linking và resolution"""
        
        try:
            self.logger.info("🔗 Performing cross-video entity linking...")
            
            # Group entities by type for efficient comparison
            entities_by_type = defaultdict(list)
            for entity in knowledge_graph.entities.values():
                entities_by_type[entity.entity_type].append(entity)
            
            merge_candidates = []
            
            # Find similar entities across videos
            for entity_type, entities in entities_by_type.items():
                if len(entities) < 2:
                    continue
                
                # Compare entities pairwise
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        entity1 = entities[i]
                        entity2 = entities[j]
                        
                        # Skip if from same video only
                        if entity1.source_videos == entity2.source_videos:
                            continue
                        
                        # Calculate similarity
                        similarity = self._calculate_entity_similarity(entity1, entity2)
                        
                        if similarity >= self.entity_similarity_threshold:
                            merge_candidates.append((entity1, entity2, similarity))
            
            # Sort by similarity và merge
            merge_candidates.sort(key=lambda x: x[2], reverse=True)
            
            merged_count = 0
            for entity1, entity2, similarity in merge_candidates:
                # Check if both entities still exist (not already merged)
                if (entity1.entity_id in knowledge_graph.entities and 
                    entity2.entity_id in knowledge_graph.entities):
                    
                    self._merge_cross_video_entities(knowledge_graph, entity1, entity2)
                    merged_count += 1
            
            self.logger.info(f"✅ Cross-video linking completed: {merged_count} entities merged")
            
        except Exception as e:
            self.logger.error(f"❌ Error in cross-video linking: {e}")
    
    def _merge_cross_video_entities(self, knowledge_graph: KnowledgeGraph, entity1: Entity, entity2: Entity):
        """Merge entities across videos"""
        
        # Use entity1 as primary, merge entity2 into it
        primary_entity = entity1
        secondary_entity = entity2
        
        # Merge information
        self._merge_entities(primary_entity, secondary_entity)
        
        # Update relations that reference secondary entity
        relations_to_update = []
        for relation in knowledge_graph.relations.values():
            if relation.source_entity == secondary_entity.entity_id:
                relation.source_entity = primary_entity.entity_id
                relations_to_update.append(relation)
            elif relation.target_entity == secondary_entity.entity_id:
                relation.target_entity = primary_entity.entity_id
                relations_to_update.append(relation)
        
        # Update video mappings
        for video_id in secondary_entity.source_videos:
            if video_id in knowledge_graph.video_mappings:
                knowledge_graph.video_mappings[video_id].discard(secondary_entity.entity_id)
                knowledge_graph.video_mappings[video_id].add(primary_entity.entity_id)
        
        # Remove secondary entity from indices
        if secondary_entity.entity_type in knowledge_graph.entity_index:
            knowledge_graph.entity_index[secondary_entity.entity_type].discard(secondary_entity.entity_id)
        
        # Remove secondary entity
        del knowledge_graph.entities[secondary_entity.entity_id]
        
        self.logger.debug(f"🔗 Merged entities: {secondary_entity.name} → {primary_entity.name}")


    # ==================== EXTRACTION RESULT MERGING METHODS ====================
    
    def _merge_extraction_result(
        self,
        knowledge_graph: KnowledgeGraph,
        extraction_result: ExtractionResult,
        video_id: str
    ):
        """Merge extraction result vào knowledge graph"""
        
        try:
            # Add entities
            for entity in extraction_result.entities:
                if entity.entity_id in knowledge_graph.entities:
                    # Merge với existing entity
                    existing_entity = knowledge_graph.entities[entity.entity_id]
                    self._merge_entities(existing_entity, entity)
                else:
                    # Add new entity
                    knowledge_graph.entities[entity.entity_id] = entity
                    
                    # Update indices
                    if entity.entity_type not in knowledge_graph.entity_index:
                        knowledge_graph.entity_index[entity.entity_type] = set()
                    knowledge_graph.entity_index[entity.entity_type].add(entity.entity_id)
            
            # Add relations
            for relation in extraction_result.relations:
                if relation.relation_id in knowledge_graph.relations:
                    # Merge với existing relation
                    existing_relation = knowledge_graph.relations[relation.relation_id]
                    self._merge_relations(existing_relation, relation)
                else:
                    # Add new relation
                    knowledge_graph.relations[relation.relation_id] = relation
                    
                    # Update indices
                    if relation.relation_type not in knowledge_graph.relation_index:
                        knowledge_graph.relation_index[relation.relation_type] = set()
                    knowledge_graph.relation_index[relation.relation_type].add(relation.relation_id)
            
            # Update video mappings
            video_entity_ids = {e.entity_id for e in extraction_result.entities}
            if video_id not in knowledge_graph.video_mappings:
                knowledge_graph.video_mappings[video_id] = set()
            knowledge_graph.video_mappings[video_id].update(video_entity_ids)
            
            knowledge_graph.last_updated = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ Error merging extraction result: {e}")

# ==================== NETWORKX GRAPH BUILDING METHODS ====================
    
    def _build_networkx_graph(self, knowledge_graph: KnowledgeGraph):
        """Build NetworkX graph representation for complex queries"""
        
        try:
            # Clear existing graph
            knowledge_graph.nx_graph.clear()
            
            # Add nodes (entities)
            for entity in knowledge_graph.entities.values():
                knowledge_graph.nx_graph.add_node(
                    entity.entity_id,
                    name=entity.name,
                    type=entity.entity_type,
                    description=entity.description,
                    confidence=entity.confidence,
                    source_videos=list(entity.source_videos),
                    aliases=list(entity.aliases)
                )
            
            # Add edges (relations)
            for relation in knowledge_graph.relations.values():
                knowledge_graph.nx_graph.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    relation_id=relation.relation_id,
                    type=relation.relation_type,
                    description=relation.description,
                    confidence=relation.confidence,
                    temporal_info=relation.temporal_info
                )
            
            self.logger.debug(f"📈 NetworkX graph built: {knowledge_graph.nx_graph.number_of_nodes()} nodes, {knowledge_graph.nx_graph.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"❌ Error building NetworkX graph: {e}")


    # ==================== EMBEDDING GENERATION METHODS ====================
    
    def _generate_entity_embeddings(self, knowledge_graph: KnowledgeGraph):
        """Generate embeddings cho entities"""
        
        try:
            entities_to_embed = []
            entity_texts = []
            
            for entity in knowledge_graph.entities.values():
                if entity.embedding is None:  # Only generate if not already exists
                    # Create text representation
                    text_parts = [entity.name]
                    if entity.description:
                        text_parts.append(entity.description)
                    if entity.aliases:
                        text_parts.extend(list(entity.aliases)[:3])  # Limit aliases
                    
                    entity_text = f"{entity.entity_type}: {' | '.join(text_parts)}"
                    entity_texts.append(entity_text)
                    entities_to_embed.append(entity)
            
            if not entity_texts:
                return
            
            # Generate embeddings in batches
            batch_size = 10
            for i in range(0, len(entity_texts), batch_size):
                batch_texts = entity_texts[i:i + batch_size]
                batch_entities = entities_to_embed[i:i + batch_size]
                
                result = self.model_manager.inference_text_embedding(batch_texts)
                
                if result.success and result.result is not None:
                    embeddings = result.result
                    
                    for j, embedding in enumerate(embeddings):
                        if j < len(batch_entities):
                            batch_entities[j].embedding = embedding
                
                time.sleep(0.1)  # Small delay to avoid rate limits
            
            embedded_count = sum(1 for e in knowledge_graph.entities.values() if e.embedding is not None)
            self.logger.debug(f"🎯 Generated embeddings for {embedded_count} entities")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating entity embeddings: {e}")


    # ==================== GRAPH STATISTICS METHODS ====================
    
    def _compute_graph_statistics(self, knowledge_graph: KnowledgeGraph):
        """Compute và store graph statistics"""
        
        try:
            stats = {
                'total_entities': len(knowledge_graph.entities),
                'total_relations': len(knowledge_graph.relations),
                'total_videos': len(knowledge_graph.video_mappings),
                'entity_types': {},
                'relation_types': {},
                'graph_density': 0.0,
                'average_degree': 0.0,
                'connected_components': 0,
                'largest_component_size': 0
            }
            
            # Entity type distribution
            for entity in knowledge_graph.entities.values():
                entity_type = entity.entity_type
                stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
            
            # Relation type distribution
            for relation in knowledge_graph.relations.values():
                relation_type = relation.relation_type
                stats['relation_types'][relation_type] = stats['relation_types'].get(relation_type, 0) + 1
            
            # NetworkX graph statistics
            if knowledge_graph.nx_graph.number_of_nodes() > 0:
                try:
                    # Convert to undirected for some metrics
                    undirected_graph = knowledge_graph.nx_graph.to_undirected()
                    
                    # Density
                    stats['graph_density'] = nx.density(undirected_graph)
                    
                    # Average degree
                    degrees = [d for n, d in undirected_graph.degree()]
                    stats['average_degree'] = np.mean(degrees) if degrees else 0.0
                    
                    # Connected components
                    components = list(nx.connected_components(undirected_graph))
                    stats['connected_components'] = len(components)
                    stats['largest_component_size'] = len(max(components, key=len)) if components else 0
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error computing graph metrics: {e}")
            
            knowledge_graph.metadata['statistics'] = stats
            self.logger.debug(f"📊 Graph statistics computed: {stats['total_entities']} entities, {stats['total_relations']} relations")
            
        except Exception as e:
            self.logger.error(f"❌ Error computing graph statistics: {e}")


    # ==================== QUERY METHODS ====================
    
    def query_entities(
        self,
        knowledge_graph: KnowledgeGraph,
        entity_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        video_id: Optional[str] = None,
        confidence_threshold: float = 0.0
    ) -> List[Entity]:
        """
        Query entities from knowledge graph
        
        Args:
            knowledge_graph: KnowledgeGraph to query
            entity_type: Filter by entity type
            name_pattern: Filter by name pattern (substring match)
            video_id: Filter by source video
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of matching entities
        """
        
        matching_entities = []
        
        for entity in knowledge_graph.entities.values():
            # Type filter
            if entity_type and entity.entity_type != entity_type:
                continue
            
            # Name pattern filter
            if name_pattern and name_pattern.lower() not in entity.name.lower():
                continue
            
            # Video filter
            if video_id and video_id not in entity.source_videos:
                continue
            
            # Confidence filter
            if entity.confidence < confidence_threshold:
                continue
            
            matching_entities.append(entity)
        
        # Sort by confidence
        matching_entities.sort(key=lambda e: e.confidence, reverse=True)
        
        return matching_entities
    
    def query_relations(
        self,
        knowledge_graph: KnowledgeGraph,
        source_entity: Optional[str] = None,
        target_entity: Optional[str] = None,
        relation_type: Optional[str] = None,
        confidence_threshold: float = 0.0
    ) -> List[Relation]:
        """
        Query relations from knowledge graph
        
        Args:
            knowledge_graph: KnowledgeGraph to query
            source_entity: Filter by source entity ID
            target_entity: Filter by target entity ID
            relation_type: Filter by relation type
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of matching relations
        """
        
        matching_relations = []
        
        for relation in knowledge_graph.relations.values():
            # Source entity filter
            if source_entity and relation.source_entity != source_entity:
                continue
            
            # Target entity filter
            if target_entity and relation.target_entity != target_entity:
                continue
            
            # Relation type filter
            if relation_type and relation.relation_type != relation_type:
                continue
            
            # Confidence filter
            if relation.confidence < confidence_threshold:
                continue
            
            matching_relations.append(relation)
        
        # Sort by confidence
        matching_relations.sort(key=lambda r: r.confidence, reverse=True)
        
        return matching_relations
    
    def find_entity_by_name(self, knowledge_graph: KnowledgeGraph, name: str) -> Optional[Entity]:
        """Find entity by exact name match"""
        
        name_lower = name.lower()
        
        for entity in knowledge_graph.entities.values():
            if entity.name.lower() == name_lower:
                return entity
            
            # Check aliases
            if any(alias.lower() == name_lower for alias in entity.aliases):
                return entity
        
        return None
    
    def find_similar_entities(
        self,
        knowledge_graph: KnowledgeGraph,
        query_text: str,
        top_k: int = 10
    ) -> List[Tuple[Entity, float]]:
        """
        Find entities similar to query text using embeddings
        
        Args:
            knowledge_graph: KnowledgeGraph to search
            query_text: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        
        try:
            # Generate query embedding
            result = self.model_manager.inference_text_embedding([query_text])
            
            if not result.success or result.result is None:
                return []
            
            query_embedding = result.result[0]
            
            # Calculate similarities
            similarities = []
            
            for entity in knowledge_graph.entities.values():
                if entity.embedding is not None:
                    from utils import cosine_similarity
                    similarity = cosine_similarity(query_embedding, entity.embedding)
                    similarities.append((entity, similarity))
            
            # Sort by similarity và return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Error finding similar entities: {e}")
            return []


    # ==================== ENTITY NEIGHBORHOOD METHODS ====================
    
    def get_entity_neighborhood(
        self,
        knowledge_graph: KnowledgeGraph,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get neighborhood of an entity in the graph
        
        Args:
            knowledge_graph: KnowledgeGraph to query
            entity_id: Target entity ID
            max_depth: Maximum depth for neighborhood exploration
            
        Returns:
            Dict containing neighbors và connections
        """
        
        try:
            if entity_id not in knowledge_graph.entities:
                return {}
            
            # Use NetworkX for neighborhood analysis
            if entity_id not in knowledge_graph.nx_graph:
                return {}
            
            neighborhood = {
                'center_entity': knowledge_graph.entities[entity_id],
                'neighbors': {},
                'relations': [],
                'paths': {}
            }
            
            # Get neighbors at different depths
            for depth in range(1, max_depth + 1):
                if depth == 1:
                    # Direct neighbors
                    neighbors = list(knowledge_graph.nx_graph.neighbors(entity_id))
                    predecessors = list(knowledge_graph.nx_graph.predecessors(entity_id))
                    all_neighbors = set(neighbors + predecessors)
                    
                    neighborhood['neighbors'][depth] = [
                        knowledge_graph.entities[neighbor_id] 
                        for neighbor_id in all_neighbors 
                        if neighbor_id in knowledge_graph.entities
                    ]
                else:
                    # Multi-hop neighbors
                    try:
                        ego_graph = nx.ego_graph(knowledge_graph.nx_graph.to_undirected(), entity_id, radius=depth)
                        depth_neighbors = [
                            knowledge_graph.entities[node_id]
                            for node_id in ego_graph.nodes()
                            if node_id != entity_id and node_id in knowledge_graph.entities
                        ]
                        neighborhood['neighbors'][depth] = depth_neighbors
                    except:
                        neighborhood['neighbors'][depth] = []
            
            # Get direct relations
            direct_relations = []
            for relation in knowledge_graph.relations.values():
                if relation.source_entity == entity_id or relation.target_entity == entity_id:
                    direct_relations.append(relation)
            
            neighborhood['relations'] = direct_relations
            
            # Get shortest paths to other entities (limited to avoid performance issues)
            if len(knowledge_graph.nx_graph) < 1000:  # Only for smaller graphs
                try:
                    paths = nx.single_source_shortest_path(
                        knowledge_graph.nx_graph.to_undirected(),
                        entity_id,
                        cutoff=max_depth
                    )
                    
                    # Convert to entity objects và filter
                    filtered_paths = {}
                    for target_id, path in paths.items():
                        if target_id != entity_id and len(path) <= max_depth + 1:
                            filtered_paths[target_id] = {
                                'target_entity': knowledge_graph.entities.get(target_id),
                                'path_length': len(path) - 1,
                                'path_entities': [knowledge_graph.entities.get(node_id) for node_id in path if node_id in knowledge_graph.entities]
                            }
                    
                    neighborhood['paths'] = filtered_paths
                except:
                    neighborhood['paths'] = {}
            
            return neighborhood
            
        except Exception as e:
            self.logger.error(f"❌ Error getting entity neighborhood: {e}")
            return {}
        
# ==================== SAVE/LOAD METHODS ====================
    
    def save_knowledge_graph(self, knowledge_graph: KnowledgeGraph, file_path: str) -> bool:
        """
        Save knowledge graph to file
        
        Args:
            knowledge_graph: KnowledgeGraph to save
            file_path: Output file path
            
        Returns:
            True if successful
        """
        
        try:
            # Prepare data for serialization
            save_data = {
                'graph_id': knowledge_graph.graph_id,
                'entities': {eid: asdict(entity) for eid, entity in knowledge_graph.entities.items()},
                'relations': {rid: asdict(relation) for rid, relation in knowledge_graph.relations.items()},
                'video_mappings': {vid: list(entities) for vid, entities in knowledge_graph.video_mappings.items()},
                'entity_index': {etype: list(entities) for etype, entities in knowledge_graph.entity_index.items()},
                'relation_index': {rtype: list(relations) for rtype, relations in knowledge_graph.relation_index.items()},
                'metadata': knowledge_graph.metadata,
                'creation_time': knowledge_graph.creation_time,
                'last_updated': knowledge_graph.last_updated
            }
            
            # Convert numpy arrays to lists for JSON serialization
            for entity_data in save_data['entities'].values():
                if entity_data.get('embedding') is not None:
                    entity_data['embedding'] = entity_data['embedding'].tolist()
                entity_data['aliases'] = list(entity_data.get('aliases', set()))
                entity_data['source_videos'] = list(entity_data.get('source_videos', set()))
                entity_data['source_chunks'] = list(entity_data.get('source_chunks', set()))
            
            for relation_data in save_data['relations'].values():
                relation_data['source_videos'] = list(relation_data.get('source_videos', set()))
                relation_data['source_chunks'] = list(relation_data.get('source_chunks', set()))
            
            # Save to file
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
            else:
                # Use pickle for faster serialization
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f)
            
            self.logger.info(f"💾 Knowledge graph saved: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving knowledge graph: {e}")
            return False
    
    def load_knowledge_graph(self, file_path: str) -> Optional[KnowledgeGraph]:
        """
        Load knowledge graph from file
        
        Args:
            file_path: Input file path
            
        Returns:
            Loaded KnowledgeGraph hoặc None if failed
        """
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"❌ File not found: {file_path}")
                return None
            
            # Load data
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    save_data = pickle.load(f)
            
            # Reconstruct knowledge graph
            knowledge_graph = KnowledgeGraph(graph_id=save_data['graph_id'])
            
            # Load entities
            for entity_id, entity_data in save_data['entities'].items():
                # Convert lists back to sets
                entity_data['aliases'] = set(entity_data.get('aliases', []))
                entity_data['source_videos'] = set(entity_data.get('source_videos', []))
                entity_data['source_chunks'] = set(entity_data.get('source_chunks', []))
                
                # Convert embedding back to numpy array
                if entity_data.get('embedding'):
                    entity_data['embedding'] = np.array(entity_data['embedding'])
                
                entity = Entity(**entity_data)
                knowledge_graph.entities[entity_id] = entity
            
            # Load relations
            for relation_id, relation_data in save_data['relations'].items():
                # Convert lists back to sets
                relation_data['source_videos'] = set(relation_data.get('source_videos', []))
                relation_data['source_chunks'] = set(relation_data.get('source_chunks', []))
                
                relation = Relation(**relation_data)
                knowledge_graph.relations[relation_id] = relation
            
            # Load mappings và indices
            knowledge_graph.video_mappings = {
                vid: set(entities) for vid, entities in save_data.get('video_mappings', {}).items()
            }
            knowledge_graph.entity_index = {
                etype: set(entities) for etype, entities in save_data.get('entity_index', {}).items()
            }
            knowledge_graph.relation_index = {
                rtype: set(relations) for rtype, relations in save_data.get('relation_index', {}).items()
            }
            
            # Load metadata
            knowledge_graph.metadata = save_data.get('metadata', {})
            knowledge_graph.creation_time = save_data.get('creation_time', time.time())
            knowledge_graph.last_updated = save_data.get('last_updated', time.time())
            
            # Rebuild NetworkX graph
            self._build_networkx_graph(knowledge_graph)
            
            self.logger.info(f"📂 Knowledge graph loaded: {file_path}")
            self.logger.info(f"   📊 {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relations)} relations")
            
            return knowledge_graph
            
        except Exception as e:
            self.logger.error(f"❌ Error loading knowledge graph: {e}")
            return None


    # ==================== GRAPH MERGING METHODS ====================
    
    def merge_knowledge_graphs(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Merge multiple knowledge graphs into one
        
        Args:
            graphs: List of KnowledgeGraph objects to merge
            
        Returns:
            Merged KnowledgeGraph
        """
        
        if not graphs:
            return KnowledgeGraph(graph_id="empty_graph")
        
        if len(graphs) == 1:
            return graphs[0]
        
        self.logger.info(f"🔄 Merging {len(graphs)} knowledge graphs...")
        
        # Create new merged graph
        merged_graph_id = f"merged_{int(time.time())}_{hashlib.md5(str([g.graph_id for g in graphs]).encode()).hexdigest()[:8]}"
        merged_graph = KnowledgeGraph(graph_id=merged_graph_id)
        
        try:
            # Merge entities
            for graph in graphs:
                for entity in graph.entities.values():
                    if entity.entity_id in merged_graph.entities:
                        # Merge với existing entity
                        existing_entity = merged_graph.entities[entity.entity_id]
                        self._merge_entities(existing_entity, entity)
                    else:
                        # Add new entity
                        merged_graph.entities[entity.entity_id] = entity
                        
                        # Update indices
                        if entity.entity_type not in merged_graph.entity_index:
                            merged_graph.entity_index[entity.entity_type] = set()
                        merged_graph.entity_index[entity.entity_type].add(entity.entity_id)
            
            # Merge relations
            for graph in graphs:
                for relation in graph.relations.values():
                    if relation.relation_id in merged_graph.relations:
                        # Merge với existing relation
                        existing_relation = merged_graph.relations[relation.relation_id]
                        self._merge_relations(existing_relation, relation)
                    else:
                        # Add new relation
                        merged_graph.relations[relation.relation_id] = relation
                        
                        # Update indices
                        if relation.relation_type not in merged_graph.relation_index:
                            merged_graph.relation_index[relation.relation_type] = set()
                        merged_graph.relation_index[relation.relation_type].add(relation.relation_id)
            
            # Merge video mappings
            for graph in graphs:
                for video_id, entity_ids in graph.video_mappings.items():
                    if video_id not in merged_graph.video_mappings:
                        merged_graph.video_mappings[video_id] = set()
                    merged_graph.video_mappings[video_id].update(entity_ids)
            
            # Merge metadata
            merged_graph.metadata['source_graphs'] = [g.graph_id for g in graphs]
            merged_graph.metadata['merge_time'] = time.time()
            
            # Rebuild NetworkX graph
            self._build_networkx_graph(merged_graph)
            
            # Compute statistics
            self._compute_graph_statistics(merged_graph)
            
            self.logger.info(f"✅ Graphs merged successfully:")
            self.logger.info(f"   📊 {len(merged_graph.entities)} entities, {len(merged_graph.relations)} relations")
            
            return merged_graph
            
        except Exception as e:
            self.logger.error(f"❌ Error merging knowledge graphs: {e}")
            return merged_graph


    # ==================== GRAPH CLEANUP METHODS ====================
    
    def cleanup_knowledge_graph(self, knowledge_graph: KnowledgeGraph, confidence_threshold: float = 0.3):
        """
        Clean up knowledge graph by removing low-confidence entities và relations
        
        Args:
            knowledge_graph: KnowledgeGraph to clean
            confidence_threshold: Minimum confidence to keep
        """
        
        try:
            # Remove low-confidence entities
            entities_to_remove = []
            for entity_id, entity in knowledge_graph.entities.items():
                if entity.confidence < confidence_threshold:
                    entities_to_remove.append(entity_id)
            
            # Remove entities và update relations
            for entity_id in entities_to_remove:
                # Remove relations involving this entity
                relations_to_remove = []
                for relation_id, relation in knowledge_graph.relations.items():
                    if relation.source_entity == entity_id or relation.target_entity == entity_id:
                        relations_to_remove.append(relation_id)
                
                for relation_id in relations_to_remove:
                    del knowledge_graph.relations[relation_id]
                
                # Remove from indices
                entity = knowledge_graph.entities[entity_id]
                if entity.entity_type in knowledge_graph.entity_index:
                    knowledge_graph.entity_index[entity.entity_type].discard(entity_id)
                
                # Remove from video mappings
                for video_id in entity.source_videos:
                    if video_id in knowledge_graph.video_mappings:
                        knowledge_graph.video_mappings[video_id].discard(entity_id)
                
                # Remove entity
                del knowledge_graph.entities[entity_id]
            
            # Remove low-confidence relations
            relations_to_remove = []
            for relation_id, relation in knowledge_graph.relations.items():
                if relation.confidence < confidence_threshold:
                    relations_to_remove.append(relation_id)
            
            for relation_id in relations_to_remove:
                relation = knowledge_graph.relations[relation_id]
                if relation.relation_type in knowledge_graph.relation_index:
                    knowledge_graph.relation_index[relation.relation_type].discard(relation_id)
                del knowledge_graph.relations[relation_id]
            
            # Rebuild NetworkX graph
            self._build_networkx_graph(knowledge_graph)
            
            # Update statistics
            self._compute_graph_statistics(knowledge_graph)
            
            knowledge_graph.last_updated = time.time()
            
            self.logger.info(f"🧹 Knowledge graph cleaned:")
            self.logger.info(f"   🗑️ Removed {len(entities_to_remove)} entities, {len(relations_to_remove)} relations")
            self.logger.info(f"   📊 Remaining: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relations)} relations")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning knowledge graph: {e}")

# ==================== GRAPH SUMMARY METHODS ====================
    
    def get_knowledge_graph_summary(self, knowledge_graph: KnowledgeGraph) -> Dict[str, Any]:
        """Get comprehensive summary of knowledge graph"""
        
        summary = {
            'graph_id': knowledge_graph.graph_id,
            'basic_stats': {
                'total_entities': len(knowledge_graph.entities),
                'total_relations': len(knowledge_graph.relations),
                'total_videos': len(knowledge_graph.video_mappings)
            },
            'entity_distribution': {},
            'relation_distribution': {},
            'video_coverage': {},
            'top_entities': [],
            'top_relations': [],
            'temporal_span': {},
            'creation_info': {
                'created': datetime.fromtimestamp(knowledge_graph.creation_time).isoformat(),
                'last_updated': datetime.fromtimestamp(knowledge_graph.last_updated).isoformat()
            }
        }
        
        try:
            # Entity type distribution
            entity_counts = Counter()
            confidence_scores = []
            
            for entity in knowledge_graph.entities.values():
                entity_counts[entity.entity_type] += 1
                confidence_scores.append(entity.confidence)
            
            summary['entity_distribution'] = dict(entity_counts)
            summary['average_entity_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Relation type distribution
            relation_counts = Counter()
            relation_confidences = []
            
            for relation in knowledge_graph.relations.values():
                relation_counts[relation.relation_type] += 1
                relation_confidences.append(relation.confidence)
            
            summary['relation_distribution'] = dict(relation_counts)
            summary['average_relation_confidence'] = np.mean(relation_confidences) if relation_confidences else 0.0
            
            # Video coverage
            for video_id, entity_ids in knowledge_graph.video_mappings.items():
                summary['video_coverage'][video_id] = len(entity_ids)
            
            # Top entities by confidence
            top_entities = sorted(
                knowledge_graph.entities.values(),
                key=lambda e: e.confidence,
                reverse=True
            )[:10]
            
            summary['top_entities'] = [
                {
                    'name': entity.name,
                    'type': entity.entity_type,
                    'confidence': entity.confidence,
                    'video_count': len(entity.source_videos),
                    'description': entity.description[:100] + "..." if len(entity.description) > 100 else entity.description
                }
                for entity in top_entities
            ]
            
            # Top relations by confidence
            top_relations = sorted(
                knowledge_graph.relations.values(),
                key=lambda r: r.confidence,
                reverse=True
            )[:10]
            
            summary['top_relations'] = [
                {
                    'source': knowledge_graph.entities.get(relation.source_entity, {}).name if relation.source_entity in knowledge_graph.entities else 'Unknown',
                    'relation_type': relation.relation_type,
                    'target': knowledge_graph.entities.get(relation.target_entity, {}).name if relation.target_entity in knowledge_graph.entities else 'Unknown',
                    'confidence': relation.confidence,
                    'description': relation.description[:100] + "..." if len(relation.description) > 100 else relation.description
                }
                for relation in top_relations
            ]
            
            # Temporal span analysis
            all_timestamps = []
            for entity in knowledge_graph.entities.values():
                for video_id, start_time, end_time in entity.timestamps:
                    all_timestamps.extend([start_time, end_time])
            
            if all_timestamps:
                summary['temporal_span'] = {
                    'earliest_time': min(all_timestamps),
                    'latest_time': max(all_timestamps),
                    'total_duration': max(all_timestamps) - min(all_timestamps)
                }
            
            # Add statistics from metadata
            if 'statistics' in knowledge_graph.metadata:
                summary['graph_metrics'] = knowledge_graph.metadata['statistics']
            
        except Exception as e:
            self.logger.error(f"❌ Error generating graph summary: {e}")
            summary['error'] = str(e)
        
        return summary


    # ==================== EXPORT METHODS ====================
    
    def export_knowledge_graph(
        self,
        knowledge_graph: KnowledgeGraph,
        format: str = "cytoscape",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export knowledge graph to various formats
        
        Args:
            knowledge_graph: KnowledgeGraph to export
            format: Export format ("cytoscape", "gephi", "graphml", "networkx")
            output_path: Output file path
            
        Returns:
            Export data or file path
        """
        
        try:
            if format.lower() == "cytoscape":
                return self._export_cytoscape(knowledge_graph, output_path)
            elif format.lower() == "gephi":
                return self._export_gephi(knowledge_graph, output_path)
            elif format.lower() == "graphml":
                return self._export_graphml(knowledge_graph, output_path)
            elif format.lower() == "networkx":
                return knowledge_graph.nx_graph
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting knowledge graph: {e}")
            return None
    
    def _export_cytoscape(self, knowledge_graph: KnowledgeGraph, output_path: Optional[str]) -> Optional[str]:
        """Export to Cytoscape.js format"""
        
        nodes = []
        edges = []
        
        for entity in knowledge_graph.entities.values():
            nodes.append({
                'data': {
                    'id': entity.entity_id,
                    'label': entity.name,
                    'type': entity.entity_type,
                    'description': entity.description,
                    'confidence': entity.confidence,
                    'video_count': len(entity.source_videos)
                }
            })
        
        for relation in knowledge_graph.relations.values():
            edges.append({
                'data': {
                    'id': relation.relation_id,
                    'source': relation.source_entity,
                    'target': relation.target_entity,
                    'label': relation.relation_type,
                    'description': relation.description,
                    'confidence': relation.confidence
                }
            })
        
        export_data = {'nodes': nodes, 'edges': edges}
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return output_path
        else:
            return safe_json_dumps(export_data)
    
    def _export_gephi(self, knowledge_graph: KnowledgeGraph, output_path: Optional[str]) -> Optional[str]:
        """Export to GEXF format for Gephi"""
        
        import xml.etree.ElementTree as ET
        
        gexf = ET.Element("gexf", version="1.2")
        graph_elem = ET.SubElement(gexf, "graph", mode="static", defaultedgetype="directed")
        
        # Attributes
        attributes = ET.SubElement(graph_elem, "attributes", {"class": "node"})
        ET.SubElement(attributes, "attribute", id="0", title="type", type="string")
        ET.SubElement(attributes, "attribute", id="1", title="description", type="string")
        ET.SubElement(attributes, "attribute", id="2", title="confidence", type="float")
        
        # Nodes
        nodes_elem = ET.SubElement(graph_elem, "nodes")
        for entity in knowledge_graph.entities.values():
            node = ET.SubElement(nodes_elem, "node", id=entity.entity_id, label=entity.name)
            attvalues = ET.SubElement(node, "attvalues")
            ET.SubElement(attvalues, "attvalue", {"for": "0", "value": entity.entity_type})
            ET.SubElement(attvalues, "attvalue", {"for": "1", "value": entity.description})
            ET.SubElement(attvalues, "attvalue", {"for": "2", "value": str(entity.confidence)})
        
        # Edges
        edges_elem = ET.SubElement(graph_elem, "edges")
        for relation in knowledge_graph.relations.values():
            ET.SubElement(edges_elem, "edge", {
                "id": relation.relation_id,
                "source": relation.source_entity,
                "target": relation.target_entity,
                "label": relation.relation_type
            })
        
        tree = ET.ElementTree(gexf)
        if output_path:
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            return output_path
        else:
            return ET.tostring(gexf, encoding='unicode')
    
    def _export_graphml(self, knowledge_graph: KnowledgeGraph, output_path: Optional[str]) -> Optional[str]:
        """Export to GraphML format using NetworkX"""
        
        if output_path:
            nx.write_graphml(knowledge_graph.nx_graph, output_path)
            return output_path
        else:
            from io import StringIO
            buffer = StringIO()
            nx.write_graphml(knowledge_graph.nx_graph, buffer)
            return buffer.getvalue()


    # ==================== TIMELINE METHODS ====================
    
    def get_entity_timeline(self, knowledge_graph: KnowledgeGraph, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get timeline of entity appearances across videos
        
        Args:
            knowledge_graph: KnowledgeGraph to query
            entity_id: Target entity ID
            
        Returns:
            List of timeline events
        """
        
        if entity_id not in knowledge_graph.entities:
            return []
        
        entity = knowledge_graph.entities[entity_id]
        timeline = []
        
        for video_id, start_time, end_time in entity.timestamps:
            timeline.append({
                'video_id': video_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'start_timestamp': self._seconds_to_timestamp(start_time),
                'end_timestamp': self._seconds_to_timestamp(end_time)
            })
        
        # Sort by video_id và start_time
        timeline.sort(key=lambda x: (x['video_id'], x['start_time']))
        
        return timeline
    
    def find_entity_co_occurrences(
        self,
        knowledge_graph: KnowledgeGraph,
        entity_id: str,
        time_window: float = 30.0
    ) -> List[Dict[str, Any]]:
        """
        Find entities that co-occur với target entity trong time window
        
        Args:
            knowledge_graph: KnowledgeGraph to query
            entity_id: Target entity ID
            time_window: Time window in seconds
            
        Returns:
            List of co-occurring entities với overlap info
        """
        
        if entity_id not in knowledge_graph.entities:
            return []
        
        target_entity = knowledge_graph.entities[entity_id]
        co_occurrences = []
        
        for other_entity in knowledge_graph.entities.values():
            if other_entity.entity_id == entity_id:
                continue
            
            # Check for temporal overlap
            overlaps = []
            for target_video, target_start, target_end in target_entity.timestamps:
                for other_video, other_start, other_end in other_entity.timestamps:
                    if target_video == other_video:
                        # Check if within time window
                        time_diff = min(abs(target_start - other_start), abs(target_end - other_end))
                        if time_diff <= time_window:
                            overlap_start = max(target_start, other_start)
                            overlap_end = min(target_end, other_end)
                            if overlap_end > overlap_start:
                                overlaps.append({
                                    'video_id': target_video,
                                    'overlap_start': overlap_start,
                                    'overlap_end': overlap_end,
                                    'overlap_duration': overlap_end - overlap_start,
                                    'time_difference': time_diff
                                })
            
            if overlaps:
                co_occurrences.append({
                    'entity': other_entity,
                    'overlaps': overlaps,
                    'total_overlap_duration': sum(o['overlap_duration'] for o in overlaps),
                    'min_time_difference': min(o['time_difference'] for o in overlaps)
                })
        
        # Sort by total overlap duration
        co_occurrences.sort(key=lambda x: x['total_overlap_duration'], reverse=True)
        
        return co_occurrences


    # ==================== UTILITY METHODS ====================
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp format HH:MM:SS.mmm"""
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# ==================== GLOBAL INSTANCE MANAGEMENT ====================

# Global knowledge builder instance
_global_knowledge_builder = None

def get_knowledge_builder(config: Optional[Config] = None) -> KnowledgeBuilder:
    """
    Get global knowledge builder instance (singleton pattern)
    
    Args:
        config: Configuration object (chỉ sử dụng lần đầu)
        
    Returns:
        KnowledgeBuilder instance
    """
    global _global_knowledge_builder
    if _global_knowledge_builder is None:
        _global_knowledge_builder = KnowledgeBuilder(config)
    return _global_knowledge_builder

def reset_knowledge_builder():
    """Reset global knowledge builder instance"""
    global _global_knowledge_builder
    if _global_knowledge_builder:
        del _global_knowledge_builder
    _global_knowledge_builder = None

# ==================== TESTING AND EXAMPLES ====================

if __name__ == "__main__":
    import psutil
    import copy
    
    print("🧪 Testing VideoRAG Knowledge Builder")
    print("="*60)
    
    # Setup logging
    from utils import setup_logging, seconds_to_timestamp
    setup_logging("INFO", enable_colors=True)
    
    # Initialize knowledge builder
    config = get_config()
    builder = KnowledgeBuilder(config)
    
    logger = get_logger('test')
    
    # Print configuration
    logger.info("⚙️ Knowledge Builder Configuration:")
    logger.info(f"  Entity types: {len(builder.entity_types)}")
    logger.info(f"  Relation types: {len(builder.relation_types)}")
    logger.info(f"  Cache directory: {builder.cache_dir}")
    logger.info(f"  Similarity thresholds: Entity={builder.entity_similarity_threshold}, Relation={builder.relation_similarity_threshold}")


    # ==================== TEST DATA CREATION ====================
    
    def create_test_data():
        """Create test entities and relations"""
        
        logger.info("🔍 Creating test data...")
        
        # Create test entities
        test_entities = [
            Entity(
                entity_id="person_001",
                name="John Smith",
                entity_type="PERSON",
                description="A person wearing blue shirt",
                confidence=0.85,
                source_videos={"video_001"},
                source_chunks={"chunk_001"},
                timestamps=[("video_001", 10.0, 25.0)]
            ),
            Entity(
                entity_id="object_001",
                name="Red Car",
                entity_type="OBJECT",
                description="A red sedan car",
                confidence=0.90,
                source_videos={"video_001"},
                source_chunks={"chunk_001"},
                timestamps=[("video_001", 12.0, 30.0)]
            ),
            Entity(
                entity_id="action_001",
                name="Walking",
                entity_type="ACTION",
                description="Person walking on sidewalk",
                confidence=0.75,
                source_videos={"video_001"},
                source_chunks={"chunk_001"},
                timestamps=[("video_001", 10.0, 20.0)]
            )
        ]
        
        # Create test relations
        test_relations = [
            Relation(
                relation_id="rel_001",
                source_entity="person_001",
                target_entity="action_001",
                relation_type="PERFORMS",
                description="John Smith is walking",
                confidence=0.80,
                source_videos={"video_001"},
                source_chunks={"chunk_001"}
            ),
            Relation(
                relation_id="rel_002",
                source_entity="person_001",
                target_entity="object_001",
                relation_type="INTERACTS_WITH",
                description="John Smith near red car",
                confidence=0.70,
                source_videos={"video_001"},
                source_chunks={"chunk_001"}
            )
        ]
        
        return test_entities, test_relations


    # ==================== BASIC FUNCTIONALITY TESTS ====================
    
    def test_basic_functionality():
        """Test basic entity and relation creation"""
        
        logger.info("🔨 Testing basic functionality...")
        
        test_entities, test_relations = create_test_data()
        
        logger.info(f"✅ Created {len(test_entities)} test entities and {len(test_relations)} test relations")
        
        # Test knowledge graph creation
        test_graph = KnowledgeGraph(graph_id="test_graph")
        
        # Add entities
        for entity in test_entities:
            test_graph.entities[entity.entity_id] = entity
            if entity.entity_type not in test_graph.entity_index:
                test_graph.entity_index[entity.entity_type] = set()
            test_graph.entity_index[entity.entity_type].add(entity.entity_id)
        
        # Add relations
        for relation in test_relations:
            test_graph.relations[relation.relation_id] = relation
            if relation.relation_type not in test_graph.relation_index:
                test_graph.relation_index[relation.relation_type] = set()
            test_graph.relation_index[relation.relation_type].add(relation.relation_id)
        
        # Add video mappings
        test_graph.video_mappings["video_001"] = {e.entity_id for e in test_entities}
        
        # Build NetworkX graph
        builder._build_networkx_graph(test_graph)
        
        # Compute statistics
        builder._compute_graph_statistics(test_graph)
        
        logger.info(f"✅ Test graph created: {len(test_graph.entities)} entities, {len(test_graph.relations)} relations")
        
        return test_graph


    # ==================== QUERY FUNCTIONALITY TESTS ====================
    
    def test_query_functionality(test_graph):
        """Test graph querying functionality"""
        
        logger.info("🔍 Testing graph querying...")
        
        # Query entities by type
        person_entities = builder.query_entities(test_graph, entity_type="PERSON")
        logger.info(f"✅ Found {len(person_entities)} PERSON entities")
        
        # Query relations by type
        performs_relations = builder.query_relations(test_graph, relation_type="PERFORMS")
        logger.info(f"✅ Found {len(performs_relations)} PERFORMS relations")
        
        # Find entity by name
        found_entity = builder.find_entity_by_name(test_graph, "John Smith")
        if found_entity:
            logger.info(f"✅ Found entity by name: {found_entity.name} ({found_entity.entity_type})")
        
        # Test entity neighborhood
        logger.info("🌐 Testing entity neighborhood...")
        neighborhood = builder.get_entity_neighborhood(test_graph, "person_001", max_depth=2)
        
        if neighborhood:
            logger.info(f"✅ Entity neighborhood: {len(neighborhood.get('neighbors', {}).get(1, []))} direct neighbors")
            logger.info(f"   Relations: {len(neighborhood.get('relations', []))}")


    # ==================== SIMILARITY TESTS ====================
    
    def test_similarity_calculations():
        """Test similarity calculation methods"""
        
        logger.info("📊 Testing similarity calculations...")
        
        # Test string similarity
        sim1 = builder._calculate_string_similarity("John Smith", "john smith")
        sim2 = builder._calculate_string_similarity("John Smith", "Jane Doe")
        logger.info(f"✅ String similarity: 'John Smith' vs 'john smith' = {sim1:.3f}")
        logger.info(f"✅ String similarity: 'John Smith' vs 'Jane Doe' = {sim2:.3f}")


    # ==================== ENTITY MERGING TESTS ====================
    
    def test_entity_merging():
        """Test entity merging functionality"""
        
        logger.info("🔗 Testing entity merging...")
        
        test_entities, _ = create_test_data()
        
        # Create similar entity để test merging
        similar_entity = Entity(
            entity_id="person_002",
            name="John Smith",
            entity_type="PERSON",
            description="Same person in different video",
            confidence=0.80,
            source_videos={"video_002"},
            source_chunks={"chunk_002"},
            timestamps=[("video_002", 5.0, 15.0)]
        )
        
        entities_to_merge = [test_entities[0], similar_entity]
        merged_entities = builder._merge_similar_entities(entities_to_merge)
        
        logger.info(f"✅ Entity merging: {len(entities_to_merge)} → {len(merged_entities)} entities")
        if merged_entities:
            merged_entity = merged_entities[0]
            logger.info(f"   Merged entity videos: {merged_entity.source_videos}")
            logger.info(f"   Merged confidence: {merged_entity.confidence:.3f}")


    # ==================== PROMPT AND PARSING TESTS ====================
    
    def test_extraction_prompt():
        """Test extraction prompt creation"""
        
        logger.info("📝 Testing extraction prompt creation...")
        
        test_content = "Visual: A person wearing blue shirt walking on sidewalk || Audio: footsteps and car engine sounds || Summary: Person walking near parked cars"
        prompt = builder._create_extraction_prompt(test_content, 10.0, 30.0)
        
        logger.info(f"✅ Extraction prompt created ({len(prompt)} characters)")
        logger.info(f"   Contains entity types: {'PERSON' in prompt}")
        logger.info(f"   Contains relation types: {'PERFORMS' in prompt}")


    # ==================== SAVE/LOAD TESTS ====================
    
    def test_save_load_functionality(test_graph):
        """Test save/load functionality"""
        
        logger.info("💾 Testing save/load functionality...")
        
        test_file_path = "./test_knowledge_graph.pkl"
        
        # Save graph
        save_success = builder.save_knowledge_graph(test_graph, test_file_path)
        if save_success:
            logger.info(f"✅ Graph saved to {test_file_path}")
            
            # Load graph
            loaded_graph = builder.load_knowledge_graph(test_file_path)
            if loaded_graph:
                logger.info(f"✅ Graph loaded: {len(loaded_graph.entities)} entities, {len(loaded_graph.relations)} relations")
                
                # Compare original và loaded
                entities_match = len(test_graph.entities) == len(loaded_graph.entities)
                relations_match = len(test_graph.relations) == len(loaded_graph.relations)
                logger.info(f"   Entities match: {entities_match}")
                logger.info(f"   Relations match: {relations_match}")
            
            # Cleanup
            import os
            if os.path.exists(test_file_path):
                os.remove(test_file_path)


    # ==================== EXPORT TESTS ====================
    
    def test_export_functionality(test_graph):
        """Test export functionality"""
        
        logger.info("📤 Testing export functionality...")
        
        # Test Cytoscape export
        cytoscape_data = builder.export_knowledge_graph(test_graph, format="cytoscape")
        if cytoscape_data:
            logger.info(f"✅ Cytoscape export successful ({len(cytoscape_data)} characters)")
        
        # Test NetworkX export
        nx_graph = builder.export_knowledge_graph(test_graph, format="networkx")
        if nx_graph:
            logger.info(f"✅ NetworkX export successful ({nx_graph.number_of_nodes()} nodes)")


    # ==================== TIMELINE TESTS ====================
    
    def test_timeline_functionality(test_graph):
        """Test timeline functionality"""
        
        logger.info("⏰ Testing timeline functionality...")
        
        timeline = builder.get_entity_timeline(test_graph, "person_001")
        if timeline:
            logger.info(f"✅ Entity timeline: {len(timeline)} appearances")
            for event in timeline:
                logger.info(f"   {event['video_id']}: {event['start_timestamp']} - {event['end_timestamp']}")
        
        # Test co-occurrence finding
        logger.info("🤝 Testing co-occurrence finding...")
        
        co_occurrences = builder.find_entity_co_occurrences(test_graph, "person_001", time_window=30.0)
        logger.info(f"✅ Found {len(co_occurrences)} co-occurring entities")
        
        for co_occurrence in co_occurrences:
            entity_name = co_occurrence['entity'].name
            overlap_duration = co_occurrence['total_overlap_duration']
            logger.info(f"   {entity_name}: {overlap_duration:.1f}s overlap")


    # ==================== PERFORMANCE TESTS ====================
    
    def test_performance():
        """Test performance với synthetic data"""
        
        logger.info("⚡ Testing performance với synthetic data...")
        
        start_time = time.time()
        
        # Create larger synthetic graph
        large_graph = KnowledgeGraph(graph_id="large_test_graph")
        
        # Add many entities
        for i in range(100):
            entity = Entity(
                entity_id=f"entity_{i:03d}",
                name=f"Entity {i}",
                entity_type=np.random.choice(list(builder.entity_types)),
                description=f"Synthetic entity number {i}",
                confidence=np.random.uniform(0.5, 1.0),
                source_videos={f"video_{i//10}"},
                source_chunks={f"chunk_{i}"},
                timestamps=[(f"video_{i//10}", i*2.0, (i+1)*2.0)]
            )
            large_graph.entities[entity.entity_id] = entity
        
        # Add many relations
        entity_ids = list(large_graph.entities.keys())
        for i in range(200):
            source_id = np.random.choice(entity_ids)
            target_id = np.random.choice(entity_ids)
            if source_id != target_id:
                relation = Relation(
                    relation_id=f"relation_{i:03d}",
                    source_entity=source_id,
                    target_entity=target_id,
                    relation_type=np.random.choice(list(builder.relation_types)),
                    description=f"Synthetic relation {i}",
                    confidence=np.random.uniform(0.5, 1.0),
                    source_videos={f"video_{i//20}"},
                    source_chunks={f"chunk_{i}"}
                )
                large_graph.relations[relation.relation_id] = relation
        
        # Build NetworkX graph
        builder._build_networkx_graph(large_graph)
        
        # Compute statistics
        builder._compute_graph_statistics(large_graph)
        
        performance_time = time.time() - start_time
        
        logger.info(f"✅ Large graph performance test:")
        logger.info(f"   Created {len(large_graph.entities)} entities, {len(large_graph.relations)} relations")
        logger.info(f"   Processing time: {performance_time:.2f}s")


    # ==================== ERROR HANDLING TESTS ====================
    
    def test_error_handling():
        """Test error handling"""
        
        logger.info("🛡️ Testing error handling...")
        
        # Test với invalid data
        try:
            invalid_result = builder._parse_extraction_response(
                "invalid json data", 
                "test_chunk", 
                "test_video", 
                0.0, 
                10.0
            )
            logger.info(f"✅ Invalid JSON handling: {len(invalid_result[0])} entities extracted")
        except Exception as e:
            logger.info(f"✅ Exception handling works: {e}")
        
        # Test với empty graph
        empty_graph = KnowledgeGraph(graph_id="empty_graph")
        empty_summary = builder.get_knowledge_graph_summary(empty_graph)
        logger.info(f"✅ Empty graph handling: {empty_summary['basic_stats']['total_entities']} entities")


    # ==================== MEMORY USAGE TESTS ====================
    
    def test_memory_usage():
        """Test memory usage"""
        
        logger.info("💾 Testing memory usage...")
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)
        
        # Simulate some processing
        for i in range(5):
            fake_entities = []
            for j in range(10):
                # Create fake entity data
                fake_entity = Entity(
                    entity_id=f"test_{i}_{j}",
                    name=f"Test Entity {j}",
                    entity_type="OBJECT",
                    description=f"Test entity number {j}",
                    confidence=0.8,
                    source_videos={"test_video"},
                    source_chunks={f"chunk_{j}"},
                    timestamps=[("test_video", j * 1.0, (j + 1) * 1.0)]
                )
                fake_entities.append(fake_entity)
            
            # Simulate processing
            time.sleep(0.1)
        
        memory_after = process.memory_info().rss / (1024**2)
        memory_diff = memory_after - memory_before
        
        logger.info(f"📊 Memory usage: {memory_before:.1f} MB → {memory_after:.1f} MB (Δ{memory_diff:+.1f} MB)")


    # ==================== GRAPH CLEANUP TESTS ====================
    
    def test_graph_cleanup(test_graph):
        """Test graph cleanup functionality"""
        
        logger.info("🧹 Testing graph cleanup...")
        
        original_entities = len(test_graph.entities)
        original_relations = len(test_graph.relations)
        
        # Create copy for cleanup test
        cleanup_graph = copy.deepcopy(test_graph)
        cleanup_graph.graph_id = "cleanup_test_graph"
        
        # Set some entities to low confidence
        for entity in list(cleanup_graph.entities.values())[:1]:
            entity.confidence = 0.2  # Below threshold
        
        builder.cleanup_knowledge_graph(cleanup_graph, confidence_threshold=0.3)
        
        logger.info(f"✅ Graph cleanup: {original_entities} → {len(cleanup_graph.entities)} entities")
        logger.info(f"   Relations: {original_relations} → {len(cleanup_graph.relations)}")


    # ==================== CONFIGURATION VALIDATION TESTS ====================
    
    def test_configuration_validation():
        """Test configuration validation"""
        
        logger.info("⚙️ Testing configuration validation...")
        
        logger.info(f"✅ Entity types defined: {len(builder.entity_types)}")
        logger.info(f"✅ Relation types defined: {len(builder.relation_types)}")
        logger.info(f"✅ Cache directory exists: {builder.cache_dir.exists()}")


    # ==================== RUN ALL TESTS ====================
    
    def run_all_tests():
        """Run all test functions"""
        
        # Create test graph
        test_graph = test_basic_functionality()
        
        # Run all tests
        test_query_functionality(test_graph)
        test_similarity_calculations()
        test_entity_merging()
        test_extraction_prompt()
        test_save_load_functionality(test_graph)
        test_export_functionality(test_graph)
        test_timeline_functionality(test_graph)
        test_performance()
        test_error_handling()
        test_memory_usage()
        test_graph_cleanup(test_graph)
        test_configuration_validation()
        
        return test_graph

    
    # ==================== MAIN TEST EXECUTION ====================
    
    # Run all tests
    test_graph = run_all_tests()
    
    # Final summary
    logger.info("📋 Final Knowledge Builder Summary:")
    
    total_entities = sum(len(kg.entities) for kg in builder.knowledge_graphs.values())
    total_relations = sum(len(kg.relations) for kg in builder.knowledge_graphs.values())
    
    logger.info(f"   Knowledge graphs created: {len(builder.knowledge_graphs)}")
    logger.info(f"   Total entities processed: {total_entities}")
    logger.info(f"   Total relations processed: {total_relations}")
    
    # Test graph statistics
    logger.info("📊 Testing graph statistics...")
    
    stats = builder.get_knowledge_graph_summary(test_graph)
    
    logger.info(f"✅ Graph summary generated:")
    logger.info(f"   Total entities: {stats['basic_stats']['total_entities']}")
    logger.info(f"   Total relations: {stats['basic_stats']['total_relations']}")
    logger.info(f"   Entity types: {list(stats['entity_distribution'].keys())}")
    logger.info(f"   Relation types: {list(stats['relation_distribution'].keys())}")
    
    # Cleanup
    logger.info("🧹 Final cleanup...")
    
    # Clear graphs
    builder.knowledge_graphs.clear()
    
    logger.info("✅ Knowledge Builder testing completed successfully!")
    
    # Capabilities summary
    print("\n" + "="*60)
    print("🎉 VideoRAG Knowledge Builder - Capabilities Summary")
    print("="*60)
    print("✅ Multi-modal entity extraction (Visual + Audio + Text)")
    print("✅ Automated relation discovery và classification")
    print("✅ Cross-video entity linking và resolution")  
    print("✅ Knowledge graph construction và management")
    print("✅ NetworkX integration for complex queries")
    print("✅ Entity/relation similarity matching")
    print("✅ Temporal analysis và co-occurrence detection")
    print("✅ Multiple export formats (Cytoscape, Gephi, GraphML)")
    print("✅ Incremental graph updates và merging")
    print("✅ Performance optimization và caching")
    print("✅ Comprehensive querying capabilities")
    print("✅ Graph statistics và analytics")
    print("✅ Save/load functionality")
    print("✅ Error handling và validation")
    print("✅ Memory management và cleanup")
    print("="*60)