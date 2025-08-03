"""
database_manager.py - Unified Database Management cho VideoRAG System
Quản lý vector DB, graph DB và metadata storage
"""

import os
import json
import time
import sqlite3
import threading
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available, using fallback vector search")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX not available, graph operations disabled")

from config import Config, DatabaseConfig, get_config
from utils import (
    get_logger, measure_time, performance_monitor,
    cosine_similarity
)
from knowledge_builder import KnowledgeGraph, Entity, Relation
from video_processor import ProcessedVideo, VideoChunk, FrameInfo


# ==================== DATA STRUCTURES ====================

@dataclass
class VideoMetadata:
    """Metadata cho video trong database"""
    video_id: str
    video_path: str
    filename: str
    duration: float
    fps: float
    resolution: str
    size_bytes: int
    processing_time: float
    chunk_count: int
    frame_count: int
    created_at: float
    updated_at: float
    success: bool = True
    error_message: str = ""


@dataclass
class EmbeddingEntry:
    """Entry cho vector database"""
    entry_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    source_type: str  # "frame", "chunk", "entity"
    source_id: str
    video_id: str
    timestamp: Optional[float] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class QueryResult:
    """Result từ database query"""
    success: bool
    results: List[Any] = None
    total_count: int = 0
    execution_time: float = 0.0
    error_message: str = ""


# ==================== VECTOR DATABASE MANAGER ====================

class VectorDatabaseManager:
    """Manager cho vector database operations"""
    
    def __init__(self, config: DatabaseConfig, logger):
        self.config = config
        self.logger = logger
        self.db_path = Path(config.connection_string)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self._init_vector_storage()
        self._init_threading()
        
        # Load existing data và initialize index
        self._load_vectors()
        
        if FAISS_AVAILABLE:
            self._init_faiss_index()
        
        self.logger.info(f"📊 Vector database initialized: {len(self.vectors)} vectors")
    
    def _init_vector_storage(self):
        """Initialize vector storage components"""
        self.vectors: Dict[str, EmbeddingEntry] = {}
        self.vector_index = None
        self.index_dirty = True
    
    def _init_threading(self):
        """Initialize threading components"""
        self._lock = threading.RLock()


    # ==================== DATA PERSISTENCE METHODS ====================
    
    def _load_vectors(self):
        """Load vectors từ disk storage"""
        vector_file = self.db_path / "vectors.pkl"
        
        if vector_file.exists():
            try:
                with open(vector_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.vectors = saved_data.get('vectors', {})
                    
                self.logger.info(f"📂 Loaded {len(self.vectors)} vectors from disk")
            except Exception as e:
                self.logger.error(f"❌ Error loading vectors: {e}")
                self.vectors = {}
    
    def _save_vectors(self):
        """Save vectors to disk storage"""
        vector_file = self.db_path / "vectors.pkl"
        
        try:
            save_data = {
                'vectors': self.vectors,
                'created_at': time.time()
            }
            
            with open(vector_file, 'wb') as f:
                pickle.dump(save_data, f)
                
            self.logger.debug(f"💾 Saved {len(self.vectors)} vectors to disk")
        except Exception as e:
            self.logger.error(f"❌ Error saving vectors: {e}")


    # ==================== FAISS INDEX METHODS ====================
    
    def _init_faiss_index(self):
        """Initialize FAISS index"""
        if not self.vectors:
            return
        
        try:
            # Get embedding dimension từ first vector
            sample_vector = next(iter(self.vectors.values())).vector
            dimension = len(sample_vector)
            
            # Create FAISS index based on configuration
            if self.config.index_type.upper() == "HNSW":
                self.vector_index = faiss.IndexHNSWFlat(dimension, 32)
                self.vector_index.hnsw.efConstruction = 40
                self.vector_index.hnsw.efSearch = 16
            else:
                self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            
            # Add vectors to index
            vectors_array = np.array([entry.vector for entry in self.vectors.values()]).astype('float32')
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors_array)
            
            self.vector_index.add(vectors_array)
            self.index_dirty = False
            
            self.logger.info(f"🚀 FAISS index initialized: {self.vector_index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing FAISS index: {e}")
            self.vector_index = None
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index when dirty"""
        if FAISS_AVAILABLE and self.vectors:
            try:
                sample_vector = next(iter(self.vectors.values())).vector
                dimension = len(sample_vector)
                
                # Recreate index
                if self.config.index_type.upper() == "HNSW":
                    self.vector_index = faiss.IndexHNSWFlat(dimension, 32)
                else:
                    self.vector_index = faiss.IndexFlatIP(dimension)
                
                # Add all vectors
                vectors_array = np.array([entry.vector for entry in self.vectors.values()]).astype('float32')
                faiss.normalize_L2(vectors_array)
                self.vector_index.add(vectors_array)
                
                self.index_dirty = False
                self.logger.debug("🔄 FAISS index rebuilt")
                
            except Exception as e:
                self.logger.error(f"❌ Error rebuilding FAISS index: {e}")


    # ==================== CRUD OPERATIONS ====================
    
    def add_embedding(self, entry: EmbeddingEntry) -> bool:
        """Add embedding to vector database"""
        with self._lock:
            try:
                self.vectors[entry.entry_id] = entry
                self.index_dirty = True
                
                # Periodically save to disk
                if len(self.vectors) % 100 == 0:
                    self._save_vectors()
                
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Error adding embedding: {e}")
                return False
    
    def add_embeddings_batch(self, entries: List[EmbeddingEntry]) -> int:
        """Add multiple embeddings in batch"""
        added_count = 0
        
        with self._lock:
            for entry in entries:
                try:
                    self.vectors[entry.entry_id] = entry
                    added_count += 1
                except Exception as e:
                    self.logger.error(f"❌ Error adding embedding {entry.entry_id}: {e}")
            
            self.index_dirty = True
            self._save_vectors()
        
        self.logger.info(f"📦 Added {added_count}/{len(entries)} embeddings to vector database")
        return added_count
    
    def get_embedding(self, entry_id: str) -> Optional[EmbeddingEntry]:
        """Get embedding by ID"""
        with self._lock:
            return self.vectors.get(entry_id)
    
    def delete_embedding(self, entry_id: str) -> bool:
        """Delete embedding by ID"""
        with self._lock:
            if entry_id in self.vectors:
                del self.vectors[entry_id]
                self.index_dirty = True
                return True
            return False


    # ==================== SEARCH METHODS ====================
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[EmbeddingEntry, float]]:
        """Search for similar vectors"""
        
        with self._lock:
            if not self.vectors:
                return []
            
            try:
                # Use FAISS if available and index is built
                if FAISS_AVAILABLE and self.vector_index is not None:
                    if self.index_dirty:
                        self._rebuild_faiss_index()
                    
                    return self._faiss_search(query_vector, top_k, threshold, filters)
                else:
                    return self._fallback_search(query_vector, top_k, threshold, filters)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in vector search: {e}")
                return []
    
    def _faiss_search(
        self, 
        query_vector: np.ndarray, 
        top_k: int,
        threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[EmbeddingEntry, float]]:
        """Search using FAISS index"""
        
        # Normalize query vector
        query_normalized = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Search
        scores, indices = self.vector_index.search(query_normalized, min(top_k * 2, len(self.vectors)))
        
        # Convert results
        results = []
        vector_list = list(self.vectors.values())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            if score < threshold:
                continue
            
            entry = vector_list[idx]
            
            # Apply filters
            if filters and not self._apply_filters(entry, filters):
                continue
            
            results.append((entry, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _fallback_search(
        self,
        query_vector: np.ndarray,
        top_k: int, 
        threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[EmbeddingEntry, float]]:
        """Fallback search without FAISS"""
        
        similarities = []
        
        for entry in self.vectors.values():
            # Apply filters first
            if filters and not self._apply_filters(entry, filters):
                continue
            
            # Calculate similarity
            similarity = cosine_similarity(query_vector, entry.vector)
            
            if similarity >= threshold:
                similarities.append((entry, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


    # ==================== UTILITY METHODS ====================
    
    def _apply_filters(self, entry: EmbeddingEntry, filters: Dict[str, Any]) -> bool:
        """Apply filters to entry"""
        for key, value in filters.items():
            if key == 'source_type' and entry.source_type != value:
                return False
            elif key == 'video_id' and entry.video_id != value:
                return False
            elif key == 'source_id' and entry.source_id != value:
                return False
            elif key in entry.metadata and entry.metadata[key] != value:
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        with self._lock:
            stats = {
                'total_vectors': len(self.vectors),
                'index_type': self.config.index_type,
                'index_dirty': self.index_dirty,
                'faiss_available': FAISS_AVAILABLE,
                'source_type_counts': {},
                'video_counts': {}
            }
            
            # Count by source type
            for entry in self.vectors.values():
                source_type = entry.source_type
                stats['source_type_counts'][source_type] = stats['source_type_counts'].get(source_type, 0) + 1
                
                video_id = entry.video_id
                stats['video_counts'][video_id] = stats['video_counts'].get(video_id, 0) + 1
            
            return stats
        
# ==================== GRAPH DATABASE MANAGER ====================

class GraphDatabaseManager:
    """Manager cho graph database operations"""
    
    def __init__(self, config: DatabaseConfig, logger):
        self.config = config
        self.logger = logger
        self.db_path = Path(config.connection_string)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self._init_storage()
        self._init_threading()
        
        # Load existing graphs
        self._load_graphs()
        
        self.logger.info(f"📊 Graph database initialized: {len(self.knowledge_graphs)} graphs")
    
    def _init_storage(self):
        """Initialize storage for knowledge graphs"""
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
    
    def _init_threading(self):
        """Initialize threading components"""
        self._lock = threading.RLock()


    # ==================== DATA PERSISTENCE METHODS ====================
    
    def _load_graphs(self):
        """Load knowledge graphs từ disk storage"""
        graph_file = self.db_path / "knowledge_graphs.pkl"
        
        if graph_file.exists():
            try:
                with open(graph_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.knowledge_graphs = saved_data.get('graphs', {})
                    
                self.logger.info(f"📂 Loaded {len(self.knowledge_graphs)} knowledge graphs from disk")
            except Exception as e:
                self.logger.error(f"❌ Error loading knowledge graphs: {e}")
                self.knowledge_graphs = {}
    
    def _save_graphs(self):
        """Save knowledge graphs to disk storage"""
        graph_file = self.db_path / "knowledge_graphs.pkl"
        
        try:
            save_data = {
                'graphs': self.knowledge_graphs,
                'created_at': time.time()
            }
            
            with open(graph_file, 'wb') as f:
                pickle.dump(save_data, f)
                
            self.logger.debug(f"💾 Saved {len(self.knowledge_graphs)} knowledge graphs to disk")
        except Exception as e:
            self.logger.error(f"❌ Error saving knowledge graphs: {e}")


    # ==================== CRUD OPERATIONS ====================
    
    def store_knowledge_graph(self, graph: KnowledgeGraph) -> bool:
        """Store knowledge graph"""
        with self._lock:
            try:
                self.knowledge_graphs[graph.graph_id] = graph
                self._save_graphs()
                return True
            except Exception as e:
                self.logger.error(f"❌ Error storing knowledge graph: {e}")
                return False
    
    def get_knowledge_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Get knowledge graph by ID"""
        with self._lock:
            return self.knowledge_graphs.get(graph_id)
    
    def list_knowledge_graphs(self) -> List[str]:
        """List all knowledge graph IDs"""
        with self._lock:
            return list(self.knowledge_graphs.keys())
    
    def delete_knowledge_graph(self, graph_id: str) -> bool:
        """Delete knowledge graph"""
        with self._lock:
            if graph_id in self.knowledge_graphs:
                del self.knowledge_graphs[graph_id]
                self._save_graphs()
                return True
            return False


    # ==================== QUERY METHODS ====================
    
    def query_entities(
        self,
        graph_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> List[Entity]:
        """Query entities across knowledge graphs"""
        with self._lock:
            results = []
            
            # Determine which graphs to search
            graphs_to_search = self._get_graphs_to_search(graph_id)
            
            # Search entities
            for graph in graphs_to_search:
                for entity in graph.entities.values():
                    # Apply filters
                    if not self._apply_entity_filters(entity, entity_type, name_pattern, video_id):
                        continue
                    
                    results.append(entity)
            
            return results
    
    def query_relations(
        self,
        graph_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        source_entity: Optional[str] = None,
        target_entity: Optional[str] = None
    ) -> List[Relation]:
        """Query relations across knowledge graphs"""
        with self._lock:
            results = []
            
            # Determine which graphs to search
            graphs_to_search = self._get_graphs_to_search(graph_id)
            
            # Search relations
            for graph in graphs_to_search:
                for relation in graph.relations.values():
                    # Apply filters
                    if not self._apply_relation_filters(relation, relation_type, source_entity, target_entity):
                        continue
                    
                    results.append(relation)
            
            return results


    # ==================== UTILITY METHODS ====================
    
    def _get_graphs_to_search(self, graph_id: Optional[str]) -> List[KnowledgeGraph]:
        """Get list of graphs to search based on graph_id filter"""
        if graph_id:
            if graph_id in self.knowledge_graphs:
                return [self.knowledge_graphs[graph_id]]
            else:
                return []
        else:
            return list(self.knowledge_graphs.values())
    
    def _apply_entity_filters(
        self, 
        entity: Entity, 
        entity_type: Optional[str],
        name_pattern: Optional[str],
        video_id: Optional[str]
    ) -> bool:
        """Apply filters to entity"""
        if entity_type and entity.entity_type != entity_type:
            return False
        
        if name_pattern and name_pattern.lower() not in entity.name.lower():
            return False
        
        if video_id and video_id not in entity.source_videos:
            return False
        
        return True
    
    def _apply_relation_filters(
        self,
        relation: Relation,
        relation_type: Optional[str],
        source_entity: Optional[str],
        target_entity: Optional[str]
    ) -> bool:
        """Apply filters to relation"""
        if relation_type and relation.relation_type != relation_type:
            return False
        
        if source_entity and relation.source_entity != source_entity:
            return False
        
        if target_entity and relation.target_entity != target_entity:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph database statistics"""
        with self._lock:
            stats = {
                'total_graphs': len(self.knowledge_graphs),
                'total_entities': 0,
                'total_relations': 0,
                'entity_types': {},
                'relation_types': {},
                'graphs_info': []
            }
            
            for graph_id, graph in self.knowledge_graphs.items():
                # Count entities and relations
                stats['total_entities'] += len(graph.entities)
                stats['total_relations'] += len(graph.relations)
                
                # Count entity types
                for entity in graph.entities.values():
                    entity_type = entity.entity_type
                    stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
                
                # Count relation types
                for relation in graph.relations.values():
                    relation_type = relation.relation_type
                    stats['relation_types'][relation_type] = stats['relation_types'].get(relation_type, 0) + 1
                
                # Graph info
                stats['graphs_info'].append({
                    'graph_id': graph_id,
                    'entities': len(graph.entities),
                    'relations': len(graph.relations),
                    'videos': len(graph.video_mappings),
                    'created': graph.creation_time,
                    'updated': graph.last_updated
                })
            
            return stats
        
# ==================== METADATA DATABASE MANAGER ====================

class MetadataDatabaseManager:
    """Manager cho metadata database operations"""
    
    def __init__(self, config: DatabaseConfig, logger):
        self.config = config
        self.logger = logger
        self.db_path = config.connection_string
        
        # Initialize core components
        self._init_threading()
        
        # Initialize SQLite database
        self._init_database()
        
        self.logger.info(f"📊 Metadata database initialized: {self.db_path}")
    
    def _init_threading(self):
        """Initialize threading components"""
        self._lock = threading.RLock()


    # ==================== DATABASE INITIALIZATION ====================
    
    def _init_database(self):
        """Initialize SQLite database với required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables
                self._create_videos_table(cursor)
                self._create_chunks_table(cursor)
                self._create_frames_table(cursor)
                self._create_audio_segments_table(cursor)
                
                # Create indices
                self._create_indices(cursor)
                
                conn.commit()
                self.logger.debug("✅ Database tables initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing database: {e}")
            raise
    
    def _create_videos_table(self, cursor):
        """Create videos table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                duration REAL NOT NULL,
                fps REAL NOT NULL,
                resolution TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                processing_time REAL NOT NULL,
                chunk_count INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                success BOOLEAN NOT NULL DEFAULT 1,
                error_message TEXT DEFAULT ''
            )
        """)
    
    def _create_chunks_table(self, cursor):
        """Create chunks table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                frame_count INTEGER NOT NULL,
                unified_description TEXT DEFAULT '',
                has_audio BOOLEAN DEFAULT 0,
                created_at REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
        """)
    
    def _create_frames_table(self, cursor):
        """Create frames table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                frame_index INTEGER NOT NULL,
                caption TEXT DEFAULT '',
                confidence REAL DEFAULT 0.0,
                image_path TEXT DEFAULT '',
                created_at REAL NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id),
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
        """)
    
    def _create_audio_segments_table(self, cursor):
        """Create audio segments table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_segments (
                segment_id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                transcript TEXT DEFAULT '',
                confidence REAL DEFAULT 0.0,
                audio_path TEXT DEFAULT '',
                created_at REAL NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id),
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
        """)
    
    def _create_indices(self, cursor):
        """Create indices for better performance"""
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks (video_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_time ON chunks (start_time, end_time)",
            "CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames (video_id)",
            "CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audio_video_id ON audio_segments (video_id)"
        ]
        
        for index_sql in indices:
            cursor.execute(index_sql)


    # ==================== CRUD OPERATIONS ====================
    
    def store_video_metadata(self, metadata: VideoMetadata) -> bool:
        """Store video metadata"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO videos (
                            video_id, video_path, filename, duration, fps, resolution,
                            size_bytes, processing_time, chunk_count, frame_count,
                            created_at, updated_at, success, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metadata.video_id, metadata.video_path, metadata.filename,
                        metadata.duration, metadata.fps, metadata.resolution,
                        metadata.size_bytes, metadata.processing_time, metadata.chunk_count,
                        metadata.frame_count, metadata.created_at, metadata.updated_at,
                        metadata.success, metadata.error_message
                    ))
                    
                    conn.commit()
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Error storing video metadata: {e}")
                return False
    
    def store_processed_video(self, processed_video: ProcessedVideo) -> bool:
        """Store complete processed video data"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    current_time = time.time()
                    
                    # Store video metadata
                    self._store_video_info(cursor, processed_video, current_time)
                    
                    # Store chunks, frames, and audio segments
                    for chunk in processed_video.chunks:
                        self._store_chunk_data(cursor, chunk, current_time)
                    
                    conn.commit()
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Error storing processed video: {e}")
                return False
    
    def _store_video_info(self, cursor, processed_video: ProcessedVideo, current_time: float):
        """Store video metadata information"""
        video_info = processed_video.video_info
        cursor.execute("""
            INSERT OR REPLACE INTO videos (
                video_id, video_path, filename, duration, fps, resolution,
                size_bytes, processing_time, chunk_count, frame_count,
                created_at, updated_at, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            processed_video.video_id, processed_video.video_path,
            video_info.get('filename', os.path.basename(processed_video.video_path)),
            processed_video.total_duration, video_info.get('fps', 0),
            video_info.get('resolution', ''), video_info.get('size_bytes', 0),
            processed_video.processing_time, len(processed_video.chunks),
            processed_video.total_frames, current_time, current_time,
            processed_video.success, processed_video.error_message
        ))
    
    def _store_chunk_data(self, cursor, chunk: VideoChunk, current_time: float):
        """Store chunk and related data"""
        # Store chunk
        cursor.execute("""
            INSERT OR REPLACE INTO chunks (
                chunk_id, video_id, start_time, end_time, frame_count,
                unified_description, has_audio, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.chunk_id, chunk.video_id, chunk.start_time, chunk.end_time,
            len(chunk.frames), chunk.unified_description,
            chunk.audio_segment is not None, current_time
        ))
        
        # Store frames
        for frame in chunk.frames:
            cursor.execute("""
                INSERT OR REPLACE INTO frames (
                    frame_id, chunk_id, video_id, timestamp, frame_index,
                    caption, confidence, image_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                frame.frame_id, chunk.chunk_id, frame.video_id,
                frame.timestamp, frame.frame_index, frame.caption,
                frame.confidence, frame.image_path or '', current_time
            ))
        
        # Store audio segment
        if chunk.audio_segment:
            audio = chunk.audio_segment
            cursor.execute("""
                INSERT OR REPLACE INTO audio_segments (
                    segment_id, chunk_id, video_id, start_time, end_time,
                    transcript, confidence, audio_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audio.segment_id, chunk.chunk_id, audio.video_id,
                audio.start_time, audio.end_time, audio.transcript,
                audio.confidence, audio.audio_path or '', current_time
            ))


    # ==================== QUERY METHODS ====================
    
    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Get video metadata by ID"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT video_id, video_path, filename, duration, fps, resolution,
                               size_bytes, processing_time, chunk_count, frame_count,
                               created_at, updated_at, success, error_message
                        FROM videos WHERE video_id = ?
                    """, (video_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        return VideoMetadata(*row)
                    return None
                    
            except Exception as e:
                self.logger.error(f"❌ Error getting video metadata: {e}")
                return None
    
    def list_videos(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        success_only: bool = True
    ) -> List[VideoMetadata]:
        """List videos with pagination"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT video_id, video_path, filename, duration, fps, resolution,
                               size_bytes, processing_time, chunk_count, frame_count,
                               created_at, updated_at, success, error_message
                        FROM videos
                    """
                    
                    params = []
                    
                    if success_only:
                        query += " WHERE success = 1"
                    
                    query += " ORDER BY created_at DESC"
                    
                    if limit:
                        query += " LIMIT ? OFFSET ?"
                        params.extend([limit, offset])
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    return [VideoMetadata(*row) for row in rows]
                    
            except Exception as e:
                self.logger.error(f"❌ Error listing videos: {e}")
                return []
            
# ==================== SEARCH METHODS ====================
    
    def search_frames_by_caption(
        self,
        search_text: str,
        video_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search frames by caption text"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT f.frame_id, f.video_id, f.timestamp, f.caption,
                               f.confidence, f.image_path, v.filename
                        FROM frames f
                        JOIN videos v ON f.video_id = v.video_id
                        WHERE f.caption LIKE ?
                    """
                    
                    params = [f"%{search_text}%"]
                    
                    if video_id:
                        query += " AND f.video_id = ?"
                        params.append(video_id)
                    
                    query += " ORDER BY f.confidence DESC, f.timestamp ASC LIMIT ?"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            'frame_id': row[0],
                            'video_id': row[1],
                            'timestamp': row[2],
                            'caption': row[3],
                            'confidence': row[4],
                            'image_path': row[5],
                            'video_filename': row[6]
                        })
                    
                    return results
                    
            except Exception as e:
                self.logger.error(f"❌ Error searching frames: {e}")
                return []
    
    def search_audio_by_transcript(
        self,
        search_text: str,
        video_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search audio segments by transcript"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT a.segment_id, a.video_id, a.start_time, a.end_time,
                               a.transcript, a.confidence, v.filename
                        FROM audio_segments a
                        JOIN videos v ON a.video_id = v.video_id
                        WHERE a.transcript LIKE ?
                    """
                    
                    params = [f"%{search_text}%"]
                    
                    if video_id:
                        query += " AND a.video_id = ?"
                        params.append(video_id)
                    
                    query += " ORDER BY a.confidence DESC, a.start_time ASC LIMIT ?"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            'segment_id': row[0],
                            'video_id': row[1],
                            'start_time': row[2],
                            'end_time': row[3],
                            'transcript': row[4],
                            'confidence': row[5],
                            'video_filename': row[6]
                        })
                    
                    return results
                    
            except Exception as e:
                self.logger.error(f"❌ Error searching audio: {e}")
                return []


    # ==================== TIMELINE METHODS ====================
    
    def get_video_timeline(self, video_id: str) -> Dict[str, Any]:
        """Get complete timeline data for a video"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get video info
                    video_info = self._get_video_info_for_timeline(cursor, video_id)
                    if not video_info:
                        return {}
                    
                    # Get timeline components
                    chunks = self._get_chunks_for_timeline(cursor, video_id)
                    frames = self._get_frames_for_timeline(cursor, video_id)
                    audio_segments = self._get_audio_for_timeline(cursor, video_id)
                    
                    return {
                        'video_info': video_info,
                        'chunks': chunks,
                        'frames': frames,
                        'audio_segments': audio_segments
                    }
                    
            except Exception as e:
                self.logger.error(f"❌ Error getting video timeline: {e}")
                return {}
    
    def _get_video_info_for_timeline(self, cursor, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video info for timeline"""
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        video_row = cursor.fetchone()
        
        if not video_row:
            return None
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        return dict(zip(column_names, video_row))
    
    def _get_chunks_for_timeline(self, cursor, video_id: str) -> List[Dict[str, Any]]:
        """Get chunks for timeline"""
        cursor.execute("""
            SELECT chunk_id, start_time, end_time, frame_count, 
                   unified_description, has_audio
            FROM chunks WHERE video_id = ? ORDER BY start_time
        """, (video_id,))
        chunks = cursor.fetchall()
        
        return [dict(zip(['chunk_id', 'start_time', 'end_time', 'frame_count', 
                         'unified_description', 'has_audio'], chunk)) for chunk in chunks]
    
    def _get_frames_for_timeline(self, cursor, video_id: str) -> List[Dict[str, Any]]:
        """Get frames for timeline"""
        cursor.execute("""
            SELECT frame_id, chunk_id, timestamp, caption, confidence
            FROM frames WHERE video_id = ? ORDER BY timestamp
        """, (video_id,))
        frames = cursor.fetchall()
        
        return [dict(zip(['frame_id', 'chunk_id', 'timestamp', 'caption', 
                         'confidence'], frame)) for frame in frames]
    
    def _get_audio_for_timeline(self, cursor, video_id: str) -> List[Dict[str, Any]]:
        """Get audio segments for timeline"""
        cursor.execute("""
            SELECT segment_id, chunk_id, start_time, end_time, transcript
            FROM audio_segments WHERE video_id = ? ORDER BY start_time
        """, (video_id,))
        audio_segments = cursor.fetchall()
        
        return [dict(zip(['segment_id', 'chunk_id', 'start_time', 'end_time', 
                         'transcript'], audio)) for audio in audio_segments]


    # ==================== MANAGEMENT METHODS ====================
    
    def delete_video(self, video_id: str) -> bool:
        """Delete video and all associated data"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Delete in reverse order due to foreign keys
                    self._delete_video_components(cursor, video_id)
                    
                    conn.commit()
                    return True
                    
            except Exception as e:
                self.logger.error(f"❌ Error deleting video: {e}")
                return False
    
    def _delete_video_components(self, cursor, video_id: str):
        """Delete all components of a video"""
        # Delete in reverse order due to foreign keys
        cursor.execute("DELETE FROM audio_segments WHERE video_id = ?", (video_id,))
        cursor.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
        cursor.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
        cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))


    # ==================== STATISTICS METHODS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metadata database statistics"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    stats = {}
                    
                    # Video statistics
                    stats.update(self._get_video_stats(cursor))
                    
                    # Component statistics
                    stats.update(self._get_component_stats(cursor))
                    
                    # Database statistics
                    stats.update(self._get_database_stats(cursor))
                    
                    return stats
                    
            except Exception as e:
                self.logger.error(f"❌ Error getting stats: {e}")
                return {}
    
    def _get_video_stats(self, cursor) -> Dict[str, Any]:
        """Get video-related statistics"""
        stats = {}
        
        # Count videos
        cursor.execute("SELECT COUNT(*) FROM videos")
        stats['total_videos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM videos WHERE success = 1")
        stats['successful_videos'] = cursor.fetchone()[0]
        
        # Total duration
        cursor.execute("SELECT SUM(duration) FROM videos WHERE success = 1")
        result = cursor.fetchone()[0]
        stats['total_duration_hours'] = (result or 0) / 3600
        
        return stats
    
    def _get_component_stats(self, cursor) -> Dict[str, Any]:
        """Get component-related statistics"""
        stats = {}
        
        # Count chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        stats['total_chunks'] = cursor.fetchone()[0]
        
        # Count frames
        cursor.execute("SELECT COUNT(*) FROM frames")
        stats['total_frames'] = cursor.fetchone()[0]
        
        # Count audio segments
        cursor.execute("SELECT COUNT(*) FROM audio_segments")
        stats['total_audio_segments'] = cursor.fetchone()[0]
        
        return stats
    
    def _get_database_stats(self, cursor) -> Dict[str, Any]:
        """Get database-related statistics"""
        stats = {}
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()[0]
        stats['database_size_mb'] = db_size / (1024 * 1024)
        
        return stats
    
# ==================== MAIN DATABASE MANAGER ====================

class DatabaseManager:
    """
    Unified Database Manager cho VideoRAG System
    Quản lý vector DB, graph DB và metadata storage
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DatabaseManager
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger('database_manager')
        
        # Initialize sub-managers
        self._init_sub_managers()
        
        self.logger.info("💾 DatabaseManager initialized")
        self.logger.info(f"📊 Components: Vector DB, Graph DB, Metadata DB")
    
    def _init_sub_managers(self):
        """Initialize database sub-managers"""
        try:
            # Vector database
            self.vector_db = self._init_vector_database()
            
            # Graph database
            self.graph_db = self._init_graph_database()
            
            # Metadata database
            self.metadata_db = self._init_metadata_database()
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing sub-managers: {e}")
            raise
    
    def _init_vector_database(self):
        """Initialize vector database"""
        vector_config = self.config.get_database_config('vector_db')
        if vector_config:
            return VectorDatabaseManager(vector_config, self.logger)
        else:
            self.logger.warning("⚠️ Vector database config not found")
            return None
    
    def _init_graph_database(self):
        """Initialize graph database"""
        graph_config = self.config.get_database_config('graph_db')
        if graph_config:
            return GraphDatabaseManager(graph_config, self.logger)
        else:
            self.logger.warning("⚠️ Graph database config not found")
            return None
    
    def _init_metadata_database(self):
        """Initialize metadata database"""
        metadata_config = self.config.get_database_config('metadata_db')
        if metadata_config:
            return MetadataDatabaseManager(metadata_config, self.logger)
        else:
            self.logger.warning("⚠️ Metadata database config not found")
            return None


    # ==================== VIDEO PROCESSING STORAGE ====================
    
    @measure_time
    def store_processed_video(self, processed_video: ProcessedVideo) -> bool:
        """
        Store complete processed video data across all databases
        
        Args:
            processed_video: ProcessedVideo object to store
            
        Returns:
            True if successful
        """
        try:
            with performance_monitor.monitor(f"store_video_{processed_video.video_id}"):
                success = True
                
                # Store in metadata database
                if self.metadata_db:
                    if not self.metadata_db.store_processed_video(processed_video):
                        self.logger.error("❌ Failed to store in metadata database")
                        success = False
                
                # Store embeddings in vector database
                if self.vector_db and processed_video.success:
                    if not self._store_video_embeddings(processed_video):
                        self.logger.error("❌ Failed to store embeddings")
                        success = False
                
                if success:
                    self.logger.info(f"✅ Stored processed video: {processed_video.video_id}")
                
                return success
                
        except Exception as e:
            self.logger.error(f"❌ Error storing processed video: {e}")
            return False
    
    def _store_video_embeddings(self, processed_video: ProcessedVideo) -> bool:
        """Store video embeddings in vector database"""
        try:
            embedding_entries = []
            
            for chunk in processed_video.chunks:
                # Store chunk embedding
                if chunk.chunk_embedding is not None:
                    chunk_entry = EmbeddingEntry(
                        entry_id=f"chunk_{chunk.chunk_id}",
                        vector=chunk.chunk_embedding,
                        metadata={
                            'chunk_id': chunk.chunk_id,
                            'start_time': chunk.start_time,
                            'end_time': chunk.end_time,
                            'description': chunk.unified_description,
                            'frame_count': len(chunk.frames)
                        },
                        source_type='chunk',
                        source_id=chunk.chunk_id,
                        video_id=processed_video.video_id,
                        timestamp=chunk.start_time
                    )
                    embedding_entries.append(chunk_entry)
                
                # Store frame embeddings (if available)
                for frame in chunk.frames:
                    if frame.embedding is not None:
                        frame_entry = EmbeddingEntry(
                            entry_id=f"frame_{frame.frame_id}",
                            vector=frame.embedding,
                            metadata={
                                'frame_id': frame.frame_id,
                                'frame_index': frame.frame_index,
                                'caption': frame.caption,
                                'confidence': frame.confidence
                            },
                            source_type='frame',
                            source_id=frame.frame_id,
                            video_id=processed_video.video_id,
                            timestamp=frame.timestamp
                        )
                        embedding_entries.append(frame_entry)
            
            # Batch store embeddings
            if embedding_entries:
                stored_count = self.vector_db.add_embeddings_batch(embedding_entries)
                self.logger.info(f"📦 Stored {stored_count} embeddings for video {processed_video.video_id}")
                return stored_count > 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing video embeddings: {e}")
            return False


    # ==================== KNOWLEDGE GRAPH STORAGE ====================
    
    def store_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> bool:
        """
        Store knowledge graph và entity embeddings
        
        Args:
            knowledge_graph: KnowledgeGraph to store
            
        Returns:
            True if successful
        """
        try:
            success = True
            
            # Store in graph database
            if self.graph_db:
                if not self.graph_db.store_knowledge_graph(knowledge_graph):
                    self.logger.error("❌ Failed to store knowledge graph")
                    success = False
            
            # Store entity embeddings in vector database
            if self.vector_db:
                if not self._store_entity_embeddings(knowledge_graph):
                    self.logger.error("❌ Failed to store entity embeddings")
                    success = False
            
            if success:
                self.logger.info(f"✅ Stored knowledge graph: {knowledge_graph.graph_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error storing knowledge graph: {e}")
            return False
    
    def _store_entity_embeddings(self, knowledge_graph: KnowledgeGraph) -> bool:
        """Store entity embeddings in vector database"""
        try:
            embedding_entries = []
            
            for entity in knowledge_graph.entities.values():
                if entity.embedding is not None:
                    entity_entry = EmbeddingEntry(
                        entry_id=f"entity_{entity.entity_id}",
                        vector=entity.embedding,
                        metadata={
                            'entity_id': entity.entity_id,
                            'name': entity.name,
                            'entity_type': entity.entity_type,
                            'description': entity.description,
                            'confidence': entity.confidence,
                            'source_videos': list(entity.source_videos),
                            'aliases': list(entity.aliases)
                        },
                        source_type='entity',
                        source_id=entity.entity_id,
                        video_id=list(entity.source_videos)[0] if entity.source_videos else '',
                        timestamp=None
                    )
                    embedding_entries.append(entity_entry)
            
            # Batch store embeddings
            if embedding_entries:
                stored_count = self.vector_db.add_embeddings_batch(embedding_entries)
                self.logger.info(f"📦 Stored {stored_count} entity embeddings for graph {knowledge_graph.graph_id}")
                return stored_count > 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing entity embeddings: {e}")
            return False


    # ==================== QUERY METHODS ====================
    
    def query_similar_embeddings(
        self,
        query_vector: np.ndarray,
        source_type: Optional[str] = None,
        video_id: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> QueryResult:
        """
        Query similar embeddings from vector database
        
        Args:
            query_vector: Query vector for similarity search
            source_type: Filter by source type ('frame', 'chunk', 'entity')
            video_id: Filter by video ID
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            QueryResult with similar embeddings
        """
        start_time = time.time()
        
        if not self.vector_db:
            return QueryResult(False, error_message="Vector database not available")
        
        try:
            # Prepare filters
            filters = {}
            if source_type:
                filters['source_type'] = source_type
            if video_id:
                filters['video_id'] = video_id
            
            # Search similar vectors
            results = self.vector_db.search_similar(
                query_vector, top_k, threshold, filters
            )
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                success=True,
                results=results,
                total_count=len(results),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Error querying similar embeddings: {e}")
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def query_knowledge_graph(
        self,
        graph_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> QueryResult:
        """
        Query knowledge graph for entities and relations
        
        Args:
            graph_id: Specific graph ID to search
            entity_type: Filter by entity type
            relation_type: Filter by relation type
            video_id: Filter by video ID
            
        Returns:
            QueryResult with entities and relations
        """
        start_time = time.time()
        
        if not self.graph_db:
            return QueryResult(False, error_message="Graph database not available")
        
        try:
            results = {
                'entities': [],
                'relations': []
            }
            
            # Query entities
            entities = self.graph_db.query_entities(
                graph_id=graph_id,
                entity_type=entity_type,
                video_id=video_id
            )
            results['entities'] = entities
            
            # Query relations
            relations = self.graph_db.query_relations(
                graph_id=graph_id,
                relation_type=relation_type
            )
            results['relations'] = relations
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                success=True,
                results=results,
                total_count=len(entities) + len(relations),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Error querying knowledge graph: {e}")
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def search_videos(
        self,
        query_text: Optional[str] = None,
        video_id: Optional[str] = None,
        success_only: bool = True,
        limit: int = 50
    ) -> QueryResult:
        """
        Search videos and their content
        
        Args:
            query_text: Search text for captions/transcripts
            video_id: Specific video ID
            success_only: Only return successfully processed videos
            limit: Maximum results to return
            
        Returns:
            QueryResult with matching videos and content
        """
        start_time = time.time()
        
        if not self.metadata_db:
            return QueryResult(False, error_message="Metadata database not available")
        
        try:
            results = {
                'videos': [],
                'frames': [],
                'audio_segments': []
            }
            
            if video_id:
                # Get specific video
                video_metadata = self.metadata_db.get_video_metadata(video_id)
                if video_metadata:
                    results['videos'] = [video_metadata]
            else:
                # List videos
                results['videos'] = self.metadata_db.list_videos(
                    limit=limit, success_only=success_only
                )
            
            if query_text:
                # Search frames by caption
                results['frames'] = self.metadata_db.search_frames_by_caption(
                    query_text, video_id, limit
                )
                
                # Search audio by transcript
                results['audio_segments'] = self.metadata_db.search_audio_by_transcript(
                    query_text, video_id, limit
                )
            
            execution_time = time.time() - start_time
            total_count = (len(results['videos']) + len(results['frames']) + 
                          len(results['audio_segments']))
            
            return QueryResult(
                success=True,
                results=results,
                total_count=total_count,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Error searching videos: {e}")
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def get_video_timeline(self, video_id: str) -> QueryResult:
        """
        Get complete timeline for a video
        
        Args:
            video_id: Video ID to get timeline for
            
        Returns:
            QueryResult with complete timeline data
        """
        start_time = time.time()
        
        if not self.metadata_db:
            return QueryResult(False, error_message="Metadata database not available")
        
        try:
            timeline_data = self.metadata_db.get_video_timeline(video_id)
            execution_time = time.time() - start_time
            
            return QueryResult(
                success=bool(timeline_data),
                results=timeline_data,
                total_count=1 if timeline_data else 0,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Error getting video timeline: {e}")
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


    # ==================== HYBRID QUERY METHODS ====================
    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: Optional[np.ndarray] = None,
        video_id: Optional[str] = None,
        top_k: int = 10,
        text_weight: float = 0.6,
        vector_weight: float = 0.4
    ) -> QueryResult:
        """
        Perform hybrid search combining text and vector similarity
        
        Args:
            query_text: Text query for semantic search
            query_vector: Vector for similarity search
            video_id: Filter by video ID
            top_k: Number of results to return
            text_weight: Weight for text-based results
            vector_weight: Weight for vector-based results
            
        Returns:
            QueryResult with combined results
        """
        start_time = time.time()
        
        try:
            combined_results = {}
            
            # Text-based search
            text_results = self.search_videos(
                query_text=query_text,
                video_id=video_id,
                limit=top_k * 2
            )
            
            if text_results.success:
                # Score text results
                self._score_text_results(combined_results, text_results, query_text, text_weight)
            
            # Vector-based search
            if query_vector is not None:
                vector_results = self.query_similar_embeddings(
                    query_vector=query_vector,
                    video_id=video_id,
                    top_k=top_k * 2
                )
                
                if vector_results.success:
                    self._score_vector_results(combined_results, vector_results, vector_weight)
            
            # Combine and sort results
            final_results = self._combine_hybrid_results(combined_results, top_k)
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                success=True,
                results=final_results,
                total_count=len(final_results),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Error in hybrid search: {e}")
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _score_text_results(self, combined_results: dict, text_results: QueryResult, query_text: str, text_weight: float):
        """Score text-based search results"""
        for frame in text_results.results.get('frames', []):
            frame_id = frame['frame_id']
            # Simple text relevance scoring
            relevance = query_text.lower().count(' '.join(frame['caption'].lower().split()[:3]))
            combined_results[frame_id] = {
                'item': frame,
                'text_score': relevance * text_weight,
                'vector_score': 0.0,
                'type': 'frame'
            }
        
        for audio in text_results.results.get('audio_segments', []):
            segment_id = audio['segment_id']
            relevance = query_text.lower().count(' '.join(audio['transcript'].lower().split()[:3]))
            combined_results[segment_id] = {
                'item': audio,
                'text_score': relevance * text_weight,
                'vector_score': 0.0,
                'type': 'audio'
            }
    
    def _score_vector_results(self, combined_results: dict, vector_results: QueryResult, vector_weight: float):
        """Score vector-based search results"""
        for entry, similarity in vector_results.results:
            item_id = entry.entry_id
            if item_id in combined_results:
                combined_results[item_id]['vector_score'] = similarity * vector_weight
            else:
                combined_results[item_id] = {
                    'item': entry,
                    'text_score': 0.0,
                    'vector_score': similarity * vector_weight,
                    'type': entry.source_type
                }
    
    def _combine_hybrid_results(self, combined_results: dict, top_k: int) -> List[Dict[str, Any]]:
        """Combine and sort hybrid search results"""
        final_results = []
        for item_id, result_data in combined_results.items():
            total_score = result_data['text_score'] + result_data['vector_score']
            final_results.append({
                'id': item_id,
                'item': result_data['item'],
                'total_score': total_score,
                'text_score': result_data['text_score'],
                'vector_score': result_data['vector_score'],
                'type': result_data['type']
            })
        
        # Sort by total score
        final_results.sort(key=lambda x: x['total_score'], reverse=True)
        return final_results[:top_k]


    # ==================== MANAGEMENT METHODS ====================
    
    def delete_video(self, video_id: str) -> bool:
        """
        Delete video and all associated data from all databases
        
        Args:
            video_id: Video ID to delete
            
        Returns:
            True if successful
        """
        try:
            success = True
            
            # Delete from metadata database
            if self.metadata_db:
                if not self.metadata_db.delete_video(video_id):
                    self.logger.error("❌ Failed to delete from metadata database")
                    success = False
            
            # Delete embeddings from vector database
            if self.vector_db:
                # Find and delete all embeddings for this video
                embeddings_to_delete = []
                for entry_id, entry in self.vector_db.vectors.items():
                    if entry.video_id == video_id:
                        embeddings_to_delete.append(entry_id)
                
                for entry_id in embeddings_to_delete:
                    self.vector_db.delete_embedding(entry_id)
                
                self.logger.info(f"🗑️ Deleted {len(embeddings_to_delete)} embeddings for video {video_id}")
            
            # Note: Knowledge graph entities are shared across videos,
            # so we don't automatically delete them
            
            if success:
                self.logger.info(f"✅ Deleted video: {video_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error deleting video: {e}")
            return False
        
# ==================== SYSTEM MANAGEMENT METHODS ====================
    
    def sync_distributed_data(self, source_db_path: str) -> bool:
        """
        Sync data from distributed processing node
        
        Args:
            source_db_path: Path to source database directory
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"🔄 Syncing data from: {source_db_path}")
            
            source_path = Path(source_db_path)
            if not source_path.exists():
                self.logger.error(f"❌ Source path does not exist: {source_db_path}")
                return False
            
            success = True
            
            # Sync vector database
            if self.vector_db:
                success &= self._sync_vector_database(source_path)
            
            # Sync graph database
            if self.graph_db:
                success &= self._sync_graph_database(source_path)
            
            # Sync metadata database
            if self.metadata_db:
                success &= self._sync_metadata_database(source_path)
            
            if success:
                self.logger.info("✅ Data sync completed successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing distributed data: {e}")
            return False
    
    def _sync_vector_database(self, source_path: Path) -> bool:
        """Sync vector database from source"""
        try:
            source_vector_file = source_path / "vectors.pkl"
            if source_vector_file.exists():
                with open(source_vector_file, 'rb') as f:
                    source_data = pickle.load(f)
                    source_vectors = source_data.get('vectors', {})
                
                # Merge vectors
                merged_count = 0
                for entry_id, entry in source_vectors.items():
                    if entry_id not in self.vector_db.vectors:
                        self.vector_db.vectors[entry_id] = entry
                        merged_count += 1
                
                self.vector_db.index_dirty = True
                self.vector_db._save_vectors()
                
                self.logger.info(f"📦 Merged {merged_count} new vectors")
                return True
        except Exception as e:
            self.logger.error(f"❌ Error syncing vectors: {e}")
            return False
    
    def _sync_graph_database(self, source_path: Path) -> bool:
        """Sync graph database from source"""
        try:
            source_graph_file = source_path / "knowledge_graphs.pkl"
            if source_graph_file.exists():
                with open(source_graph_file, 'rb') as f:
                    source_data = pickle.load(f)
                    source_graphs = source_data.get('graphs', {})
                
                # Merge graphs
                merged_count = 0
                for graph_id, graph in source_graphs.items():
                    if graph_id not in self.graph_db.knowledge_graphs:
                        self.graph_db.knowledge_graphs[graph_id] = graph
                        merged_count += 1
                
                self.graph_db._save_graphs()
                
                self.logger.info(f"📊 Merged {merged_count} new knowledge graphs")
                return True
        except Exception as e:
            self.logger.error(f"❌ Error syncing graphs: {e}")
            return False
    
    def _sync_metadata_database(self, source_path: Path) -> bool:
        """Sync metadata database from source"""
        try:
            source_metadata_file = source_path / "metadata.db"
            if source_metadata_file.exists():
                # Copy new videos from source database
                with sqlite3.connect(str(source_metadata_file)) as source_conn:
                    source_cursor = source_conn.cursor()
                    
                    # Get all videos from source
                    source_cursor.execute("SELECT * FROM videos")
                    source_videos = source_cursor.fetchall()
                    
                    merged_count = 0
                    with sqlite3.connect(self.metadata_db.db_path) as target_conn:
                        target_cursor = target_conn.cursor()
                        
                        for video_row in source_videos:
                            video_id = video_row[0]  # First column is video_id
                            
                            # Check if video already exists
                            target_cursor.execute("SELECT COUNT(*) FROM videos WHERE video_id = ?", (video_id,))
                            if target_cursor.fetchone()[0] == 0:
                                # Insert new video
                                placeholders = ','.join(['?'] * len(video_row))
                                target_cursor.execute(f"INSERT INTO videos VALUES ({placeholders})", video_row)
                                
                                # Copy associated chunks, frames, and audio segments
                                self._copy_video_related_data(source_conn, target_conn, video_id)
                                merged_count += 1
                        
                        target_conn.commit()
                
                self.logger.info(f"📋 Merged {merged_count} new videos with metadata")
                return True
        except Exception as e:
            self.logger.error(f"❌ Error syncing metadata: {e}")
            return False
    
    def _copy_video_related_data(self, source_conn, target_conn, video_id: str):
        """Copy chunks, frames, and audio segments for a video"""
        source_cursor = source_conn.cursor()
        target_cursor = target_conn.cursor()
        
        # Copy chunks
        source_cursor.execute("SELECT * FROM chunks WHERE video_id = ?", (video_id,))
        chunks = source_cursor.fetchall()
        
        for chunk_row in chunks:
            placeholders = ','.join(['?'] * len(chunk_row))
            target_cursor.execute(f"INSERT INTO chunks VALUES ({placeholders})", chunk_row)
        
        # Copy frames
        source_cursor.execute("SELECT * FROM frames WHERE video_id = ?", (video_id,))
        frames = source_cursor.fetchall()
        
        for frame_row in frames:
            placeholders = ','.join(['?'] * len(frame_row))
            target_cursor.execute(f"INSERT INTO frames VALUES ({placeholders})", frame_row)
        
        # Copy audio segments
        source_cursor.execute("SELECT * FROM audio_segments WHERE video_id = ?", (video_id,))
        audio_segments = source_cursor.fetchall()
        
        for audio_row in audio_segments:
            placeholders = ','.join(['?'] * len(audio_row))
            target_cursor.execute(f"INSERT INTO audio_segments VALUES ({placeholders})", audio_row)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all databases
        
        Returns:
            Dictionary containing statistics from all database components
        """
        try:
            stats = {
                'timestamp': time.time(),
                'vector_db': {},
                'graph_db': {},
                'metadata_db': {},
                'system_info': {}
            }
            
            # Vector database stats
            if self.vector_db:
                stats['vector_db'] = self.vector_db.get_stats()
            
            # Graph database stats
            if self.graph_db:
                stats['graph_db'] = self.graph_db.get_stats()
            
            # Metadata database stats
            if self.metadata_db:
                stats['metadata_db'] = self.metadata_db.get_stats()
            
            # System info
            stats['system_info'] = {
                'faiss_available': FAISS_AVAILABLE,
                'networkx_available': NETWORKX_AVAILABLE,
                'config_databases': len(self.config.databases),
                'active_connections': sum([
                    1 for db in [self.vector_db, self.graph_db, self.metadata_db] 
                    if db is not None
                ])
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system stats: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up old data from databases
        
        Args:
            days_old: Remove data older than this many days
            
        Returns:
            Dictionary with cleanup counts
        """
        try:
            cleanup_stats = {
                'videos_deleted': 0,
                'embeddings_deleted': 0,
                'graphs_cleaned': 0
            }
            
            cutoff_time = time.time() - (days_old * 24 * 3600)
            
            # Clean up old videos from metadata DB
            if self.metadata_db:
                cleanup_stats['videos_deleted'] = self._cleanup_old_videos(cutoff_time)
            
            # Clean up orphaned embeddings
            if self.vector_db and self.metadata_db:
                cleanup_stats['embeddings_deleted'] = self._cleanup_orphaned_embeddings()
            
            self.logger.info(f"🧹 Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_videos(self, cutoff_time: float) -> int:
        """Clean up old failed videos"""
        videos_deleted = 0
        
        with sqlite3.connect(self.metadata_db.db_path) as conn:
            cursor = conn.cursor()
            
            # Find old videos
            cursor.execute("""
                SELECT video_id FROM videos 
                WHERE created_at < ? AND success = 0
            """, (cutoff_time,))
            
            old_videos = [row[0] for row in cursor.fetchall()]
            
            for video_id in old_videos:
                if self.delete_video(video_id):
                    videos_deleted += 1
        
        return videos_deleted
    
    def _cleanup_orphaned_embeddings(self) -> int:
        """Clean up orphaned embeddings"""
        embeddings_deleted = 0
        
        with sqlite3.connect(self.metadata_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT video_id FROM videos")
            valid_video_ids = {row[0] for row in cursor.fetchall()}
        
        embeddings_to_delete = []
        for entry_id, entry in self.vector_db.vectors.items():
            if entry.video_id not in valid_video_ids:
                embeddings_to_delete.append(entry_id)
        
        for entry_id in embeddings_to_delete:
            if self.vector_db.delete_embedding(entry_id):
                embeddings_deleted += 1
        
        return embeddings_deleted
    
    def backup_databases(self, backup_dir: str) -> bool:
        """
        Create backup of all databases
        
        Args:
            backup_dir: Directory to store backups
            
        Returns:
            True if successful
        """
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_subdir = backup_path / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            success = True
            
            # Backup each database component
            success &= self._backup_vector_database(backup_subdir)
            success &= self._backup_graph_database(backup_subdir)
            success &= self._backup_metadata_database(backup_subdir)
            
            # Create backup manifest
            self._create_backup_manifest(backup_subdir, timestamp)
            
            if success:
                self.logger.info(f"✅ Database backup completed: {backup_subdir}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error creating database backup: {e}")
            return False
    
    def _backup_vector_database(self, backup_subdir: Path) -> bool:
        """Backup vector database"""
        if not self.vector_db:
            return True
        
        try:
            vector_backup = backup_subdir / "vector_db"
            vector_backup.mkdir(exist_ok=True)
            
            # Save current vectors
            self.vector_db._save_vectors()
            
            # Copy vector files
            source_vector_file = self.vector_db.db_path / "vectors.pkl"
            if source_vector_file.exists():
                import shutil
                shutil.copy2(source_vector_file, vector_backup / "vectors.pkl")
            
            self.logger.info(f"📦 Vector database backed up to {vector_backup}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error backing up vector database: {e}")
            return False
    
    def _backup_graph_database(self, backup_subdir: Path) -> bool:
        """Backup graph database"""
        if not self.graph_db:
            return True
        
        try:
            graph_backup = backup_subdir / "graph_db"
            graph_backup.mkdir(exist_ok=True)
            
            # Save current graphs
            self.graph_db._save_graphs()
            
            # Copy graph files
            source_graph_file = self.graph_db.db_path / "knowledge_graphs.pkl"
            if source_graph_file.exists():
                import shutil
                shutil.copy2(source_graph_file, graph_backup / "knowledge_graphs.pkl")
            
            self.logger.info(f"📊 Graph database backed up to {graph_backup}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error backing up graph database: {e}")
            return False
    
    def _backup_metadata_database(self, backup_subdir: Path) -> bool:
        """Backup metadata database"""
        if not self.metadata_db:
            return True
        
        try:
            metadata_backup = backup_subdir / "metadata_db"
            metadata_backup.mkdir(exist_ok=True)
            
            # Copy SQLite database
            source_db_file = Path(self.metadata_db.db_path)
            if source_db_file.exists():
                import shutil
                shutil.copy2(source_db_file, metadata_backup / "metadata.db")
            
            self.logger.info(f"📋 Metadata database backed up to {metadata_backup}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error backing up metadata database: {e}")
            return False
    
    def _create_backup_manifest(self, backup_subdir: Path, timestamp: str):
        """Create backup manifest file"""
        manifest = {
            'timestamp': timestamp,
            'backup_time': time.time(),
            'databases': {
                'vector_db': self.vector_db is not None,
                'graph_db': self.graph_db is not None,
                'metadata_db': self.metadata_db is not None
            },
            'stats': self.get_system_stats()
        }
        
        with open(backup_subdir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def restore_databases(self, backup_dir: str) -> bool:
        """
        Restore databases from backup
        
        Args:
            backup_dir: Directory containing backup
            
        Returns:
            True if successful
        """
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                self.logger.error(f"❌ Backup directory does not exist: {backup_dir}")
                return False
            
            # Read manifest
            manifest_file = backup_path / "manifest.json"
            if not manifest_file.exists():
                self.logger.error(f"❌ Backup manifest not found: {manifest_file}")
                return False
            
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            self.logger.info(f"🔄 Restoring backup from {manifest.get('timestamp', 'unknown')}")
            
            success = True
            
            # Restore each database component
            success &= self._restore_vector_database(backup_path, manifest)
            success &= self._restore_graph_database(backup_path, manifest)
            success &= self._restore_metadata_database(backup_path, manifest)
            
            if success:
                self.logger.info("✅ Database restoration completed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error restoring databases: {e}")
            return False
    
    def _restore_vector_database(self, backup_path: Path, manifest: dict) -> bool:
        """Restore vector database from backup"""
        if not (self.vector_db and manifest['databases'].get('vector_db')):
            return True
        
        try:
            vector_backup = backup_path / "vector_db" / "vectors.pkl"
            if vector_backup.exists():
                import shutil
                target_file = self.vector_db.db_path / "vectors.pkl"
                shutil.copy2(vector_backup, target_file)
                
                # Reload vectors
                self.vector_db._load_vectors()
                self.vector_db._init_faiss_index()
                
                self.logger.info("📦 Vector database restored")
                return True
        except Exception as e:
            self.logger.error(f"❌ Error restoring vector database: {e}")
            return False
    
    def _restore_graph_database(self, backup_path: Path, manifest: dict) -> bool:
        """Restore graph database from backup"""
        if not (self.graph_db and manifest['databases'].get('graph_db')):
            return True
        
        try:
            graph_backup = backup_path / "graph_db" / "knowledge_graphs.pkl"
            if graph_backup.exists():
                import shutil
                target_file = self.graph_db.db_path / "knowledge_graphs.pkl"
                shutil.copy2(graph_backup, target_file)
                
                # Reload graphs
                self.graph_db._load_graphs()
                
                self.logger.info("📊 Graph database restored")
                return True
        except Exception as e:
            self.logger.error(f"❌ Error restoring graph database: {e}")
            return False
    
    def _restore_metadata_database(self, backup_path: Path, manifest: dict) -> bool:
        """Restore metadata database from backup"""
        if not (self.metadata_db and manifest['databases'].get('metadata_db')):
            return True
        
        try:
            metadata_backup = backup_path / "metadata_db" / "metadata.db"
            if metadata_backup.exists():
                import shutil
                shutil.copy2(metadata_backup, self.metadata_db.db_path)
                
                self.logger.info("📋 Metadata database restored")
                return True
        except Exception as e:
            self.logger.error(f"❌ Error restoring metadata database: {e}")
            return False
    
    def __del__(self):
        """Cleanup when DatabaseManager is destroyed"""
        try:
            # Save any pending data
            if hasattr(self, 'vector_db') and self.vector_db:
                self.vector_db._save_vectors()
            
            if hasattr(self, 'graph_db') and self.graph_db:
                self.graph_db._save_graphs()
                
        except:
            pass


# ==================== GLOBAL INSTANCE MANAGEMENT ====================

# Global database manager instance
_global_database_manager = None

def get_database_manager(config: Optional[Config] = None) -> DatabaseManager:
    """
    Get global database manager instance (singleton pattern)
    
    Args:
        config: Configuration object (chỉ sử dụng lần đầu)
        
    Returns:
        DatabaseManager instance
    """
    global _global_database_manager
    if _global_database_manager is None:
        _global_database_manager = DatabaseManager(config)
    return _global_database_manager

def reset_database_manager():
    """Reset global database manager instance"""
    global _global_database_manager
    if _global_database_manager:
        del _global_database_manager
    _global_database_manager = None