/**
 * QMD Store Types - Type definitions for the store module
 *
 * This module contains all type definitions used across the store.
 * No methods or implementations - types only.
 */

import type { Database } from "bun:sqlite";

// =============================================================================
// Configuration Constants
// =============================================================================

export const DEFAULT_EMBED_MODEL = "embeddinggemma";
export const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
export const DEFAULT_QUERY_MODEL = "Qwen/Qwen3-1.7B";
export const DEFAULT_GLOB = "**/*.md";
export const DEFAULT_MULTI_GET_MAX_BYTES = 10 * 1024; // 10KB

// Chunking: 800 tokens per chunk with 15% overlap
export const CHUNK_SIZE_TOKENS = 800;
export const CHUNK_OVERLAP_TOKENS = Math.floor(CHUNK_SIZE_TOKENS * 0.15);  // 120 tokens (15% overlap)
// Fallback char-based approximation for sync chunking (~4 chars per token)
export const CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4;  // 3200 chars
export const CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4;  // 480 chars

// =============================================================================
// Virtual Path Types
// =============================================================================

export type VirtualPath = {
  collectionName: string;
  path: string;  // relative path within collection
};

// =============================================================================
// Document Types
// =============================================================================

/**
 * Unified document result type with all metadata.
 * Body is optional - use getDocumentBody() to load it separately if needed.
 */
export type DocumentResult = {
  filepath: string;           // Full filesystem path
  displayPath: string;        // Short display path (e.g., "docs/readme.md")
  title: string;              // Document title (from first heading or filename)
  context: string | null;     // Folder context description if configured
  hash: string;               // Content hash for caching/change detection
  docid: string;              // Short docid (first 6 chars of hash) for quick reference
  collectionName: string;     // Parent collection name
  modifiedAt: string;         // Last modification timestamp
  bodyLength: number;         // Body length in bytes (useful before loading)
  body?: string;              // Document body (optional, load with getDocumentBody)
};

/**
 * Error result when document is not found
 */
export type DocumentNotFound = {
  error: "not_found";
  query: string;
  similarFiles: string[];
};

/**
 * Result from multi-get operations
 */
export type MultiGetResult = {
  doc: DocumentResult;
  skipped: false;
} | {
  doc: Pick<DocumentResult, "filepath" | "displayPath">;
  skipped: true;
  skipReason: string;
};

// =============================================================================
// Search Types
// =============================================================================

/**
 * Search result extends DocumentResult with score and source info
 */
export type SearchResult = DocumentResult & {
  score: number;              // Relevance score (0-1)
  source: "fts" | "vec";      // Search source (full-text or vector)
  chunkPos?: number;          // Character position of matching chunk (for vector search)
};

/**
 * Ranked result for RRF fusion (simplified, used internally)
 */
export type RankedResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
};

// =============================================================================
// Collection Types
// =============================================================================

export type CollectionInfo = {
  name: string;
  path: string;
  pattern: string;
  documents: number;
  lastUpdated: string;
};

// =============================================================================
// Index Types
// =============================================================================

export type IndexStatus = {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: CollectionInfo[];
};

export type IndexHealthInfo = {
  needsEmbedding: number;
  totalDocs: number;
  daysStale: number | null;
};

// =============================================================================
// Snippet Types
// =============================================================================

export type SnippetResult = {
  line: number;           // 1-indexed line number of best match
  snippet: string;        // The snippet text with diff-style header
  linesBefore: number;    // Lines in document before snippet
  linesAfter: number;     // Lines in document after snippet
  snippetLines: number;   // Number of lines in snippet
};

// =============================================================================
// Store Interface
// =============================================================================

export type StoreOptions = {
  dbName: string
}

export type StoreDb = {
  name: 'sqlite3' | 'turso';
  db: any;
  exec(query: string, ...params: any): Promise<{ lastInsertRowid: number }>;
  get(query: string, ...params: any): Promise<any>;
  all(query: string, ...params: any): Promise<any[]>;
}

export type Store = {
  db: StoreDb;
  dbPath: string;
  close: () => void;
  ensureVecTable: (dimensions: number) => Promise<void>;

  // Index health
  getHashesNeedingEmbedding: () => Promise<number>;
  getIndexHealth: () => Promise<IndexHealthInfo>;
  getStatus: () => Promise<IndexStatus>;

  // Caching (getCacheKey is sync - pure computation)
  getCacheKey: (url: string, body: object) => string;
  getCachedResult: (cacheKey: string) => Promise<string | null>;
  setCachedResult: (cacheKey: string, result: string) => Promise<void>;
  clearCache: () => Promise<void>;

  // Cleanup and maintenance
  deleteLLMCache: () => Promise<number>;
  deleteInactiveDocuments: () => Promise<number>;
  cleanupOrphanedContent: () => Promise<number>;
  cleanupOrphanedVectors: () => Promise<number>;
  vacuumDatabase: () => Promise<void>;

  // Context
  getContextForFile: (filepath: string) => Promise<string | null>;
  getContextForPath: (collectionName: string, path: string) => Promise<string | null>;
  getCollectionByName: (name: string) => { name: string; pwd: string; glob_pattern: string } | null;
  getCollectionsWithoutContext: () => Promise<{ name: string; pwd: string; doc_count: number }[]>;
  getTopLevelPathsWithoutContext: (collectionName: string) => Promise<string[]>;

  // Virtual paths (sync - pure computation, no DB access)
  parseVirtualPath: (virtualPath: string) => VirtualPath | null;
  buildVirtualPath: (collectionName: string, path: string) => string;
  isVirtualPath: (path: string) => boolean;
  resolveVirtualPath: (virtualPath: string) => string | null;
  toVirtualPath: (absolutePath: string) => Promise<string | null>;

  // Search
  searchFTS: (query: string, limit?: number, collectionId?: number) => Promise<SearchResult[]>;
  searchVec: (generate: () => Promise<number[] | null>, limit?: number, collectionName?: string) => Promise<SearchResult[]>;

  // Query expansion & reranking
  expandQuery: (query: string, model?: string) => Promise<string[]>;
  rerank: (query: string, documents: { file: string; text: string }[], model?: string) => Promise<{ file: string; score: number }[]>;

  // Document retrieval
  findDocument: (filename: string, options?: { includeBody?: boolean }) => Promise<DocumentResult | DocumentNotFound>;
  getDocumentBody: (doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number) => Promise<string | null>;
  findDocuments: (pattern: string, options?: { includeBody?: boolean; maxBytes?: number }) => Promise<{ docs: MultiGetResult[]; errors: string[] }>;

  // Fuzzy matching and docid lookup
  findSimilarFiles: (query: string, maxDistance?: number, limit?: number) => Promise<string[]>;
  matchFilesByGlob: (pattern: string) => Promise<{ filepath: string; displayPath: string; bodyLength: number }[]>;
  findDocumentByDocid: (docid: string) => Promise<{ filepath: string; hash: string } | null>;

  // Document indexing operations
  insertContent: (hash: string, content: string, createdAt: string) => Promise<void>;
  insertDocument: (collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string) => Promise<number>;
  findActiveDocument: (collectionName: string, path: string) => Promise<{ id: number; hash: string; title: string } | null>;
  updateDocumentTitle: (documentId: number, title: string, modifiedAt: string) => Promise<void>;
  updateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => Promise<void>;
  deactivateDocument: (collectionName: string, path: string) => Promise<void>;
  getActiveDocumentPaths: (collectionName: string) => Promise<string[]>;

  // Vector/embedding operations
  getHashesForEmbedding: () => Promise<{ hash: string; body: string; path: string }[]>;
  clearAllEmbeddings: () => Promise<void>;
  insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string) => Promise<void>;

  // Collection management
  listCollections: () => Promise<{ name: string; pwd: string; glob_pattern: string; doc_count: number; active_count: number; last_modified: string | null }[]>;
  removeCollection: (name: string) => Promise<{ deletedDocs: number; cleanedHashes: number }>;
  renameCollection: (oldName: string, newName: string) => Promise<void>;
  getAllCollections: () => Promise<{ name: string }[]>;

  // Context management
  insertContext: (collectionId: number, pathPrefix: string, context: string) => Promise<void>;
  deleteContext: (collectionName: string, pathPrefix: string) => Promise<number>;
  deleteGlobalContexts: () => Promise<number>;
  listPathContexts: () => Promise<{ collection_name: string; path_prefix: string; context: string }[]>;
};
