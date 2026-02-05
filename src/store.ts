/**
 * QMD Store - Core data access and retrieval functions
 *
 * This module provides all database operations, search functions, and document
 * retrieval for QMD. It returns raw data structures that can be formatted by
 * CLI or MCP consumers.
 *
 * Usage:
 *   const store = createStore("/path/to/db.sqlite");
 *   // or use default path:
 *   const store = createStore();
 *
 * Architecture:
 *   - store_types.ts: Type definitions (interfaces, types, constants)
 *   - store_util.ts: Generic helpers (path utils, chunking, hashing, fuzzy matching)
 *   - store_sqlite.ts: SQLite implementation (database init, queries, store factory)
 *   - store.ts: This file - imports and re-exports all public APIs
 */

import { connectTursoDb, createTursoStore } from "./store_turso";
import { createSqliteStore } from "./store_sqlite";
import type { Store, StoreOptions } from "./store_types";

// =============================================================================
// Re-export types
// =============================================================================

export type {
  VirtualPath,
  DocumentResult,
  DocumentNotFound,
  MultiGetResult,
  SearchResult,
  RankedResult,
  CollectionInfo,
  IndexStatus,
  IndexHealthInfo,
  SnippetResult,
  Store,
} from "./store_types";

export {
  DEFAULT_EMBED_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_GLOB,
  DEFAULT_MULTI_GET_MAX_BYTES,
  CHUNK_SIZE_TOKENS,
  CHUNK_OVERLAP_TOKENS,
  CHUNK_SIZE_CHARS,
  CHUNK_OVERLAP_CHARS,
} from "./store_types";

// =============================================================================
// Re-export utilities
// =============================================================================

export {
  // Environment
  homedir,

  // Path utilities
  isAbsolutePath,
  normalizePathSeparators,
  getRelativePathFromPrefix,
  resolve,
  getPwd,
  getRealPath,

  // Virtual path utilities
  normalizeVirtualPath,
  parseVirtualPath,
  buildVirtualPath,
  isVirtualPath,

  // Hashing and caching
  hashContent,
  getCacheKey,

  // Document helpers
  getDocid,
  normalizeDocid,
  isDocid,
  extractTitle,
  handelize,

  // Chunking
  chunkDocument,
  chunkDocumentByTokens,

  // Fuzzy matching
  levenshtein,

  // Reciprocal Rank Fusion
  reciprocalRankFusion,

  // Snippet extraction
  extractSnippet,
} from "./store_util";

// =============================================================================
// Re-export SQLite implementation
// =============================================================================

export {
  enableProductionMode,
  getDefaultDbPath,
} from "./store_sqlite";

// =============================================================================
// Re-export LLM formatting functions (for backward compatibility)
// =============================================================================

export { formatQueryForEmbedding, formatDocForEmbedding } from "./llm";

// =============================================================================
// Re-export Turso implementation
// =============================================================================

export {
  createTursoStore,
  type TursoDatabase,
} from "./store_turso";

export async function createStore(dbPath?: string, options?: StoreOptions): Promise<Store> {
  if (options?.dbName == 'turso') {
    const { db, path } = await connectTursoDb(dbPath);
    return createTursoStore(db, path);
  } else {
    return createSqliteStore(dbPath);
  }
}