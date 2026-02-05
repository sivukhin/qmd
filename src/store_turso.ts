/**
 * QMD Store Turso - Turso database implementation
 *
 * This module provides a Turso-backed store implementation using Turso's native
 * Full-Text Search (Tantivy-based) and Vector Search capabilities.
 *
 * Users provide their own Turso database instance (local or synced).
 *
 * Usage:
 *   import { connect } from '@tursodatabase/database';
 *   const db = await connect(':memory:'); // or file path
 *   const store = await createTursoStore(db);
 *
 * Features:
 *   - Native FTS using Turso's Tantivy-based fts_match/fts_score functions
 *   - Native vector search using vector32 and vector_distance_cos
 *   - No external extensions required (no sqlite-vec, no FTS5)
 *
 * Note: All methods are async due to Turso's async API.
 */

import { connect } from "@tursodatabase/database";
import { Glob } from "bun";
import {
  type Store,
  type DocumentResult,
  type DocumentNotFound,
  type MultiGetResult,
  type SearchResult,
  type IndexStatus,
  type IndexHealthInfo,
  DEFAULT_RERANK_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_MULTI_GET_MAX_BYTES,
} from "./store_types";
import {
  parseVirtualPath,
  buildVirtualPath,
  isVirtualPath,
  getDocid,
  normalizeDocid,
  isDocid,
  levenshtein,
  homedir,
  resolve,
  getCacheKey,
} from "./store_util";
import {
  getCollection,
  listCollections as collectionsListCollections,
  loadConfig as collectionsLoadConfig,
  addContext as collectionsAddContext,
  removeContext as collectionsRemoveContext,
  listAllContexts as collectionsListAllContexts,
  setGlobalContext,
} from "./collections";
import {
  getDefaultLlamaCpp,
  type RerankDocument,
} from "./llm";

// =============================================================================
// Turso Database Type
// =============================================================================

export type TursoDatabase = Awaited<ReturnType<typeof connect>>;

// =============================================================================
// Query Plan Utility
// =============================================================================

export async function explainQuery(db: TursoDatabase, sql: string, params?: any[]): Promise<string> {
  try {
    const rows = await db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...(params || [])) as any[];
    if (rows.length === 0) return "(empty plan)";

    const lines: string[] = [];
    for (const row of rows) {
      const indent = "  ".repeat(row.id || 0);
      const detail = row.detail || row.opcode || JSON.stringify(row);
      lines.push(`${indent}${detail}`);
    }
    return lines.join("\n");
  } catch (e: any) {
    return `(explain failed: ${e.message})`;
  }
}

export async function printQueryPlan(db: TursoDatabase, sql: string, params?: any[]): Promise<void> {
  const plan = await explainQuery(db, sql, params);
  console.log("=== Query Plan ===");
  console.log(sql.trim());
  console.log("---");
  console.log(plan);
  console.log("==================\n");
}

// =============================================================================
// Database Initialization
// =============================================================================

async function initializeDatabase(db: TursoDatabase): Promise<void> {
  await db.exec("PRAGMA journal_mode = WAL");
  await db.exec("PRAGMA foreign_keys = ON");

  await db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      hash TEXT PRIMARY KEY,
      doc TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  await db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection TEXT NOT NULL,
      path TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
      UNIQUE(collection, path)
    )
  `);

  await db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active)`);
  await db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)`);
  await db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active)`);

  await db.exec(`
    CREATE TABLE IF NOT EXISTS llm_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Store embeddings as BLOB (raw binary format)
  await db.exec(`
    CREATE TABLE IF NOT EXISTS content_vectors (
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL DEFAULT 0,
      pos INTEGER NOT NULL DEFAULT 0,
      embedding BLOB NOT NULL,
      model TEXT NOT NULL,
      embedded_at TEXT NOT NULL,
      PRIMARY KEY (hash, seq)
    )
  `);

  await db.exec(`
    CREATE TABLE IF NOT EXISTS documents_search (
      doc_id INTEGER PRIMARY KEY,
      filepath TEXT NOT NULL,
      title TEXT NOT NULL,
      body TEXT NOT NULL
    )
  `);

  await db.exec(`
      CREATE INDEX IF NOT EXISTS fts_documents ON documents_search
      USING fts (filepath, title, body)
      WITH (tokenizer = 'default', weights = 'title=2.0,filepath=1.5,body=1.0')
  `);
}

// =============================================================================
// Index Health
// =============================================================================

async function getHashesNeedingEmbedding(db: TursoDatabase): Promise<number> {
  const result = await db.prepare(`
    SELECT COUNT(DISTINCT d.hash) as count
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
  `).get() as { count: number } | undefined;
  return result?.count ?? 0;
}

async function getIndexHealth(db: TursoDatabase): Promise<IndexHealthInfo> {
  const needsEmbedding = await getHashesNeedingEmbedding(db);
  const totalResult = await db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number } | undefined;
  const totalDocs = totalResult?.count ?? 0;

  const mostRecent = await db.prepare(`SELECT MAX(modified_at) as latest FROM documents WHERE active = 1`).get() as { latest: string | null } | undefined;
  let daysStale: number | null = null;
  if (mostRecent?.latest) {
    const lastUpdate = new Date(mostRecent.latest);
    daysStale = Math.floor((Date.now() - lastUpdate.getTime()) / (24 * 60 * 60 * 1000));
  }

  return { needsEmbedding, totalDocs, daysStale };
}

// =============================================================================
// Caching
// =============================================================================

async function getCachedResult(db: TursoDatabase, cacheKey: string): Promise<string | null> {
  const row = await db.prepare(`SELECT result FROM llm_cache WHERE hash = ?`).get(cacheKey) as { result: string } | undefined;
  return row?.result || null;
}

async function setCachedResult(db: TursoDatabase, cacheKey: string, result: string): Promise<void> {
  const now = new Date().toISOString();
  await db.prepare(`INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)`).run(cacheKey, result, now);
}

async function clearCache(db: TursoDatabase): Promise<void> {
  await db.exec(`DELETE FROM llm_cache`);
}

// =============================================================================
// Cleanup and Maintenance
// =============================================================================

async function deleteLLMCache(db: TursoDatabase): Promise<number> {
  const result = await db.prepare(`DELETE FROM llm_cache`).run();
  return result.changes;
}

async function deleteInactiveDocuments(db: TursoDatabase): Promise<number> {
  const result = await db.prepare(`DELETE FROM documents WHERE active = 0`).run();
  return result.changes;
}

async function cleanupOrphanedContent(db: TursoDatabase): Promise<number> {
  const result = await db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();
  return result.changes;
}

async function cleanupOrphanedVectors(db: TursoDatabase): Promise<number> {
  // Count orphaned vectors first
  const countResult = await db.prepare(`
    SELECT COUNT(*) as c FROM content_vectors cv
    WHERE NOT EXISTS (
      SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
    )
  `).get() as { c: number } | undefined;

  const count = countResult?.c ?? 0;
  if (count === 0) {
    return 0;
  }

  // Delete orphaned vectors
  await db.exec(`
    DELETE FROM content_vectors WHERE hash NOT IN (
      SELECT hash FROM documents WHERE active = 1
    )
  `);

  return count;
}

async function vacuumDatabase(db: TursoDatabase): Promise<void> {
  await db.exec(`VACUUM`);
}

// =============================================================================
// Status
// =============================================================================

async function getStatus(db: TursoDatabase): Promise<IndexStatus> {
  const yamlCollections = collectionsListCollections();

  const collections = [];
  for (const col of yamlCollections) {
    const stats = await db.prepare(`
      SELECT
        COUNT(*) as active_count,
        MAX(modified_at) as last_doc_update
      FROM documents
      WHERE collection = ? AND active = 1
    `).get(col.name) as { active_count: number; last_doc_update: string | null } | undefined;

    collections.push({
      name: col.name,
      path: col.path,
      pattern: col.pattern,
      documents: stats?.active_count ?? 0,
      lastUpdated: stats?.last_doc_update || new Date().toISOString(),
    });
  }

  // Sort by last update time (most recent first)
  collections.sort((a, b) => {
    if (!a.lastUpdated) return 1;
    if (!b.lastUpdated) return -1;
    return new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime();
  });

  const totalResult = await db.prepare(`SELECT COUNT(*) as c FROM documents WHERE active = 1`).get() as { c: number } | undefined;
  const totalDocs = totalResult?.c ?? 0;
  const needsEmbedding = await getHashesNeedingEmbedding(db);

  // Check if any vectors exist
  const vecResult = await db.prepare(`SELECT 1 FROM content_vectors LIMIT 1`).get();
  const hasVectors = !!vecResult;

  return {
    totalDocuments: totalDocs,
    needsEmbedding,
    hasVectorIndex: hasVectors,
    collections,
  };
}

// =============================================================================
// Document Indexing Operations
// =============================================================================

async function insertContent(db: TursoDatabase, hash: string, content: string, createdAt: string): Promise<void> {
  await db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`).run(hash, content, createdAt);
}

async function insertDocument(
  db: TursoDatabase,
  collectionName: string,
  path: string,
  title: string,
  hash: string,
  createdAt: string,
  modifiedAt: string
): Promise<number> {
  const { id } = await db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, 1)
    RETURNING id
  `).get(collectionName, path, title, hash, createdAt, modifiedAt);

  const doc = await db.prepare(`SELECT id FROM documents WHERE collection = ? AND path = ? AND active = 1`).get(collectionName, path) as { id: number } | undefined;

  if (doc) {
    const content = await db.prepare(`SELECT doc FROM content WHERE hash = ?`).get(hash) as { doc: string } | undefined;
    if (content) {
      const filepath = `${collectionName}/${path}`;
      await db.prepare(`INSERT OR REPLACE INTO documents_search (doc_id, filepath, title, body) VALUES (?, ?, ?, ?)`).run(doc.id, filepath, title, content.doc);
    }
  }
  return id;
}

async function findActiveDocument(db: TursoDatabase, collectionName: string, path: string): Promise<{ id: number; hash: string; title: string } | null> {
  const result = await db.prepare(`SELECT id, hash, title FROM documents WHERE collection = ? AND path = ? AND active = 1`).get(collectionName, path) as { id: number; hash: string; title: string } | undefined;
  return result ?? null;
}

async function deactivateDocument(db: TursoDatabase, collectionName: string, path: string): Promise<void> {
  const doc = await db.prepare(`SELECT id FROM documents WHERE collection = ? AND path = ? AND active = 1`).get(collectionName, path) as { id: number } | undefined;
  await db.prepare(`UPDATE documents SET active = 0 WHERE collection = ? AND path = ? AND active = 1`).run(collectionName, path);
  if (doc) {
    await db.prepare(`DELETE FROM documents_search WHERE doc_id = ?`).run(doc.id);
  }
}

async function updateDocumentTitle(db: TursoDatabase, documentId: number, title: string, modifiedAt: string): Promise<void> {
  await db.prepare(`UPDATE documents SET title = ?, modified_at = ? WHERE id = ?`).run(title, modifiedAt, documentId);

  // Also update the FTS search table
  await db.prepare(`UPDATE documents_search SET title = ? WHERE doc_id = ?`).run(title, documentId);
}

async function updateDocument(db: TursoDatabase, documentId: number, title: string, hash: string, modifiedAt: string): Promise<void> {
  await db.prepare(`UPDATE documents SET title = ?, hash = ?, modified_at = ? WHERE id = ?`).run(title, hash, modifiedAt, documentId);

  // Update the FTS search table with new content
  const content = await db.prepare(`SELECT doc FROM content WHERE hash = ?`).get(hash) as { doc: string } | undefined;
  const doc = await db.prepare(`SELECT collection, path FROM documents WHERE id = ?`).get(documentId) as { collection: string; path: string } | undefined;
  if (content && doc) {
    const filepath = `${doc.collection}/${doc.path}`;
    await db.prepare(`INSERT OR REPLACE INTO documents_search (doc_id, filepath, title, body) VALUES (?, ?, ?, ?)`).run(documentId, filepath, title, content.doc);
  }
}

async function getActiveDocumentPaths(db: TursoDatabase, collectionName: string): Promise<string[]> {
  const rows = await db.prepare(`SELECT path FROM documents WHERE collection = ? AND active = 1`).all(collectionName) as { path: string }[];
  return rows.map(r => r.path);
}

// =============================================================================
// Vector/Embedding Operations
// =============================================================================

// Convert embedding array to vector32 string format: '[1.0, 2.0, 3.0]'
function toVector32String(embedding: number[]): string {
  return `[${embedding.join(', ')}]`;
}

async function insertEmbedding(db: TursoDatabase, hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string): Promise<void> {
  const vectorStr = toVector32String(Array.from(embedding));
  await db.prepare(`INSERT OR REPLACE INTO content_vectors (hash, seq, pos, embedding, model, embedded_at) VALUES (?, ?, ?, vector32(?), ?, ?)`).run(hash, seq, pos, vectorStr, model, embeddedAt);
}

async function getHashesForEmbedding(db: TursoDatabase): Promise<{ hash: string; body: string; path: string }[]> {
  const rows = await db.prepare(`
    SELECT d.hash, c.doc as body, MIN(d.path) as path
    FROM documents d
    JOIN content c ON d.hash = c.hash
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
    GROUP BY d.hash
  `).all() as { hash: string; body: string; path: string }[];
  return rows;
}

async function clearAllEmbeddings(db: TursoDatabase): Promise<void> {
  await db.exec(`DELETE FROM content_vectors`);
}

// =============================================================================
// Fuzzy Matching and Docid Lookup
// =============================================================================

async function findDocumentByDocid(db: TursoDatabase, docid: string): Promise<{ filepath: string; hash: string } | null> {
  const shortHash = normalizeDocid(docid);
  if (shortHash.length < 1) return null;

  const result = await db.prepare(`
    SELECT 'qmd://' || collection || '/' || path as filepath, hash
    FROM documents WHERE hash LIKE ? AND active = 1 LIMIT 1
  `).get(`${shortHash}%`) as { filepath: string; hash: string } | undefined;
  return result ?? null;
}

async function findSimilarFiles(db: TursoDatabase, query: string, maxDistance: number = 3, limit: number = 5): Promise<string[]> {
  const allFiles = await db.prepare(`SELECT path FROM documents WHERE active = 1`).all() as { path: string }[];
  const queryLower = query.toLowerCase();
  return allFiles
    .map(f => ({ path: f.path, dist: levenshtein(f.path.toLowerCase(), queryLower) }))
    .filter(f => f.dist <= maxDistance)
    .sort((a, b) => a.dist - b.dist)
    .slice(0, limit)
    .map(f => f.path);
}

// =============================================================================
// Context Functions
// =============================================================================

async function getContextForFile(db: TursoDatabase, filepath: string): Promise<string | null> {
  if (!filepath) return null;

  const collections = collectionsListCollections();
  const config = collectionsLoadConfig();

  let collectionName: string | null = null;
  let relativePath: string | null = null;

  const parsedVirtual = filepath.startsWith('qmd://') ? parseVirtualPath(filepath) : null;
  if (parsedVirtual) {
    collectionName = parsedVirtual.collectionName;
    relativePath = parsedVirtual.path;
  } else {
    for (const coll of collections) {
      if (!coll?.path) continue;
      if (filepath.startsWith(coll.path + '/') || filepath === coll.path) {
        collectionName = coll.name;
        relativePath = filepath.startsWith(coll.path + '/') ? filepath.slice(coll.path.length + 1) : '';
        break;
      }
    }
    if (!collectionName || relativePath === null) return null;
  }

  const coll = getCollection(collectionName);
  if (!coll) return null;

  const doc = await db.prepare(`SELECT path FROM documents WHERE collection = ? AND path = ? AND active = 1 LIMIT 1`).get(collectionName, relativePath);
  if (!doc) return null;

  const contexts: string[] = [];
  if (config.global_context) contexts.push(config.global_context);

  if (coll.context) {
    const normalizedPath = relativePath.startsWith("/") ? relativePath : `/${relativePath}`;
    const matching = Object.entries(coll.context)
      .filter(([prefix]) => {
        const np = prefix.startsWith("/") ? prefix : `/${prefix}`;
        return normalizedPath.startsWith(np);
      })
      .sort((a, b) => a[0].length - b[0].length);
    for (const [, ctx] of matching) contexts.push(ctx);
  }

  return contexts.length > 0 ? contexts.join('\n\n') : null;
}

function getCollectionByName(name: string): { name: string; pwd: string; glob_pattern: string } | null {
  const collection = getCollection(name);
  if (!collection) return null;
  return { name: collection.name, pwd: collection.path, glob_pattern: collection.pattern };
}

async function getContextForPath(db: TursoDatabase, collectionName: string, path: string): Promise<string | null> {
  const config = collectionsLoadConfig();
  const coll = getCollection(collectionName);

  if (!coll) return null;

  const contexts: string[] = [];

  // Add global context if present
  if (config.global_context) {
    contexts.push(config.global_context);
  }

  // Add all matching path contexts (from most general to most specific)
  if (coll.context) {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;

    const matchingContexts: { prefix: string; context: string }[] = [];
    for (const [prefix, context] of Object.entries(coll.context)) {
      const normalizedPrefix = prefix.startsWith("/") ? prefix : `/${prefix}`;
      if (normalizedPath.startsWith(normalizedPrefix)) {
        matchingContexts.push({ prefix: normalizedPrefix, context });
      }
    }

    matchingContexts.sort((a, b) => a.prefix.length - b.prefix.length);

    for (const match of matchingContexts) {
      contexts.push(match.context);
    }
  }

  return contexts.length > 0 ? contexts.join('\n\n') : null;
}

async function getCollectionsWithoutContext(db: TursoDatabase): Promise<{ name: string; pwd: string; doc_count: number }[]> {
  const yamlCollections = collectionsListCollections();
  const result: { name: string; pwd: string; doc_count: number }[] = [];

  for (const coll of yamlCollections) {
    if (!coll.context || Object.keys(coll.context).length === 0) {
      const stats = await db.prepare(`
        SELECT COUNT(d.id) as doc_count
        FROM documents d
        WHERE d.collection = ? AND d.active = 1
      `).get(coll.name) as { doc_count: number } | undefined;

      result.push({
        name: coll.name,
        pwd: coll.path,
        doc_count: stats?.doc_count || 0,
      });
    }
  }

  return result.sort((a, b) => a.name.localeCompare(b.name));
}

async function getTopLevelPathsWithoutContext(db: TursoDatabase, collectionName: string): Promise<string[]> {
  const paths = await db.prepare(`
    SELECT DISTINCT path FROM documents
    WHERE collection = ? AND active = 1
  `).all(collectionName) as { path: string }[];

  const yamlColl = getCollection(collectionName);
  if (!yamlColl) return [];

  const contextPrefixes = new Set<string>();
  if (yamlColl.context) {
    for (const prefix of Object.keys(yamlColl.context)) {
      contextPrefixes.add(prefix);
    }
  }

  const topLevelDirs = new Set<string>();
  for (const { path } of paths) {
    const parts = path.split('/').filter(Boolean);
    if (parts.length > 1) {
      const dir = parts[0];
      if (dir) topLevelDirs.add(dir);
    }
  }

  const missing: string[] = [];
  for (const dir of topLevelDirs) {
    let hasContext = false;
    for (const prefix of contextPrefixes) {
      if (prefix === '' || prefix === dir || dir.startsWith(prefix + '/')) {
        hasContext = true;
        break;
      }
    }
    if (!hasContext) {
      missing.push(dir);
    }
  }

  return missing.sort();
}

// =============================================================================
// Virtual Path Functions
// =============================================================================

function resolveVirtualPath(virtualPath: string): string | null {
  const parsed = parseVirtualPath(virtualPath);
  if (!parsed) return null;

  const coll = getCollectionByName(parsed.collectionName);
  if (!coll) return null;

  return resolve(coll.pwd, parsed.path);
}

async function toVirtualPath(db: TursoDatabase, absolutePath: string): Promise<string | null> {
  const collections = collectionsListCollections();

  for (const coll of collections) {
    if (absolutePath.startsWith(coll.path + '/') || absolutePath === coll.path) {
      const relativePath = absolutePath.startsWith(coll.path + '/')
        ? absolutePath.slice(coll.path.length + 1)
        : '';

      const doc = await db.prepare(`
        SELECT d.path
        FROM documents d
        WHERE d.collection = ? AND d.path = ? AND d.active = 1
        LIMIT 1
      `).get(coll.name, relativePath);

      if (doc) {
        return buildVirtualPath(coll.name, relativePath);
      }
    }
  }

  return null;
}

// =============================================================================
// Document Retrieval
// =============================================================================

async function findDocument(db: TursoDatabase, filename: string, options: { includeBody?: boolean } = {}): Promise<DocumentResult | DocumentNotFound> {
  let filepath = filename;
  const colonMatch = filepath.match(/:(\d+)$/);
  if (colonMatch) filepath = filepath.slice(0, -colonMatch[0].length);

  if (isDocid(filepath)) {
    const docidMatch = await findDocumentByDocid(db, filepath);
    if (docidMatch) filepath = docidMatch.filepath;
    else return { error: "not_found", query: filename, similarFiles: [] };
  }

  if (filepath.startsWith('~/')) filepath = homedir() + filepath.slice(1);

  const bodyCol = options.includeBody ? `, content.doc as body` : ``;
  const selectCols = `'qmd://' || d.collection || '/' || d.path as virtual_path, d.collection || '/' || d.path as display_path, d.title, d.hash, d.collection, d.modified_at, LENGTH(content.doc) as body_length ${bodyCol}`;

  let doc = await db.prepare(`SELECT ${selectCols} FROM documents d JOIN content ON content.hash = d.hash WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1`).get(filepath) as any;

  if (!doc) {
    doc = await db.prepare(`SELECT ${selectCols} FROM documents d JOIN content ON content.hash = d.hash WHERE 'qmd://' || d.collection || '/' || d.path LIKE ? AND d.active = 1 LIMIT 1`).get(`%${filepath}`) as any;
  }

  if (!doc && !filepath.startsWith('qmd://')) {
    for (const coll of collectionsListCollections()) {
      let relativePath: string | null = null;
      if (filepath.startsWith(coll.path + '/')) relativePath = filepath.slice(coll.path.length + 1);
      else if (!filepath.startsWith('/')) relativePath = filepath;

      if (relativePath) {
        doc = await db.prepare(`SELECT ${selectCols} FROM documents d JOIN content ON content.hash = d.hash WHERE d.collection = ? AND d.path = ? AND d.active = 1`).get(coll.name, relativePath) as any;
        if (doc) break;
      }
    }
  }

  if (!doc) {
    const similar = await findSimilarFiles(db, filepath, 5, 5);
    return { error: "not_found", query: filename, similarFiles: similar };
  }

  const virtualPath = doc.virtual_path || `qmd://${doc.collection}/${doc.display_path}`;
  const context = await getContextForFile(db, virtualPath);

  return {
    filepath: virtualPath,
    displayPath: doc.display_path,
    title: doc.title,
    context,
    hash: doc.hash,
    docid: getDocid(doc.hash),
    collectionName: doc.collection,
    modifiedAt: doc.modified_at,
    bodyLength: doc.body_length,
    ...(options.includeBody && doc.body !== undefined && { body: doc.body }),
  };
}

async function getDocumentBody(db: TursoDatabase, doc: { filepath: string }, fromLine?: number, maxLines?: number): Promise<string | null> {
  const filepath = doc.filepath;
  let row: { body: string } | undefined = undefined;

  if (filepath.startsWith('qmd://')) {
    row = await db.prepare(`SELECT content.doc as body FROM documents d JOIN content ON content.hash = d.hash WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1`).get(filepath) as { body: string } | undefined;
  }

  if (!row) {
    for (const coll of collectionsListCollections()) {
      if (filepath.startsWith(coll.path + '/')) {
        const relativePath = filepath.slice(coll.path.length + 1);
        row = await db.prepare(`SELECT content.doc as body FROM documents d JOIN content ON content.hash = d.hash WHERE d.collection = ? AND d.path = ? AND d.active = 1`).get(coll.name, relativePath) as { body: string } | undefined;
        if (row) break;
      }
    }
  }

  if (!row) return null;

  let body = row.body;
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = body.split('\n');
    const start = (fromLine || 1) - 1;
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    body = lines.slice(start, end).join('\n');
  }
  return body;
}

async function matchFilesByGlob(db: TursoDatabase, pattern: string): Promise<{ filepath: string; displayPath: string; bodyLength: number }[]> {
  const allFiles = await db.prepare(`
    SELECT
      'qmd://' || d.collection || '/' || d.path as virtual_path,
      LENGTH(content.doc) as body_length,
      d.path,
      d.collection
    FROM documents d
    JOIN content ON content.hash = d.hash
    WHERE d.active = 1
  `).all() as { virtual_path: string; body_length: number; path: string; collection: string }[];

  const glob = new Glob(pattern);
  return allFiles
    .filter(f => glob.match(f.virtual_path) || glob.match(f.path))
    .map(f => ({
      filepath: f.virtual_path,
      displayPath: f.path,
      bodyLength: f.body_length
    }));
}

type DbDocRow = {
  virtual_path: string;
  display_path: string;
  title: string;
  hash: string;
  collection: string;
  path: string;
  modified_at: string;
  body_length: number;
  body?: string;
};

async function findDocuments(
  db: TursoDatabase,
  pattern: string,
  options: { includeBody?: boolean; maxBytes?: number } = {}
): Promise<{ docs: MultiGetResult[]; errors: string[] }> {
  const isCommaSeparated = pattern.includes(',') && !pattern.includes('*') && !pattern.includes('?');
  const errors: string[] = [];
  const maxBytes = options.maxBytes ?? DEFAULT_MULTI_GET_MAX_BYTES;

  const bodyCol = options.includeBody ? `, content.doc as body` : ``;
  const selectCols = `
    'qmd://' || d.collection || '/' || d.path as virtual_path,
    d.collection || '/' || d.path as display_path,
    d.title,
    d.hash,
    d.collection,
    d.modified_at,
    LENGTH(content.doc) as body_length
    ${bodyCol}
  `;

  let fileRows: DbDocRow[];

  if (isCommaSeparated) {
    const names = pattern.split(',').map(s => s.trim()).filter(Boolean);
    fileRows = [];
    for (const name of names) {
      let doc = await db.prepare(`
        SELECT ${selectCols}
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1
      `).get(name) as DbDocRow | undefined;
      if (!doc) {
        doc = await db.prepare(`
          SELECT ${selectCols}
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE 'qmd://' || d.collection || '/' || d.path LIKE ? AND d.active = 1
          LIMIT 1
        `).get(`%${name}`) as DbDocRow | undefined;
      }
      if (doc) {
        fileRows.push(doc);
      } else {
        const similar = await findSimilarFiles(db, name, 5, 3);
        let msg = `File not found: ${name}`;
        if (similar.length > 0) {
          msg += ` (did you mean: ${similar.join(', ')}?)`;
        }
        errors.push(msg);
      }
    }
  } else {
    const matched = await matchFilesByGlob(db, pattern);
    if (matched.length === 0) {
      errors.push(`No files matched pattern: ${pattern}`);
      return { docs: [], errors };
    }
    const virtualPaths = matched.map(m => m.filepath);
    const placeholders = virtualPaths.map(() => '?').join(',');
    fileRows = await db.prepare(`
      SELECT ${selectCols}
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE 'qmd://' || d.collection || '/' || d.path IN (${placeholders}) AND d.active = 1
    `).all(...virtualPaths) as DbDocRow[];
  }

  const results: MultiGetResult[] = [];

  for (const row of fileRows) {
    const virtualPath = row.virtual_path || `qmd://${row.collection}/${row.display_path}`;
    const context = await getContextForFile(db, virtualPath);

    if (row.body_length > maxBytes) {
      results.push({
        doc: { filepath: virtualPath, displayPath: row.display_path },
        skipped: true,
        skipReason: `File too large (${Math.round(row.body_length / 1024)}KB > ${Math.round(maxBytes / 1024)}KB)`,
      });
      continue;
    }

    results.push({
      doc: {
        filepath: virtualPath,
        displayPath: row.display_path,
        title: row.title || row.display_path.split('/').pop() || row.display_path,
        context,
        hash: row.hash,
        docid: getDocid(row.hash),
        collectionName: row.collection,
        modifiedAt: row.modified_at,
        bodyLength: row.body_length,
        ...(options.includeBody && row.body !== undefined && { body: row.body }),
      },
      skipped: false,
    });
  }

  return { docs: results, errors };
}

// =============================================================================
// Query Expansion & Reranking
// =============================================================================

async function expandQuery(db: TursoDatabase, query: string, model: string = DEFAULT_QUERY_MODEL): Promise<string[]> {
  const cacheKey = getCacheKey("expandQuery", { query, model });
  const cached = await getCachedResult(db, cacheKey);
  if (cached) {
    const lines = cached.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    return [query, ...lines.slice(0, 2)];
  }

  const llm = getDefaultLlamaCpp();
  const results = await llm.expandQuery(query);
  const queryTexts = results.map(r => r.text);

  const expandedOnly = queryTexts.filter(t => t !== query);
  if (expandedOnly.length > 0) {
    await setCachedResult(db, cacheKey, expandedOnly.join('\n'));
  }

  return Array.from(new Set([query, ...queryTexts]));
}

async function rerank(db: TursoDatabase, query: string, documents: { file: string; text: string }[], model: string = DEFAULT_RERANK_MODEL): Promise<{ file: string; score: number }[]> {
  const cachedResults: Map<string, number> = new Map();
  const uncachedDocs: RerankDocument[] = [];

  for (const doc of documents) {
    const cacheKey = getCacheKey("rerank", { query, file: doc.file, model });
    const cached = await getCachedResult(db, cacheKey);
    if (cached !== null) {
      cachedResults.set(doc.file, parseFloat(cached));
    } else {
      uncachedDocs.push({ file: doc.file, text: doc.text });
    }
  }

  if (uncachedDocs.length > 0) {
    const llm = getDefaultLlamaCpp();
    const rerankResult = await llm.rerank(query, uncachedDocs, { model });

    for (const result of rerankResult.results) {
      const cacheKey = getCacheKey("rerank", { query, file: result.file, model });
      await setCachedResult(db, cacheKey, result.score.toString());
      cachedResults.set(result.file, result.score);
    }
  }

  return documents
    .map(doc => ({ file: doc.file, score: cachedResults.get(doc.file) || 0 }))
    .sort((a, b) => b.score - a.score);
}

// =============================================================================
// Collection Management
// =============================================================================

async function listCollections(db: TursoDatabase): Promise<{ name: string; pwd: string; glob_pattern: string; doc_count: number; active_count: number; last_modified: string | null }[]> {
  const collections = collectionsListCollections();

  const result = [];
  for (const coll of collections) {
    const stats = await db.prepare(`
      SELECT
        COUNT(d.id) as doc_count,
        SUM(CASE WHEN d.active = 1 THEN 1 ELSE 0 END) as active_count,
        MAX(d.modified_at) as last_modified
      FROM documents d
      WHERE d.collection = ?
    `).get(coll.name) as { doc_count: number; active_count: number; last_modified: string | null } | undefined;

    result.push({
      name: coll.name,
      pwd: coll.path,
      glob_pattern: coll.pattern,
      doc_count: stats?.doc_count || 0,
      active_count: stats?.active_count || 0,
      last_modified: stats?.last_modified || null,
    });
  }

  return result;
}

async function removeCollection(db: TursoDatabase, collectionName: string): Promise<{ deletedDocs: number; cleanedHashes: number }> {
  // Delete from FTS search table first
  await db.exec(`
    DELETE FROM documents_search WHERE doc_id IN (
      SELECT id FROM documents WHERE collection = '${collectionName}'
    )
  `);

  // Delete documents from database
  const docResult = await db.prepare(`DELETE FROM documents WHERE collection = ?`).run(collectionName);

  // Clean up orphaned content hashes
  const cleanupResult = await db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();

  return {
    deletedDocs: docResult.changes,
    cleanedHashes: cleanupResult.changes
  };
}

async function renameCollection(db: TursoDatabase, oldName: string, newName: string): Promise<void> {
  // Update all documents with the new collection name in database
  await db.prepare(`UPDATE documents SET collection = ? WHERE collection = ?`).run(newName, oldName);

  // Update FTS table - need to update filepath which contains collection name
  await db.exec(`
    UPDATE documents_search
    SET filepath = '${newName}' || substr(filepath, ${oldName.length + 1})
    WHERE filepath LIKE '${oldName}/%'
  `);
}

async function getAllCollections(db: TursoDatabase): Promise<{ name: string }[]> {
  const collections = collectionsListCollections();
  return collections.map(c => ({ name: c.name }));
}

// =============================================================================
// Context Management
// =============================================================================

async function insertContext(db: TursoDatabase, collectionId: number, pathPrefix: string, context: string): Promise<void> {
  // Get collection name from ID
  const coll = await db.prepare(`SELECT collection as name FROM documents WHERE id = ? LIMIT 1`).get(collectionId) as { name: string } | undefined;
  if (!coll) {
    throw new Error(`Collection with id ${collectionId} not found`);
  }

  // Use collections.ts to add context
  collectionsAddContext(coll.name, pathPrefix, context);
}

async function deleteContext(collectionName: string, pathPrefix: string): Promise<number> {
  const success = collectionsRemoveContext(collectionName, pathPrefix);
  return success ? 1 : 0;
}

async function deleteGlobalContexts(): Promise<number> {
  let deletedCount = 0;

  setGlobalContext(undefined);
  deletedCount++;

  const collections = collectionsListCollections();
  for (const coll of collections) {
    const success = collectionsRemoveContext(coll.name, '');
    if (success) {
      deletedCount++;
    }
  }

  return deletedCount;
}

async function listPathContexts(): Promise<{ collection_name: string; path_prefix: string; context: string }[]> {
  const allContexts = collectionsListAllContexts();

  return allContexts.map(ctx => ({
    collection_name: ctx.collection,
    path_prefix: ctx.path,
    context: ctx.context,
  })).sort((a, b) => {
    if (a.collection_name !== b.collection_name) {
      return a.collection_name.localeCompare(b.collection_name);
    }
    if (a.path_prefix.length !== b.path_prefix.length) {
      return b.path_prefix.length - a.path_prefix.length;
    }
    return a.path_prefix.localeCompare(b.path_prefix);
  });
}

// =============================================================================
// Vector Search
// =============================================================================

async function searchVec(db: TursoDatabase, generate: () => Promise<number[] | null>, limit: number = 20, collectionName?: string): Promise<SearchResult[]> {
  const embedding = await generate();
  if (!embedding) {
    return [];
  }
  const vectorStr = toVector32String(embedding);

  let sql = `
    SELECT cv.hash, cv.pos,
      vector_distance_cos(cv.embedding, vector32(?)) as distance,
      'qmd://' || d.collection || '/' || d.path as filepath,
      d.collection || '/' || d.path as display_path,
      d.title, d.collection, content.doc as body
    FROM content_vectors cv
    JOIN documents d ON d.hash = cv.hash AND d.active = 1
    JOIN content ON content.hash = d.hash
  `;

  const params: (string | number)[] = [vectorStr];
  if (collectionName) {
    sql += ` WHERE d.collection = ?`;
    params.push(collectionName);
  }
  sql += ` ORDER BY distance ASC LIMIT ?`;
  params.push(limit * 3); // Fetch extra for deduplication

  const rows = await db.prepare(sql).all(...params) as any[];

  // Deduplicate by filepath, keeping the best distance
  const seen = new Map<string, any>();
  for (const row of rows) {
    const existing = seen.get(row.filepath);
    if (!existing || row.distance < existing.distance) seen.set(row.filepath, row);
  }

  // Sort and limit
  const sortedResults = Array.from(seen.values())
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit);

  const results: SearchResult[] = [];
  for (const row of sortedResults) {
    results.push({
      filepath: row.filepath,
      displayPath: row.display_path,
      title: row.title,
      hash: row.hash,
      docid: getDocid(row.hash),
      collectionName: row.collection,
      modifiedAt: "",
      bodyLength: row.body.length,
      body: row.body,
      context: await getContextForFile(db, row.filepath),
      score: 1 - row.distance,
      source: "vec",
      chunkPos: row.pos,
    });
  }
  return results;
}

// =============================================================================
// Full-Text Search (using Turso native FTS with fts_match/fts_score)
// =============================================================================

function escapeTantivySpecialChars(query: string): string {
  const special: string[] = ['+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/'];
  const escaped: string[] = [];
  for (const c of query) {
    if (special.includes(c)) {
      escaped.push(`\\${c}`);
    } else {
      escaped.push(c);
    }
  }
  return escaped.join('');
}

async function searchFts(db: TursoDatabase, query: string, limit: number = 20, collectionName?: string): Promise<SearchResult[]> {
  const queryEscaped = escapeTantivySpecialChars(query);
  // Use Turso's native FTS with fts_match for filtering and fts_score for ranking
  let sql = `
      SELECT score,
        'qmd://' || d.collection || '/' || d.path as filepath,
        d.collection || '/' || d.path as display_path,
        d.title, d.hash, d.collection, content.doc as body
      FROM (
        SELECT 
          fts_score(ds.filepath, ds.title, ds.body, ?1) as score,
          doc_id
        FROM documents_search ds
        WHERE fts_match(ds.filepath, ds.title, ds.body, ?1)
        ORDER BY score DESC LIMIT ?2
      ) ds
      JOIN documents d ON d.id = ds.doc_id AND d.active = 1
      JOIN content ON content.hash = d.hash
    `;
  const params: (string | number)[] = [queryEscaped, limit];

  if (collectionName) {
    sql += ` AND d.collection = ?3`;
    params.push(collectionName);
  }
  const rows = await db.prepare(sql).all(...params) as any[];

  const results: SearchResult[] = [];
  for (const row of rows) {
    results.push({
      filepath: row.filepath,
      displayPath: row.display_path,
      title: row.title,
      hash: row.hash,
      docid: getDocid(row.hash),
      collectionName: row.collection,
      modifiedAt: "",
      bodyLength: row.body.length,
      body: row.body,
      context: await getContextForFile(db, row.filepath),
      score: row.score,
      source: "fts",
    } as any);
  }
  return results;
}

// =============================================================================
// Store Factory
// =============================================================================

let _productionMode = false;

export function enableProductionMode(): void {
  _productionMode = true;
}

export function getDefaultDbPath(indexName: string = "index"): string {
  // Always allow override via INDEX_PATH (for testing)
  if (Bun.env.INDEX_PATH) {
    return Bun.env.INDEX_PATH;
  }

  // In non-production mode (tests), require explicit path
  if (!_productionMode) {
    throw new Error(
      "Database path not set. Tests must set INDEX_PATH env var or use createStore() with explicit path. " +
      "This prevents tests from accidentally writing to the global index."
    );
  }

  const cacheDir = Bun.env.XDG_CACHE_HOME || resolve(homedir(), ".cache");
  const qmdCacheDir = resolve(cacheDir, "qmd");
  try { Bun.spawnSync(["mkdir", "-p", qmdCacheDir]); } catch { }
  return resolve(qmdCacheDir, `${indexName}.turso`);
}

export async function connectTursoDb(dbPath?: string): Promise<{ db: TursoDatabase, path: string }> {
  const resolvedPath = dbPath || getDefaultDbPath();
  const db = await connect(resolvedPath, { experimental: ["index_method"] });
  return { db, path: resolvedPath };
}

export async function createTursoStore(db: TursoDatabase, dbPath: string = ":memory:"): Promise<Store> {
  await initializeDatabase(db);

  return {
    db: {
      name: 'turso',
      db: db,
      async exec(query, ...params) { return await db.prepare(query).run(...params); },
      async get(query, ...params) { return await db.prepare(query).get(...params); },
      async all(query, ...params) { return await db.prepare(query).all(...params); },
    },
    dbPath,
    close: () => db.close(),
    ensureVecTable: async () => {
      // Turso uses native vector support, no separate table needed
    },

    // Index health
    getHashesNeedingEmbedding: () => getHashesNeedingEmbedding(db),
    getIndexHealth: () => getIndexHealth(db),
    getStatus: () => getStatus(db),

    // Caching
    getCacheKey,
    getCachedResult: (key) => getCachedResult(db, key),
    setCachedResult: (key, value) => setCachedResult(db, key, value),
    clearCache: () => clearCache(db),

    // Cleanup and maintenance
    deleteLLMCache: () => deleteLLMCache(db),
    deleteInactiveDocuments: () => deleteInactiveDocuments(db),
    cleanupOrphanedContent: () => cleanupOrphanedContent(db),
    cleanupOrphanedVectors: () => cleanupOrphanedVectors(db),
    vacuumDatabase: () => vacuumDatabase(db),

    // Context
    getContextForFile: (fp) => getContextForFile(db, fp),
    getContextForPath: (collectionName, path) => getContextForPath(db, collectionName, path),
    getCollectionByName,
    getCollectionsWithoutContext: () => getCollectionsWithoutContext(db),
    getTopLevelPathsWithoutContext: (collectionName) => getTopLevelPathsWithoutContext(db, collectionName),

    // Virtual paths
    parseVirtualPath,
    buildVirtualPath,
    isVirtualPath,
    resolveVirtualPath,
    toVirtualPath: (absolutePath) => toVirtualPath(db, absolutePath),

    // Search
    searchFTS: (q, lim, coll) => searchFts(db, q, lim, coll ? String(coll) : undefined),
    searchVec: (generate, lim, coll) => searchVec(db, generate, lim, coll),

    // Query expansion & reranking
    expandQuery: (query, model) => expandQuery(db, query, model),
    rerank: (query, documents, model) => rerank(db, query, documents, model),

    // Document retrieval
    findDocument: (fn, opts) => findDocument(db, fn, opts),
    getDocumentBody: (doc, from, max) => getDocumentBody(db, doc, from, max),
    findDocuments: (pattern, opts) => findDocuments(db, pattern, opts),

    // Fuzzy matching and docid lookup
    findSimilarFiles: (q, dist, lim) => findSimilarFiles(db, q, dist, lim),
    matchFilesByGlob: (pattern) => matchFilesByGlob(db, pattern),
    findDocumentByDocid: (id) => findDocumentByDocid(db, id),

    // Document indexing operations
    insertContent: (h, c, t) => insertContent(db, h, c, t),
    insertDocument: (coll, path, title, hash, created, modified) => insertDocument(db, coll, path, title, hash, created, modified),
    findActiveDocument: (coll, path) => findActiveDocument(db, coll, path),
    updateDocumentTitle: (documentId, title, modifiedAt) => updateDocumentTitle(db, documentId, title, modifiedAt),
    updateDocument: (documentId, title, hash, modifiedAt) => updateDocument(db, documentId, title, hash, modifiedAt),
    deactivateDocument: (coll, path) => deactivateDocument(db, coll, path),
    getActiveDocumentPaths: (coll) => getActiveDocumentPaths(db, coll),

    // Vector/embedding operations
    insertEmbedding: (h, s, p, e, m, t) => insertEmbedding(db, h, s, p, e, m, t),
    getHashesForEmbedding: () => getHashesForEmbedding(db),
    clearAllEmbeddings: () => clearAllEmbeddings(db),

    // Collection management
    listCollections: () => listCollections(db),
    removeCollection: (name) => removeCollection(db, name),
    renameCollection: (oldName, newName) => renameCollection(db, oldName, newName),
    getAllCollections: () => getAllCollections(db),

    // Context management
    insertContext: (collectionId, pathPrefix, context) => insertContext(db, collectionId, pathPrefix, context),
    deleteContext: (collectionName, pathPrefix) => deleteContext(collectionName, pathPrefix),
    deleteGlobalContexts: () => deleteGlobalContexts(),
    listPathContexts: () => listPathContexts(),
  };
}
