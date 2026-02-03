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

import type { connect } from "@tursodatabase/database";
import {
  type DocumentResult,
  type DocumentNotFound,
  type SearchResult,
  type IndexHealthInfo,
} from "./store_types";
import {
  parseVirtualPath,
  getDocid,
  normalizeDocid,
  isDocid,
  levenshtein,
  homedir,
} from "./store_util";
import {
  getCollection,
  listCollections as collectionsListCollections,
  loadConfig as collectionsLoadConfig,
} from "./collections";

// =============================================================================
// Turso Database Type
// =============================================================================

export type TursoDatabase = Awaited<ReturnType<typeof connect>>;

// Statement type from Turso
export type TursoStatement = ReturnType<TursoDatabase["prepare"]>;

// =============================================================================
// Turso Store Interface (Async)
// =============================================================================

export type TursoStore = {
  db: TursoDatabase;
  close: () => void;

  // Index health
  getHashesNeedingEmbedding: () => Promise<number>;
  getIndexHealth: () => Promise<IndexHealthInfo>;

  // Caching
  getCachedResult: (cacheKey: string) => Promise<string | null>;
  setCachedResult: (cacheKey: string, result: string) => Promise<void>;
  clearCache: () => Promise<void>;

  // Context
  getContextForFile: (filepath: string) => Promise<string | null>;
  getCollectionByName: (name: string) => { name: string; pwd: string; glob_pattern: string } | null;

  // Document retrieval
  findDocument: (filename: string, options?: { includeBody?: boolean }) => Promise<DocumentResult | DocumentNotFound>;
  getDocumentBody: (doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number) => Promise<string | null>;

  // Fuzzy matching and docid lookup
  findSimilarFiles: (query: string, maxDistance?: number, limit?: number) => Promise<string[]>;
  findDocumentByDocid: (docid: string) => Promise<{ filepath: string; hash: string } | null>;

  // Document indexing operations
  insertContent: (hash: string, content: string, createdAt: string) => Promise<void>;
  insertDocument: (collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string) => Promise<void>;
  findActiveDocument: (collectionName: string, path: string) => Promise<{ id: number; hash: string; title: string } | null>;
  deactivateDocument: (collectionName: string, path: string) => Promise<void>;
  getActiveDocumentPaths: (collectionName: string) => Promise<string[]>;

  // Vector/embedding operations
  insertEmbedding: (hash: string, seq: number, pos: number, embedding: number[], model: string, embeddedAt: string) => Promise<void>;
  getHashesForEmbedding: () => Promise<{ hash: string; body: string; path: string }[]>;
  clearAllEmbeddings: () => Promise<void>;

  // Search
  searchVec: (queryEmbedding: number[], limit?: number, collectionName?: string) => Promise<SearchResult[]>;
  searchFts: (query: string, limit?: number, collectionName?: string) => Promise<SearchResult[]>;
};

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

  try {
    await db.exec(`
      CREATE INDEX IF NOT EXISTS fts_documents ON documents_search
      USING fts (filepath, title, body)
      WITH (tokenizer = 'default', weights = 'title=2.0,filepath=1.5,body=1.0')
    `);
  } catch {
    // FTS index might already exist or not supported in test environment
  }
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
): Promise<void> {
  await db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, 1)
  `).run(collectionName, path, title, hash, createdAt, modifiedAt);

  const doc = await db.prepare(`SELECT id FROM documents WHERE collection = ? AND path = ? AND active = 1`).get(collectionName, path) as { id: number } | undefined;

  if (doc) {
    const content = await db.prepare(`SELECT doc FROM content WHERE hash = ?`).get(hash) as { doc: string } | undefined;
    if (content) {
      const filepath = `${collectionName}/${path}`;
      await db.prepare(`INSERT OR REPLACE INTO documents_search (doc_id, filepath, title, body) VALUES (?, ?, ?, ?)`).run(doc.id, filepath, title, content.doc);
    }
  }
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

async function insertEmbedding(db: TursoDatabase, hash: string, seq: number, pos: number, embedding: number[], model: string, embeddedAt: string): Promise<void> {
  const vectorStr = toVector32String(embedding);
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

// =============================================================================
// Vector Search
// =============================================================================

async function searchVec(db: TursoDatabase, queryEmbedding: number[], limit: number = 20, collectionName?: string): Promise<SearchResult[]> {
  const vectorStr = toVector32String(queryEmbedding);

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

async function searchFts(db: TursoDatabase, query: string, limit: number = 20, collectionName?: string): Promise<SearchResult[]> {
  // Use Turso's native FTS with fts_match for filtering and fts_score for ranking
  // Note: FTS functions may not be available in all Turso builds (experimental feature)
  try {
    let sql = `
      SELECT
        fts_score(ds.filepath, ds.title, ds.body, ?) as score,
        'qmd://' || d.collection || '/' || d.path as filepath,
        d.collection || '/' || d.path as display_path,
        d.title, d.hash, d.collection, content.doc as body
      FROM documents_search ds
      JOIN documents d ON d.id = ds.doc_id AND d.active = 1
      JOIN content ON content.hash = d.hash
      WHERE fts_match(ds.filepath, ds.title, ds.body, ?)
    `;

    const params: (string | number)[] = [query, query];
    if (collectionName) {
      sql += ` AND d.collection = ?`;
      params.push(collectionName);
    }
    sql += ` ORDER BY score DESC LIMIT ?`;
    params.push(limit);

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
      });
    }
    return results;
  } catch (err: any) {
    // FTS functions not available - fall back to LIKE-based search
    if (err?.message?.includes("no such function")) {
      return searchFtsLikeFallback(db, query, limit, collectionName);
    }
    throw err;
  }
}

// Fallback FTS using LIKE when native fts_match/fts_score are not available
async function searchFtsLikeFallback(db: TursoDatabase, query: string, limit: number = 20, collectionName?: string): Promise<SearchResult[]> {
  const searchTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  if (searchTerms.length === 0) return [];

  let sql = `
    SELECT
      'qmd://' || d.collection || '/' || d.path as filepath,
      d.collection || '/' || d.path as display_path,
      d.title, d.hash, d.collection, content.doc as body
    FROM documents d
    JOIN content ON content.hash = d.hash
    WHERE d.active = 1
  `;

  const params: (string | number)[] = [];

  // Add LIKE conditions for each search term
  for (const term of searchTerms) {
    sql += ` AND (LOWER(d.title) LIKE ? OR LOWER(content.doc) LIKE ?)`;
    params.push(`%${term}%`, `%${term}%`);
  }

  if (collectionName) {
    sql += ` AND d.collection = ?`;
    params.push(collectionName);
  }
  sql += ` LIMIT ?`;
  params.push(limit);

  const rows = await db.prepare(sql).all(...params) as any[];

  const results: SearchResult[] = [];
  for (const row of rows) {
    // Compute simple relevance score based on term frequency
    const bodyLower = row.body.toLowerCase();
    const titleLower = row.title.toLowerCase();
    let score = 0;
    for (const term of searchTerms) {
      const bodyMatches = (bodyLower.match(new RegExp(term, 'g')) || []).length;
      const titleMatches = (titleLower.match(new RegExp(term, 'g')) || []).length;
      score += bodyMatches + titleMatches * 2; // Title matches weighted higher
    }

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
      score,
      source: "fts",
    });
  }

  // Sort by score descending
  results.sort((a, b) => b.score - a.score);
  return results;
}

// =============================================================================
// Store Factory
// =============================================================================

export async function createTursoStore(db: TursoDatabase): Promise<TursoStore> {
  await initializeDatabase(db);

  return {
    db,
    close: () => db.close(),

    getHashesNeedingEmbedding: () => getHashesNeedingEmbedding(db),
    getIndexHealth: () => getIndexHealth(db),

    getCachedResult: (key) => getCachedResult(db, key),
    setCachedResult: (key, value) => setCachedResult(db, key, value),
    clearCache: () => clearCache(db),

    getContextForFile: (fp) => getContextForFile(db, fp),
    getCollectionByName,

    findDocument: (fn, opts) => findDocument(db, fn, opts),
    getDocumentBody: (doc, from, max) => getDocumentBody(db, doc, from, max),

    findSimilarFiles: (q, dist, lim) => findSimilarFiles(db, q, dist, lim),
    findDocumentByDocid: (id) => findDocumentByDocid(db, id),

    insertContent: (h, c, t) => insertContent(db, h, c, t),
    insertDocument: (coll, path, title, hash, created, modified) => insertDocument(db, coll, path, title, hash, created, modified),
    findActiveDocument: (coll, path) => findActiveDocument(db, coll, path),
    deactivateDocument: (coll, path) => deactivateDocument(db, coll, path),
    getActiveDocumentPaths: (coll) => getActiveDocumentPaths(db, coll),

    insertEmbedding: (h, s, p, e, m, t) => insertEmbedding(db, h, s, p, e, m, t),
    getHashesForEmbedding: () => getHashesForEmbedding(db),
    clearAllEmbeddings: () => clearAllEmbeddings(db),

    searchVec: (emb, lim, coll) => searchVec(db, emb, lim, coll),
    searchFts: (q, lim, coll) => searchFts(db, q, lim, coll),
  };
}
