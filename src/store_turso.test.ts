/**
 * store_turso.test.ts - Unit tests for the Turso store implementation
 *
 * Run with: bun test store_turso.test.ts
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "bun:test";
import { connect } from "@tursodatabase/database";
import { mkdtemp, writeFile, unlink, readdir, rmdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createTursoStore, type TursoStore, type TursoDatabase } from "./store_turso";
import type { CollectionConfig } from "./collections";

// =============================================================================
// Test Utilities
// =============================================================================

let testDir: string;
let testConfigDir: string;

async function setupTestConfig(): Promise<void> {
  const configPrefix = join(testDir, `config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  testConfigDir = await mkdtemp(configPrefix);
  process.env.QMD_CONFIG_DIR = testConfigDir;

  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(join(testConfigDir, "index.yml"), YAML.stringify(emptyConfig));
}

async function setupTestConfigWithCollection(name: string, path: string): Promise<void> {
  const config: CollectionConfig = {
    collections: {
      [name]: { path, pattern: "**/*.md" },
    },
  };
  await writeFile(join(testConfigDir, "index.yml"), YAML.stringify(config));
}

async function cleanupTestConfig(): Promise<void> {
  try {
    const files = await readdir(testConfigDir);
    for (const file of files) {
      await unlink(join(testConfigDir, file));
    }
    await rmdir(testConfigDir);
  } catch {}
  delete process.env.QMD_CONFIG_DIR;
}

// =============================================================================
// Tests
// =============================================================================

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-turso-test-"));
});

afterAll(async () => {
  try {
    const files = await readdir(testDir);
    for (const file of files) {
      try {
        await unlink(join(testDir, file));
      } catch {
        try {
          const subFiles = await readdir(join(testDir, file));
          for (const subFile of subFiles) {
            await unlink(join(testDir, file, subFile));
          }
          await rmdir(join(testDir, file));
        } catch {}
      }
    }
    await rmdir(testDir);
  } catch {}
});

describe("TursoStore", () => {
  let db: TursoDatabase;
  let store: TursoStore;

  beforeEach(async () => {
    await setupTestConfig();
    db = await connect(":memory:");
    store = await createTursoStore(db);
  });

  afterEach(async () => {
    store.close();
    await cleanupTestConfig();
  });

  describe("Content Operations", () => {
    test("insertContent stores content with hash", async () => {
      const hash = "abc123def456";
      const content = "# Test Document\n\nThis is test content.";
      const now = new Date().toISOString();

      await store.insertContent(hash, content, now);

      const row = await db.prepare("SELECT * FROM content WHERE hash = ?").get(hash) as any;
      expect(row.hash).toBe(hash);
      expect(row.doc).toBe(content);
    });

    test("insertContent ignores duplicate hashes", async () => {
      const hash = "abc123def456";
      const now = new Date().toISOString();

      await store.insertContent(hash, "First content", now);
      await store.insertContent(hash, "Second content", now);

      const row = await db.prepare("SELECT doc FROM content WHERE hash = ?").get(hash) as any;
      expect(row.doc).toBe("First content");
    });
  });

  describe("Document Operations", () => {
    const testHash = "testhash123456";
    const testContent = "# Test\n\nContent here.";
    const now = new Date().toISOString();

    beforeEach(async () => {
      await store.insertContent(testHash, testContent, now);
    });

    test("insertDocument creates a document record", async () => {
      await store.insertDocument("test-collection", "docs/test.md", "Test Title", testHash, now, now);

      const doc = await db.prepare("SELECT * FROM documents WHERE collection = ? AND path = ?")
        .get("test-collection", "docs/test.md") as any;

      expect(doc.collection).toBe("test-collection");
      expect(doc.path).toBe("docs/test.md");
      expect(doc.title).toBe("Test Title");
      expect(doc.active).toBe(1);
    });

    test("findActiveDocument returns existing document", async () => {
      await store.insertDocument("test-collection", "docs/test.md", "Test Title", testHash, now, now);

      const found = await store.findActiveDocument("test-collection", "docs/test.md");
      expect(found?.hash).toBe(testHash);
      expect(found?.title).toBe("Test Title");
    });

    test("findActiveDocument returns null for non-existent document", async () => {
      const found = await store.findActiveDocument("test-collection", "nonexistent.md");
      expect(found).toBeNull();
    });

    test("deactivateDocument marks document as inactive", async () => {
      await store.insertDocument("test-collection", "docs/test.md", "Test Title", testHash, now, now);
      await store.deactivateDocument("test-collection", "docs/test.md");

      const found = await store.findActiveDocument("test-collection", "docs/test.md");
      expect(found).toBeNull();
    });

    test("getActiveDocumentPaths returns all active paths", async () => {
      await store.insertDocument("coll", "a.md", "A", testHash, now, now);
      await store.insertDocument("coll", "b.md", "B", testHash, now, now);
      await store.insertDocument("coll", "c.md", "C", testHash, now, now);
      await store.deactivateDocument("coll", "b.md");

      const paths = await store.getActiveDocumentPaths("coll");
      expect(paths).toContain("a.md");
      expect(paths).toContain("c.md");
      expect(paths).not.toContain("b.md");
    });
  });

  describe("Caching", () => {
    test("setCachedResult and getCachedResult work together", async () => {
      await store.setCachedResult("key", "value");
      expect(await store.getCachedResult("key")).toBe("value");
    });

    test("getCachedResult returns null for missing key", async () => {
      expect(await store.getCachedResult("nonexistent")).toBeNull();
    });

    test("clearCache removes all entries", async () => {
      await store.setCachedResult("key1", "value1");
      await store.setCachedResult("key2", "value2");
      await store.clearCache();

      expect(await store.getCachedResult("key1")).toBeNull();
      expect(await store.getCachedResult("key2")).toBeNull();
    });
  });

  describe("Index Health", () => {
    test("getIndexHealth returns correct stats", async () => {
      const now = new Date().toISOString();
      await store.insertContent("hash1", "Content 1", now);
      await store.insertContent("hash2", "Content 2", now);
      await store.insertDocument("coll", "a.md", "A", "hash1", now, now);
      await store.insertDocument("coll", "b.md", "B", "hash2", now, now);

      const health = await store.getIndexHealth();
      expect(health.totalDocs).toBe(2);
      expect(health.needsEmbedding).toBe(2);
    });
  });

  describe("Embedding Operations", () => {
    test("insertEmbedding stores embedding", async () => {
      const now = new Date().toISOString();
      await store.insertEmbedding("hash1", 0, 0, [0.1, 0.2, 0.3], "model", now);

      const row = await db.prepare("SELECT * FROM content_vectors WHERE hash = ?").get("hash1") as any;
      expect(row.hash).toBe("hash1");
      expect(row.model).toBe("model");
    });

    test("getHashesForEmbedding returns docs without embeddings", async () => {
      const now = new Date().toISOString();
      await store.insertContent("hash1", "Content 1", now);
      await store.insertContent("hash2", "Content 2", now);
      await store.insertDocument("coll", "a.md", "A", "hash1", now, now);
      await store.insertDocument("coll", "b.md", "B", "hash2", now, now);

      const hashes = await store.getHashesForEmbedding();
      expect(hashes.map(h => h.hash)).toContain("hash1");
      expect(hashes.map(h => h.hash)).toContain("hash2");

      await store.insertEmbedding("hash1", 0, 0, [0.1], "model", now);
      const remaining = await store.getHashesForEmbedding();
      expect(remaining.map(h => h.hash)).not.toContain("hash1");
      expect(remaining.map(h => h.hash)).toContain("hash2");
    });

    test("clearAllEmbeddings removes all embeddings", async () => {
      const now = new Date().toISOString();
      await store.insertEmbedding("hash1", 0, 0, [0.1], "model", now);
      await store.insertEmbedding("hash2", 0, 0, [0.2], "model", now);

      await store.clearAllEmbeddings();

      const count = await db.prepare("SELECT COUNT(*) as c FROM content_vectors").get() as any;
      expect(count.c).toBe(0);
    });
  });

  describe("Document Retrieval", () => {
    const testHash = "docretrieval123";
    const testContent = "# Retrieval Test\n\nDocument content.";
    const now = new Date().toISOString();

    beforeEach(async () => {
      await setupTestConfigWithCollection("mybooks", "/tmp/mybooks");
      await store.insertContent(testHash, testContent, now);
      await store.insertDocument("mybooks", "chapter1.md", "Chapter One", testHash, now, now);
    });

    test("findDocument by virtual path", async () => {
      const result = await store.findDocument("qmd://mybooks/chapter1.md");

      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.title).toBe("Chapter One");
        expect(result.hash).toBe(testHash);
      }
    });

    test("findDocument with includeBody", async () => {
      const result = await store.findDocument("qmd://mybooks/chapter1.md", { includeBody: true });

      if (!("error" in result)) {
        expect(result.body).toBe(testContent);
      }
    });

    test("findDocument returns not_found for missing", async () => {
      const result = await store.findDocument("nonexistent.md");
      expect("error" in result).toBe(true);
    });

    test("getDocumentBody retrieves content", async () => {
      const doc = await store.findDocument("qmd://mybooks/chapter1.md");
      if (!("error" in doc)) {
        expect(await store.getDocumentBody(doc)).toBe(testContent);
      }
    });
  });

  describe("Docid Lookup", () => {
    test("findDocumentByDocid finds by short hash", async () => {
      const now = new Date().toISOString();
      const hash = "abcdef123456789";
      await store.insertContent(hash, "Content", now);
      await store.insertDocument("coll", "file.md", "Title", hash, now, now);

      const result = await store.findDocumentByDocid("abcdef");
      expect(result?.hash).toBe(hash);
    });

    test("findDocumentByDocid handles # prefix", async () => {
      const now = new Date().toISOString();
      const hash = "xyz789abcdef123";
      await store.insertContent(hash, "Content", now);
      await store.insertDocument("coll", "file.md", "Title", hash, now, now);

      const result = await store.findDocumentByDocid("#xyz789");
      expect(result?.hash).toBe(hash);
    });
  });

  describe("Vector Search", () => {
    test("searchVec finds similar documents", async () => {
      const now = new Date().toISOString();
      const hash = "vechash123";
      await store.insertContent(hash, "Vector content", now);
      await store.insertDocument("coll", "vec.md", "Vec Doc", hash, now, now);
      await store.insertEmbedding(hash, 0, 0, [0.1, 0.2, 0.3, 0.4], "model", now);

      const results = await store.searchVec([0.1, 0.2, 0.3, 0.4], 10);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].hash).toBe(hash);
      expect(results[0].source).toBe("vec");
    });

    test("searchVec returns results ordered by similarity", async () => {
      const now = new Date().toISOString();

      await store.insertContent("close", "Close content", now);
      await store.insertContent("far", "Far content", now);
      await store.insertDocument("coll", "close.md", "Close", "close", now, now);
      await store.insertDocument("coll", "far.md", "Far", "far", now, now);

      // Embedding very similar to query
      await store.insertEmbedding("close", 0, 0, [0.9, 0.1, 0.0, 0.0], "model", now);
      // Embedding very different from query
      await store.insertEmbedding("far", 0, 0, [0.0, 0.0, 0.9, 0.1], "model", now);

      const results = await store.searchVec([1.0, 0.0, 0.0, 0.0], 10);

      expect(results[0].hash).toBe("close");
    });
  });

  describe("Full-Text Search", () => {
    test("searchFts finds documents by content", async () => {
      const now = new Date().toISOString();
      await store.insertContent("fts1", "Machine learning is transforming industries", now);
      await store.insertContent("fts2", "Database optimization techniques", now);
      await store.insertDocument("coll", "ml.md", "Machine Learning", "fts1", now, now);
      await store.insertDocument("coll", "db.md", "Database Guide", "fts2", now, now);

      const results = await store.searchFts("machine learning", 10);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].hash).toBe("fts1");
      expect(results[0].source).toBe("fts");
    });

    test("searchFts ranks by relevance", async () => {
      const now = new Date().toISOString();
      await store.insertContent("high", "Database database database optimization", now);
      await store.insertContent("low", "Some database info here", now);
      await store.insertDocument("coll", "high.md", "Database Heavy", "high", now, now);
      await store.insertDocument("coll", "low.md", "Light Mention", "low", now, now);

      const results = await store.searchFts("database", 10);

      // Document with more occurrences should rank higher
      expect(results.length).toBe(2);
      expect(results[0].hash).toBe("high");
    });
  });
});
