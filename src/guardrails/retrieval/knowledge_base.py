import os
from pathlib import Path
from typing import List, Tuple
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer, util
import chromadb


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from PDF."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = page_text.replace("-\n", "").replace("\n", " ")
                text += page_text + " "
    except Exception as e:
        print(f"[PDF Extract] ⚠️ Failed to read {pdf_path.name}: {e}")
    return text.strip()


def simple_chunker(text: str, max_words: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks (exactly like the zip RAG)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks


class KnowledgeBase:
    """
    Identical to the ZIP RAG version:
    - Loads PDFs or TXTs under /userrag
    - Converts PDFs to .txt if needed
    - Splits text into overlapping chunks
    - Creates embeddings with SentenceTransformer(all-MiniLM-L6-v2)
    - Stores vectors in Chroma
    - Uses cosine similarity for retrieval
    """

    def __init__(self, kb_dir: str = "/workspace/safeguarddev/userrag"):
        self.kb_path = Path(kb_dir)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        self.collection_name = "knowledge_base_collection_v1"

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.chunks = []
        self.embeddings = None
        self._load_and_embed_dir()

    def _load_and_embed_dir(self):
        """Load all text data from /userrag and embed."""
        if not self.kb_path.exists():
            print(f"[KnowledgeBase] ❌ Folder not found: {self.kb_path}")
            return

        # Step 1: Convert all PDFs to .txt
        for pdf in self.kb_path.glob("*.pdf"):
            txt_path = self.kb_path / f"{pdf.stem}.txt"
            if not txt_path.exists():
                text = extract_text_from_pdf(pdf)
                if text:
                    txt_path.write_text(text, encoding="utf-8")
                    print(
                        f"[RAGLoader] ✅ Converted {pdf.name} → {txt_path.name} ({len(text)} chars)"
                    )
            else:
                print(f"[RAGLoader] ℹ️ Found existing TXT: {txt_path.name}")

        # Step 2: Collect all .txt
        txt_files = list(self.kb_path.glob("*.txt"))
        if not txt_files:
            print("[KnowledgeBase] ❌ No text files found.")
            return

        print(
            f"[KnowledgeBase] 📄 Found {len(txt_files)} text files: {[f.name for f in txt_files]}"
        )

        # Step 3: Read & chunk
        all_chunks = []
        all_metadatas = []  # <-- [修复] 跟踪元数据

        for f in txt_files:
            text = f.read_text(encoding="utf-8")
            chunks = simple_chunker(text)
            all_chunks.extend(chunks)

            # <-- [修复] 为这个文件的每个 chunk 添加来源
            for _ in chunks:
                all_metadatas.append({"source": f.name})

        if not all_chunks:
            print("[KnowledgeBase] ⚠️ No valid chunks.")
            return

        self.chunks = all_chunks
        print(f"[KnowledgeBase] 🧩 Total chunks: {len(all_chunks)}")

        # Step 4: Embed + Add to Chroma
        self.embeddings = self.encoder.encode(
            all_chunks, convert_to_tensor=True, normalize_embeddings=True
        )
        ids = [f"chunk-{i}" for i in range(len(all_chunks))]

        # <-- [修复] 将元数据添加到 collection 中
        self.collection.add(
            embeddings=self.embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas,  # <-- 添加
            ids=ids,
        )
        print(
            f"[KnowledgeBase] ✅ Embedded {len(all_chunks)} chunks with SentenceTransformer."
        )

    # <-- [修复] 定义正确的返回类型 List[Tuple[str, float, str]]
    async def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Retrieve relevant chunks (like in zip)."""
        if not self.chunks or self.embeddings is None:
            print("[KnowledgeBase] ⚠️ Knowledge base empty.")
            return []

        query_emb = self.encoder.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )

        # <-- [修复] 确保在 query 时也请求 metadatas
        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=top_k,
            include=["documents", "distances", "metadatas"],  # <-- 确保包含 metadatas
        )

        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]  # <-- 获取 metadatas

        sims = [1 - d for d in distances]
        print(f"[RAG] sims={sims}  max={max(sims) if sims else 0:.3f}")

        # <-- [修复] 组合成 (text, score, source) 格式
        combined_results = []
        for i in range(len(docs)):
            source_name = metadatas[i].get("source", "unknown")
            combined_results.append((docs[i], sims[i], source_name))

        return combined_results
