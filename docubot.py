"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "it", "its", "this", "that",
    "and", "or", "but", "not", "no", "if", "so", "as", "can", "will",
    "my", "your", "i", "we", "they", "he", "she", "how", "what", "when",
    "where", "which", "who", "why",
}

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Chunk documents into paragraphs for finer-grained retrieval
        self.chunks = self.chunk_documents(self.documents)

        # Build a retrieval index over chunks (implemented in Phase 1)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------

    def chunk_documents(self, documents):
        """
        Split each document into paragraph-level chunks.
        Returns a list of (filename, paragraph_text) tuples.
        Paragraphs are separated by one or more blank lines.
        """
        chunks = []
        for filename, text in documents:
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if para:
                    chunks.append((filename, para))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        TODO (Phase 1):
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for filename, text in documents:
            words = text.lower().split()
            for word in words:
                token = word.strip(".,!?;:()[]{}\"'`#*-_/\\<>")
                if not token:
                    continue
                if token not in index:
                    index[token] = []
                if filename not in index[token]:
                    index[token].append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a relevance score for how well the text matches the query.

        - Tokenizes both query and text on word boundaries
        - Ignores stop words so common words don't inflate scores
        - Counts how many meaningful query words appear in the text
        """
        text_words = set(re.findall(r"[a-z0-9]+", text.lower()))
        query_words = re.findall(r"[a-z0-9]+", query.lower())
        meaningful = [w for w in query_words if w not in STOP_WORDS]
        score = sum(1 for word in meaningful if word in text_words)
        return score

    def retrieve(self, query, top_k=3):
        """
        TODO (Phase 1):
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        query_words = query.lower().split()
        candidate_files = set()
        for word in query_words:
            if word in self.index:
                candidate_files.update(self.index[word])

        results = []
        for filename, text in self.chunks:
            if filename in candidate_files:
                score = self.score_document(query, text)
                if score > 0:
                    results.append((score, filename, text))

        results.sort(key=lambda x: x[0], reverse=True)
        return [(filename, text) for _, filename, text in results[:top_k]]

    # -----------------------------------------------------------
    # Evidence Guardrail
    # -----------------------------------------------------------

    def _has_sufficient_evidence(self, query, snippets):
        """
        Decide whether the best retrieved snippet is strong enough to
        justify answering.  Returns False when the evidence is too weak.

        Thresholds are based on meaningful (non-stop) query words:
        - 1 meaningful word:  score >= 1
        - 2+ meaningful words: score >= at least a third (minimum 2)
        - 0 meaningful words (all stop words): always refuse
        """
        if not snippets:
            return False

        meaningful = [w for w in re.findall(r"[a-z0-9]+", query.lower())
                      if w not in STOP_WORDS]
        if not meaningful:
            return False

        best_score = self.score_document(query, snippets[0][1])

        if len(meaningful) == 1:
            return best_score >= 1

        min_score = max(2, len(meaningful) // 3)
        return best_score >= min_score

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not self._has_sufficient_evidence(query, snippets):
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not self._has_sufficient_evidence(query, snippets):
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
