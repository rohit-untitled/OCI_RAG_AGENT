# import os
# import json
# import numpy as np
# import faiss
# import logging
# from sentence_transformers import SentenceTransformer

# logger = logging.getLogger(__name__)

# class EmbeddingService:
#     def __init__(self):
#         self.model_name = "all-MiniLM-L6-v2"
#         logger.info(f"🔍 Loading embedding model: {self.model_name}")
#         self.model = SentenceTransformer(self.model_name)

#         # Paths for saving index and metadata
#         self.index_path = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index.index")
#         self.metadata_path = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")

#         self.index = None
#         self.metadata = []

#     # EMBEDDING CREATION
#     def create_embeddings_and_index(self, chunks, persist=False):
#         """Generate embeddings, create FAISS index, and optionally persist."""
#         texts = [c["text"] for c in chunks]
#         embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

#         dimension = embeddings.shape[1]
#         logger.info(f"Creating FAISS index of dimension {dimension}")

#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(embeddings)
#         self.metadata = chunks

#         if persist:
#             self.save_index()
#             logger.info(f"FAISS index saved to {self.index_path}")

#     # SEARCH FUNCTION
#     def search(self, query, top_k=3):
#         """Search top_k similar chunks using FAISS similarity."""
#         if self.index is None:
#             raise RuntimeError("FAISS index is not loaded or created yet.")

#         query_emb = self.model.encode([query], convert_to_numpy=True)
#         distances, indices = self.index.search(query_emb, top_k)

#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx < len(self.metadata):
#                 results.append({
#                     "text": self.metadata[idx]["text"],
#                     "source": self.metadata[idx].get("file_path", "unknown"),
#                     "distance": float(dist)
#                 })
#         return results

#     # SAVE INDEX
#     def save_index(self):
#         """Persist FAISS index and metadata to disk."""
#         os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
#         faiss.write_index(self.index, self.index_path)

#         with open(self.metadata_path, "w", encoding="utf-8") as f:
#             json.dump(self.metadata, f, ensure_ascii=False, indent=2)

#     # LOAD INDEX
#     def load_index(self):
#         """Load FAISS index and metadata if available."""
#         if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
#             logger.info("📂 Loading FAISS index and metadata...")
#             self.index = faiss.read_index(self.index_path)
#             with open(self.metadata_path, "r", encoding="utf-8") as f:
#                 self.metadata = json.load(f)
#             logger.info(f"✅ FAISS index loaded with {len(self.metadata)} entries.")
#         else:
#             logger.warning("⚠️ No saved FAISS index found. Starting fresh.")


# import os
# import json
# import numpy as np
# import faiss
# import logging
# import oci

# from oci.generative_ai_inference import GenerativeAiInferenceClient
# from oci.generative_ai_inference.models import (
#     EmbedTextDetails,
#     OnDemandServingMode
# )

# logger = logging.getLogger(__name__)


# class EmbeddingService:
#     def __init__(self):

#         # ------------------------------
#         # Load OCI Config
#         # ------------------------------
#         logger.info("Loading OCI config profile 'GC3TEST02'...")
#         self.config = oci.config.from_file("~/.oci/config", profile_name="GC3TEST02")

#         # ------------------------------
#         # Endpoint (Hyderabad region)
#         # ------------------------------
#         self.endpoint = "https://inference.generativeai.ap-hyderabad-1.oci.oraclecloud.com"

#         logger.info("Initializing OCI Generative AI Client")
#         self.client = GenerativeAiInferenceClient(
#             config=self.config,
#             service_endpoint=self.endpoint,
#             retry_strategy=oci.retry.NoneRetryStrategy(),
#             timeout=(10, 240)
#         )

#         # ------------------------------
#         # Model (from Oracle sample)
#         # ------------------------------
#         self.serving_mode = OnDemandServingMode(
#             model_id="cohere.embed-multilingual-image-v3.0"
#         )

#         # ------------------------------
#         # Compartment OCID (yours)
#         # ------------------------------
#         self.compartment_id = "ocid1.compartment.oc1..aaaaaaaa2pf2tel6ftytyrdkwaareqpcjfyfit6s62v4qdukfjiflqhlmura"

#         # ------------------------------
#         # File paths for FAISS
#         # ------------------------------
#         base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
#         os.makedirs(base_dir, exist_ok=True)

#         self.index_path = os.path.join(base_dir, "faiss_index.index")
#         self.metadata_path = os.path.join(base_dir, "metadata.json")

#         self.index = None
#         self.metadata = []

#     # =====================================================================
#     #  OCI Embedding Call
#     # =====================================================================
#     def embed_texts(self, texts):
#         """
#         texts: list[str]
#         Returns numpy array of shape (N, D)
#         """

#         if isinstance(texts, str):
#             texts = [texts]

#         logger.info(f"📡 Requesting OCI embeddings for {len(texts)} text(s)...")

#         request = EmbedTextDetails(
#             serving_mode=self.serving_mode,
#             inputs=texts,            # MUST be a list
#             truncate="NONE",
#             compartment_id=self.compartment_id
#         )

#         response = self.client.embed_text(request)

#         # OCI returns list[list[floats]] for Cohere embeddings
#         vectors = response.data.embeddings

#         return np.array(vectors, dtype="float32")


#     # =====================================================================
#     #  Create & Save FAISS Index
#     # =====================================================================
#     def create_embeddings_and_index(self, chunks, persist=False):
#         texts = [c["text"] for c in chunks]
#         embeddings = self.embed_texts(texts)

#         dim = embeddings.shape[1]
#         logger.info(f"Creating FAISS index with dimension {dim}")

#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(embeddings)

#         self.metadata = chunks

#         if persist:
#             self.save_index()

#     # =====================================================================
#     #  Search FAISS Index
#     # =====================================================================
#     def search(self, query, top_k=3):
#         if self.index is None:
#             raise RuntimeError("FAISS index not loaded.")

#         query_vec = self.embed_texts([query])
#         distances, indices = self.index.search(query_vec, top_k)

#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx < len(self.metadata):
#                 results.append({
#                     "text": self.metadata[idx]["text"],
#                     "source": self.metadata[idx].get("file_path", "unknown"),
#                     "distance": float(dist),
#                 })

#         return results

#     # =====================================================================
#     #  Save / Load Index
#     # =====================================================================
#     def save_index(self):
#         faiss.write_index(self.index, self.index_path)
#         with open(self.metadata_path, "w", encoding="utf-8") as f:
#             json.dump(self.metadata, f, ensure_ascii=False, indent=2)

#         logger.info("FAISS index saved successfully.")

#     def load_index(self):
#         if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
#             self.index = faiss.read_index(self.index_path)
#             with open(self.metadata_path, "r") as f:
#                 self.metadata = json.load(f)
#             logger.info("FAISS index loaded successfully.")
#         else:
#             logger.warning("No FAISS index found.")




import oci
import logging
import numpy as np
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails, OnDemandServingMode
)

logger = logging.getLogger(__name__)


class OCIEmbeddingService:
    def __init__(self):
        logger.info("Loading OCI config profile 'GC3TEST02'...")

        self.config = oci.config.from_file("~/.oci/config", profile_name="GC3TEST02")

        self.endpoint = "https://inference.generativeai.ap-hyderabad-1.oci.oraclecloud.com"

        logger.info("Initializing OCI Generative AI Client")
        self.client = GenerativeAiInferenceClient(
            config=self.config,
            service_endpoint=self.endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )

        self.serving_mode = OnDemandServingMode(
            model_id="cohere.embed-multilingual-image-v3.0"
        )

        self.compartment_id = "ocid1.compartment.oc1..aaaaaaaa2pf2tel6ftytyrdkwaareqpcjfyfit6s62v4qdukfjiflqhlmura"

    # -----------------------------------------
    # Split text into <450-word safe subchunks
    # -----------------------------------------
    def _split_long_text(self, text, max_words=200):
        """
        Split text into safe subchunks.
        Reduce max_words if chunks are still too large for OCI model.
        """
        import re

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)

        return chunks

    # -----------------------------------------
    # SAFE embed function
    # - auto splits large text
    # - embeds multiple segments
    # - returns mean pooled vector
    # -----------------------------------------
    def embed_text(self, text: str):
        if not text.strip():
            return []

        # Split long texts to avoid OCI errors
        sub_chunks = self._split_long_text(text)
        all_embeddings = []

        for idx, part in enumerate(sub_chunks, start=1):
            try:
                request = EmbedTextDetails(
                    inputs=[part],
                    serving_mode=self.serving_mode,
                    compartment_id=self.compartment_id,
                )
                response = self.client.embed_text(request)
                emb = response.data.embeddings[0]
                all_embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed embedding sub-chunk {idx}: {e}")
                continue

        if not all_embeddings:
            return None  # return None instead of empty list

        # Mean pooling to combine them into 1 embedding
        final_vec = np.mean(np.array(all_embeddings), axis=0).tolist()
        return final_vec
