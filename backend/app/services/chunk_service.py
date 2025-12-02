import os
import json

def chunk_anonymized_documents(base_dir: str, chunk_size: int = 400):
    """
    Loads anonymized text files, chunks them safely for OCI embedding
    (max ~508 token limit, so 400-word chunks are safe).
    """
    anonymized_dir = os.path.join(base_dir, "anonymized")
    chunk_dir = os.path.join(base_dir, "chunks")

    os.makedirs(chunk_dir, exist_ok=True)

    all_chunks = []

    for filename in os.listdir(anonymized_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(anonymized_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        # Split into approx word tokens
        words = text.split()
        current_chunk = []

        for word in words:
            current_chunk.append(word)

            # When chunk_limit hits, save it
            if len(current_chunk) >= chunk_size:
                all_chunks.append({
                    "source_file": filename,
                    "chunk": " ".join(current_chunk)
                })
                current_chunk = []

        # Add final leftover chunk
        if current_chunk:
            all_chunks.append({
                "source_file": filename,
                "chunk": " ".join(current_chunk)
            })

    # Save output
    output_file = os.path.join(chunk_dir, "chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    return {
        "message": "Chunking complete",
        "total_chunks": len(all_chunks),
        "output_file": output_file,
    }
