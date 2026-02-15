import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from pathlib import Path

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-4o-mini"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = ChatOpenAI(model=llm_model)
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Answer the following question using only the context provided. Be direct and concise.\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"""
        response = self.llm.invoke([prompt])
        return response.content


## surya - CLIP-based Image Search with LLM description
from src.clip_store import CLIPImageStore

class ImageRAGSearch:
    def __init__(self, persist_dir: str = "faiss_store_images"):
        self.clip_store = CLIPImageStore(persist_dir)
        clip_path = os.path.join(persist_dir, "clip.index")
        meta_path = os.path.join(persist_dir, "clip_meta.pkl")
        if os.path.exists(clip_path) and os.path.exists(meta_path):
            self.clip_store.load()
        else:
            from src.data_loader import load_images
            image_paths = load_images("data")
            if not image_paths:
                print("[WARN] No images found in data/ folder.")
            else:
                self.clip_store.build_from_images(image_paths)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        print("[INFO] Image RAG initialized (CLIP + LLM)")

    def describe_image(self, image_path: str, query: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            ]
        )
        response = self.llm.invoke([message])
        return response.content

    def search_and_summarize(self, query: str, top_k: int = 3) -> dict:
        # extract_prompt = f"Extract the main subject from this image search query. Reply with ONLY the subject, nothing else.\n\nQuery: {query}"
        # clean_query = self.llm.invoke([extract_prompt]).content.strip()
        # print(f"[DEBUG] Clean query: '{clean_query}'")
        
        results = self.clip_store.query_text(query, top_k=top_k)
        for r in results:
            print(f"[DEBUG] score={r['score']:.4f} path={r['metadata']['image_path']}")

        if not results:
            return {"images": []}

        seen = set()
        images = []

        # Check filename match first
        for r in results:
            path = r["metadata"]["image_path"]
            filename = Path(path).stem.lower()
            filename_words = set(filename.replace("_", " ").replace("-", " ").split())
            query_words = set(query.lower().split())
            # Remove common short words to avoid false matches
            stop_words = {"a", "an", "the", "in", "of", "on", "to", "me", "my", "is", "it", "give", "show", "get"}
            query_words = query_words - stop_words
            name_match = bool(query_words & filename_words)
            if name_match and path not in seen:
                seen.add(path)
                description = self.describe_image(path, query)
                images.append({"path": path, "score": round(r["score"], 4), "description": description})

        # If no filename match, trust CLIP ranking — return top result
        if not images:
            best = results[0]
            path = best["metadata"]["image_path"]
            description = self.describe_image(path, query)
            images.append({"path": path, "score": round(best["score"], 4), "description": description})

        return {"images": images}
## surya end


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)