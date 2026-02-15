import argparse
from src.search import RAGSearch
import subprocess


## surya - CLI flag support
def run_pdfs(query):
    rag_search = RAGSearch()
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

def run_images(query):
    from src.search import ImageRAGSearch
    rag_search = ImageRAGSearch()
    result = rag_search.search_and_summarize(query, top_k=3)
    if result["images"]:
        print(f"\nResults for: '{query}'")
        for img in result["images"]:
            print(f"\n  Image: {img['path']}")
            print(f"  Score: {img['score']}")
            print(f"  Description: {img['description']}")
            ## surya - open image on Mac
            subprocess.run(["chafa", "--size=40x20", img["path"]])
            ## surya end
    else:
        print("No relevant images found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Search")
    parser.add_argument("--pdfs", action="store_true", help="Search through PDFs")
    parser.add_argument("--images", action="store_true", help="Search through images")
    parser.add_argument("--query", type=str, default="What is attention mechanism?", help="Search query")
    args = parser.parse_args()

    if args.images:
        run_images(args.query)
    else:
        run_pdfs(args.query)
## surya end