import os
import argparse
from dotenv import load_dotenv
from pipeline import RAGPipeline

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index")
    p_index.add_argument("files", nargs="+")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("query")

    args = parser.parse_args()
    rag = RAGPipeline(api_key)

    if args.cmd == "index":
        rag.index_files(args.files)
        print("Indexed successfully!")
    elif args.cmd == "ask":
        rag.load_index()
        ans, sources = rag.ask(args.query)
        print("\n=== Answer ===")
        print(ans)
        print("\n=== Sources ===")
        for i, (c, s) in enumerate(sources):
            print(f"[{i+1}] {c.source} {c.text[:200]} ({s:.3f})")

if __name__ == "__main__":
    main()
