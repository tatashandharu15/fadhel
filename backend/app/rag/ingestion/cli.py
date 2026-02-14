import argparse
import sys
import os

# Ensure backend module is in python path
sys.path.append(os.getcwd())

from backend.app.rag.ingestion.pipeline import IngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="Automotive RAG Data Ingestion CLI")
    parser.add_argument("--path", type=str, required=True, help="Path to data directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist.")
        sys.exit(1)
        
    pipeline = IngestionPipeline(data_path=args.path)
    pipeline.run()

if __name__ == "__main__":
    main()
