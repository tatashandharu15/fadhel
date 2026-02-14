import os
from typing import Iterator, Tuple

class FileLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load_files(self) -> Iterator[Tuple[str, str]]:
        """
        Yields (filename, content) for supported file types (.txt, .md)
        """
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith(('.txt', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            yield file, content
                    except Exception as e:
                        print(f"Error reading file {file}: {e}")
