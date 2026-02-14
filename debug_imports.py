
import sys
import os

print("Python executable:", sys.executable)
print("System path:", sys.path)

try:
    import torch
    print("Torch version:", torch.__version__)
    print("Torch file:", torch.__file__)
except ImportError as e:
    print("Failed to import torch:", e)

try:
    import transformers
    print("Transformers version:", transformers.__version__)
    print("Transformers file:", transformers.__file__)
except ImportError as e:
    print("Failed to import transformers:", e)

try:
    import sentence_transformers
    print("Sentence Transformers version:", sentence_transformers.__version__)
    print("Sentence Transformers file:", sentence_transformers.__file__)
except ImportError as e:
    print("Failed to import sentence_transformers:", e)
