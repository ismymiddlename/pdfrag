import os

# Tests stub generation, so the 2.4 GB answer model must not be preloaded.
os.environ["PRELOAD_ANSWER_MODEL"] = "0"
