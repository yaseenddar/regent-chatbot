import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import hashlib
import pandas as pd
from config import Config
from utils.nltk import NLTK
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client.models import ScorePoint
from langchain_core.documents import Document
from vector_db.chunker import DocumentChunker
