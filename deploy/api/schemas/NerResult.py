from typing_extensions import Literal
from typing import Any, Optional, List,Dict
from pydantic import BaseModel
class OneNer(BaseModel):
    start: int = None
    end: int = None
    text: str = None

class NerResult(BaseModel):
    speakers: List[List[OneNer]] = None