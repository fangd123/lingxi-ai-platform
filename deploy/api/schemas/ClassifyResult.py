from typing_extensions import Literal
from typing import Any, Optional, List,Dict
from pydantic import BaseModel

class ClassifyResult(BaseModel):
    categories: List[int]
    title_elements: Optional[List]
    probabilities: Optional[List[List[float]]]
