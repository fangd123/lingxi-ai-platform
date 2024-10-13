from typing import Optional, List

from pydantic import BaseModel
from typing_extensions import Literal


class Sentences(BaseModel):
    texts: List[str]
    debug: bool = False
