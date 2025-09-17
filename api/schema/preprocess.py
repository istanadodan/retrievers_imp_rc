from pydantic import BaseModel, Field


class DocumentSplitSchema(BaseModel):
    chunk_size: int = Field(500, ge=1)
    chunk_overlap: int = Field(0, ge=0)

    class Config:
        from_attributes = True