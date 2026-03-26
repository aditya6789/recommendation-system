"""User request/response schemas."""

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    name: str
    email: EmailStr


class UserRead(BaseModel):
    id: int
    name: str
    email: EmailStr

    model_config = {"from_attributes": True}
