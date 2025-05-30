from pydantic import BaseModel, Field, field_validator

class TokenRequest(BaseModel):
    user_id: str = Field(
        ...,  # ... means required
        # min_length=3,
        # max_length=50,
        description="Unique identifier for the user (Supabase User ID - UUID format)"
    )

    # @field_validator('user_id')
    # @classmethod
    # def validate_user_id(cls, v: str) -> str:
    #     if not v.strip():
    #         raise ValueError('user_id cannot be empty or whitespace')
    #     if not v.isalnum():
    #         raise ValueError('user_id must be alphanumeric')
    #     return v


class SupabaseTokenExchangeRequest(BaseModel):
    supabase_token: str = Field(..., description="The Supabase access token obtained from frontend authentication.")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = Field(default="bearer", pattern="^bearer$")