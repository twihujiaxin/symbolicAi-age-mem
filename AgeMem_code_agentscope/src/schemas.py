# -*- coding: utf-8 -*-
"""Pydantic schemas for AgeMem tools."""
from pydantic import BaseModel


class GenerateResponseSchema(BaseModel):
    """Structured schema for generate_response."""
    reply: str = ""
