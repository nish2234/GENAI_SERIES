"""
03 — Pydantic Models: Complex Schemas as Python Classes
========================================================

Instead of writing JSON Schema dicts by hand, define Pydantic models.
Benefits:
  - Clean Python syntax with type hints
  - Auto-generates JSON Schema for the API
  - Validates model output with proper types
  - Supports nested models, optional fields, enums
  - IDE autocomplete on the parsed result

Run:
  python 03_pydantic_models.py
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from enum import Enum

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"


# --- Simple Pydantic Model ---

class PersonInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    city: str = Field(description="City of residence")
    occupation: str | None = Field(default=None, description="Job title if mentioned")


def demo_simple_pydantic():
    """Use a Pydantic model as response schema."""
    print("=" * 60)
    print("SIMPLE PYDANTIC MODEL — PersonInfo")
    print("=" * 60)

    text = "Dr. James Wilson, 45, is a cardiologist based in Chicago."

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Extract person info:\n\n{text}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=PersonInfo,
        ),
    )

    raw = json.loads(response.text)
    person = PersonInfo.model_validate(raw)

    print(f"\nInput: {text}")
    print(f"\nValidated Pydantic object:")
    print(f"  Name:       {person.name}")
    print(f"  Age:        {person.age}")
    print(f"  City:       {person.city}")
    print(f"  Occupation: {person.occupation}")
    print(f"\n  Type check: age is {type(person.age).__name__} ✓")


# --- Nested Pydantic Models ---

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class ContactInfo(BaseModel):
    email: str | None = None
    phone: str | None = None
    address: Address | None = None


class EmployeeProfile(BaseModel):
    name: str
    title: str
    department: str
    contact: ContactInfo
    years_at_company: int


def demo_nested_models():
    """Nested Pydantic models — complex structures made simple."""
    print("\n" + "=" * 60)
    print("NESTED MODELS — EmployeeProfile with ContactInfo + Address")
    print("=" * 60)

    text = """
    Maria Garcia is a Senior Software Engineer in the Platform team.
    She's been with the company for 6 years. You can reach her at
    maria.garcia@company.com or (555) 123-4567. Her office is at
    100 Main Street, Austin, TX 78701.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Extract the employee profile:\n\n{text}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=EmployeeProfile,
        ),
    )

    raw = json.loads(response.text)
    employee = EmployeeProfile.model_validate(raw)

    print(f"\n  Name:       {employee.name}")
    print(f"  Title:      {employee.title}")
    print(f"  Department: {employee.department}")
    print(f"  Years:      {employee.years_at_company}")
    print(f"  Email:      {employee.contact.email}")
    print(f"  Phone:      {employee.contact.phone}")
    if employee.contact.address:
        addr = employee.contact.address
        print(f"  Address:    {addr.street}, {addr.city}, {addr.state} {addr.zip_code}")


# --- Enum + List Models ---

class Priority(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class TaskItem(BaseModel):
    task: str = Field(description="The task description")
    priority: Priority = Field(description="Priority level")
    estimated_hours: float = Field(description="Estimated time in hours")


class TaskList(BaseModel):
    tasks: list[TaskItem]
    total_hours: float


def demo_enum_and_lists():
    """Pydantic with enums and lists."""
    print("\n" + "=" * 60)
    print("ENUMS + LISTS — TaskList with Priority enum")
    print("=" * 60)

    text = """
    Sprint planning notes: We need to fix the login bug urgently (about 2 hours),
    refactor the payment module (medium priority, maybe 8 hours), and update the
    docs for the API changes (low priority, 3 hours). Also need to deploy the
    hotfix ASAP — that's about 1 hour.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Extract all tasks with priorities and time estimates:\n\n{text}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=TaskList,
        ),
    )

    raw = json.loads(response.text)
    task_list = TaskList.model_validate(raw)

    print(f"\n  Extracted {len(task_list.tasks)} tasks:")
    for t in task_list.tasks:
        emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[t.priority.value]
        print(f"    {emoji} [{t.priority.value:6}] {t.task} ({t.estimated_hours}h)")
    print(f"\n  Total estimated hours: {task_list.total_hours}h")


if __name__ == "__main__":
    demo_simple_pydantic()
    demo_nested_models()
    demo_enum_and_lists()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
Pydantic models give you:
  1. Clean Python class syntax (no hand-written JSON Schema dicts)
  2. Nested structures (models inside models)
  3. Enums for constrained choices
  4. Type validation on the parsed output
  5. IDE autocomplete and type checking

Next: Let's combine everything into a full entity extractor!
""")
