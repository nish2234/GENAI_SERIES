# CrewAI — quick notes

- CrewAI is a Python framework for role-based multi-agent "crews".
- Its `crewai_tools` package ships a big library of ready-made tools:
  web search (Serper), website scraping, file and directory readers/writers,
  PDF/CSV/JSON search, YouTube search, GitHub search, and more.
- Each CrewAI tool is a class with a `name`, a `description` and an `args_schema`.
- Many tools need no API key (file tools, website scraping). Search tools usually need one
  (SerperDevTool needs SERPER_API_KEY).
