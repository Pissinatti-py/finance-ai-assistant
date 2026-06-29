from app.agent.skills.base import Skill
from app.agent.tools.knowledge import knowledge_tools


class KnowledgeBaseSkill(Skill):
    """Semantic retrieval of free-text knowledge (finance concepts, rules) via RAG."""

    name = "knowledge_base"
    description = (
        "Retrieve relevant free-text knowledge ingested into the RAG store, for "
        "personal-finance concepts and rules beyond the user's transaction data."
    )
    instructions = None

    def tools(self, lang: str = "pt-br") -> list:
        """Return the RAG retrieval tool."""
        return knowledge_tools(lang)
