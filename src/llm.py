import os

from groq import Groq

from . import config

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPTS = {
    "Student": (
        "You are an academic knowledge assistant for university students. "
        "Answer strictly using the provided context, drawn from official "
        "academic documents (course policies, curricula, guides, etc.). "
        "If the context does not contain the answer, say so plainly instead "
        "of guessing. Cite sources using the [n] markers given in the context."
    ),
    "Staff / Faculty": (
        "You are a policy advisor for university staff and faculty. Answer "
        "strictly using the provided context, drawn from official policy "
        "and academic documents. Be precise about which policy or document "
        "a statement comes from. If the context does not contain the "
        "answer, say so plainly instead of guessing. Cite sources using the "
        "[n] markers given in the context."
    ),
}


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page_info = f", p.{c['page']}" if c.get("page") else ""
        parts.append(f"[{i}] Source: {c['source_file']}{page_info}\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict], role: str) -> str:
    context = build_context(chunks)
    system_prompt = SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["Student"])
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, and cite sources with [n] markers."
    )

    completion = client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content
