from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()



def get_llm():
    try:
        return ChatGroq(
            model="openai/gpt-oss-120b",
            timeout=30,
            max_retries=2,
        )
    except Exception as e:
        raise RuntimeError(f"LLM setup error: {e}") from e

    

def generator(query, final_docs, llm):
    if not query.strip():
        raise ValueError('query can not be empty')

    if not final_docs:
        return 'Not found relevant docs'

    context = "\n\n".join(result.doc.page_content.strip() for result in final_docs if result.doc.page_content.strip())

    if not context:
        return 'No valid context found in the provided documents.'

    prompt = f"""
You are a reliable AI news assistant.

Your task is to answer the USER QUESTION using ONLY the information provided in the CONTEXT.

IMPORTANT:
- The CONTEXT is the only source of truth.
- You must carefully read ALL provided context before answering.
- The answer may require combining facts from multiple passages.
- If the context contains enough information to answer the question, you MUST answer it.
- Do NOT say that the information is insufficient if the answer can be reasonably derived from the provided context.
- Do NOT use outside knowledge, assumptions, guesses, or prior knowledge.
- Do NOT invent or infer unsupported facts.

ANSWERING RULES:
1. Understand exactly what the USER QUESTION is asking.
2. Find all relevant facts from the CONTEXT.
3. Combine relevant facts from multiple passages when necessary.
4. Answer the question directly and clearly.
5. Preserve numbers, dates, percentages, names, and other factual details exactly as stated.
6. If the question asks for a reason or cause, explain the cause using only facts supported by the CONTEXT.
7. If the question asks for a comparison, clearly mention the relevant values and periods.
8. If the question asks "how much", "when", "who", "where", or similar factual details, provide the exact information available in the CONTEXT.
9. Do not add unrelated information.
10. Do not repeat the same information unnecessarily.
11. Answer entirely in natural Bangla.
12. Keep the answer concise but complete.
13. Use short paragraphs or bullet points when appropriate.
14. If the CONTEXT contains conflicting information, mention the conflict instead of guessing.

INSUFFICIENT INFORMATION:
Only when the CONTEXT genuinely does not contain enough information to answer the USER QUESTION, respond exactly:

"প্রদত্ত তথ্যের ভিত্তিতে এই প্রশ্নের সম্পূর্ণ উত্তর দেওয়া সম্ভব নয়।"

Before using the above response, carefully check ALL passages in the CONTEXT and make sure the answer cannot be derived by combining the available information.

OUTPUT:
- Give only the answer to the USER QUESTION.
- Do not mention the CONTEXT.
- Do not mention RAG, retrieval, documents, prompts, or the AI model.
- Do not describe your reasoning process.

CONTEXT:
{context}

USER QUESTION:
{query}
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error during response generation: {e}") from e
