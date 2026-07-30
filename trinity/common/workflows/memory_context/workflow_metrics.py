from __future__ import annotations

import json
import math
import re
import os
from typing import Dict, List

# Global tokenizer cache to avoid repeated loading.
_tokenizer_cache = None
_tokenizer_initialized = False


def _get_tokenizer():
    """Get or initialize the default tokenizer."""
    global _tokenizer_cache, _tokenizer_initialized

    if _tokenizer_initialized:
        return _tokenizer_cache

    _tokenizer_initialized = True

    try:
        from transformers import AutoTokenizer

        # Open-source friendly: allow external tokenizer path via env.
        tokenizer_path = os.getenv("TOKENIZER_PATH", "bert-base-uncased")
        _tokenizer_cache = AutoTokenizer.from_pretrained(tokenizer_path)
        return _tokenizer_cache
    except Exception:
        _tokenizer_cache = None
        return None


def tokenize(text: str) -> List[str]:
    """Tokenize text into a token list."""
    if not text:
        return []

    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            cleaned_tokens = []
            for token in tokens:
                cleaned = token.replace("##", "").replace("▁", " ").strip()
                if cleaned:
                    cleaned_tokens.append(cleaned.lower())
            return cleaned_tokens
        except Exception:
            raise Exception("Failed to tokenize text")

    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def calculate_token_f1(predicted: str, expected: str) -> float:
    """Token-level F1 score."""
    if not predicted or not expected:
        return 0.0

    pred_tokens = tokenize(predicted)
    exp_tokens = tokenize(expected)

    if not pred_tokens or not exp_tokens:
        return 0.0

    exp_token_counts = {}
    for token in exp_tokens:
        exp_token_counts[token] = exp_token_counts.get(token, 0) + 1

    pred_token_counts = {}
    for token in pred_tokens:
        pred_token_counts[token] = pred_token_counts.get(token, 0) + 1

    matches = 0
    for token, pred_count in pred_token_counts.items():
        if token in exp_token_counts:
            matches += min(pred_count, exp_token_counts[token])

    precision = matches / len(pred_tokens) if pred_tokens else 0.0
    recall = matches / len(exp_tokens) if exp_tokens else 0.0

    if precision + recall == 0.0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_answer_em(predicted: str, expected: str) -> float:
    """Compute exact match for answers."""
    if not predicted or not expected:
        return 0.0

    pred_normalized = str(predicted).strip().lower()
    exp_normalized = str(expected).strip().lower()

    pred_normalized = re.sub(r'^(answer|the answer is|the answer)[:：]\s*', '', pred_normalized, flags=re.IGNORECASE)
    exp_normalized = re.sub(r'^(answer|the answer is|the answer)[:：]\s*', '', exp_normalized, flags=re.IGNORECASE)

    pred_normalized = re.sub(r'^["\']|["\']$', '', pred_normalized).strip()
    exp_normalized = re.sub(r'^["\']|["\']$', '', exp_normalized).strip()

    return 1.0 if pred_normalized == exp_normalized else 0.0


def calculate_supporting_facts_metrics_vs_expected(
    predicted_facts_sentences: List[str],
    expected_facts_sentences: List[str],
) -> Dict[str, float]:
    """Precision/recall/f1 for supporting facts using token-level similarity."""
    if not expected_facts_sentences:
        if not predicted_facts_sentences:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not predicted_facts_sentences:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_predicted = set()
    matched_expected = set()
    similarity_threshold = 0.7

    for pred_idx, pred_sent in enumerate(predicted_facts_sentences):
        best_match_score = 0.0
        best_match_idx = -1

        for exp_idx, exp_sent in enumerate(expected_facts_sentences):
            similarity = calculate_token_f1(pred_sent, exp_sent)
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_idx = exp_idx

        if best_match_score >= similarity_threshold:
            matched_predicted.add(pred_idx)
            if best_match_idx >= 0:
                matched_expected.add(best_match_idx)

    precision = len(matched_predicted) / len(predicted_facts_sentences) if predicted_facts_sentences else 0.0
    recall = len(matched_expected) / len(expected_facts_sentences) if expected_facts_sentences else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calculate_supporting_facts_f1(
    predicted_facts_sentences: List[str],
    expected_facts_sentences: List[str],
) -> Dict[str, float]:
    """F1 metrics wrapper for supporting facts."""
    return calculate_supporting_facts_metrics_vs_expected(
        predicted_facts_sentences,
        expected_facts_sentences,
    )


async def get_supporting_facts_llm_judge_score(
    question: str,
    answer: str,
    predicted_facts_sentences: List[str],
    expected_facts_sentences: List[str],
    memory_sentences: List[str],
    chat_client,
) -> float:
    """LLM-as-a-judge score for supporting facts."""
    if not expected_facts_sentences:
        return 1.0 if not predicted_facts_sentences else 0.0

    def format_sentences(sentences, label):
        if not sentences:
            return f"{label}: None"
        return f"{label}:\n" + "\n".join([f"- {sent}" for sent in sentences])

    judge_prompt = f"""You are an expert judge evaluating the quality of supporting facts for question answering.

Question: {question}
Answer: {answer}

Ground Truth Supporting Facts (the facts that should be identified):
{format_sentences(expected_facts_sentences, 'Expected Supporting Facts')}

Model Predicted Supporting Facts (the facts identified by the model):
{format_sentences(predicted_facts_sentences, 'Predicted Supporting Facts')}

Retrieved Memory (for reference - sentences available in the memory system):
{format_sentences(memory_sentences, 'Retrieved Memory')}

Please evaluate how well the predicted supporting facts match the ground truth expected facts:
1. Are all expected facts covered by the predictions?
2. Are the predicted facts actually relevant to answering the question?
3. Are there any irrelevant facts in the predictions?

Score on a scale of 0.0 to 1.0:
- 1.0: Perfect match - all expected facts are correctly identified, no irrelevant facts
- 0.8-0.9: Mostly correct with minor omissions or one irrelevant fact
- 0.6-0.7: Partially correct - some relevant facts identified but missing important ones
- 0.4-0.5: Some correct elements but significant errors or omissions
- 0.2-0.3: Mostly incorrect with few correct elements
- 0.0-0.1: Completely incorrect or irrelevant

Respond with only a number between 0.0 and 1.0 (e.g., "0.85")."""

    try:
        judge_response = chat_client.chat(
            messages=[{"role": "user", "content": judge_prompt}],
            model_name="qwen-max",
        )

        judge_text = judge_response.strip()
        score_match = re.search(r'(\d+\.?\d*)', judge_text)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        return calculate_supporting_facts_f1(
            predicted_facts_sentences, expected_facts_sentences
        ).get("f1", 0.0)
    except Exception:
        return calculate_supporting_facts_f1(
            predicted_facts_sentences, expected_facts_sentences
        ).get("f1", 0.0)


def extract_supporting_facts_from_context(
    question: str,
    answer: str,
    context_info: dict,
    chat_client,
) -> dict:
    """LLM extraction of supporting facts when dataset lacks them."""
    if not context_info:
        return {"title": [], "sent_id": []}

    titles = context_info.get("title", [])
    sentences_list = context_info.get("sentences", [])

    if not titles or not sentences_list:
        return {"title": [], "sent_id": []}

    context_text = ""
    for idx, (title, sents) in enumerate(zip(titles, sentences_list)):
        context_text += f"\n[{idx}] Title: {title}\n"
        for sent_idx, sent in enumerate(sents):
            context_text += f"  Sentence {sent_idx}: {sent}\n"

    extract_prompt = f"""You are an expert at identifying supporting facts for question answering.

Question: {question}
Answer: {answer}

Context:
{context_text}

Please identify the supporting facts (title and sentence ID) that are most relevant to answering the question. 
Supporting facts should be sentences from the context that directly support or contain information needed to answer the question.

Output format: A JSON object with "title" (array of strings) and "sent_id" (array of integers).
The title array and sent_id array must have the same length, where title[i] and sent_id[i] form a pair.
Example: {{"title": ["Title Name", "Another Title"], "sent_id": [0, 3]}}

Only output the JSON object, nothing else."""

    try:
        response = chat_client.chat(
            messages=[{"role": "user", "content": extract_prompt}],
            model_name="qwen-max",
        )

        response = response.strip()
        response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^```\s*|\s*```$', '', response).strip()

        try:
            facts = json.loads(response)
            if isinstance(facts, dict) and "title" in facts and "sent_id" in facts:
                title_list = facts.get("title", [])
                sent_id_list = facts.get("sent_id", [])
                if isinstance(title_list, list) and isinstance(sent_id_list, list):
                    if len(title_list) == len(sent_id_list):
                        valid_titles = [str(t).strip() for t in titles]
                        normalized_titles = []
                        normalized_sent_ids = []

                        for title, sent_id in zip(title_list, sent_id_list):
                            title_str = str(title).strip()
                            sent_id_int = int(sent_id)
                            if title_str in valid_titles:
                                normalized_titles.append(title_str)
                                normalized_sent_ids.append(sent_id_int)

                        return {"title": normalized_titles, "sent_id": normalized_sent_ids}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return {"title": [], "sent_id": []}

    except Exception:
        return {"title": [], "sent_id": []}


def extract_sentences_from_supporting_facts(
    supporting_facts: dict,
    context_info: dict,
) -> List[str]:
    """Convert supporting fact indices into sentence texts."""
    if not supporting_facts or not context_info:
        return []

    if not isinstance(supporting_facts, dict):
        return []

    title_list = supporting_facts.get("title", [])
    sent_id_list = supporting_facts.get("sent_id", [])

    if not title_list or not sent_id_list:
        return []

    if len(title_list) != len(sent_id_list):
        return []

    titles = context_info.get("title", [])
    sentences_list = context_info.get("sentences", [])

    if not titles or not sentences_list:
        return []

    title_to_idx = {}
    for idx, title in enumerate(titles):
        title_str = str(title).strip()
        title_to_idx[title_str] = idx

    extracted_sentences = []
    for title, sent_id in zip(title_list, sent_id_list):
        fact_title = str(title).strip()
        sent_id_int = int(sent_id)

        if fact_title in title_to_idx and sent_id_int >= 0:
            idx = title_to_idx[fact_title]
            if idx < len(sentences_list) and sent_id_int < len(sentences_list[idx]):
                sentence = sentences_list[idx][sent_id_int]
                extracted_sentences.append(sentence)

    return extracted_sentences


async def get_answer_llm_judge_score(
    question: str,
    predicted_answer: str,
    expected_answer: str,
    chat_client,
) -> float:
    """LLM-as-a-judge score for answers."""
    if not expected_answer:
        return 0.0

    judge_prompt = f"""You are an expert judge evaluating the correctness of answers to questions.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {predicted_answer}

Please evaluate the generated answer on a scale of 0.0 to 1.0:
- 1.0: Perfect match or equivalent correct answer
- 0.8-0.9: Mostly correct with minor differences
- 0.6-0.7: Partially correct or close approximation
- 0.4-0.5: Some correct elements but significant errors
- 0.2-0.3: Mostly incorrect with few correct elements
- 0.0-0.1: Completely incorrect or irrelevant

Respond with only a number between 0.0 and 1.0 (e.g., "0.85")."""

    try:
        judge_response = chat_client.chat(
            messages=[{"role": "user", "content": judge_prompt}],
            model_name="qwen-max",
        )

        judge_text = judge_response.strip()
        score_match = re.search(r'(\d+\.?\d*)', judge_text)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        return calculate_answer_em(predicted_answer, expected_answer)
    except Exception:
        return calculate_answer_em(predicted_answer, expected_answer)


def calculate_joint_llm_judge(
    answer_llm_score: float,
    supporting_facts_llm_score: float,
) -> float:
    """Joint LLM judge via geometric mean."""
    if answer_llm_score == 0.0 or supporting_facts_llm_score == 0.0:
        return 0.0
    return math.sqrt(answer_llm_score * supporting_facts_llm_score)

