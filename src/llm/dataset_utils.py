import os
import random
import logging
from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)

def load_dataset_by_name(dataset_name, dataset_dir=None):
    """
    Load dataset by name. Tries to load from local disk first, then Hugging Face Hub.
    """
    if dataset_dir:
        dataset_path = os.path.join(dataset_dir, dataset_name)
        if os.path.exists(dataset_path):
            logger.info(f"Loading {dataset_name} from local disk: {dataset_path}")
            try:
                dataset = load_from_disk(dataset_path)
                if "test" in dataset:
                    return dataset["test"]
                elif "validation" in dataset:
                    return dataset["validation"]
                else:
                    return dataset[list(dataset.keys())[0]]
            except Exception as e:
                logger.warning(f"Failed to load from disk: {e}. Trying Hugging Face Hub.")

    logger.info(f"Loading {dataset_name} from Hugging Face Hub")
    try:
        if dataset_name == "gsm8k":
            dataset = load_dataset("openai/gsm8k", "main")
        elif dataset_name == "mmlu":
            dataset = load_dataset("cais/mmlu", "all")
        elif dataset_name == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag")
        elif dataset_name == "humaneval":
            dataset = load_dataset("openai/humaneval")
        elif dataset_name == "mgsm":
            dataset = load_dataset("juletxara/mgsm")
        elif dataset_name == "arc_challenge":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
            
        if "test" in dataset:
            return dataset["test"]
        elif "validation" in dataset:
            return dataset["validation"]
        else:
            return dataset[list(dataset.keys())[0]]
            
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return None

def format_question(sample, dataset_name):
    """
    Format the question prompt based on the dataset type.
    """
    question = ""
    if dataset_name in ["gsm8k", "mgsm"]:
        question = sample.get("question", "")
    elif dataset_name == "mmlu":
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        if choices:
            question += "\nOptions:\n"
            for i, choice in enumerate(choices):
                question += f"{chr(65+i)}. {choice}\n"
    elif dataset_name == "hellaswag":
        question = sample.get("ctx", "") + " " + sample.get("activity_label", "")
    elif dataset_name == "humaneval":
        question = sample.get("prompt", "")
    elif dataset_name == "arc_challenge":
        question = sample.get("question", "")
    
    return question

def select_questions(dataset, dataset_name, num_questions=30, seed=42):
    """
    Select a fixed subset of questions from the dataset.
    """
    if dataset is None or len(dataset) == 0:
        return []
    
    random.seed(seed)
    
    total_samples = len(dataset)
    if total_samples <= num_questions:
        indices = list(range(total_samples))
    else:
        indices = random.sample(range(total_samples), num_questions)
    
    questions = []
    for idx in indices:
        sample = dataset[idx]
        q_text = format_question(sample, dataset_name)
        if q_text:
            questions.append(q_text)
            
    return questions
