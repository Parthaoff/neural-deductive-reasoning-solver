# src/data_generator.py

import random
import itertools
import json
import os
import string
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_VARIABLES = 10
VARIABLES = list(string.ascii_uppercase[:NUM_VARIABLES])

TRAIN_SIZE = 20000
TEST_SIZE = 3000
MAX_PREMISES = 6
MAX_CHAIN_LENGTH = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data")

# ==========================================================
# LOGIC ENGINE
# ==========================================================

def evaluate_expression(expr, assignment):
    """
    Evaluates propositional expression under assignment.
    Supported:
    - A
    - ~A
    - A -> B
    """
    expr = expr.strip()

    if "->" in expr:
        left, right = expr.split("->")
        return (not evaluate_expression(left.strip(), assignment)) \
               or evaluate_expression(right.strip(), assignment)

    if expr.startswith("~"):
        var = expr[1:].strip()
        return not assignment.get(var, False)

    return assignment.get(expr, False)


def check_entailment(premises, query):
    """
    Returns True, False, or Unknown
    via full truth table enumeration.
    """
    all_assignments = list(itertools.product([False, True], repeat=NUM_VARIABLES))

    entails_query = True
    entails_neg_query = True

    for values in all_assignments:
        assignment = dict(zip(VARIABLES, values))

        if all(evaluate_expression(p, assignment) for p in premises):
            if evaluate_expression(query, assignment):
                entails_neg_query = False
            else:
                entails_query = False

    if entails_query:
        return "True"
    elif entails_neg_query:
        return "False"
    else:
        return "Unknown"


# ==========================================================
# RANDOM LOGIC GENERATION
# ==========================================================

def random_literal():
    var = random.choice(VARIABLES)
    if random.random() < 0.3:
        return "~" + var
    return var


def random_implication():
    left = random.choice(VARIABLES)
    right = random.choice(VARIABLES)
    while right == left:
        right = random.choice(VARIABLES)
    return f"{left} -> {right}"


def generate_reasoning_chain():
    """
    Creates multi-hop implication chain:
    A -> B -> C -> D
    """
    chain_length = random.randint(2, MAX_CHAIN_LENGTH)
    chain_vars = random.sample(VARIABLES, min(chain_length, len(VARIABLES)))

    premises = []

    for i in range(len(chain_vars) - 1):
        premises.append(f"{chain_vars[i]} -> {chain_vars[i+1]}")

    premises.append(chain_vars[0])  # Starting fact

    return premises, chain_vars[-1]  # Return premises and expected conclusion


def generate_sample():
    premises = []
    target_query = None

    # 60% chance create reasoning chain
    if random.random() < 0.6:
        chain_premises, chain_conclusion = generate_reasoning_chain()
        premises.extend(chain_premises)
        
        # Sometimes use chain conclusion as query
        if random.random() < 0.5:
            target_query = chain_conclusion

    # Add random distractors
    num_extra = random.randint(0, MAX_PREMISES - len(premises))
    for _ in range(num_extra):
        if random.random() < 0.6:
            premises.append(random_implication())
        else:
            premises.append(random.choice(VARIABLES))

    # Generate query
    if target_query is None:
        target_query = random.choice(VARIABLES)

    # Ensure we have at least one premise
    if len(premises) == 0:
        premises.append(random_implication())

    label = check_entailment(premises, target_query)

    return {
        "premises": premises,
        "query": target_query,
        "label": label
    }


# ==========================================================
# DATASET CREATION
# ==========================================================

def generate_dataset(size):
    dataset = []
    label_count = {"True": 0, "False": 0, "Unknown": 0}

    for _ in tqdm(range(size), desc="Generating"):
        sample = generate_sample()
        dataset.append(sample)
        label_count[sample["label"]] += 1

    print("Label distribution:", label_count)
    return dataset


def save_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating training set...")
    train_data = generate_dataset(TRAIN_SIZE)

    print("Generating test set...")
    test_data = generate_dataset(TEST_SIZE)

    train_path = os.path.join(OUTPUT_DIR, "train.json")
    test_path = os.path.join(OUTPUT_DIR, "test.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"\nDataset successfully created.")
    print(f"Train size: {len(train_data)} -> {train_path}")
    print(f"Test size: {len(test_data)} -> {test_path}")


if __name__ == "__main__":
    save_dataset()
