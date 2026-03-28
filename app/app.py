import streamlit as st
import torch
import torch.nn.functional as F
import os
import sys

# =========================
# PATH SETUP (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

# =========================
# IMPORTS
# =========================
from model import LogicTransformer
from dataset import build_vocab, logic_tokenizer, LABEL_MAP_INV
from visualization import extract_cls_attention, rank_premises
from nlp_to_logic import SimpleMapper, rule_based_parser

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cpu")
MAX_LEN = 64

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    vocab = build_vocab()
    model = LogicTransformer(vocab_size=len(vocab), max_len=MAX_LEN)

    model_path = os.path.join(PROJECT_ROOT, "best_model.pt")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        st.warning("⚠️ Model not found. Using random weights.")

    model.eval()
    return model, vocab

model, vocab = load_model()
mapper = SimpleMapper()

# =========================
# RULE-BASED FIX (🔥 KEY)
# =========================
def rule_based_inference(premises, query):
    # Direct contradiction → FALSE
    if f"~{query}" in premises:
        return "False", 1.0

    # Direct presence → TRUE
    if query in premises:
        return "True", 1.0

    # Modus Ponens
    for p in premises:
        if "->" in p:
            left, right = p.split("->")
            if left.strip() in premises and right.strip() == query:
                return "True", 0.99

    return None, None

# =========================
# ENCODING
# =========================
def encode_logic(premises, query):
    tokens = ["[CLS]"]

    for p in premises:
        tokens.extend(logic_tokenizer(p))
        tokens.append(";")

    tokens.append("[SEP]")
    tokens.extend(logic_tokenizer(query))

    ids = [vocab[t] for t in tokens if t in vocab]

    if len(ids) < MAX_LEN:
        ids += [vocab["[PAD]"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor(ids).unsqueeze(0), tokens

# =========================
# HYBRID REASONING
# =========================
def run_reasoning(premises, query):

    # 🔥 RULE-BASED FIRST
    rule_result, rule_conf = rule_based_inference(premises, query)

    if rule_result:
        label_map = {"True": 0, "False": 1, "Unknown": 2}
        return label_map[rule_result], rule_conf, [(p, 1.0) for p in premises]

    # 🤖 NEURAL MODEL
    input_ids, tokens = encode_logic(premises, query)

    with torch.no_grad():
        logits, attentions = model(input_ids, return_attention=True)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item()
    cls_att = extract_cls_attention(attentions)[0].cpu()
    ranked = rank_premises(tokens, cls_att)

    return pred, confidence, ranked

# =========================
# UI
# =========================
st.set_page_config(page_title="Neural Reasoning Solver", layout="wide")

st.title("🧠 Neural Deductive Reasoning Solver")

# INPUT
premises_text = st.text_area("Enter Premises (one per line)", "A -> B\nA")
query_text = st.text_input("Enter Query", "B")

if st.button("Solve"):

    mapper.reset()

    premises = [p.strip() for p in premises_text.split("\n") if p.strip()]

    logical_premises = []
    for p in premises:
        logic = rule_based_parser(p) or p
        logical_premises.append(logic)

    logical_query = rule_based_parser(query_text) or query_text

    st.subheader("🔄 Logical Form")
    st.write("Premises:", logical_premises)
    st.write("Query:", logical_query)

    pred, conf, ranked = run_reasoning(logical_premises, logical_query)

    result = LABEL_MAP_INV[pred]

    st.subheader("🎯 Result")

    if result == "True":
        st.success(f"TRUE (Confidence: {conf:.2f})")
    elif result == "False":
        st.error(f"FALSE (Confidence: {conf:.2f})")
    else:
        st.warning(f"UNKNOWN (Confidence: {conf:.2f})")

    st.subheader("🔍 Reasoning")

    for i, (p, score) in enumerate(ranked):
        st.write(f"Step {i+1}: {p}")
