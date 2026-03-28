import streamlit as st
import torch
import torch.nn.functional as F
import os
import sys

# =========================
# PATH SETUP
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

st.set_page_config(page_title="Neural Reasoning Solver", layout="wide")

# =========================
# CUSTOM CSS (🔥 PREMIUM UI)
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

.result-true {
    color: #22c55e;
    font-size: 28px;
    font-weight: bold;
}

.result-false {
    color: #ef4444;
    font-size: 28px;
    font-weight: bold;
}

.result-unknown {
    color: #facc15;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

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

    model.eval()
    return model, vocab

model, vocab = load_model()
mapper = SimpleMapper()

# =========================
# RULE FIX
# =========================
def rule_based_inference(premises, query):
    if f"~{query}" in premises:
        return "False", 1.0
    if query in premises:
        return "True", 1.0
    for p in premises:
        if "->" in p:
            l, r = p.split("->")
            if l.strip() in premises and r.strip() == query:
                return "True", 0.99
    return None, None

# =========================
# ENCODE
# =========================
def encode_logic(premises, query):
    tokens = ["[CLS]"]
    for p in premises:
        tokens.extend(logic_tokenizer(p))
        tokens.append(";")
    tokens.append("[SEP]")
    tokens.extend(logic_tokenizer(query))

    ids = [vocab[t] for t in tokens if t in vocab]
    ids = ids[:MAX_LEN] + [vocab["[PAD]"]] * (MAX_LEN - len(ids))

    return torch.tensor(ids).unsqueeze(0), tokens

# =========================
# MODEL
# =========================
def run_reasoning(premises, query):
    input_ids, tokens = encode_logic(premises, query)

    with torch.no_grad():
        logits, attn = model(input_ids, return_attention=True)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    conf = probs[0][pred].item()
    cls_att = extract_cls_attention(attn)[0].cpu()
    ranked = rank_premises(tokens, cls_att)

    return pred, conf, ranked, probs[0].cpu().numpy()

# =========================
# UI
# =========================
st.title("🧠 Neural Deductive Reasoning Solver")
st.markdown("### ⚡ AI-powered Logical Inference with Explainability")

col1, col2 = st.columns(2)

with col1:
    premises_text = st.text_area("Premises", "A -> B\nA")

with col2:
    query_text = st.text_input("Query", "B")

if st.button("🚀 Solve", use_container_width=True):

    premises = [p.strip() for p in premises_text.split("\n") if p.strip()]
    logical_premises = [rule_based_parser(p) or p for p in premises]
    logical_query = rule_based_parser(query_text) or query_text

    rule_res, rule_conf = rule_based_inference(logical_premises, logical_query)

    if rule_res:
        result = rule_res
        confidence = rule_conf
        probs = [1,0,0] if result=="True" else [0,1,0]
        ranked = [(p,1.0) for p in logical_premises]
    else:
        pred, confidence, ranked, probs = run_reasoning(logical_premises, logical_query)
        result = LABEL_MAP_INV[pred]

    # =========================
    # RESULT CARD
    # =========================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if result == "True":
        st.markdown(f'<p class="result-true">✅ TRUE ({confidence:.2f})</p>', unsafe_allow_html=True)
    elif result == "False":
        st.markdown(f'<p class="result-false">❌ FALSE ({confidence:.2f})</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="result-unknown">⚠️ UNKNOWN ({confidence:.2f})</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # GRAPH (🔥 HERE IT IS)
    # =========================
    st.subheader("📊 Confidence Graph")

    st.bar_chart({
        "True": probs[0],
        "False": probs[1],
        "Unknown": probs[2]
    })

    # =========================
    # REASONING
    # =========================
    st.subheader("🔍 Reasoning Steps")

    for i, (p, score) in enumerate(ranked):
        st.markdown(f"**Step {i+1}:** {p}")
