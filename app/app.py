# app/app.py

import streamlit as st
import torch
import torch.nn.functional as F
import os
import sys

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from model import LogicTransformer
from dataset import build_vocab, logic_tokenizer, LABEL_MAP_INV
from visualization import extract_cls_attention, rank_premises
from nlp_to_logic import SimpleMapper, rule_based_parser, clean_logic_output

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Neural Deductive Reasoning Solver",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_logic_model():
    vocab = build_vocab()
    
    model = LogicTransformer(
        vocab_size=len(vocab),
        d_model=256,
        n_heads=8,
        num_layers=6,
        max_len=MAX_LEN
    ).to(DEVICE)

    model_path = os.path.join(PROJECT_ROOT, "best_model.pt")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        st.success(f"✅ Model loaded successfully!")
    else:
        st.warning(f"⚠️ No trained model found at {model_path}. Please train the model first.")
    
    model.eval()
    return model, vocab


# Initialize
model, vocab = load_logic_model()
mapper = SimpleMapper()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def convert_to_logic(sentence):
    """Convert English sentence to logical form."""
    # First try rule-based (for direct logical input)
    rule_result = rule_based_parser(sentence)
    if rule_result:
        return rule_result, "rule"
    
    # Then try semantic mapping
    semantic_result = mapper.convert(sentence)
    if semantic_result:
        return semantic_result, "semantic"
    
    return None, None


def encode_logic(premises, query):
    """Encode logical premises and query for the model."""
    tokens = ["[CLS]"]

    for p in premises:
        p_tokens = logic_tokenizer(p)
        if p_tokens:
            tokens.extend(p_tokens)
            tokens.append(";")

    tokens.append("[SEP]")
    
    q_tokens = logic_tokenizer(query)
    if q_tokens:
        tokens.extend(q_tokens)

    # Convert to IDs
    ids = [vocab[t] for t in tokens if t in vocab]

    # Pad or truncate
    if len(ids) < MAX_LEN:
        ids += [vocab["[PAD]"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor(ids).unsqueeze(0).to(DEVICE), tokens


def run_reasoning(premises, query):
    """Run the neural reasoning model."""
    input_ids, tokens = encode_logic(premises, query)

    with torch.no_grad():
        logits, attentions = model(input_ids, return_attention=True)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    confidence = probs[0][prediction].item()
    cls_attention = extract_cls_attention(attentions)[0].cpu()
    ranked = rank_premises(tokens, cls_attention)

    return prediction, confidence, ranked, probs[0].cpu().numpy()


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🧠 Neural Deductive Reasoning Solver")
st.markdown("*A Transformer-based Logical Inference System*")

st.markdown("---")

# Device info
col1, col2 = st.columns([3, 1])
with col2:
    if DEVICE.type == 'cuda':
        st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("💻 Running on CPU")

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📝 Input")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Enter Premises** (one per line)")
    st.markdown("*Examples: `All A are B`, `If B then C`, `A`, `A -> B`*")
    
    premises_text = st.text_area(
        "Premises",
        value="All A are B\nIf B then C\nA",
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Enter Query**")
    st.markdown("*Examples: `Is C true?`, `C`*")
    
    query_text = st.text_input(
        "Query",
        value="Is C true?",
        label_visibility="collapsed"
    )

# Mode selection
input_mode = st.radio(
    "Input Mode",
    ["Natural Language", "Direct Logic"],
    horizontal=True,
    help="Natural Language: converts English to logic. Direct Logic: uses input as-is."
)

st.markdown("---")

# ==========================================
# SOLVE BUTTON
# ==========================================

if st.button("🔮 Solve", type="primary", use_container_width=True):
    
    # Reset mapper for new problem
    mapper.reset()
    
    premises_lines = [p.strip() for p in premises_text.split("\n") if p.strip()]
    
    if not premises_lines:
        st.error("Please enter at least one premise.")
    elif not query_text.strip():
        st.error("Please enter a query.")
    else:
        # ==========================================
        # CONVERSION
        # ==========================================
        
        st.subheader("🔄 Step 1: Converting to Logic")
        
        logical_premises = []
        conversion_table = []
        
        for p in premises_lines:
            if input_mode == "Direct Logic":
                logic = rule_based_parser(p) or p
                method = "direct"
            else:
                logic, method = convert_to_logic(p)
            
            if logic:
                logical_premises.append(logic)
                conversion_table.append({
                    "Input": p,
                    "Logic": logic,
                    "Method": method or "direct"
                })
            else:
                conversion_table.append({
                    "Input": p,
                    "Logic": "❌ INVALID",
                    "Method": "failed"
                })

        # Convert Query
        if input_mode == "Direct Logic":
            logical_query = rule_based_parser(query_text) or query_text
            query_method = "direct"
        else:
            logical_query, query_method = convert_to_logic(query_text)

        if not logical_query:
            st.error("❌ Failed to convert query into logic.")
            st.stop()

        # Show conversion table
        st.markdown("### 📊 Conversion Results")
        st.table(conversion_table)

        st.markdown(f"**Query Conversion:** `{query_text}` → `{logical_query}`")

        # ==========================================
        # STEP 2: MODEL INFERENCE
        # ==========================================
        st.markdown("---")
        st.subheader("🧠 Step 2: Neural Reasoning")

        prediction, confidence, ranked, probs = run_reasoning(
            logical_premises,
            logical_query
        )

        result_label = LABEL_MAP_INV[prediction]

        # ==========================================
        # RESULT DISPLAY
        # ==========================================
        st.markdown("### 🎯 Result")

        if result_label == "True":
            st.success(f"✅ TRUE (Confidence: {confidence:.2f})")
        elif result_label == "False":
            st.error(f"❌ FALSE (Confidence: {confidence:.2f})")
        else:
            st.warning(f"⚠️ UNKNOWN (Confidence: {confidence:.2f})")

        # ==========================================
        # PROBABILITY DISTRIBUTION
        # ==========================================
        st.markdown("### 📊 Prediction Confidence")

        prob_dict = {
            "True": float(probs[0]),
            "False": float(probs[1]),
            "Unknown": float(probs[2])
        }

        st.bar_chart(prob_dict)

        # ==========================================
        # PROOF PATH / REASONING
        # ==========================================
        st.markdown("### 🔍 Reasoning Path (Attention-based)")

        if ranked:
            for i, (premise, score) in enumerate(ranked):
                st.write(f"**Step {i+1}:** `{premise}`  → Importance: `{score:.4f}`")
        else:
            st.info("No reasoning steps available.")

        # ==========================================
        # FINAL SUMMARY
        # ==========================================
        st.markdown("---")
        st.subheader("📌 Final Summary")

        st.markdown(
            f"""
            **Input:** {', '.join(premises_lines)}  
            **Query:** {query_text}  

            **Logical Form:**  
            Premises → `{logical_premises}`  
            Query → `{logical_query}`  

            **Result:** **{result_label.upper()}**  
            **Confidence:** **{confidence:.2f}**
            """
        )
