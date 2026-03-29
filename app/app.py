import streamlit as st
import torch
import torch.nn.functional as F
import os
import sys
import re

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from model import LogicTransformer
from dataset import build_vocab, logic_tokenizer, LABEL_MAP_INV
from visualization import extract_cls_attention, rank_premises
from nlp_to_logic import SimpleMapper, rule_based_parser, clean_logic_output

# PAGE CONFIG

st.set_page_config(
    page_title="Neural Deductive Reasoning Solver",
    page_icon="🧠",
    layout="wide"
)

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64

# LOAD MODEL

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

# SYMBOLIC FORWARD CHAINING

def extract_atoms(expr: str) -> set:
    """Extract all single-letter variable names from a logical expression."""
    return set(re.findall(r'\b([A-Z][a-z0-9]*|[A-Z])\b', expr))


def parse_implication(expr: str):
    """
    Parse an implication from a logical expression string.
    Handles: A -> B, A => B, A --> B, A ==> B
    Returns (antecedent, consequent) or None if not an implication.
    """
    m = re.match(r'^(.+?)\s*(?:->|=>|-->|==>)\s*(.+)$', expr.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def symbolic_forward_chain(logical_premises: list, logical_query: str):
    """
    Symbolic forward-chaining reasoner for propositional logic.

    Rules handled:
      - Atomic facts:        'A'
      - Implications:        'A -> B'
      - Biconditionals:      'A <-> B'  (split into two implications)
      - Conjunctions:        'A & B'    (split into two facts)

    Returns:
      (label, explanation)
        label       : 'True' | 'False' | 'Unknown'
        explanation : human-readable derivation steps
    """
    facts = set()
    rules = []          # list of (antecedent_str, consequent_str)
    steps = []          # derivation log

    # ── 1. Parse premises ────────────────────────────────────────────
    for p in logical_premises:
        p = p.strip()
        if not p:
            continue

        # Biconditional: A <-> B  →  A->B and B->A
        bic = re.match(r'^(.+?)\s*<[-=]?>\s*(.+)$', p)
        if bic:
            a, b = bic.group(1).strip(), bic.group(2).strip()
            rules.append((a, b))
            rules.append((b, a))
            continue

        # Conjunction as premise: A & B  →  fact A, fact B
        if '&' in p and not parse_implication(p):
            for part in p.split('&'):
                facts.add(part.strip())
            continue

        # Implication
        impl = parse_implication(p)
        if impl:
            rules.append(impl)
            continue

        # Plain fact
        facts.add(p)

    initial_facts = set(facts)

    # ── 2. Forward chain ─────────────────────────────────────────────
    changed = True
    iterations = 0
    while changed and iterations < 200:
        changed = False
        iterations += 1
        for ant, cons in rules:
            if ant in facts and cons not in facts:
                facts.add(cons)
                steps.append(f"• `{ant}` is known → derived `{cons}`")
                changed = True

            # Handle conjunctive antecedents: 'A & B -> C'
            if '&' in ant:
                parts = [x.strip() for x in ant.split('&')]
                if all(p in facts for p in parts) and cons not in facts:
                    facts.add(cons)
                    steps.append(f"• `{' & '.join(parts)}` all known → derived `{cons}`")
                    changed = True

    # ── 3. Clean query atom ──────────────────────────────────────────
    q = logical_query.strip()
    q = re.sub(r'[?!.\s]+$', '', q).strip()   # strip trailing punctuation/spaces

    # ── 4. Evaluate query ────────────────────────────────────────────
    if q in facts:
        return 'True', steps

    # Collect every variable that appears anywhere in the premises
    all_premise_vars = set()
    for p in logical_premises:
        all_premise_vars.update(extract_atoms(p))

    query_vars = extract_atoms(q)

    # Query references variables that are completely absent from all premises
    if query_vars and query_vars.isdisjoint(all_premise_vars):
        steps.append(
            f"• Variable(s) `{', '.join(query_vars)}` never appear in any premise → cannot be derived → **False**"
        )
        return 'False', steps

    # Query variable exists in premises but was never derived
    steps.append(
        f"• `{q}` could not be derived from the given premises → **Unknown**"
    )
    return 'Unknown', steps


# HELPER FUNCTIONS

def convert_to_logic(sentence):
    """Convert English sentence to logical form."""
    rule_result = rule_based_parser(sentence)
    if rule_result:
        return rule_result, "rule"

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

    ids = [vocab[t] for t in tokens if t in vocab]

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


LABEL_TO_IDX = {"True": 0, "False": 1, "Unknown": 2}

def hybrid_predict(logical_premises, logical_query):
    """
    Hybrid symbolic + neural prediction.

    Priority:
      1. If symbolic gives 'True'  or 'False' with high certainty → trust it.
      2. If symbolic gives 'Unknown' → defer to neural model.
      3. If neural confidence < threshold → fall back to symbolic.

    Returns:
      (result_label, confidence, ranked, probs, symbolic_label, sym_steps, source)
    """
    NEURAL_CONFIDENCE_THRESHOLD = 0.60   # minimum confidence to trust neural alone

    # ── Symbolic ──────────────────────────────────────────────────────
    sym_label, sym_steps = symbolic_forward_chain(logical_premises, logical_query)

    # ── Neural ────────────────────────────────────────────────────────
    neural_pred, neural_conf, ranked, probs = run_reasoning(logical_premises, logical_query)
    neural_label = LABEL_MAP_INV[neural_pred]

    # ── Decision logic ────────────────────────────────────────────────

    # Case 1: Symbolic is definitive True/False → override neural
    if sym_label in ('True', 'False'):
        sym_idx = LABEL_TO_IDX[sym_label]
        # Build a near-certain probability vector for display
        display_probs = [0.05, 0.05, 0.05]
        display_probs[sym_idx] = 0.90
        return (
            sym_label,
            0.90,
            ranked,
            display_probs,
            sym_label,
            sym_steps,
            "symbolic"
        )

    # Case 2: Symbolic is Unknown but neural is confident
    if neural_conf >= NEURAL_CONFIDENCE_THRESHOLD:
        return (
            neural_label,
            neural_conf,
            ranked,
            list(probs),
            sym_label,
            sym_steps,
            "neural"
        )

    # Case 3: Both uncertain → report Unknown
    return (
        "Unknown",
        max(neural_conf, 0.5),
        ranked,
        list(probs),
        sym_label,
        sym_steps,
        "hybrid-uncertain"
    )


# STREAMLIT UI

st.title("🧠 Neural Deductive Reasoning Solver")
st.markdown("*A Transformer-based Logical Inference System with Symbolic Grounding*")

st.markdown("---")

col1, col2 = st.columns([3, 1])
with col2:
    if DEVICE.type == 'cuda':
        st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("💻 Running on CPU")

# INPUT SECTION

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
    st.markdown("*Examples: `Is C true?`, `C`, `Is Z true?`*")

    query_text = st.text_input(
        "Query",
        value="Is C true?",
        label_visibility="collapsed"
    )

input_mode = st.radio(
    "Input Mode",
    ["Natural Language", "Direct Logic"],
    horizontal=True,
    help="Natural Language: converts English to logic. Direct Logic: uses input as-is."
)

st.markdown("---")

# SOLVE BUTTON

if st.button("🔮 Solve", type="primary", use_container_width=True):

    mapper.reset()

    premises_lines = [p.strip() for p in premises_text.split("\n") if p.strip()]

    if not premises_lines:
        st.error("Please enter at least one premise.")
    elif not query_text.strip():
        st.error("Please enter a query.")
    else:

        # ── Step 1: Conversion ─────────────────────────────────────────
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
                conversion_table.append({"Input": p, "Logic": logic, "Method": method or "direct"})
            else:
                conversion_table.append({"Input": p, "Logic": "❌ INVALID", "Method": "failed"})

        if input_mode == "Direct Logic":
            logical_query = rule_based_parser(query_text) or query_text
            query_method = "direct"
        else:
            logical_query, query_method = convert_to_logic(query_text)

        if not logical_query:
            st.error("❌ Failed to convert query into logic.")
            st.stop()

        st.markdown("### 📊 Conversion Results")
        st.table(conversion_table)
        st.markdown(f"**Query Conversion:** `{query_text}` → `{logical_query}`")

        # ── Step 2: Hybrid Reasoning ───────────────────────────────────
        st.markdown("---")
        st.subheader("🧠 Step 2: Hybrid Symbolic + Neural Reasoning")

        (result_label, confidence, ranked,
         probs, sym_label, sym_steps, source) = hybrid_predict(
            logical_premises, logical_query
        )

        # ── Symbolic derivation trace ──────────────────────────────────
        with st.expander("🔣 Symbolic Derivation Trace", expanded=True):
            if sym_steps:
                for step in sym_steps:
                    st.markdown(step)
            else:
                st.info("No symbolic derivation steps (query matched an initial fact).")

            badge = {
                "symbolic":         "🔣 Symbolic override (definitive)",
                "neural":           "🤖 Neural model (high confidence)",
                "hybrid-uncertain": "⚠️ Hybrid – low confidence on both"
            }.get(source, source)
            st.markdown(f"**Decision source:** {badge}")

        # ── Result ────────────────────────────────────────────────────
        st.markdown("### 🎯 Result")

        if result_label == "True":
            st.success(f"✅ TRUE  (Confidence: {confidence:.2f})")
        elif result_label == "False":
            st.error(f"❌ FALSE  (Confidence: {confidence:.2f})")
        else:
            st.warning(f"⚠️ UNKNOWN  (Confidence: {confidence:.2f})")

        # ── Probability chart ─────────────────────────────────────────
        st.markdown("### 📊 Prediction Confidence")

        prob_dict = {
            "True":    float(probs[0]),
            "False":   float(probs[1]),
            "Unknown": float(probs[2])
        }
        st.bar_chart(prob_dict)

        # ── Attention-based reasoning path ────────────────────────────
        st.markdown("### 🔍 Reasoning Path (Attention-based)")

        if ranked:
            for i, (premise, score) in enumerate(ranked):
                st.write(f"**Step {i+1}:** `{premise}`  → Importance: `{score:.4f}`")
        else:
            st.info("No reasoning steps available.")

        # ── Final summary ─────────────────────────────────────────────
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
            **Decided by:** {source}
            """
          )
