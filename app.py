import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn


@st.cache_resource
def load_artifacts():
    with open("lr_model.pkl", "rb") as f:
        lr_model_obj = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        svm_model_obj = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler_obj = pickle.load(f)
    with open("le_type.pkl", "rb") as f:
        le_type_obj = pickle.load(f)
    with open("le_sender.pkl", "rb") as f:
        le_sender_obj = pickle.load(f)
    with open("le_receiver.pkl", "rb") as f:
        le_receiver_obj = pickle.load(f)
    with open("median.pkl", "rb") as f:
        median_obj = pickle.load(f)
    with open("lr_acc.pkl", "rb") as f:
        lr_acc_obj = pickle.load(f)
    with open("svm_acc.pkl", "rb") as f:
        svm_acc_obj = pickle.load(f)


    return {
        "lr_model": lr_model_obj,
        "svm_model": svm_model_obj,
        "scaler": scaler_obj,
        "le_type": le_type_obj,
        "le_sender": le_sender_obj,
        "le_receiver": le_receiver_obj,
        "median": median_obj,
        "lr_acc": lr_acc_obj,
        "svm_acc": svm_acc_obj,

    }

artifacts = load_artifacts()
lr_model = artifacts["lr_model"]
svm_model = artifacts["svm_model"]
scaler = artifacts["scaler"]
le_type = artifacts["le_type"]
le_sender = artifacts["le_sender"]
le_receiver = artifacts["le_receiver"]
median_amount = artifacts["median"]
lr_acc = artifacts["lr_acc"]
svm_acc = artifacts["svm_acc"]

st.set_page_config(page_title="UPI Fraud Detection", page_icon="💳", layout="wide")



for key, default in {
    "prediction_done": False,
    "lr_prob": None,
    "svm_prob": None,
    "lr_pred": None,
    "svm_pred": None,
    "input_summary": {},
    "scaled_vector": None,
    "encoded": {},
    "derived": {},
    "raw_input": {},
    "user_verified": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "history" not in st.session_state:
    st.session_state.history = []

st.title("💳 UPI Fraud Detection System")

left_col, right_col = st.columns([1, 3])

# ── LEFT: Input only ──────────────────────────────────────────────────────────
with left_col:
    st.subheader("📥 Transaction Details")

    hour = st.slider("🕐 Hour", 0, 23, 12)
    amount = st.number_input("💰 Amount (₹)", min_value=1, value=500)
    type_trans = st.selectbox("🔁 Type", ["P2P", "Merchant", "Recharge"])
    sender = st.text_input("📤 Sender UPI", value="user@upi")
    receiver = st.text_input("📥 Receiver UPI", value="merchant@upi")

    is_night = hour < 6
    is_high = amount > median_amount
    is_p2p = type_trans == "P2P"

    st.markdown("**Derived Flags**")
    st.write(f"🌙 Night Time: **{'Yes' if is_night else 'No'}**")
    st.write(f"📈 High Amount: **{'Yes' if is_high else 'No'}**")
    st.write(f"🔗 P2P: **{'Yes' if is_p2p else 'No'}**")

    st.markdown("---")

    def is_reverse_transaction(sender, receiver, amount):
        for txn in st.session_state.history:
            if txn["sender"] == receiver and txn["receiver"] == sender:
                if abs(txn["amount"] - amount) < 100:
                    return True
        return False

    if st.button("🔍 Predict", width="stretch", type="primary"):
        st.session_state.user_verified = None

        df = pd.DataFrame({
            'hour_of_day': [hour],
            'type': [type_trans],
            'amount': [amount],
            'sender_address': [sender],
            'receiver_address': [receiver]
        })

        df['night_transaction'] = 1 if is_night else 0
        df['high_amount'] = 1 if is_high else 0
        df['p2p_transaction'] = 1 if is_p2p else 0

        sender_history = [txn for txn in st.session_state.history if txn["sender"] == sender]
        txn_count_sender = len(sender_history) + 1
        if sender_history:
            avg_amount_sender = (sum(txn["amount"] for txn in sender_history) + amount) / txn_count_sender
        else:
            avg_amount_sender = median_amount
        amount_deviation = amount - avg_amount_sender
        is_frequent_receiver = 1 + sum(1 for txn in sender_history if txn["receiver"] == receiver) if sender_history else 0

        df['txn_count_sender'] = txn_count_sender
        df['avg_amount_sender'] = avg_amount_sender
        df['amount_deviation'] = amount_deviation
        df['is_frequent_receiver'] = is_frequent_receiver

        mapping_type = {cls: idx for idx, cls in enumerate(le_type.classes_)}
        mapping_sender = {cls: idx for idx, cls in enumerate(le_sender.classes_)}
        mapping_receiver = {cls: idx for idx, cls in enumerate(le_receiver.classes_)}

        type_enc = mapping_type.get(type_trans, -1)
        sender_enc = mapping_sender.get(sender, -1)
        receiver_enc = mapping_receiver.get(receiver, -1)

        df['type'] = type_enc
        df['sender_address'] = sender_enc
        df['receiver_address'] = receiver_enc

        reverse_detected = is_reverse_transaction(sender, receiver, amount)
        df_scaled = scaler.transform(df)
        st.session_state.scaled_vector = df_scaled[0]

        if reverse_detected:
            st.success("✅ LEGIT (Return Transaction Detected)")
            st.session_state.lr_pred = np.array([0])
            st.session_state.lr_prob = np.array([1.0, 0.0])
            st.session_state.svm_pred = np.array([0])
            st.session_state.svm_prob = np.array([1.0, 0.0])
        else:
            st.session_state.lr_pred = lr_model.predict(df_scaled)
            st.session_state.lr_prob = lr_model.predict_proba(df_scaled)[0]
            st.session_state.svm_prob = svm_model.predict_proba(df_scaled)[0]
            st.session_state.svm_pred = np.array([1 if st.session_state.svm_prob[1] >= 0.5 else 0])

        st.session_state.prediction_done = True
        st.session_state.encoded = {
            "type": (type_trans, type_enc),
            "sender_address": (sender, sender_enc),
            "receiver_address": (receiver, receiver_enc)
        }
        st.session_state.derived = {
            "night_transaction": (f"hour={hour} < 6", 1 if is_night else 0),
            "high_amount": (f"amount={amount} > {median_amount:.2f}", 1 if is_high else 0),
            "p2p_transaction": (f"type == P2P", 1 if is_p2p else 0)
        }
        st.session_state.raw_input = {
            "hour_of_day": hour, "type": type_trans,
            "amount": amount, "sender_address": sender,
            "receiver_address": receiver,
            "txn_count_sender": txn_count_sender,
            "avg_amount_sender": avg_amount_sender,
            "amount_deviation": amount_deviation,
            "is_frequent_receiver": is_frequent_receiver
        }
        st.session_state.input_summary = {
            "Hour": hour, "Amount": f"₹{amount}", "Type": type_trans,
            "Night": "Yes" if is_night else "No",
            "High Amt": "Yes" if is_high else "No",
            "P2P": "Yes" if is_p2p else "No"
        }
        st.session_state.history.append({
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "hour": hour
        })
        if not reverse_detected:
            st.success("Done! See results →")

    # ── USER CONFIRMATION (shown only when fraud flagged) ─────────────────────
    if st.session_state.prediction_done:
        lr_pred = st.session_state.lr_pred
        svm_pred = st.session_state.svm_pred
        both_fraud = lr_pred[0] == 1 and svm_pred[0] == 1
        one_fraud = lr_pred[0] != svm_pred[0]

        if both_fraud or one_fraud:
            st.markdown("---")
            st.warning("⚠️ This transaction was flagged as suspicious.")
            st.markdown("**Is this an intentional/emergency payment?**")
            st.caption("Example: sending money to a friend in midnight emergency")

            confirm = st.radio(
                "Confirm your intent:",
                ["— Select —", "✅ Yes, I know this person and it is urgent", "❌ No, cancel this transaction"],
                key="confirm_radio"
            )

            if confirm == "✅ Yes, I know this person and it is urgent":
                st.success("Transaction verified by you. Marked as Genuine.")
                st.session_state.user_verified = True
                st.caption("🏦 Real banking equivalent: OTP confirmation passed.")

            elif confirm == "❌ No, cancel this transaction":
                st.error("Transaction cancelled by user.")
                st.session_state.user_verified = False
                st.caption("🏦 Real banking equivalent: User declined — transaction blocked.")


# ── RIGHT: Output + About ─────────────────────────────────────────────────────
with right_col:
    output_tab, about_tab = st.tabs(["📊 Output", "🧠 About"])

    # ── OUTPUT TAB ────────────────────────────────────────────────────────────
    with output_tab:
        if not st.session_state.prediction_done:
            st.info("Run a prediction from the Input panel to see results here.")
        else:
            lr_prob = st.session_state.lr_prob
            svm_prob = st.session_state.svm_prob
            lr_pred = st.session_state.lr_pred
            svm_pred = st.session_state.svm_pred

           

            st.markdown("**Logistic Regression**")
            if lr_pred[0] == 1:
                st.error(f"🚨 FRAUD {lr_prob[1]*100:.2f}%")
            else:
                st.success(f"✅ LEGIT {lr_prob[0]*100:.2f}%")
            st.caption(f"Accuracy: {lr_acc*100:.2f}%")

            st.markdown("**SVM**")
            if svm_pred[0] == 1:
                st.error(f"🚨 FRAUD {svm_prob[1]*100:.2f}%")
            else:
                st.success(f"✅ LEGIT {svm_prob[0]*100:.2f}%")
            st.caption(f"Accuracy: {svm_acc*100:.2f}%")

            st.markdown("---")
            x = np.arange(2)
            fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))
            ax1.bar(x - 0.2, [lr_prob[0]*100, lr_prob[1]*100], width=0.35,
                    color=["#2ecc71", "#e74c3c"], label="LR")
            ax1.bar(x + 0.2, [svm_prob[0]*100, svm_prob[1]*100], width=0.35,
                    color=["#27ae60", "#c0392b"], alpha=0.75, label="SVM")
            ax1.set_xticks(x)
            ax1.set_xticklabels(["Legit", "Fraud"])
            ax1.set_ylabel("%")
            ax1.set_ylim(0, 115)
            ax1.legend(fontsize=7)
            ax1.set_title("Fraud vs Legit", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
            bars = ax2.bar(["LR", "SVM"], [lr_acc*100, svm_acc*100],
                           color=["#3498db", "#9b59b6"], width=0.4)
            for bar, val in zip(bars, [lr_acc*100, svm_acc*100]):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f"{val:.1f}%", ha="center", fontsize=8)
            ax2.set_ylim(0, 105)
            ax2.set_ylabel("%")
            ax2.set_title("Model Accuracy", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            max_fraud = max(lr_prob[1], svm_prob[1]) * 100
            fig3, ax3 = plt.subplots(figsize=(3.5, 2.2))
            ax3.set_xlim(-1.2, 1.2)
            ax3.set_ylim(-0.2, 1.2)
            ax3.set_aspect("equal")
            ax3.axis("off")
            for s, e, c in [(0, 40, "#2ecc71"), (40, 70, "#f39c12"), (70, 100, "#e74c3c")]:
                t = np.linspace(np.radians(180 - e*1.8), np.radians(180 - s*1.8), 100)
                xs = np.concatenate([np.cos(t), 0.6*np.cos(t[::-1])])
                ys = np.concatenate([np.sin(t), 0.6*np.sin(t[::-1])])
                ax3.fill(xs, ys, color=c, alpha=0.75)
            na = np.radians(180 - max_fraud*1.8)
            ax3.plot([0, 0.75*np.cos(na)], [0, 0.75*np.sin(na)], color="black", lw=2.5)
            ax3.add_patch(plt.Circle((0, 0), 0.07, color="black", zorder=5))
            ax3.text(0, -0.15, f"{max_fraud:.1f}%", ha="center", fontsize=11, fontweight="bold")
            ax3.text(0, 1.1, "Max Fraud %", ha="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            st.markdown("---")

            # ── FINAL VERDICT ─────────────────────────────────────────────────
            if st.session_state.user_verified is True:
                st.success("✅ GENUINE — User confirmed this transaction")
                st.info("🏦 Reason: User verified intent (emergency/known person). Equivalent to OTP confirmation in real banking.")
                st.caption(f"Original model flags — LR: {'🚨 FRAUD' if lr_pred[0]==1 else '✅ LEGIT'} | SVM: {'🚨 FRAUD' if svm_pred[0]==1 else '✅ LEGIT'}")

            elif st.session_state.user_verified is False:
                st.error("❌ BLOCKED — User cancelled this transaction")
                st.caption("User chose to cancel after fraud warning.")

            else:
                if lr_pred[0] == svm_pred[0]:
                    verdict = "🚨 FRAUD" if lr_pred[0] == 1 else "✅ LEGIT"
                    st.info(f"Both models agree: **{verdict}**")
                else:
                    st.warning("⚠️ Models disagree — trust the higher-accuracy model.")

    # ── ABOUT TAB ─────────────────────────────────────────────────────────────
    with about_tab:
        st.subheader("🧠 Step-by-Step Walkthrough")

        if not st.session_state.prediction_done:
            st.info("Run a prediction from the Input panel to see the live calculation walkthrough here.")

        with st.expander("Step 1 — Raw Input Received", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                raw = st.session_state.raw_input
                st.markdown("The app receives these raw values from the form:")
                st.dataframe(pd.DataFrame([raw]), width="stretch")
            else:
                st.markdown("Collects: `hour_of_day`, `type`, `amount`, `sender_address`, `receiver_address` from the form.")

        with st.expander("Step 2 — Feature Engineering", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                derived = st.session_state.derived
                rows = [{"Feature": f, "Rule": r, "Result": v} for f, (r, v) in derived.items()]
                st.dataframe(pd.DataFrame(rows), width="stretch")
                st.markdown(f"""
- `night_transaction = {derived['night_transaction'][1]}` — hour {'< 6 ✅' if derived['night_transaction'][1] else '>= 6, not night ❌'}
- `high_amount = {derived['high_amount'][1]}` — ₹{st.session_state.raw_input['amount']} vs median ₹{median_amount:.2f} → {'above ✅' if derived['high_amount'][1] else 'below ❌'}
- `p2p_transaction = {derived['p2p_transaction'][1]}` — type is {'P2P ✅' if derived['p2p_transaction'][1] else 'not P2P ❌'}
                """)
            else:
                st.markdown("Three binary flags: `night_transaction` (hour < 6), `high_amount` (amount > median), `p2p_transaction` (type == P2P).")

        with st.expander("Step 3 — Label Encoding", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                enc = st.session_state.encoded
                rows = [{"Column": col, "Original": orig, "Encoded": code,
                         "Status": "Known ✅" if code != -1 else "Unseen → -1 ⚠️"}
                        for col, (orig, code) in enc.items()]
                st.dataframe(pd.DataFrame(rows), width="stretch")
                for col, (orig, code) in enc.items():
                    if code == -1:
                        st.warning(f"`{col}` = `{orig}` not in training data → encoded as **-1**")
            else:
                st.markdown("`LabelEncoder` converts string columns to integers. Unseen values → **-1**.")

        with st.expander("Step 4 — Pre-Scaling Feature Vector", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                raw = st.session_state.raw_input
                derived = st.session_state.derived
                enc = st.session_state.encoded
                vec = {
                    "hour_of_day": raw["hour_of_day"],
                    "type": enc["type"][1],
                    "amount": raw["amount"],
                    "sender_address": enc["sender_address"][1],
                    "receiver_address": enc["receiver_address"][1],
                    "night_transaction": derived["night_transaction"][1],
                    "high_amount": derived["high_amount"][1],
                    "p2p_transaction": derived["p2p_transaction"][1],
                    "txn_count_sender": raw.get("txn_count_sender", "N/A"),
                    "avg_amount_sender": raw.get("avg_amount_sender", "N/A"),
                    "amount_deviation": raw.get("amount_deviation", "N/A"),
                    "is_frequent_receiver": raw.get("is_frequent_receiver", "N/A")
                }
                st.markdown("Full 12-feature vector before scaling:")
                st.dataframe(pd.DataFrame([vec]), width="stretch")
            else:
                st.markdown("12 features combined: `[hour_of_day, type, amount, sender_address, receiver_address, night_transaction, high_amount, p2p_transaction, txn_count_sender, avg_amount_sender, amount_deviation, is_frequent_receiver]`")

        with st.expander("Step 5 — StandardScaler Normalization", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                scaled = st.session_state.scaled_vector
                raw = st.session_state.raw_input
                derived = st.session_state.derived
                enc = st.session_state.encoded
                cols = ["hour_of_day", "type", "amount", "sender_address", "receiver_address",
                        "night_transaction", "high_amount", "p2p_transaction", "txn_count_sender",
                        "avg_amount_sender", "amount_deviation", "is_frequent_receiver"]
                means = scaler.mean_
                stds = scaler.scale_
                raw_vals = [
                    raw["hour_of_day"], enc["type"][1], raw["amount"],
                    enc["sender_address"][1], enc["receiver_address"][1],
                    derived["night_transaction"][1], derived["high_amount"][1], derived["p2p_transaction"][1],
                    raw["txn_count_sender"], raw["avg_amount_sender"], raw["amount_deviation"], raw["is_frequent_receiver"]
                ]
                rows = [{"Feature": cols[i], "x": round(raw_vals[i], 4),
                         "μ (mean)": round(means[i], 4), "σ (std)": round(stds[i], 4),
                         "z = (x−μ)/σ": round(scaled[i], 4)} for i in range(len(cols))]
                st.markdown("Formula: `z = (x − μ) / σ`")
                st.dataframe(pd.DataFrame(rows), width="stretch")
            else:
                st.markdown("Normalizes each feature: `z = (x − μ) / σ`. μ and σ from training data.")

        with st.expander("Step 6 — Logistic Regression Decision", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                lr_prob = st.session_state.lr_prob
                lr_pred = st.session_state.lr_pred
                scaled = st.session_state.scaled_vector
                w = lr_model.coef_[0]
                b = lr_model.intercept_[0]
                z = float(np.dot(w, scaled) + b)
                sigmoid = 1 / (1 + np.exp(-z))
                cols = ["hour_of_day", "type", "amount", "sender_address", "receiver_address",
                        "night_transaction", "high_amount", "p2p_transaction", "txn_count_sender",
                        "avg_amount_sender", "amount_deviation", "is_frequent_receiver"]
                w_df = pd.DataFrame({"Feature": cols, "Weight w": [round(x, 6) for x in w],
                                     "Scaled x": [round(v, 6) for v in scaled],
                                     "w × x": [round(w[i]*scaled[i], 6) for i in range(len(w))]})
                st.dataframe(w_df, width="stretch")
                st.markdown(f"""
**z = Σ(w × x) + b**
- Σ(w × x) = `{sum(w[i]*scaled[i] for i in range(len(w))):.6f}`
- b (bias) = `{b:.6f}`
- z = `{z:.6f}`

**P(fraud) = 1 / (1 + e⁻ᶻ) = {sigmoid:.4f} ({sigmoid*100:.2f}%)**

Threshold = 0.5 → Prediction: **{"🚨 FRAUD" if lr_pred[0] == 1 else "✅ LEGIT"}**
Model accuracy: **{lr_acc*100:.2f}%**
                """)
            else:
                st.markdown("""
1. `z = w₁x₁ + ... + w₈x₈ + b`
2. `P(fraud) = 1 / (1 + e⁻ᶻ)`
3. P ≥ 0.5 → Fraud, else Legit
                """)

        with st.expander("Step 7 — SVM Decision", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                svm_prob = st.session_state.svm_prob
                svm_pred = st.session_state.svm_pred
                scaled = st.session_state.scaled_vector
                decision = svm_model.decision_function([scaled])[0]
                n_sv = svm_model.n_support_
                st.markdown(f"""
**RBF Kernel:** `K(x, y) = exp(−γ ||x−y||²)`

**Decision function score:** `{decision:.6f}`
- Score > 0 → Fraud side of hyperplane
- Score < 0 → Legit side of hyperplane
- This score: {'> 0 → leans Fraud' if decision > 0 else '< 0 → leans Legit'}

**Support vectors:** {n_sv[0]} Legit + {n_sv[1]} Fraud = {sum(n_sv)} total

**Platt scaling:**
- P(fraud) = **{svm_prob[1]:.4f}** ({svm_prob[1]*100:.2f}%)
- P(legit) = **{svm_prob[0]:.4f}** ({svm_prob[0]*100:.2f}%)

Threshold = 0.5 → Prediction: **{"🚨 FRAUD" if svm_pred[0] == 1 else "✅ LEGIT"}**
Model accuracy: **{svm_acc*100:.2f}%**
                """)
            else:
                st.markdown("""
1. RBF kernel maps input to higher-dimensional space
2. Hyperplane maximises margin between Fraud/Legit
3. Decision score → Platt scaling → probability
4. P ≥ 0.5 → Fraud, else Legit
                """)

        with st.expander("Step 8 — Final Comparison", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                lr_prob = st.session_state.lr_prob
                svm_prob = st.session_state.svm_prob
                lr_pred = st.session_state.lr_pred
                svm_pred = st.session_state.svm_pred
                st.dataframe(pd.DataFrame({
                    "Model": ["Logistic Regression", "SVM"],
                    "Fraud %": [f"{lr_prob[1]*100:.2f}%", f"{svm_prob[1]*100:.2f}%"],
                    "Legit %": [f"{lr_prob[0]*100:.2f}%", f"{svm_prob[0]*100:.2f}%"],
                    "Prediction": ["🚨 FRAUD" if lr_pred[0] == 1 else "✅ LEGIT",
                                   "🚨 FRAUD" if svm_pred[0] == 1 else "✅ LEGIT"],
                    "Accuracy": [f"{lr_acc*100:.2f}%", f"{svm_acc*100:.2f}%"]
                }), width="stretch")
                if lr_pred[0] == svm_pred[0]:
                    st.info(f"Both models agree: **{'🚨 FRAUD' if lr_pred[0] == 1 else '✅ LEGIT'}**")
                else:
                    st.warning("⚠️ Models disagree — trust the higher-accuracy model.")
                if lr_acc > svm_acc:
                    st.success("🏆 LR has higher accuracy on this dataset.")
                elif svm_acc > lr_acc:
                    st.success("🏆 SVM has higher accuracy on this dataset.")
            else:
                st.markdown("Both predictions compared side by side. Higher-accuracy model is preferred.")

        with st.expander("Step 9 — User Confirmation & Real Banking Logic", expanded=st.session_state.prediction_done):
            if st.session_state.prediction_done:
                lr_pred = st.session_state.lr_pred
                svm_pred = st.session_state.svm_pred
                both_fraud = lr_pred[0] == 1 and svm_pred[0] == 1
                one_fraud = lr_pred[0] != svm_pred[0]

                if both_fraud or one_fraud:
                    st.markdown("**Why this was flagged:**")
                    derived = st.session_state.derived
                    enc = st.session_state.encoded
                    raw = st.session_state.raw_input
                    flags = []
                    if derived["night_transaction"][1]:
                        flags.append("🌙 Midnight transaction (hour < 6) — unusual timing")
                    if derived["high_amount"][1]:
                        flags.append(f"💰 High amount ₹{raw['amount']} > median ₹{median_amount:.0f}")
                    if derived["p2p_transaction"][1]:
                        flags.append("🔗 Direct P2P transfer — higher risk category")
                    if enc["sender_address"][1] == -1:
                        flags.append("⚠️ Sender UPI not seen in training data")
                    if enc["receiver_address"][1] == -1:
                        flags.append("⚠️ Receiver UPI not seen in training data")
                    for flag in flags:
                        st.markdown(f"- {flag}")

                    st.markdown("---")
                    st.markdown("**What a real bank does at this point:**")
                    max_fraud_prob = max(st.session_state.lr_prob[1], st.session_state.svm_prob[1]) * 100
                    if max_fraud_prob < 40:
                        st.success("🟢 Low Risk — Transaction allowed directly.")
                    elif max_fraud_prob < 70:
                        st.warning("🟡 Medium Risk — Bank sends OTP: 'Are you sending ₹X to Y?'")
                    else:
                        st.error("🔴 High Risk — Transaction held. Bank notification + security question triggered.")

                    st.markdown("---")
                    if st.session_state.user_verified is True:
                        st.success("✅ User confirmed → Transaction marked Genuine (OTP equivalent passed)")
                    elif st.session_state.user_verified is False:
                        st.error("❌ User cancelled → Transaction blocked")
                    else:
                        st.info("⏳ Waiting for user confirmation from the Input panel.")

                    st.caption(
                        "Note: ML models predict risk probability — not absolute fraud. "
                        "Real banking always allows user override via OTP/confirmation for genuine emergencies."
                    )
                else:
                    st.success("✅ No fraud flagged by either model — no confirmation needed.")
            else:
                st.markdown(
                    "When fraud is detected, real banks don't block instantly. "
                    "They ask for OTP/confirmation — allowing genuine emergencies (like midnight payments to friends) to go through."
                )