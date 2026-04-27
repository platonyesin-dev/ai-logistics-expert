# 🚛 AI Logistics Expert (Hybrid RAG)

This tool is an advanced AI consultant designed to assist logistics teams by bridging the gap between **Static Manuals** and **Real-World Experience**. 

Most bots only read a PDF; this bot reads your official manual **and** your recent client emails to provide the most up-to-date advice possible.

---

## 🚀 General Functionality

The bot uses a **Priority-Based RAG (Retrieval-Augmented Generation)** architecture:
1.  **The Brain:** Powered by `Llama 3.3` (via Groq) for high-speed reasoning.
2.  **The Memory:** A `Pinecone` Vector Database storing 3072-dimensional embeddings.
3.  **The Knowledge Tiers:**
    * **Tier 1 (Official Manual):** The bot first checks the `company_info.pdf` for standard rules.
    * **Tier 2 (Gmail Experience):** If the manual is silent or an email contains a newer update, the bot pulls from your Gmail history to provide "Experience-based" answers.

---

## 📸 Bot in Action

### Test Case 1: Manual Extraction (PDF Knowledge)
**Scenario:** A user asks about standard company policy. The bot must retrieve structured data from the official PDF and ignore the Gmail archive if no changes exist.

* **Query:** "Can you list all the service plans available at Atlas Prime?"
* **Result:** The bot correctly identifies the hierarchy (Bronze, Silver, Gold) and cites specific sections of the manual.

| Official Manual Source | Bot Output & Reasoning |

Firstly bot reads our Query. After he translates text to the vector type data and looks for the answer in the PDF guide:

<img width="779" height="739" alt="Screenshot 2026-04-27 at 20 57 01" src="https://github.com/user-attachments/assets/ed375cfd-91f4-4d51-8ff8-29a280855a98" />

As we can see here bot managed successfully identify correct chunck, and he also mentioned changes that were made via mail:

<img width="620" height="299" alt="Screenshot 2026-04-27 at 20 58 00" src="https://github.com/user-attachments/assets/3086b70b-4363-42a1-ac06-abe8e70eefa8" />


### Test Case 2: Conflict Resolution (Gmail Override)
**Scenario:** A policy has changed since the manual was published. The bot must identify the contradiction and prioritize the most recent information from Gmail.

* **Step 1:** The user asks about shipping to Brazil. Initially, the bot says "No" based on the outdated manual (request happened before mail was sent).
* **Step 2:** An internal update email is ingested into the system.
* **Step 3:** Upon re-querying, the bot identifies the new route and alerts the user of the policy change.

**1. Outdated Manual Response:**
<img width="1036" height="332" alt="Screenshot 2026-04-27 at 19 54 34" src="https://github.com/user-attachments/assets/42569fae-bef7-4b4b-8b7c-0d6fbc11b495" />

**2. Real-Time Update Source (Gmail):**
<img width="1130" height="272" alt="Screenshot 2026-04-27 at 21 00 39" src="https://github.com/user-attachments/assets/60e854ac-194d-4d42-afeb-15a818f7db33" />

**3. Hybrid RAG Final Output:**
<img width="890" height="326" alt="Screenshot 2026-04-27 at 19 55 58" src="https://github.com/user-attachments/assets/df4f75b7-8590-41ef-b8a9-6f3aff98ae5c" />


---

## 🛠️ Setup Guide

Follow these steps to run the Atlas Prime Expert on your local machine.

### 1. Prerequisites
- **Python 3.11+**
- A **Pinecone** Account (Index dimension: 3072)
- A **Google Cloud** Project (with Gmail API enabled)
- A **Groq** API Key

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/platonyesin-dev/ai-logistics-expert.git](https://github.com/platonyesin-dev/ai-logistics-expert.git)
cd ai-logistics-expert
python3.11 -m pip install -r requirements.txt
```
### 3. API Configuration
Create a .env file in the root directory and add your credentials:

Plaintext
```bash
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_gemini_key
```
### 4. Gmail Authentication
1. Download your credentials.json from the Google Cloud Console (Desktop App type).

2. Place it in the project folder.

3. Run the ingestion script to authorize and sync emails:

```bash
python3.11 ingest_gmail.py
```

5. Launch the App
```bash
python3.11 -m streamlit run app.py
```

🛡️ Security Note
This project uses .gitignore to ensure that credentials.json, token.json, and .env files are never uploaded to GitHub. Always keep your API keys private.
