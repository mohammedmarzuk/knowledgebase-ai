# Expanded Dataset Catalog

This folder contains a fictional, synthetic company knowledge base designed for learning and testing a local RAG application. None of the rules or numbers are official company policies or legal advice.

## Documents
- `company_policy.txt` — original broad fictional company handbook.
- `01_hr_employment.txt` — recruitment, onboarding, probation, performance, records.
- `02_leave_attendance.txt` — leave, attendance, sample schedule, overtime.
- `03_payroll_benefits.txt` — payroll, payslips, benefits, expenses.
- `04_remote_work.txt` — hybrid work, cross-border approval, equipment, privacy.
- `05_it_cybersecurity.txt` — passwords, phishing, devices, incident response.
- `06_privacy_data_governance.txt` — classification, personal data, retention, data quality.
- `07_ai_usage_rag_policy.txt` — generative AI, RAG, prompt injection, evaluation.
- `08_code_of_conduct.txt` — workplace respect, conflicts, gifts, reporting.
- `09_finance_procurement.txt` — purchasing, vendor onboarding, invoices, budgets.
- `10_customer_support.txt` — ticketing, sample service targets, escalation.
- `11_software_engineering_standards.txt` — code review, testing, secrets, releases.
- `12_data_engineering_etl_sql.txt` — ETL, SQL, data quality, incremental loads.
- `13_ml_model_lifecycle.txt` — ML metrics, experimentation, deployment, monitoring.
- `14_project_management.txt` — scope, milestones, status, risk, acceptance.
- `15_business_continuity_incidents.txt` — incident response, recovery, continuity.
- `16_training_facilities_safety.txt` — training, visitors, workplace safety.
- `17_vendor_customer_contracts.txt` — contract review, customer commitments, renewal.
- `18_glossary_faq.txt` — glossary, FAQs, and evaluation prompts.

## Re-index after adding the documents
1. Open the project in VS Code.
2. Activate the virtual environment and run the Streamlit app as usual.
3. In the sidebar, click **Build / refresh knowledge base**.
4. Wait for indexing to finish, then ask questions and inspect the retrieved source passages.

The app indexes supported files in the `data/` directory. Because this is a refresh operation, it rebuilds the index from all supported files currently in that folder. Keep only documents you want included.

## Suggested tests
- How many annual leave days are available in the sample policy?
- How early should planned annual leave be requested?
- What is the fictional payroll target?
- Can I work remotely from another country without approval?
- What should I do if I accidentally shared my password?
- What is the difference between a risk and an issue?
- What is the CEO's name? (The correct behaviour is to abstain.)
- What is the universal expense approval cap? (The documents do not specify one.)
