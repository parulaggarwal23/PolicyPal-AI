# PolicyPal AI — Evaluation Dataset
## Week 4, Exercise 2

---

**Purpose:** Evaluate the same PolicyPal RAG application using three different LLM generation models:
- `codellama:latest`
- `qwen2.5:0.5b`
- `tinyllama:1.1b`

**Rules:** Send the IDENTICAL 25 questions to all three models. Do not change the RAG pipeline, embeddings, retrieval, prompts, or chunking between runs. Only the LLM generation model changes.

**Embedding model (fixed):** `nomic-embed-text`  
**Retrieval:** Cosine similarity · Top-3 chunks · 590 total chunks

---

## Knowledge Base Coverage

| Document | Category | Chunks | Questions |
|---|---|---|---|
| `cgst_act.pdf` | GST | 335 | Q01–Q09 |
| `consumer_act.pdf` | Consumer Protection | 103 | Q10–Q15 |
| `it_act.pdf` | Information Technology | 100 | Q16–Q20 |
| `essential_commodities.pdf` | Essential Commodities | 52 | Q21–Q24 |
| *(not in KB)* | Out-of-scope | — | Q25 |

---

## Dataset Summary

| Attribute | Count |
|---|---|
| Total questions | 25 |
| KB-supported (answer in documents) | 23 |
| Hallucination traps (answer NOT in KB) | 2 |
| Tests retrieval quality | 8 |
| Easy | 8 |
| Medium | 11 |
| Hard | 6 |

---

## Questions by Type

| Type | Questions |
|---|---|
| Definition | Q01, Q02, Q05, Q10, Q12, Q16, Q20, Q21 |
| Policy / Legal Rule | Q06, Q07, Q11, Q17, Q18, Q22, Q23 |
| Eligibility | Q03, Q08, Q13, Q15 |
| Compliance | Q04, Q13, Q19, Q24 |
| Tax / GST | Q09, Q25 |

---

## Hallucination Trap Questions

| ID | Reason |
|---|---|
| Q09 | GST export rate schedules are not in the CGST Act text chunks — model must not invent rates |
| Q25 | Income Tax slabs are entirely outside the knowledge base — correct answer is "not found" |

---

## Retrieval Quality Test Questions

| ID | Why it tests retrieval |
|---|---|
| Q03 | Must locate Sections 22 & 24 of CGST Act from among 335 chunks |
| Q04 | Must retrieve multi-condition Section 16 ITC provisions |
| Q06 | Must combine Sections 74 and 122 (penalty + fraud) |
| Q11 | Must retrieve the six rights listed in Section 2(9) of Consumer Act |
| Q13 | Must find jurisdiction provisions spread across consumer_act chunks |
| Q14 | Must retrieve Chapter VI product liability introduction |
| Q17 | Must locate Section 43 penalty among IT Act chunks |
| Q18 | Must retrieve Section 70 protected system definition |
| Q22 | Must find Section 3 powers across Essential Commodities chunks |

---

## Full Question Set

---

### Q01 — Definition · Easy · GST

**Question:** What is the Central Goods and Services Tax (CGST) Act?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
The Central Goods and Services Tax Act, 2017 (Act No. 12 of 2017) is an Act enacted by the Parliament of India to make a provision for levy and collection of tax on intra-State supply of goods or services or both by the Central Government.

---

### Q02 — Definition · Easy · GST

**Question:** What is the meaning of 'aggregate turnover' under the CGST Act?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
Aggregate turnover means the aggregate value of all taxable supplies (excluding inward supplies on which tax is payable on reverse charge basis), exempt supplies, exports of goods or services or both, and inter-State supplies of persons having the same PAN, computed on an all-India basis but excluding CGST, SGST, UTGST, IGST, and cess.

---

### Q03 — Eligibility · Medium · GST

**Question:** Who is liable for registration under the CGST Act?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
Every supplier whose aggregate turnover exceeds the notified threshold is liable to register. Additionally, casual taxable persons, non-resident taxable persons, and persons liable to pay tax under reverse charge are compulsorily required to register regardless of turnover.

---

### Q04 — Compliance · Hard · GST

**Question:** What are the conditions a registered person must satisfy to claim Input Tax Credit (ITC) under the CGST Act?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
A registered person is entitled to ITC only if: (a) they possess a valid tax invoice or debit note from a registered supplier; (aa) the invoice details are furnished in the statement of outward supplies; (b) they have received the goods or services; (c) the tax charged has actually been paid to the government by the supplier; and (d) they have furnished a return.

---

### Q05 — Definition · Medium · GST

**Question:** What is a 'composite supply' under the CGST Act and how is it taxed?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
A composite supply is a supply made by a taxable person to a recipient consisting of two or more taxable supplies naturally bundled in the ordinary course of business, one of which is the principal supply. It is taxed at the rate applicable to the principal supply.

---

### Q06 — Policy/Legal Rule · Hard · GST

**Question:** What are the penalties for committing fraud or wilful misstatement under the CGST Act?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
Under Section 74, where tax has not been paid or short paid due to fraud or wilful misstatement or suppression of facts, the person is liable to pay the tax, interest, and a penalty equal to the tax amount. Section 122 prescribes additional penalties for specific offences.

---

### Q07 — Policy/Legal Rule · Medium · GST

**Question:** What is the validity period of registration for a casual taxable person under the CGST Act?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
The certificate of registration issued to a casual taxable person is valid for the period specified in the application for registration or ninety days from the effective date of registration, whichever is earlier.

---

### Q08 — Eligibility · Easy · GST

**Question:** Under the CGST Act, is an agriculturist required to register for GST on the supply of agricultural produce?

**Expected source:** `cgst_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
No. An agriculturist, to the extent of supply of produce out of cultivation of land, is not liable to be registered under the CGST Act.

---

### Q09 — Tax/GST · Medium · GST ⚠️ HALLUCINATION TRAP

**Question:** What is the GST rate applicable to export of goods from India?

**Expected source:** none  
**KB supported:** No  
**Tests hallucination:** ✓ Yes  
**Tests retrieval quality:** No

**Ground truth:**
The CGST Act text in the knowledge base does not contain specific rate schedules. The correct response is that this information could not be found in the provided policy documents. A model that states a specific percentage (e.g. 0%, 18%) without KB support is hallucinating.

---

### Q10 — Definition · Easy · Consumer Protection

**Question:** Who is defined as a 'consumer' under the Consumer Protection Act, 2019?

**Expected source:** `consumer_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
A consumer is any person who buys goods for consideration (not for resale or commercial purpose) or hires/avails of any service for consideration. It includes the user of goods or beneficiary of services other than the buyer, provided such use is with the approval of the buyer.

---

### Q11 — Policy/Legal Rule · Medium · Consumer Protection

**Question:** What are the six consumer rights recognised under the Consumer Protection Act, 2019?

**Expected source:** `consumer_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
The six rights are: (i) right to be protected from hazardous goods and services; (ii) right to be informed about quality, quantity, potency, purity, standard and price; (iii) right to access variety of goods at competitive prices; (iv) right to be heard; (v) right to seek redressal against unfair trade practices; (vi) right to consumer awareness/education.

---

### Q12 — Definition · Medium · Consumer Protection

**Question:** What constitutes an 'unfair trade practice' under the Consumer Protection Act, 2019?

**Expected source:** `consumer_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
Unfair trade practices include: falsely representing goods as new when rebuilt/old, false claims about quality or grade, false representation of sponsorship or approval, misleading representation about the need for goods or services, and giving false facts about price.

---

### Q13 — Compliance · Medium · Consumer Protection

**Question:** How can a consumer file a complaint under the Consumer Protection Act, 2019, and where should it be filed?

**Expected source:** `consumer_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
A complaint should be filed before the District Consumer Disputes Redressal Commission having jurisdiction where the opposite party resides or carries on business, where the cause of action arose, or where the complainant resides or personally works for gain.

---

### Q14 — Definition · Hard · Consumer Protection

**Question:** What is product liability under the Consumer Protection Act, 2019?

**Expected source:** `consumer_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
Product liability is the responsibility of a product manufacturer, product service provider, or product seller for compensation to a consumer for any harm caused by a defective product. The Consumer Protection Act, 2019 includes a dedicated Chapter VI on product liability.

---

### Q15 — Eligibility · Hard · Consumer Protection

**Question:** What compensation can a consumer receive under the Consumer Protection Act if they suffer loss due to a defective product?

**Expected source:** `consumer_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
The Consumer Protection Act, 2019 provides for product liability actions where a consumer can claim compensation from the manufacturer, service provider, or seller. The quantum of compensation is determined by the Consumer Disputes Redressal Commission.

---

### Q16 — Definition · Easy · Information Technology

**Question:** What is the purpose of the Information Technology Act, 2000?

**Expected source:** `it_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
The IT Act, 2000 was enacted to provide legal recognition for transactions carried out by means of electronic data interchange and other electronic communication (electronic commerce), to facilitate electronic filing of documents with Government agencies, and to amend related Acts including the Indian Penal Code.

---

### Q17 — Policy/Legal Rule · Medium · Information Technology

**Question:** What is the penalty for unauthorised access to a computer system under the IT Act, 2000?

**Expected source:** `it_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
Under Section 43, a person who without permission accesses a computer, computer system or computer network is liable to pay damages by way of compensation to the affected person. Section 66 provides for imprisonment up to 3 years and/or a fine up to 5 lakh rupees for computer-related offences done dishonestly or fraudulently.

---

### Q18 — Policy/Legal Rule · Hard · Information Technology

**Question:** What is a 'protected system' under the IT Act, 2000, and what are the consequences of unauthorised access to it?

**Expected source:** `it_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
A protected system is any computer resource declared by the appropriate Government by notification in the Official Gazette that directly or indirectly affects the facility of Critical Information Infrastructure. Unauthorised access to a protected system constitutes an offence under the IT Act.

---

### Q19 — Compliance · Hard · Information Technology

**Question:** Under the IT Act, 2000, can a company be held liable for a cyber offence committed by one of its employees?

**Expected source:** `it_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
Yes. Under Section 85, if a contravention is committed by a company, every person in charge of and responsible for the conduct of business at the time of the contravention is deemed guilty. However, no liability attaches if the person proves the contravention occurred without their knowledge or that they exercised all due diligence.

---

### Q20 — Definition · Easy · Information Technology

**Question:** What is the meaning of 'cyber security' as defined in the IT Act, 2000?

**Expected source:** `it_act.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
Cyber security means protecting information, equipment, devices, computers, computer resources, communication devices and information stored therein from unauthorised access, use, disclosure, disruption, modification or destruction.

---

### Q21 — Definition · Easy · Essential Commodities

**Question:** What is the Essential Commodities Act, 1955, and what is its purpose?

**Expected source:** `essential_commodities.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
The Essential Commodities Act, 1955 (Act No. 10 of 1955) is an Act to provide, in the interest of the general public, for the control of the production, supply and distribution of, and trade and commerce in, certain essential commodities. It came into force on 1st April 1955.

---

### Q22 — Policy/Legal Rule · Medium · Essential Commodities

**Question:** What powers does the Central Government have to control the supply and distribution of essential commodities?

**Expected source:** `essential_commodities.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** ✓ Yes

**Ground truth:**
Under Section 3, the Central Government may issue orders to control production, supply, distribution and trade of essential commodities including provisions for licensing, permits, price fixation, and regulation of stocks, if it considers it necessary or expedient for maintaining supplies or securing equitable distribution.

---

### Q23 — Policy/Legal Rule · Medium · Essential Commodities

**Question:** What is the punishment for violating an order issued under the Essential Commodities Act, 1955?

**Expected source:** `essential_commodities.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
A person who contravenes an order made under Section 3 shall be punishable with imprisonment for a term not less than six months but which may extend to seven years, and shall also be liable to a fine. The court may, for adequate and special reasons recorded in the judgment, impose a lesser sentence of less than six months.

---

### Q24 — Compliance · Hard · Essential Commodities

**Question:** Can a company be held liable for an offence committed under the Essential Commodities Act?

**Expected source:** `essential_commodities.pdf`  
**KB supported:** Yes  
**Tests hallucination:** No  
**Tests retrieval quality:** No

**Ground truth:**
Yes. If an offence is committed by a company and it is proved that it occurred with the consent or connivance of, or is attributable to neglect by, any director, manager, secretary or other officer of the company, that officer shall also be deemed guilty and liable to be punished.

---

### Q25 — Tax/GST · Easy · Out-of-scope ⚠️ HALLUCINATION TRAP

**Question:** What is the income tax slab rate for individuals earning between 5 lakh and 10 lakh rupees per year in India?

**Expected source:** none  
**KB supported:** No  
**Tests hallucination:** ✓ Yes  
**Tests retrieval quality:** No

**Ground truth:**
This information is not available in the PolicyPal knowledge base. The KB contains the CGST Act, Consumer Protection Act, IT Act, and Essential Commodities Act. Income tax slabs are defined in the Income Tax Act, which is not part of the current knowledge base. The correct model response is: "I could not find this information in the policy documents."

---

## Evaluation Instructions

1. Start the PolicyPal application: `venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080`
2. Select each model from the UI dropdown (or set `LLM_MODEL` env var)
3. Send each of the 25 questions via the `/api/rag-ask` or `/api/rag-ask-pipeline` endpoint
4. Record: answer text, retrieved chunks, similarity scores, latency
5. Score each answer against the ground truth
6. Compare results across the three models

### Suggested Scoring Rubric

| Score | Meaning |
|---|---|
| 2 | Correct and grounded in retrieved context |
| 1 | Partially correct or vague but not wrong |
| 0 | Incorrect, hallucinated, or refused when answer exists in KB |
| -1 | Hallucinated a specific answer for a question marked KB-supported = No |

### Maximum possible score per model: 50 (25 questions × 2)
