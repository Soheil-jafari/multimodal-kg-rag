# Chain-of-thought — paired comparison

Enhanced retrieval (`+rerank`, the frozen configuration), one retrieval per question answered twice: CoT off vs on. 37 questions.

| subset | n | Correct OFF | Correct ON | delta | improved | regressed |
|---|---:|---:|---:|---:|---:|---:|
| multi_hop (hypothesis) | 17 | 0.38 | 0.41 | +0.03 | 3 | 2 |
| single_fact (control) | 20 | 0.88 | 0.80 | -0.07 | 1 | 3 |
| **combined** | 37 | 0.65 | 0.62 | -0.03 | 4 | 5 |

Abstention (a CoT prompt can make the model more cautious, which would show up as correctness loss without any reasoning failure):

| subset | abstained OFF | abstained ON |
|---|---:|---:|
| multi_hop | 0.24 | 0.29 |
| single_fact | 0.10 | 0.15 |

## Questions whose verdict changed (9)

**q059** (multi_hop, improved) — What drugs are used to treat breast cancer that occurs in women?

- expected: `drugs treats breast cancer; breast cancer occurs in women`
- OFF: Drugs used to treat breast cancer in women include doxorubicin, paclitaxel, epirubicin, cyclophosphamide, capecitabine, vinorelbine, letrozole, and tamoxifen [val00001:379772:r1], [val00001:379772:r4], [val00002:356703:r
- ON: Doxorubicin, paclitaxel, epirubicin, docetaxel, cyclophosphamide, capecitabine, vinorelbine, and tamoxifen are used to treat breast cancer in women [val00001:379772:r1][val00001:379772:r4][val00002:356703:r7][val00002:35

**q061** (multi_hop, improved) — How might paracetamol help manage pain in cancer patients who also experience osteoarthritis?

- expected: `people with OA who take paracetamol decreases pain; pain occurs in cancer patients`
- OFF: Paracetamol may help manage pain in cancer patients who also experience osteoarthritis by providing symptomatic relief from joint pain, stiffness, and loss of motion associated with osteoarthritis. It is used as a simple
- ON: Paracetamol may help manage pain in cancer patients with osteoarthritis by providing relief from OA-related joint pain, as it is effective in reducing pain and is recommended as a first-line therapy for OA [val00001:4253

**q062** (multi_hop, regressed) — How does ondansetron, which is used to treat patients undergoing chemotherapy, also lead to hypokalemia?

- expected: `ondansetron treats chemotherapy administration; ondansetron causes hypokalemia`
- OFF: Ondansetron can lead to hypokalemia by affecting renal tubular physiology. It downregulates the Na-K-2Cl (NKCC2) cotransporter at the Loop of Henle, which increases sodium delivery to the distal nephron. This necessitate
- ON: Ondansetron can lead to hypokalemia by affecting renal tubular physiology, specifically by downregulating the NKCC2 cotransporter and upregulating the Na-K ATPase, which increases sodium delivery and entry, necessitating

**q065** (multi_hop, regressed) — What is the effect of MTZ on spermatogenesis and spermatogenic activity?

- expected: `MTZ causes suppressive effect on spermatogenesis; MTZ inhibits spermatogenic activity`
- OFF: MTZ impairs fertility in males by inhibiting spermatogenic activity and altering sperm indices. It causes suppressive effects on spermatogenesis by altering the number of germ cells at specific stages of the seminiferous
- ON: MTZ impairs fertility by inhibiting spermatogenic activity and altering sperm indices, including germ cell numbers, sperm count, motility, and morphology [val00003:411246:r3][val00003:411246:r1][val00003:411246:r0].

**q072** (multi_hop, improved) — What type of treatment do patients receive for a condition that can also be treated with bisphosphonates?

- expected: `patients treats condition; bisphosphonates treats condition`
- OFF: Patients receive letrozole 2.5 mg/day as a treatment for a condition that can also be treated with bisphosphonates [val00002:356703:r3].
- ON: Patients with bone metastases can receive treatment with bisphosphonates and letrozole, which is a form of endocrine therapy [val00002:356703:r3][val00002:356703:r14].

**q021** (single_fact, regressed) — What process does sodium perchlorate inhibit by interfering with its production?

- expected: `TH production`
- OFF: Sodium perchlorate inhibits the process of thyroid hormone production by interfering with thyroidal iodide uptake. It competitively inhibits the iodide transporter protein, blocking iodine uptake and thus inhibiting thyr
- ON: Sodium perchlorate inhibits thyroid hormone production by interfering with iodine uptake, which is essential for thyroid function [val00000:415272:r3][val00000:415766:r0][val00000:415213:r2][val00000:415213:r8].

**q023** (single_fact, regressed) — What does blood transfusion increase the risk of?

- expected: `risk of PTE`
- OFF: Blood transfusion increases the risk of postoperative pulmonary thromboembolism (PTE) [val00000:410047:r2].
- ON: Blood transfusion increases the risk of pulmonary thromboembolism (PTE) and coronary artery disease (CAD) due to iron overload [val00000:410047:r2][val00000:410049:r0][val00004:392348:r0].

**q035** (single_fact, improved) — What aspect of the Aroclor mixtures was diminished by dechlorination?

- expected: `potency`
- OFF: The aspect of the Aroclor mixtures that was diminished by dechlorination was both the potency and efficacy of induction of cytochrome P4501A activity. The decrease in potency was dependent on the extent of removal of the
- ON: The potency and efficacy of induction of the Aroclor mixtures were diminished by dechlorination, consistent with the loss of coplanar and heavily chlorinated noncoplanar PCBs [val00000:415036:r2][val00000:415036:r6][val0

**q037** (single_fact, regressed) — What is the term for the product that results from the dechlorination of Aroclor mixtures?

- expected: `dechlorinated product`
- OFF: The term for the product that results from the dechlorination of Aroclor mixtures is "dechlorinated product" or "products of reductive dechlorination" [val00000:415036:r1].
- ON: *abstained*

## Example reasoning trace

**q059** (multi_hop) — What drugs are used to treat breast cancer that occurs in women?

```
1. [val00001:379772:r1] mentions that breast cancer treatment includes hormone therapy, chemotherapy, and targeted therapies, with doxorubicin being one of the drugs used.
2. [val00001:379772:r4] discusses the use of doxorubicin in combination with paclitaxel for metastatic breast cancer.
3. [val00002:356703:r7] and [val00002:356703:r5] mention the use of epirubicin, docetaxel, cyclophosphamide, capecitabine, and vinorelbine in chemotherapy for breast cancer.
4. [val00002:356703:r0] mentions the use of tamoxifen as part of endocrine therapy for breast cancer.
```

Answer: Doxorubicin, paclitaxel, epirubicin, docetaxel, cyclophosphamide, capecitabine, vinorelbine, and tamoxifen are used to treat breast cancer in women [val00001:379772:r1][val00001:379772:r4][val00002:356703:r7][val00002:356703:r5][val00002:356703:r0].

Reasoning traces captured for 37/37 answers, median 475 characters.
