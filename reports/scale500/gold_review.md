# Gold-set spot-check — scale500

`domain_packs/biomed/gold/gold_set_scale500.jsonl` — 122 questions. Automated filters only; nothing here is human-verified yet.

For each item: the question, the answer the builder derived from the graph, and the SOURCE it was derived from. Check that the question is answerable from that source, that the direction is not reversed, and that no qualifier in the question is absent from the source (the q061 defect class).

| category | n |
|---|---|
| single_fact | 56 |
| multi_hop | 17 |
| figure | 12 |
| figure_value | 3 |
| text_derived | 23 |
| unanswerable | 11 |

## single_fact  (4 of 56 shown)

**q001** — What treatment was used for the majority of the patients?

- **expected answer:** `chemotherapy`
- **from predicate(s):** treats
- **val00002:356704:r11** [text]
    - evidence: *the majority of the patients was treated with chemotherapy*
    - region text: After reviewing the literature, data on treatment are fragmentary, and however patients underwent surgery when possible; the majority of the patients was treated with chem- otherapy, one of them with endocrine therapy (especially after chemotherapy as maintenance therapy), and very rarely radiotherapy was performed. In…

**q015** — What process is inhibited by Pb according to recent studies?

- **expected answer:** `fracture-healing`
- **from predicate(s):** inhibits
- **val00000:414920:r5** [text]
    - evidence: *Pb inhibits fracture-healing remains a topic for future investigation.*
    - region text: Pb inhibits fracture-healing remains a topic for Although the precise mechanism by which future investigation, the phenotype is very eproducible and is somcwhat reminiscent of phenotypes described in other mouse models. I he increased chondrogenesis observed is simi ar to that seen in the fracrure callus of parathy roi…

**q029** — What was significantly lower in patients treated with IPC according to the evidence provided?

- **expected answer:** `rate of PTE`
- **from predicate(s):** decreases
- **val00000:410047:r2** [text]
    - evidence: *A significantly lower rate of PTE was observed in the patients treated with IPC[risk ratio of 0.396].*
    - region text: As shown in Table 3, multivariate analysis was done for a total of 12 variables, including the use of IPC. Significant associations with PTE were observed in the case of surgery for malignant tumors,blood transfusion, BMl >25 km/ m2, and BMI ≥28 kg/m2, with their risk ratios being 2.860, 3.834, 2.718, and 3.922, respec…

**q043** — In the context of patients undergoing AraSns for gynecologic malignancies, which group of patients is specifically mentioned as having a higher rate of PTE development?

- **expected answer:** `postoperative patients`
- **from predicate(s):** occurs_in
- **val00000:410047:r3** [text]
    - evidence: *Thirty-two patients among 42 developed PTE had malignant tumors postoperative patients who s (76%) compared to an overall rate of PTE development of 2.21% (32/1,451) among patients undergoing AraSns for gynecologic malignancies.*
    - region text: Thirty-two patients among 42 developed PTE had malignant tumors postoperative patients who s (76%) compared to an overall rate of PTE development of 2.21% (32/1,451) among patients undergoing AraSns for gynecologic malignancies.

## multi_hop  (4 of 17 shown)

**q057** — What type of benign disease do patients receive chemotherapy to treat?

- **expected answer:** `patients occurs in benign disease; patients treats chemotherapy`
- **from predicate(s):** occurs_in, treats
- **val00000:410047:r5** [text]
    - region text: On the other hand, 10 of the 42 patients (24%) who developed PTE had benign disease compared to an overall rate of PTE development of 0.32% (10/3,158) benign among all patients undergoing Aragins for gynecologic iumors.
- **val00002:356704:r11** [text]
    - region text: After reviewing the literature, data on treatment are fragmentary, and however patients underwent surgery when possible; the majority of the patients was treated with chem- otherapy, one of them with endocrine therapy (especially after chemotherapy as maintenance therapy), and very rarely radiotherapy was performed. In…

**q061** — How might paracetamol help manage pain in cancer patients who also experience osteoarthritis?

- **expected answer:** `people with OA who take paracetamol decreases pain; pain occurs in cancer patients`
- **from predicate(s):** decreases, occurs_in
- **val00001:425355:r3** [text]
    - region text: Paracetamol (acetaminophen) is a simple analgesic used in OA for decades that has both analgesic and antipyretic actions. It has a narrow therapeutic window but in recom- mended doses (1 g three to four times daily) is of favourable efficacy and very safe. Studies comparing paracetamol to placebo show that people with …
- **val00001:425924:r0** [text]
    - region text: Chronic pain is experienced by as many as 90% of cancer patients at some point during the disease. This pain can be directly cancer related or arise from a sensory neuropathy related to cherotherapy. Major pharmacological agents used to treat cancer pain often lack anatomical specificity and can have off-target effects…

**q065** — What is the effect of MTZ on spermatogenesis and spermatogenic activity?

- **expected answer:** `MTZ causes suppressive effect on spermatogenesis; MTZ inhibits spermatogenic activity`
- **from predicate(s):** causes, inhibits
- **val00003:411246:r1** [text]
    - region text: Quantitative studies have indicated marked al- terations in the number of germ cells at stage I, V and Xll following intraperitonial administration of 130 mg/kgBW/day of MTZ for seven days in mice 9) while the drug at the doses of 200 mg/kgBW/ day and 400 mg/kgB W/day for 60 days causes sup- pressive effect on spermato…
- **val00003:411246:r3** [text]
    - region text: From the foregoing it is clearly seen that MTZ at various doses impairs fertility in the males by in- hibiting spermatogenic activity and sperm indices. However, a detailed study regarding the effects of therapeutic dose of MTZ for long duration, such us for 4-8 wecks on the male reproductive organs and fertility is st…

**q069** — How does the mortality rate of hemodynamically unstable patients change for individuals over 40 years old?

- **expected answer:** `mortality rate of hemodynamically unstable patients increases mortality; mortality increases every decade over 40 years old`
- **from predicate(s):** increases, increases
- **val00001:358775:r1** [text]
    - region text: Trauma is one of the leading causes of death in patients under the age of 45 years [1]. Pelvic fractures occur in 4.0%—9.3% of patients with blunt trauma [2, 3]. These fractures should be considered severe, since mortality in these patients is high, ranging from 5.6% to 15.0% [2 8]. The mortality rate of hemodynamicall…
- **val00001:380166:r5** [text]
    - region text: ocal and regional recurrence, and 30% local and distance recurrence. Symptoms were evident only in 26%. They conclude that survival can be predicted by age under 45 years old, subclinical or local recurrence, and the ability of maintaining the disease-free situation. Mazzaferri and Jhiang [14] have published a follow-u…

## figure  (4 of 12 shown)

**q074** — What effects does in vivo lead exposure have on osteoclast precursor frequency and function?

- **expected answer:** `Figure 8. Lack of effects on osteoclast precursors after in vivo Pb exposure. M1, gate used to distinguish between CD11b-positive and -negative cells on the FACScalibur cytometer. Splenocytes from group B (n = 6/group) w`
- **val00000:414920:r8** [figure] crop=`artifacts/scale500/crops/val00000_414920_r8.png`
    - region text: (no OCR text)
- **val00000:414920:r7** [text]
    - region text: Figure 8. Lack of effects on osteoclast precursors after in vivo Pb exposure. M1, gate used to distinguish between CD11b-positive and -negative cells on the FACScalibur cytometer. Splenocytes from group B (n = 6/group) were used to determine the CD11b* 0CP frequency by flow cytometry analysis (4) or cultured observed n…

**q077** — How do chlorpyrifos surface loadings compare between high and low homes in main play areas?

- **expected answer:** `Figure 2. Box plots for chlorpyrifos surface loadings (main play areas, LWWA) (ng/cm°) for (4) *high" homes (H1H7) and (B) ~low* homes (HBH10). Note that the y-axis on each plot is not the same.`
- **val00000:415287:r10** [figure] crop=`artifacts/scale500/crops/val00000_415287_r10.png`
    - region text: (no OCR text)
- **val00000:415287:r6** [text]
    - region text: Figure 2. Box plots for chlorpyrifos surface loadings (main play areas, LWWA) (ng/cm°) for (4) *high" homes (H1H7) and (B) ~low* homes (HBH10). Note that the y-axis on each plot is not the same.

**q080** — What was the outcome of the vertebroplasty treatment for the painful haemangioma in the 56-year-old man?

- **expected answer:** `FiGukz 1: Painful haemangioma of right-hand aspect of T8 (arrowed) in a 56-year-old man treated with unipedicular vertebroplasty. Sagittal (a) and axial (b) TSE T2-weighted images. (c): Vertebroplasty spot fluoroscopy im`
- **val00001:358817:r4** [figure] crop=`artifacts/scale500/crops/val00001_358817_r4.png`
    - region text: (no OCR text)
- **val00001:358817:r3** [text]
    - region text: FiGukz 1: Painful haemangioma of right-hand aspect of T8 (arrowed) in a 56-year-old man treated with unipedicular vertebroplasty. Sagittal (a) and axial (b) TSE T2-weighted images. (c): Vertebroplasty spot fluoroscopy image, AP projection. Postprocedure, the patient reported complete resolution of pain.

**q083** — What does the WHO scale indicate about the likelihood of moderate body weight change?

- **expected answer:** `Figure 1 WHO scale predicting moderate (5-10%) body weight change`
- **val00001:413967:r7** [figure] crop=`artifacts/scale500/crops/val00001_413967_r7.png`
    - region text: (no OCR text)
- **val00001:413967:r3** [text]
    - region text: Figure 1 WHO scale predicting moderate (5-10%) body weight change

## figure_value  (3 of 3 shown)

**q086** — What does CYP1A inhibit in terms of protein expression?

- **expected answer:** `CYP1A protein expression`
- **from predicate(s):** inhibits
- ⚠ **value NOT verified** — derived from table OCR, which phase 9 showed misplaces digits. Verify against the crop before use.
- **crop:** `artifacts/scale500/crops/val00000_415406_r6.png`
- **val00000:415406:r6** [table] crop=`artifacts/scale500/crops/val00000_415406_r6.png`
    - evidence: *modestly lowers CYP1A protein expression in vivo*
    - region text: Compound Type Structure Mechanism of action Sample refererces sgsiuofe HHV BNF Synthetically derived Matsuda et al. 1995; Ronisz and model PAH Forlin 1998 BaP relevant PAH Environmentally Chaloupka et al. 1993; Fent and Batscher 2000; Van Veld et al. 1997 PCB-126 Environmentaly Abnet et al. 1999; Dabrowska et al. relev…

**q087** — Which molecule is increased by P. falciparum according to the table?

- **expected answer:** `ICAM-1`
- **from predicate(s):** increases
- ⚠ **value NOT verified** — derived from table OCR, which phase 9 showed misplaces digits. Verify against the crop before use.
- **crop:** `artifacts/scale500/crops/val00003_420785_r1.png`
- **val00003:420785:r1** [table] crop=`artifacts/scale500/crops/val00003_420785_r1.png`
    - evidence: *- increased ICAM-1 and E-selectin*
    - region text: Endothelial cell type Plosmodium strain Evaluated parameters Endothelial phenotype Ref. Porcine brain capillary endothelial cels (PSCEC) P. falc(parum - ICAM-1, E-selectin expression; - increased ICAM-1 and E-selectin [99] - TEER - decreased BBB function; - tight junction expression - tight junction disruption Human um…

**q088** — What is the specific strain of P. berghei associated with the mouse in the study?

- **expected answer:** `P. berghei (K173)`
- **from predicate(s):** occurs_in
- ⚠ **value NOT verified** — derived from table OCR, which phase 9 showed misplaces digits. Verify against the crop before use.
- **crop:** `artifacts/scale500/crops/val00003_420786_r5.png`
- **val00003:420786:r5** [table] crop=`artifacts/scale500/crops/val00003_420786_r5.png`
    - evidence: *Mouse P. berghei (K173) Histochemical and histological evaluation of cerebral lesions and their distribution*
    - region text: Animal source Plasmodium Method to evaluate B8B integrity strain Degree of impairment Reference Rhesus monkey P. knowfes/ Examination of moverment of proteins across the BBB by radiometric and Increase of BBB permeability [109-111] (Mgcgcz mulotta) fluorimetric methods Rhesus monkey P. frogile Electron microscopy, immu…

## text_derived  (4 of 23 shown)

**q089** — How many women were included in the analysis of the cohort?

- **expected answer:** `4320`
- **val00000:409598:r3** [text]
    - region text: Table 1: Distribution of 10 clinically and 5 genetically relevant variables in a cohort of young women at baseline of the observational period 1993 = 2003. The total number of women in this analysis is 4320, i.e. excluding 17 women with a final diagnosis of a possible/ potential VTE. Deviations from this number are due…

**q094** — What factors were examined in relation to the effect of water contaminants?

- **expected answer:** `History of IUGR, primiparity, and smoking.`
- **val00000:415312:r4** [text]
    - region text: history' c of IUGR, primiparity, and smoking during environment interactions, that is, whether the fect of water contaminants (total ’T'HMs and chloroform in tap watet) was modified by : rssirw pue u1oq new- variant alleles vs. none), using a heterogeneity gcnetic variants (one or two chi-square test (Hills and De Stav…

**q099** — What was found in serum samples from human males chronically exposed to PCBs?

- **expected answer:** `A decrease of total estrogenic activity and increased dioxin-like activity were found.`
- **val00000:415907:r5** [text]
    - region text: Ir pizro bioassays are a suitable tool for exposure assessment of dioxin-like and (anti)estrogenic compounds (van den Berg rt al. 1998; Zacharewski 1997). However, currently only limited data are available on Hioxin-like activities found in human female erum and follicular fluid; the TEQs deter mined by the DR-CALUX as…

**q104** — What online resource was used for predicting peptide secondary structures and folding?

- **expected answer:** `PEP-Fold.`
- **val00001:379632:r5** [text]
    - region text: 2.3. Structure and Folding Analysis of the Peptides, The pep- tide secondary structures and folding were predicted by PEP- Fold, an online resource for de novo peptide structure diction [9]. PEP-Fold provided pdb (protein data bank) files pre- with representation of macromolecular structure. Pdb files were visualized b…

## unanswerable  (4 of 11 shown)

**q112** — Which gene mutation causes cystic fibrosis?

- **expected answer:** `insufficient evidence in the corpus`
- **absence check:** top cosine vs corpus = 0.617 (lower = more clearly absent)

**q114** — How many chambers does the human heart have?

- **expected answer:** `insufficient evidence in the corpus`
- **absence check:** top cosine vs corpus = 0.6044 (lower = more clearly absent)

**q116** — What is the incubation period of the measles virus?

- **expected answer:** `insufficient evidence in the corpus`
- **absence check:** top cosine vs corpus = 0.5791 (lower = more clearly absent)

**q118** — Which blood type is the universal donor for red cell transfusion?

- **expected answer:** `insufficient evidence in the corpus`
- **absence check:** top cosine vs corpus = 0.5871 (lower = more clearly absent)
