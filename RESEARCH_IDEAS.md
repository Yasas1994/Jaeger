# Research Questions and Ideas: ArcFace / Metric Learning for Metagenomic Reads

This file collects publishable research questions and project ideas that emerge from the Jaeger training pipeline, especially the self-supervised pretraining and ArcFace metric-learning components.

## Core questions the current setup can answer

1. **Open-set taxonomic classification**
   - Can reads from taxa never seen during training be rejected rather than misclassified?
   - What distance/cosine threshold best separates known from unknown taxa?

2. **Novel phage / virus discovery**
   - Do viral reads from novel clades fall far from all known class centroids?
   - Can ArcFace embeddings flag divergent phages in assembled contigs?

3. **Class imbalance and rare taxa**
   - Does ArcFace handle imbalanced metagenomic training data better than standard cross-entropy?
   - Can it improve recall for rare or under-represented taxa?

4. **Hierarchical classification**
   - Can a single embedding space predict multiple taxonomic ranks (species → genus → family → phylum) using hierarchical margins?
   - Does enforcing hierarchy improve low-rank predictions?

5. **Representation quality across sequence types**
   - Which input representation (nucleotide, translated, codon, k-mer) produces the most discriminative embeddings?
   - How short can reads be while still forming separable clusters?

6. **Embedding-based data cleaning**
   - Can mislabeled or contaminated training fragments be detected as outliers in embedding space?
   - Can this be used to curate training sets automatically?

7. **Cross-environment generalization**
   - Do embeddings trained on one environment (e.g., seawater) transfer to another (e.g., soil, human gut)?
   - Which environments share embedding structure?

8. **Few-shot and zero-shot extension**
   - Can new taxa be added by computing centroids from a handful of examples instead of retraining?
   - How many examples are needed for a stable centroid?

9. **Confidence calibration**
   - Is cosine similarity a better-calibrated confidence score than softmax probability?
   - Can it be used to decide when to abstain from prediction?

10. **Contig / MAG-level aggregation**
    - How should read-level embeddings be aggregated for contig- or MAG-level classification (mean, voting, attention)?
    - Does this improve prophage detection?

## New research ideas

1. **Open-world phage discovery benchmark**
   Build a benchmark where known phages are training classes and novel phages (or environmental contigs) are the test set. Use ArcFace distance to rank novelty.

2. **Hierarchical ArcFace loss for taxonomic ranks**
   Extend ArcFace with hierarchical margins so the embedding space respects the tree of life. This could replace separate classifiers per rank.

3. **Self-supervised + ArcFace pretraining on unlabeled metagenomes**
   Pretrain on massive unlabeled read sets with a contrastive/ArcFace objective, then fine-tune on curated labels. Test whether this improves rare-taxon recall.

4. **Metagenomic read retrieval with embedding indexes**
   Use ArcFace embeddings to build a Faiss/ANN index for fast similarity search across read databases, enabling k-NN classification at scale.

5. **Metric-learning ensemble for robust phage detection**
   Train multiple ArcFace models with different architectures/receptive fields and combine their distance spaces. Compare against single-model softmax ensembles.

6. **Embedding-driven active learning for metagenomics**
   Use embedding-space uncertainty to select the most informative unlabeled reads for manual annotation, reducing labeling cost.

7. **Domain-adversarial ArcFace for cross-environment transfer**
   Add a domain-adversarial component so the embedding space is invariant to sequencing platform or environment while preserving taxonomic structure.

8. **ArcFace for plasmid vs. chromosome vs. phage disentanglement**
   Explicitly learn an embedding space that separates mobile genetic elements (plasmids, phages, transposons) from chromosomal DNA across diverse bacteria.

9. **Long-read / contig ArcFace with Hyena**
   Leverage the Hyena block integration to embed long contigs directly (no fragmenting) and study how classification improves with context length.

10. **Interpretability of viral embeddings**
    Analyze which motifs/positions drive embedding similarity. Do learned centroids correspond to known functional or taxonomic markers?
