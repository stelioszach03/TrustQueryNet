# TrustQueryNet References Used

This list covers the primary references explicitly grounding the current manuscript claims, datasets, backbone choice, and baseline methods.

## Datasets and External Validation

1. Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. "The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Scientific Data*, 2018.
   - Used for the primary internal dataset and lesion-level split context.
2. Marc Combalia et al. "Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images: the 2019 International Skin Imaging Collaboration Grand Challenge." *The Lancet Digital Health*, 2022.
   - Used to ground the official ISIC 2019 external validation setting and the broader external-shift framing.

## Backbone and Training

3. Zhuang Liu et al. "A ConvNet for the 2020s." *CVPR*, 2022.
   - Used for the ConvNeXt-Tiny backbone.
4. Christian Szegedy et al. "Rethinking the Inception Architecture for Computer Vision." *CVPR*, 2016.
   - Used as the standard citation for label smoothing in the training recipe.

## Calibration and Selective Prediction

5. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. "On Calibration of Modern Neural Networks." *ICML*, 2017.
   - Used for temperature scaling and calibration framing.
6. Yonatan Geifman and Ran El-Yaniv. "Selective Classification for Deep Neural Networks." *NeurIPS*, 2017.
   - Used for selective prediction, risk-coverage framing, and AURC-style evaluation context.

## Noisy-Label Baseline

7. Zhilu Zhang and Mert R. Sabuncu. "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels." *NeurIPS*, 2018.
   - Used for the GCE baseline.

## Note

This is the current paper-facing bibliography set. It is intentionally focused on the works explicitly used in the manuscript, not a full literature review of trustworthy medical AI.
