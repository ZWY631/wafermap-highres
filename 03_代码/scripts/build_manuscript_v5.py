#!/usr/bin/env python3
"""Transform the audited v4 manuscript into the submission-ready v5 draft.

Changes applied on top of v4 (which remains untouched):
1. Reference expansion 25 -> 29 entries (four new, Crossref-verified DOIs),
   with full citation renumbering so the list is in first-appearance order.
2. Language-consistency pass: '+/-' -> '±', 'x 10^-n' -> '× 10⁻ⁿ',
   'N x M' grid notation -> 'N × M', ONNX/HighRes first-use definitions,
   abstract wording alignment, and declaration placeholders replaced with
   the default statements agreed for submission.
"""

from __future__ import annotations

import re
from pathlib import Path

MANUSCRIPT_DIR = Path(__file__).resolve().parents[2] / "06_论文" / "manuscript"
V4 = MANUSCRIPT_DIR / "05_英文论文领域对照增强版_v4.md"
V5 = MANUSCRIPT_DIR / "06_英文论文投稿定稿版_v5.md"

# old -> new reference numbering (first-appearance order after expansion)
RENUMBER = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 18, 9: 19, 10: 20, 11: 22, 12: 10, 13: 8, 14: 9,
    15: 11, 16: 27, 17: 12, 18: 13, 19: 29, 20: 28,
    21: 21, 22: 26, 23: 23, 24: 24, 25: 25,
}
NEW_REF_NUMBERS = {14, 15, 16, 17}

REFERENCE_ENTRIES = [
    "Wu MJ, Jang JSR, Chen JL. Wafer map failure pattern recognition and similarity ranking for large-scale data sets. IEEE Transactions on Semiconductor Manufacturing. 2015;28(1):1-12. https://doi.org/10.1109/TSM.2014.2364237",
    "Ma J, Zhang T, Yang C, Cao Y, Xie L, Tian H, Li X. Review of wafer surface defect detection methods. Electronics. 2023;12(8):1787. https://doi.org/10.3390/electronics12081787",
    "Zheng H, Sherazi SWA, Son SH, Lee JY. A deep convolutional neural network-based multi-class image classification for automatic wafer map failure recognition in semiconductor manufacturing. Applied Sciences. 2021;11(20):9769. https://doi.org/10.3390/app11209769",
    "Park S, You C. Deep convolutional generative adversarial networks-based data augmentation method for classifying class-imbalanced defect patterns in wafer bin map. Applied Sciences. 2023;13(9):5507. https://doi.org/10.3390/app13095507",
    "Chen Y, Zhao M, Xu Z, Li K, Ji J. Wafer defect recognition method based on multi-scale feature fusion. Frontiers in Neuroscience. 2023;17:1202985. https://doi.org/10.3389/fnins.2023.1202985",
    "Jeong I, Lee SY, Park K, Kim I, Huh H, Lee S. Wafer map failure pattern classification using geometric transformation-invariant convolutional neural network. Scientific Reports. 2023;13(1):8127. https://doi.org/10.1038/s41598-023-34147-2",
    "He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition; 2016. p. 770-778. https://doi.org/10.1109/CVPR.2016.90",
    "Tan M, Le QV. EfficientNet: Rethinking model scaling for convolutional neural networks. In: Proceedings of the 36th International Conference on Machine Learning. 2019;97:6105-6114. https://proceedings.mlr.press/v97/tan19a.html",
    "Howard AG, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W, Zhu Y, Pang R, Vasudevan V, Le QV, Adam H. Searching for MobileNetV3. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019. p. 1314-1324. https://doi.org/10.1109/ICCV.2019.00140",
    "Kang H, Kang S. A stacking ensemble classifier with handcrafted and convolutional features for wafer map pattern classification. Computers in Industry. 2021;129:103450. https://doi.org/10.1016/j.compind.2021.103450",
    "Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. In: International Conference on Learning Representations; 2015. https://arxiv.org/abs/1409.1556",
    "He H, Garcia EA. Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering. 2009;21(9):1263-1284. https://doi.org/10.1109/TKDE.2008.239",
    "Buda M, Maki A, Mazurowski MA. A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks. 2018;106:249-259. https://doi.org/10.1016/j.neunet.2018.07.011",
    "Kim T, Behdinan K. Advances in machine learning and deep learning applications towards wafer map defect recognition and classification: a review. Journal of Intelligent Manufacturing. 2023;34(8):3215-3247. https://doi.org/10.1007/s10845-022-01994-1",
    "Nakazawa T, Kulkarni DV. Wafer map defect pattern classification and image retrieval using convolutional neural network. IEEE Transactions on Semiconductor Manufacturing. 2018;31(2):309-314. https://doi.org/10.1109/TSM.2018.2795466",
    "Manivannan S. An ensemble-based deep semi-supervised learning for the classification of Wafer Bin Maps defect patterns. Computers & Industrial Engineering. 2022;172:108614. https://doi.org/10.1016/j.cie.2022.108614",
    "Kyeong K, Kim H. Classification of mixed-type defect patterns in wafer bin maps using convolutional neural networks. IEEE Transactions on Semiconductor Manufacturing. 2018;31(3):395-402. https://doi.org/10.1109/TSM.2018.2841416",
    "Ma N, Zhang X, Zheng HT, Sun J. ShuffleNet V2: Practical guidelines for efficient CNN architecture design. In: Computer Vision - ECCV 2018. p. 122-138. https://doi.org/10.1007/978-3-030-01264-9_8",
    "Wang Q, Wu B, Zhu P, Li P, Zuo W, Hu Q. ECA-Net: Efficient channel attention for deep convolutional neural networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition; 2020. p. 11531-11539. https://doi.org/10.1109/CVPR42600.2020.01155",
    "Lin TY, Goyal P, Girshick R, He K, Dollar P. Focal loss for dense object detection. In: 2017 IEEE International Conference on Computer Vision. p. 2999-3007. https://doi.org/10.1109/ICCV.2017.324",
    "Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer P, Weiss R, Dubourg V, Vanderplas J, Passos A, Cournapeau D, Brucher M, Perrot M, Duchesnay E. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html",
    "Loshchilov I, Hutter F. Decoupled weight decay regularization. In: International Conference on Learning Representations; 2019. https://arxiv.org/abs/1711.05101",
    "McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika. 1947;12:153-157. https://doi.org/10.1007/BF02295996",
    "Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall; 1993. https://doi.org/10.1007/978-1-4899-4541-9",
    "Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics. 1979;6:65-70.",
    "Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, Lin Z, Gimelshein N, Antiga L, Desmaison A, Kopf A, Yang E, DeVito Z, Raison M, Tejani A, Chilamkurthy S, Steiner B, Fang L, Bai J, Chintala S. PyTorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems. 2019;32:8024-8035. https://papers.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
    "Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision. 2017. p. 618-626. https://doi.org/10.1109/ICCV.2017.74",
    "ONNX Runtime developers. ONNX Runtime documentation: cross-platform machine-learning accelerator. Microsoft; accessed 17 August 2026. https://onnxruntime.ai/docs/",
    "Jacob B, Kligys S, Chen B, Zhu M, Tang M, Howard A, Adam H, Kalenichenko D. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2018. p. 2704-2713. https://doi.org/10.1109/CVPR.2018.00286",
]


def renumber_citations(text: str) -> str:
    """Rewrite in-text citation tokens [n] / [a,b] per RENUMBER."""
    pattern = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    def replace(match: re.Match) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        if any(n == 0 for n in numbers):
            return match.group(0)  # layer names [0], normalization [0,1]
        mapped = [str(RENUMBER.get(n, n)) for n in numbers]
        return "[" + ", ".join(mapped) + "]"

    return pattern.sub(replace, text)


def main() -> None:
    text = V4.read_text(encoding="utf-8")

    # ---- status header ----
    text = text.replace(
        "**Manuscript status:** Domain-comparison enhanced draft (v4), audited 17 August 2026",
        "**Manuscript status:** Submission-ready draft (v5), reference-expanded and language-consistency pass, 21 August 2026",
    )

    # ---- abstract: define HighRes on first use ----
    text = text.replace(
        "HighRes exceeds this domain comparator by 0.1914",
        "The high-resolution model (HighRes) exceeds this domain comparator by 0.1914",
        1,
    )

    # ---- notation consistency ----
    text = text.replace("+/-", "±")
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    text = re.sub(
        r"(\d+(?:\.\d+)?) x 10\^-(\d+)",
        lambda m: f"{m.group(1)} × 10⁻{m.group(2).translate(superscripts)}",
        text,
    )
    text = re.sub(r"(?<=\d) x (?=\d)", " × ", text)

    # ---- first-use definitions ----
    text = text.replace(
        "The model is exported to ONNX opset 17 with a dynamic batch dimension.",
        "The model is exported to the Open Neural Network Exchange (ONNX) format, opset 17, with a dynamic batch dimension.",
    )
    text = text.replace(
        "The final network contains 1,262,397 trainable parameters.",
        "The final network contains 1,262,397 trainable parameters. This configuration is referred to as HighRes ShuffleNetV2 (HighRes) below.",
    )

    # ---- new references inserted in Related Work 2.1 ----
    # Placeholders {{REFn}} survive the renumbering pass and are converted to
    # final bracket numbers afterwards.
    text = text.replace(
        "whereas the present study assigns one label to an entire die-test map and produces no bounding box or pixel-level defect mask.",
        "whereas the present study assigns one label to an entire die-test map and produces no bounding box or pixel-level defect mask. A more recent review surveys machine-learning and deep-learning wafer-map recognition for manufacturing practice {{REF14}}.",
    )
    text = text.replace(
        "but their sampled class setting differs from the naturally imbalanced full labeled subset studied here.",
        "but their sampled class setting differs from the naturally imbalanced full labeled subset studied here. Nakazawa and Kulkarni demonstrated that CNNs can classify wafer-map defect patterns and retrieve similar maps from learned representations {{REF15}}.",
    )
    text = text.replace(
        "The present work instead retains the empirical distribution, uses simple label-preserving geometric augmentation, and reports class-balanced metrics.",
        "The present work instead retains the empirical distribution, uses simple label-preserving geometric augmentation, and reports class-balanced metrics. Semi-supervised ensembles have also been proposed to exploit unlabeled wafer maps {{REF16}}; the present study does not follow this direction because the controlled comparison is defined entirely on labeled maps.",
    )
    text = text.replace(
        "Our training-time flips and rotations follow the same label-invariance motivation, although the present model does not claim architectural rotation invariance.",
        "Our training-time flips and rotations follow the same label-invariance motivation, although the present model does not claim architectural rotation invariance. A single wafer map can also contain several recurring failure signatures, and CNN classifiers have been extended to such mixed-type patterns {{REF17}}; the present nine-class single-label protocol does not address co-occurring labels.",
    )

    # ---- figure reference style (Springer: Fig. N) ----
    text = re.sub(r"Figure (\d)", r"Fig. \1", text)

    # ---- declarations: default statements ----
    text = text.replace(
        "[To be completed by the author. If no funding supported the work, use: \"The authors received no specific funding for this work.\"]",
        "The authors received no specific funding for this work.",
    )
    text = text.replace(
        "[To be completed by the author. Suggested wording if applicable: \"The authors have no relevant financial or non-financial interests to disclose.\"]",
        "The authors have no relevant financial or non-financial interests to disclose.",
    )
    text = text.replace(
        "[Add an anonymized repository link before peer review if required.]",
        "All code, the fixed lot-disjoint split manifest, and the frozen results are "
        "publicly available under the MIT License at "
        "https://github.com/ZWY631/wafermap-highres.",
    )
    text = text.replace(
        "[Add the final repository link and software license before submission.]",
        "Source code and processing pipelines are publicly available under the "
        "MIT License at https://github.com/ZWY631/wafermap-highres.",
    )

    # ---- split text at the References section ----
    head, sep, tail = text.partition("\n## References\n")
    assert sep, "References section not found"
    # Drop the original v4 reference entries from the tail: keep only the
    # horizontal rule and everything after it (the internal checklist).
    rule_index = tail.index("\n---")
    tail = tail[rule_index:]

    head = renumber_citations(head)
    for number in (14, 15, 16, 17):
        head = head.replace(f"{{{{REF{number}}}}}", f"[{number}]")

    new_references = "\n".join(
        f"{i}. {entry}" for i, entry in enumerate(REFERENCE_ENTRIES, start=1)
    )
    # tail keeps the horizontal rule and the internal checklist
    body = head + "\n## References\n\n" + new_references + "\n" + tail

    V5.write_text(body, encoding="utf-8")
    print(f"Wrote {V5}")


if __name__ == "__main__":
    main()
