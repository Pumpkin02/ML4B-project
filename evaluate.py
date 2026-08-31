"""
evaluate.py
-----------
Evaluates the two similarity-based classification strategies (Mean Similarity
and KNN Voting) on a held-out split of the LABELLED news corpus.

Why this is the only honest evaluation available:
The German Bundestag tweets have no ground-truth labels, so accuracy on tweets
cannot be computed. The reference corpus (ISOT: Fake.csv / True.csv) IS
labelled, so we hold out part of it, treat the rest as the reference library,
and run exactly the same decision rules the app uses.

"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report

PKL_PATH = "news_embeddings.pkl"
N_TEST = 2000       # held-out items to classify (kept small for memory)
N_REF = 10000       # reference library size
SEED = 42
MEAN_THRESHOLD = 0.01   # same value as in app.py
K = 5                   # same value as in app.py


def main():
    with open(PKL_PATH, "rb") as f:
        d = pickle.load(f)

    emb = np.asarray(d["embeddings"], dtype=np.float32)
    lab = np.asarray(d["labels"])

    print("=" * 60)
    print("REFERENCE CORPUS")
    print("=" * 60)
    print(f"Total articles : {len(lab)}")
    print(f"Fake (label 0) : {(lab == 0).sum()}")
    print(f"True (label 1) : {(lab == 1).sum()}")
    print(f"Embedding dim  : {emb.shape[1]}")
    print()

    # Held-out split: reference library vs. test items
    ref_i, test_i = train_test_split(
        np.arange(len(lab)), test_size=0.2, stratify=lab, random_state=SEED
    )

    rng = np.random.default_rng(SEED)
    test_i = rng.choice(test_i, size=min(N_TEST, len(test_i)), replace=False)
    ref_i = rng.choice(ref_i, size=min(N_REF, len(ref_i)), replace=False)

    E_test, y_test = emb[test_i], lab[test_i]
    E_ref, y_ref = emb[ref_i], lab[ref_i]

    print(f"Test items     : {len(y_test)}")
    print(f"Reference items: {len(y_ref)}")
    print()

    fake_vecs = E_ref[y_ref == 0]
    true_vecs = E_ref[y_ref == 1]

    # ---------- Method 1: Mean Similarity ----------
    sim_f = cosine_similarity(E_test, fake_vecs).mean(axis=1)
    sim_t = cosine_similarity(E_test, true_vecs).mean(axis=1)
    diff = np.abs(sim_f - sim_t)
    pred_mean = np.where(
        diff < MEAN_THRESHOLD, "Unclear",
        np.where(sim_f > sim_t, "Fake", "True")
    )

    # ---------- Method 2: KNN Voting ----------
    S = cosine_similarity(E_test, E_ref)
    topk = np.argsort(S, axis=1)[:, -K:]
    pred_knn = []
    for idx in topk:
        vf = (y_ref[idx] == 0).sum()
        vt = (y_ref[idx] == 1).sum()
        if abs(vf - vt) < 2:
            pred_knn.append("Unclear")
        elif vf > vt:
            pred_knn.append("Fake")
        else:
            pred_knn.append("True")
    pred_knn = np.array(pred_knn)

    # ---------- Evaluation ----------
    truth = np.where(y_test == 0, "Fake", "True")

    for name, pred in [("MEAN SIMILARITY", pred_mean), ("KNN VOTING (K=5)", pred_knn)]:
        decided = pred != "Unclear"
        print("=" * 60)
        print(name)
        print("=" * 60)
        print(f"Coverage (non-Unclear) : {decided.mean():.1%}")

        if decided.sum() == 0:
            print("All predictions were 'Unclear' - no accuracy computable.")
            print()
            continue

        sel_acc = accuracy_score(truth[decided], pred[decided])
        sel_f1 = f1_score(truth[decided], pred[decided], average="macro")
        forced_acc = accuracy_score(truth, pred)  # Unclear counted as wrong

        print(f"Selective Accuracy     : {sel_acc:.1%}")
        print(f"Macro-F1 (on decided)  : {sel_f1:.3f}")
        print(f"Accuracy, Unclear=wrong: {forced_acc:.1%}")
        print()
        print(classification_report(truth[decided], pred[decided], digits=3))
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
