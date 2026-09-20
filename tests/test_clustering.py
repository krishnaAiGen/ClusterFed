"""Tests for labeling/clustering.py: orientation, scoring, and Stage-2 confidence."""
import numpy as np

from clusterfed.labeling import clustering, evaluate


def _two_blobs(seed=0):
    rng = np.random.default_rng(seed)
    benign = rng.normal(0.0, 0.3, size=(200, 4))
    attack = rng.normal(5.0, 0.3, size=(200, 4))
    X = np.vstack([benign, attack])
    names = np.array(["BENIGN"] * 200 + ["DoS Hulk"] * 200)
    y_bin = np.array([0] * 200 + [1] * 200)
    return X, names, y_bin


def test_orientation_makes_benign_zero():
    X, names, y_bin = _two_blobs()
    g = clustering.known_benign_point(X, names, n=50, seed=0)
    cfg = {"n_clusters": 2, "n_init": 10}
    y_pseudo, _, _ = clustering.kmeans_label(X, g, cfg, seed=0)
    acc = (y_pseudo == y_bin).mean()
    assert acc > 0.95


def test_score_perfect():
    y = np.array([0, 1, 0, 1, 1])
    m = evaluate.score(y, y)
    assert m["f1"] == 1.0 and m["accuracy"] == 1.0


def test_known_benign_point_is_near_benign_mean():
    X, names, _ = _two_blobs()
    g = clustering.known_benign_point(X, names, n=50, seed=0)
    benign_mean = X[:200].mean(axis=0)
    assert np.linalg.norm(g - benign_mean) < 0.5


def test_multicluster_recovers_mass_imbalanced_multimodal_attack():
    """The exact failure k=2 had on DDoS/Hulk (P~0.98, R~0.45): attack mass much
    larger than benign AND multimodal. Binary k-means splits the attack mass;
    multi-cluster + benign-density orientation must recover both attack modes."""
    rng = np.random.default_rng(7)
    benign = rng.normal(0.0, 0.3, size=(400, 6))
    attack_mode1 = rng.normal(4.0, 0.3, size=(2000, 6))
    attack_mode2 = rng.normal(-4.0, 0.3, size=(2000, 6))   # second mode on the far side
    X = np.vstack([benign, attack_mode1, attack_mode2])
    names = np.array(["BENIGN"] * 400 + ["DDoS"] * 4000)
    y_bin = np.array([0] * 400 + [1] * 4000)

    cfg = {"n_clusters": 8, "n_init": 10, "known_benign_n": 100}
    y_pseudo, conf = clustering.bootstrap_labels(X, names, cfg, seed=7)

    from sklearn.metrics import f1_score, recall_score
    assert recall_score(y_bin, y_pseudo) > 0.95   # both attack modes found
    assert f1_score(y_bin, y_pseudo) > 0.9
    assert conf.shape == (len(X),)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_bootstrap_labels_k2_matches_legacy_binary_path():
    """n_clusters=2 must dispatch to the v1-legacy binary path unchanged."""
    X, names, y_bin = _two_blobs()
    cfg2 = {"n_clusters": 2, "n_init": 10, "known_benign_n": 50}
    y_pseudo, conf = clustering.bootstrap_labels(X, names, cfg2, seed=0)
    g = clustering.known_benign_point(X, names, n=50, seed=0)
    y_legacy, _, _ = clustering.kmeans_label(X, g, cfg2, seed=0)
    assert np.array_equal(y_pseudo, y_legacy)
    assert conf.shape == (len(X),)


def test_benign_density_orientation_never_all_one_class():
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, size=(500, 4))     # no real structure at all
    names = np.array(["BENIGN"] * 250 + ["Bot"] * 250)
    cfg = {"n_clusters": 6, "n_init": 5, "known_benign_n": 50}
    y_pseudo, _ = clustering.bootstrap_labels(X, names, cfg, seed=3)
    assert 0 < y_pseudo.sum() < len(y_pseudo)   # both classes always present


def test_confidence_low_near_boundary_high_far_away():
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 0.3, size=(200, 2))
    attack = rng.normal(6.0, 0.3, size=(200, 2))
    boundary = np.full((1, 2), 3.0)   # midpoint between the two cluster means
    far = np.zeros((1, 2))            # deep inside the benign cluster
    X = np.vstack([benign, attack, boundary, far])
    names = np.array(["BENIGN"] * 200 + ["DoS Hulk"] * 200 + ["BENIGN"] * 2)

    g = clustering.known_benign_point(X, names, n=50, seed=0)
    cfg = {"n_clusters": 2, "n_init": 10}
    y_pseudo, _, centroids = clustering.kmeans_label(X, g, cfg, seed=0)
    conf = clustering.confidence_scores(X, centroids, y_pseudo)

    assert conf.shape == (402,)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()
    assert conf[400] < conf[401]  # boundary point far less confident than the far one
