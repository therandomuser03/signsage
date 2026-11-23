# SignSage FDMSE-ISL isolated-word model

* Architecture: SignGRU, 1,537,619 parameters, 2002 classes
* Input: sign_features_v4, 60 x 122
* Confidence threshold: 0.53

## Data
Trained on FDMSE-ISL (Patra et al., Pattern Analysis and Applications 28(3):148, 2025), used with the authors' permission. Weights follow the dataset's terms and do not inherit the code's MIT licence.

## Results
```json
{
  "test": {
    "top1": 0.820719370550768,
    "top5": 0.9508555014362433,
    "loss": 0.9627783894538879,
    "macro_recall": 0.8206585081585082,
    "n": 16014,
    "n_classes_present": 2002
  },
  "validation": {
    "top1": 0.8578566075443418,
    "top5": 0.9665251061703722,
    "loss": 0.7788709402084351,
    "macro_recall": 0.8576423576423576,
    "n": 4003,
    "n_classes_present": 2002
  },
  "data": {
    "n_classes": 2002,
    "converted": 40033,
    "usable": 40033,
    "per_split_converted": {
      "train": 20016,
      "test": 16014,
      "validation": 4003
    },
    "per_split_usable": {
      "train": 20016,
      "test": 16014,
      "validation": 4003
    },
    "total_in_manifest": 40033,
    "feature_files": 801,
    "classes_in_train": 2002,
    "classes_in_validation": 2002,
    "classes_in_test": 2002
  },
  "test_at_threshold": {
    "coverage": 0.6678531285125515,
    "precision": 0.944179523141655
  },
  "rejection_checks": {
    "all zeros": {
      "max_confidence": 0.1813054382801056,
      "rejected": true
    },
    "random noise": {
      "max_confidence": 0.039251528680324554,
      "rejected": true
    },
    "frozen pose (one frame repeated)": {
      "max_confidence": 0.5248780846595764,
      "rejected": false
    },
    "5 frames only (rest zeros)": {
      "max_confidence": 0.08746436983346939,
      "rejected": true
    }
  }
}
```

## Notes
Deployment model: trained on all signers for 40 epochs. The reported metrics come from the first run (official split, test used once).

## Limits
Studio-recorded signers; webcam performance is unmeasured. Not validated for real-world use. Not promoted until the promotion gates in docs/MODELS_AND_DATA.md pass.