/**
 * Aggregate retrieval health score.
 *
 * Combines the per-chunk alignment bell curve with RAG-specific signals
 * (rank quality, diversity, score calibration, bimodality) into a single
 * scalar in [0, 1] plus a regime label.
 *
 *   1.0  → ideal retrieval: high alignment, well-ranked, diverse
 *   0.0  → severely pathological retrieval
 *
 * Structure: "primary signal + bounded penalties" so one noisy signal
 * does not tank the score. Mean alignment is primary; everything else
 * deducts.
 */

const DEFAULT_TOLERANCE = {
  meanAlignment:  0.55,   // below this, retrieval is failing on relevance
  rankQualityR:   0.30,   // want positive; below this = rank issue
  diversity:      0.40,   // want high; below this = redundant
  scoreCalibration: 0.30, // want positive; below this = retriever score drift
  bimodalSignal:  0.40,   // above this penalises
};

// Profile-aware tolerance presets — pair with the same-named profile in
// pathologies.js. Dense (embedding) retrievers produce intrinsically lower
// TF-IDF mean alignment than sparse retrievers do, so the alignment tolerance
// must drop or healthy retrieval tops out around 0.35 health.
const TOLERANCE_PROFILES = {
  tfidf: DEFAULT_TOLERANCE,
  dense: {
    meanAlignment:  0.20,
    rankQualityR:   0.30,
    diversity:      0.40,
    scoreCalibration: 0.30,
    bimodalSignal:  0.40,
  },
};

const PENALTY_WEIGHT = 0.15;  // max health reduction per secondary signal

function scoreFromSignals(signals, tolerance = DEFAULT_TOLERANCE) {
  const primary = signals.meanAlignment != null
    ? clip(signals.meanAlignment / tolerance.meanAlignment)
    : 1;

  let penalty = 0;

  if (signals.rankQualityR != null && signals.rankQualityR < tolerance.rankQualityR) {
    penalty += PENALTY_WEIGHT * clip(
      (tolerance.rankQualityR - signals.rankQualityR) / (1 + tolerance.rankQualityR)
    );
  }
  if (signals.diversity != null && signals.diversity < tolerance.diversity) {
    penalty += PENALTY_WEIGHT * clip(
      (tolerance.diversity - signals.diversity) / tolerance.diversity
    );
  }
  if (signals.scoreCalibrationR != null && signals.scoreCalibrationR < tolerance.scoreCalibration) {
    penalty += PENALTY_WEIGHT * clip(
      (tolerance.scoreCalibration - signals.scoreCalibrationR) / (1 + tolerance.scoreCalibration)
    );
  }
  if (signals.bimodalSignal != null && signals.bimodalSignal > tolerance.bimodalSignal) {
    penalty += PENALTY_WEIGHT * clip(
      (signals.bimodalSignal - tolerance.bimodalSignal) / (1 - tolerance.bimodalSignal)
    );
  }

  return Math.max(0, primary - penalty);
}

function regime(score) {
  if (score >= 0.80) return 'healthy';
  if (score >= 0.55) return 'drift';
  if (score >= 0.30) return 'contamination';
  return 'rot';
}

function clip(v) { return Math.max(0, Math.min(1, v)); }

module.exports = { scoreFromSignals, regime, DEFAULT_TOLERANCE, TOLERANCE_PROFILES };
