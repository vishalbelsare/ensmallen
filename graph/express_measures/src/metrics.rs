use crate::validation::*;
use rayon::prelude::*;
use std::collections::HashMap;
use unzip_n::unzip_n;

unzip_n!(pub(crate) 3);

#[derive(Debug, Clone, Copy)]
pub struct BinaryConfusionMatrix {
    true_positives: usize,
    true_negatives: usize,
    false_positives: usize,
    false_negatives: usize,
}

unsafe impl Sync for BinaryConfusionMatrix {}
unsafe impl Send for BinaryConfusionMatrix {}

impl BinaryConfusionMatrix {
    /// Create a new Binary Confusion Matrix from the provided tuple.
    ///
    /// # Arguments
    /// * `ground_truth`: bool - The ground truth binary value.
    /// * `prediction`: bool - The prediction binary value.
    fn from_tuple(ground_truth: bool, prediction: bool) -> BinaryConfusionMatrix {
        if ground_truth == prediction {
            if ground_truth {
                BinaryConfusionMatrix {
                    true_positives: 1,
                    true_negatives: 0,
                    false_positives: 0,
                    false_negatives: 0,
                }
            } else {
                BinaryConfusionMatrix {
                    true_positives: 0,
                    true_negatives: 1,
                    false_positives: 0,
                    false_negatives: 0,
                }
            }
        } else {
            if prediction {
                BinaryConfusionMatrix {
                    true_positives: 0,
                    true_negatives: 0,
                    false_positives: 1,
                    false_negatives: 0,
                }
            } else {
                BinaryConfusionMatrix {
                    true_positives: 0,
                    true_negatives: 0,
                    false_positives: 0,
                    false_negatives: 1,
                }
            }
        }
    }

    /// Compute the binary confusion matrix from the values we have in the
    /// compute_auc method. This is for internal uses only and is not intedned
    /// to be exposed.
    fn form_auc_values(
        total_positives: usize, 
        total_negatives: usize, 
        current_total_samples: usize, 
        current_total_positives: usize
    ) -> Self {
        let false_positives = current_total_samples - current_total_positives;
        BinaryConfusionMatrix{
            true_positives:  current_total_positives,
            false_negatives: total_positives - current_total_positives,

            false_positives: false_positives,
            true_negatives:  total_negatives - false_positives,
        }
    }

    /// Create a new Binary Confusion Matrix from the provided slices.
    ///
    /// # Arguments
    /// * `ground_truths`: &[bool] - The ground truths binary values.
    /// * `predictions`: &[bool] - The predictions binary values.
    ///
    /// # Raises
    /// * When the slices are not compatible (i.e. do not have the same length).
    pub fn from_slices(
        ground_truths: &[bool],
        predictions: &[bool],
    ) -> Result<BinaryConfusionMatrix, String> {
        validate_vectors_length(ground_truths.len(), predictions.len())?;
        Ok(ground_truths
            .par_iter()
            .copied()
            .zip(predictions.par_iter().copied())
            .map(|(ground_truth, prediction)| {
                BinaryConfusionMatrix::from_tuple(ground_truth, prediction)
            })
            .sum())
    }

    /// Returns the total number of true positive values.
    pub fn get_number_of_true_positives(&self) -> usize {
        self.true_positives
    }

    /// Returns the total number of true negative values.
    pub fn get_number_of_true_negatives(&self) -> usize {
        self.true_negatives
    }

    /// Returns the total number of false positive values.
    pub fn get_number_of_false_positives(&self) -> usize {
        self.false_positives
    }

    /// Returns the total number of false negative values.
    pub fn get_number_of_false_negatives(&self) -> usize {
        self.false_negatives
    }

    /// Returns the total number of positive values.
    pub fn get_number_of_positive_values(&self) -> usize {
        self.true_positives + self.false_negatives
    }

    /// Returns the total number of negative values.
    pub fn get_number_of_negative_values(&self) -> usize {
        self.true_negatives + self.false_positives
    }

    /// Returns the total number of positive predictions.
    pub fn get_number_of_positive_predictions(&self) -> usize {
        self.true_positives + self.false_positives
    }

    /// Returns the total number of negative predictions.
    pub fn get_number_of_negative_predictions(&self) -> usize {
        self.true_negatives + self.false_negatives
    }

    /// Returns the total number of correct predictions.
    pub fn get_number_of_correct_predictions(&self) -> usize {
        self.true_positives + self.true_negatives
    }

    /// Returns the total number of incorrect predictions.
    pub fn get_number_of_incorrect_predictions(&self) -> usize {
        self.false_positives + self.false_negatives
    }

    /// Returns the total number of samples.
    pub fn get_number_of_samples(&self) -> usize {
        self.get_number_of_correct_predictions() + self.get_number_of_incorrect_predictions()
    }

    /// Returns whether there are positive samples.
    pub fn has_positive_samples(&self) -> bool {
        self.get_number_of_positive_values() > 0
    }

    /// Returns whether there are negative samples.
    pub fn has_negative_samples(&self) -> bool {
        self.get_number_of_negative_values() > 0
    }

    /// Returns whether there are positive predictions.
    pub fn has_positive_predictions(&self) -> bool {
        self.get_number_of_positive_predictions() > 0
    }

    /// Returns whether there are negative predictions.
    pub fn has_negative_predictions(&self) -> bool {
        self.get_number_of_negative_predictions() > 0
    }

    /// Returns the accuracy.
    pub fn get_binary_accuracy(&self) -> f32 {
        self.get_number_of_correct_predictions() as f32 / self.get_number_of_samples() as f32
    }

    /// Returns the binary recall.
    pub fn get_binary_recall(&self) -> f32 {
        if !self.has_positive_samples() {
            return f32::NAN;
        }
        self.get_number_of_true_positives() as f32 / self.get_number_of_positive_values() as f32
    }

    /// Returns the binary specificity.
    pub fn get_binary_specificity(&self) -> f32 {
        if !self.has_negative_samples() {
            return f32::NAN;
        }
        self.get_number_of_true_negatives() as f32 / self.get_number_of_negative_values() as f32
    }

    /// Returns the binary miss rate.
    pub fn get_binary_miss_rate(&self) -> f32 {
        1.0 - self.get_binary_recall()
    }

    /// Returns the binary fall-out.
    pub fn get_binary_fall_out(&self) -> f32 {
        1.0 - self.get_binary_specificity()
    }

    /// Returns the binary informedness.
    pub fn get_binary_informedness(&self) -> f32 {
        self.get_binary_recall() + self.get_binary_specificity() - 1.0
    }

    /// Returns the binary prevalence threshold.
    pub fn get_binary_prevalence_threshold(&self) -> f32 {
        ((self.get_binary_recall() * self.get_binary_fall_out()).sqrt()
            - self.get_binary_fall_out())
            / (self.get_binary_recall() - self.get_binary_fall_out())
    }

    /// Returns the binary prevalence.
    pub fn get_binary_prevalence(&self) -> f32 {
        self.get_number_of_positive_values() as f32 / self.get_number_of_samples() as f32
    }

    /// Returns the binary balanced accuracy.
    pub fn get_binary_balanced_accuracy(&self) -> f32 {
        (self.get_binary_recall() + self.get_binary_specificity()) / 2.0
    }

    /// Returns the binary precision.
    pub fn get_binary_precision(&self) -> f32 {
        if !self.has_positive_predictions() {
            return f32::NAN;
        }
        self.get_number_of_true_positives() as f32
            / self.get_number_of_positive_predictions() as f32
    }

    /// Returns the binary false discovery rate.
    pub fn get_binary_false_discovery_rate(&self) -> f32 {
        1.0 - self.get_binary_precision()
    }

    /// Returns the binary false omission rate.
    pub fn get_binary_false_omission_rate(&self) -> f32 {
        if !self.has_negative_predictions() {
            return f32::NAN;
        }
        self.get_number_of_false_negatives() as f32
            / self.get_number_of_negative_predictions() as f32
    }

    /// Returns the binary negative predictive value.
    pub fn get_binary_negative_predictive_value(&self) -> f32 {
        1.0 - self.get_binary_false_omission_rate()
    }

    /// Returns the binary positive likelyhood ratio.
    pub fn get_binary_positive_likelyhood_ratio(&self) -> f32 {
        self.get_binary_recall() / self.get_binary_fall_out()
    }

    /// Returns the binary negative likelyhood ratio.
    pub fn get_binary_negative_likelyhood_ratio(&self) -> f32 {
        self.get_binary_miss_rate() / self.get_binary_specificity()
    }

    /// Returns the binary markedness.
    pub fn get_binary_markedness(&self) -> f32 {
        self.get_binary_precision() + self.get_binary_negative_predictive_value() - 1.0
    }

    /// Returns the binary diagnostic odds ratio.
    pub fn get_binary_diagnostic_odds_ratio(&self) -> f32 {
        self.get_binary_positive_likelyhood_ratio() / self.get_binary_negative_likelyhood_ratio()
    }

    /// Returns the binary F1 score
    pub fn get_binary_f1_score(&self) -> f32 {
        (2 * self.get_number_of_true_positives()) as f32
            / (2 * self.get_number_of_true_positives() + self.get_number_of_incorrect_predictions())
                as f32
    }

    /// Returns the binary Fowlkes-Mallows index
    pub fn get_binary_fowlkes_mallows_index(&self) -> f32 {
        (self.get_binary_precision() * self.get_binary_recall()).sqrt()
    }

    /// Returns the binary Threat score.
    pub fn get_binary_threat_score(&self) -> f32 {
        self.get_number_of_true_positives() as f32
            / (self.get_number_of_true_positives() + self.get_number_of_incorrect_predictions())
                as f32
    }

    /// Returns the binary Matthews correlation coefficient.
    pub fn get_binary_matthews_correlation_coefficient(&self) -> f32 {
        (self.get_binary_recall()
            * self.get_binary_specificity()
            * self.get_binary_precision()
            * self.get_binary_negative_predictive_value())
        .sqrt()
            - (self.get_binary_miss_rate()
                * self.get_binary_fall_out()
                * self.get_binary_false_omission_rate()
                * self.get_binary_false_discovery_rate()).sqrt()
    }

    /// Returns hashmap with all available binary metrics.
    pub fn get_all_binary_metrics(&self) -> HashMap<String, f32> {
        [
            ("accuracy", self.get_binary_accuracy()),
            ("recall", self.get_binary_recall()),
            ("specificity", self.get_binary_specificity()),
            ("miss_rate", self.get_binary_miss_rate()),
            ("fall_out", self.get_binary_fall_out()),
            ("informedness", self.get_binary_informedness()),
            (
                "prevalence_threshold",
                self.get_binary_prevalence_threshold(),
            ),
            ("prevalence", self.get_binary_prevalence()),
            ("balanced_accuracy", self.get_binary_balanced_accuracy()),
            ("precision", self.get_binary_precision()),
            (
                "false_discovery_rate",
                self.get_binary_false_discovery_rate(),
            ),
            ("false_omission_rate", self.get_binary_false_omission_rate()),
            (
                "negative_predictive_value",
                self.get_binary_negative_predictive_value(),
            ),
            (
                "positive_likelyhood_ratio",
                self.get_binary_positive_likelyhood_ratio(),
            ),
            (
                "negative_likelyhood_ratio",
                self.get_binary_negative_likelyhood_ratio(),
            ),
            ("markedness", self.get_binary_markedness()),
            (
                "diagnostic_odds_ratio",
                self.get_binary_diagnostic_odds_ratio(),
            ),
            ("f1_score", self.get_binary_f1_score()),
            (
                "fowlkes_mallows_index",
                self.get_binary_fowlkes_mallows_index(),
            ),
            ("threat_score", self.get_binary_threat_score()),
            (
                "matthews_correlation_coefficient",
                self.get_binary_matthews_correlation_coefficient(),
            ),
        ]
        .into_iter()
        .map(|(name, result)| {
            (name.to_string(), result)
        })
        .collect()
    }
}

impl Default for BinaryConfusionMatrix {
    fn default() -> BinaryConfusionMatrix {
        BinaryConfusionMatrix {
            true_positives: 0,
            true_negatives: 0,
            false_positives: 0,
            false_negatives: 0,
        }
    }
}

impl core::ops::Add for BinaryConfusionMatrix {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            true_positives: self.true_positives + other.true_positives,
            true_negatives: self.true_negatives + other.true_negatives,
            false_positives: self.false_positives + other.false_positives,
            false_negatives: self.false_negatives + other.false_negatives,
        }
    }
}

impl core::ops::AddAssign for BinaryConfusionMatrix {
    fn add_assign(&mut self, other: Self) {
        self.true_positives += other.true_positives;
        self.true_negatives += other.true_negatives;
        self.false_positives += other.false_positives;
        self.false_negatives += other.false_negatives;
    }
}

impl core::iter::Sum<Self> for BinaryConfusionMatrix {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |mut a, b| {
            a += b;
            a
        })
    }
}


/// Returns binary auroc score for the provided ground truths and predictions.
/// 
/// # Arguments
/// * `ground_truths`: &[bool] - The ground truths binary values.
/// * `predictions`: &[f32] - The predictions binary values.
///
/// # Raises
/// * When the slices are not compatible (i.e. do not have the same length).
pub fn get_binary_auroc(
    ground_truths: &[bool], 
    predictions: &[f32], 
) -> Result<f32, String> {
    get_binary_auc(
        ground_truths, 
        predictions,
        |previous: &BinaryConfusionMatrix, current: &BinaryConfusionMatrix| {
            // trapezoidal approximation for rinneman integral
            (current.get_binary_recall() + previous.get_binary_recall()) *
                (current.get_binary_fall_out() - previous.get_binary_fall_out()) 
                / 2.0
        }
    )
}

/// Returns binary auprc score for the provided ground truths and predictions.
/// 
/// # Arguments
/// * `ground_truths`: &[bool] - The ground truths binary values.
/// * `predictions`: &[f32] - The predictions binary values.
///
/// # Raises
/// * When the slices are not compatible (i.e. do not have the same length).
pub fn get_binary_auprc(
    ground_truths: &[bool], 
    predictions: &[f32], 
) -> Result<f32, String> {
    get_binary_auc(
        ground_truths, 
        predictions,
        |previous: &BinaryConfusionMatrix, current: &BinaryConfusionMatrix| {
            // trapezoidal approximation for rinneman integral
            (current.get_binary_precision() + previous.get_binary_precision()) *
                (current.get_binary_recall() - previous.get_binary_recall()) 
                / 2.0
        }
    )
}

/// Returns binary auc score for the provided ground truths and predictions, 
/// of the curve specified by the callable `curve`.
/// 
/// # Arguments
/// * `ground_truths`: &[bool] - The ground truths binary values.
/// * `predictions`: &[f32] - The predictions binary values.
/// * `curve`:  fn(previous: &BinaryConfusionMatrix, current: &BinaryConfusionMatrix) -> f32 - 
///     The function that, given the previous and current binary confusion metrices,
///     (at the variation of the threshold), should compute the area of this slice
///     of the curve. E.g for AUPRC it should compute the difference of recall
///     multiplied by the current precision.
///
/// # Raises
/// * When the slices are not compatible (i.e. do not have the same length).
fn get_binary_auc(
    ground_truths: &[bool], 
    predictions: &[f32], 
    curve: fn(previous: &BinaryConfusionMatrix, current: &BinaryConfusionMatrix) -> f32,
) -> Result<f32, String> {
    // First, we check that the two vectors have the expected length.
    validate_vectors_length(ground_truths.len(), predictions.len())?;

    // Secondly, we sort the provided predictions by decreasing
    // order, using a reverse index.
    let mut reverse_predictions_index: Vec<usize> = (0..ground_truths.len()).collect();
    reverse_predictions_index.par_sort_unstable_by(|&a, &b| {
        predictions[b]
            .partial_cmp(&predictions[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // We compute the comulative sum of the positive labels.
    let positive_labels_running_sum: Vec<usize> = reverse_predictions_index
        .into_iter()
        .map(|index| ground_truths[index])
        .scan(0, |current_total, label| {
            if label {
                *current_total += 1;
            }
            Some(*current_total)
        })
        .collect();
    
    // We get the total positives and negatives.
    let number_of_predictions = positive_labels_running_sum.len();
    let total_positives = *positive_labels_running_sum.last().unwrap();
    let total_negatives = number_of_predictions - total_positives;

    if total_positives == 0 {
        return Err(concat!(
            "We could not compute the given AUC because the given data ",
            "has no posive labels",
        ).to_string());
    }

    if total_negatives == 0 {
        return Err(concat!(
            "We could not compute the given AUC because the given data ",
            "has no negative labels",
        ).to_string());
    }

    // And finally, we can compute the AUC integral.
    Ok(positive_labels_running_sum.par_windows(2)
        .enumerate()
        .map(|(i, positive_labels_sum_window)| {
            let previous = BinaryConfusionMatrix::form_auc_values(
                total_positives,
                total_negatives,
                i + 1,
                positive_labels_sum_window[0],
            );
            let current = BinaryConfusionMatrix::form_auc_values(
                total_positives,
                total_negatives,
                i + 2,
                positive_labels_sum_window[1],
            );

            curve(
                &previous,
                &current,
            )
        })
        .sum::<f32>()
    )
}
