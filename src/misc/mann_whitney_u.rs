use std::f64;

// The null hypothesis: a >= b
pub fn mann_whitney_u(a: &[f64], b: &[f64], p: f64) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }

    let n1 = a.len();
    let n2 = b.len();
    let mut ranks = Vec::with_capacity(n1 + n2);

    for &v in a {
        ranks.push((v, 0));
    }
    for &v in b {
        ranks.push((v, 1));
    }

    ranks.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

    let mut sorted_ranks = vec![0.0; ranks.len()];
    let mut i = 0;
    while i < ranks.len() {
        let mut j = i + 1;
        while j < ranks.len() && (ranks[i].0 - ranks[j].0).abs() < 1e-8 {
            j += 1;
        }
        let rank_val = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            sorted_ranks[k] = rank_val;
        }
        i = j;
    }

    let r1: f64 = sorted_ranks
        .iter()
        .zip(&ranks)
        .filter(|(_, (_, group))| *group == 0)
        .map(|(rank, _)| *rank)
        .sum();

    let u1 = r1 - (n1 * (n1 + 1) / 2) as f64;

    let mean_u = (n1 * n2) as f64 / 2.0;
    let std_u = (((n1 * n2 * (n1 + n2 + 1)) as f64) / 12.0).sqrt();
    let z = (u1 - mean_u + 0.5) / std_u;

    normal_cdf(z) < p
}

fn normal_cdf(z: f64) -> f64 {
    (1.0 + erf(z / 2f64.sqrt())) / 2.0
}

fn erf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * z.abs());
    let tau = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();

    if z >= 0.0 {
        1.0 - tau
    } else {
        tau - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_mann_whitney_u_obvious_difference() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 11.0, 12.0];
        assert!(mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_no_difference() {
        let a = vec![5.0, 6.0, 7.0];
        let b = vec![5.0, 6.0, 7.0];
        assert!(!mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_edge_case_empty() {
        let a = vec![];
        let b = vec![1.0, 2.0];
        assert!(!mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_random_1000_samples() {
        let a: Vec<f64> = (0..1000).map(|_| rand::thread_rng().gen::<f64>()).collect();
        let b: Vec<f64> = (0..1000).map(|_| rand::thread_rng().gen::<f64>()).collect();
        assert!(!mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_random_1000_samples_with_difference_bias() {
        let a: Vec<f64> = (0..1000).map(|_| rand::thread_rng().gen::<f64>()).collect();
        let b: Vec<f64> = (0..1000)
            .map(|_| rand::thread_rng().gen::<f64>() + 0.1)
            .collect();
        assert!(mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_random_500_vs_1000_samples() {
        let a: Vec<f64> = (0..500).map(|_| rand::thread_rng().gen::<f64>()).collect();
        let b: Vec<f64> = (0..1000).map(|_| rand::thread_rng().gen::<f64>()).collect();
        assert!(!mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_random_500_vs_1000_samples_with_difference_bias() {
        let a: Vec<f64> = (0..500).map(|_| rand::thread_rng().gen::<f64>()).collect();
        let b: Vec<f64> = (0..1000)
            .map(|_| rand::thread_rng().gen::<f64>() + 0.1)
            .collect();
        assert!(mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_random_1000_samples_with_difference_var() {
        let a: Vec<f64> = (0..1000)
            .map(|_| rand::thread_rng().gen::<f64>() * 2.0)
            .collect();
        let b: Vec<f64> = (0..1000)
            .map(|_| rand::thread_rng().gen::<f64>() + rand::thread_rng().gen::<f64>())
            .collect();
        assert!(!mann_whitney_u(&a, &b, 0.05));
    }

    #[test]
    fn test_mann_whitney_u_random_1000_samples_with_difference_bias_and_var() {
        let a: Vec<f64> = (0..1000)
            .map(|_| rand::thread_rng().gen::<f64>() * 2.0)
            .collect();
        let b: Vec<f64> = (0..1000)
            .map(|_| rand::thread_rng().gen::<f64>() + rand::thread_rng().gen::<f64>() + 0.1)
            .collect();
        assert!(mann_whitney_u(&a, &b, 0.05));
    }
}
