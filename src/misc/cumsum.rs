// Cumulative sum considering a per-step discount rate.
pub fn cumsum(seq: &[f64], gamma: &[f64]) -> Vec<f64> {
    assert_eq!(seq.len(), gamma.len());
    seq.iter()
        .zip(gamma.iter())
        .scan(0.0, |cumsum_x, (&x, &g)| {
            *cumsum_x *= g;
            *cumsum_x += x;
            Some(*cumsum_x)
        })
        .collect()
}

pub fn cumsum_rev(seq: &[f64], gamma: &[f64]) -> Vec<f64> {
    assert_eq!(seq.len(), gamma.len());
    seq.iter()
        .zip(gamma.iter())
        .rev()
        .scan(0.0, |cumsum_x, (&x, &g)| {
            *cumsum_x *= g;
            *cumsum_x += x;
            Some(*cumsum_x)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumsum_variable_gamma() {
        let seq = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0, 0.5, 0.25];
        // step0: 0*1.0 + 1.0 = 1.0
        // step1: 1.0*0.5 + 2.0 = 2.5
        // step2: 2.5*0.25 + 3.0 = 3.625
        let expected = vec![1.0, 2.5, 3.625];
        assert_eq!(cumsum(&seq, &gamma), expected);
    }

    #[test]
    fn test_cumsum_rev_variable_gamma() {
        let seq = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0, 0.5, 0.25];
        // step2: 0*0.25 + 3.0 = 3.0
        // step1: 3.0*0.5 + 2.0 = 3.5
        // step0: 3.5*1.0 + 1.0 = 4.5
        let expected = vec![4.5, 3.5, 3.0];
        assert_eq!(cumsum_rev(&seq, &gamma), expected);
    }

    #[test]
    fn test_cumsum_rev_variable_gamma_include_0() {
        let seq = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let gamma = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.0];
        let expected = vec![4.5, 3.5, 3.0, 4.5, 3.5, 3.0];
        assert_eq!(cumsum_rev(&seq, &gamma), expected);
    }

    #[test]
    fn test_cumsum_rev_empty() {
        let seq = vec![];
        let gamma = vec![];
        let expected: Vec<f64> = vec![];
        assert_eq!(cumsum_rev(&seq, &gamma), expected);
    }
}
