// Cumulative sum considering the discount rate.
pub fn cumsum(seq: &[f64], gamma: f64) -> Vec<f64>{
    seq.iter().scan(0.0, |cumsum_x, &x| {
        *cumsum_x *= gamma;
        *cumsum_x += x;
        Some(*cumsum_x)
    }).collect::<Vec<f64>>()
}

pub fn cumsum_rev(seq: &[f64], gamma: f64) -> Vec<f64>{
    seq.iter().rev().scan(0.0, |cumsum_x, &x| {
        *cumsum_x *= gamma;
        *cumsum_x += x;
        Some(*cumsum_x)
    }).collect::<Vec<f64>>().into_iter().rev().collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumsum() {
        let seq: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result: Vec<f64> = cumsum(&seq, 1.0);
        let expected: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cumsum_empty() {
        let seq: Vec<f64> = vec![];
        let result: Vec<f64> = cumsum(&seq, 1.0);
        let expected: Vec<f64> = vec![];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cumsum_with_gamma() {
        let seq: Vec<f64> = vec![0.0, 0.0, 10.0, 0.0, 20.0];
        let result: Vec<f64> = cumsum(&seq, 0.5);
        let expected: Vec<f64> = vec![0.0, 0.0, 10.0, 5.0, 22.5];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cumsum_rev() {
        let seq: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result: Vec<f64> = cumsum_rev(&seq, 1.0);
        let expected: Vec<f64> = vec![15.0, 14.0, 12.0, 9.0, 5.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cumsum_rev_empty() {
        let seq: Vec<f64> = vec![];
        let result: Vec<f64> = cumsum_rev(&seq, 1.0);
        let expected: Vec<f64> = vec![];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cumsum_rev_with_gamma() {
        let seq: Vec<f64> = vec![0.0, 0.0, 10.0, 0.0, 20.0];
        let result: Vec<f64> = cumsum_rev(&seq, 0.5);
        let expected: Vec<f64> = vec![3.75, 7.5, 15.0, 10.0, 20.0];
        assert_eq!(result, expected);
    }
}
