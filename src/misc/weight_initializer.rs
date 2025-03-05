use candle_nn::Init;

/// Xavier Initialization
pub fn xavier_init(nin: i64, nout: i64) -> Init {
    let lo = -(6.0 / (nin + nout) as f64).sqrt();
    let up = (6.0 / (nin + nout) as f64).sqrt();
    Init::Uniform { lo, up }
}

/// He Initialization
pub fn he_init(nin: i64) -> Init {
    let mean = 0.0;
    let stdev = (2.0 / nin as f64).sqrt();
    Init::Randn { mean, stdev }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_init() {
        let nin = 4;
        let nout = 6;
        let expected_init = xavier_init(nin, nout);

        match expected_init {
            Init::Uniform { lo, up } => {
                let expected_lo = -(6.0 / (nin + nout) as f64).sqrt();
                let expected_up = (6.0 / (nin + nout) as f64).sqrt();
                assert!((lo - expected_lo).abs() < 1e-6, "Lo value mismatch");
                assert!((up - expected_up).abs() < 1e-6, "Up value mismatch");
            }
            _ => panic!("Expected Uniform initialization"),
        }
    }

    #[test]
    fn test_he_init() {
        let nin = 4;
        let expected_init = he_init(nin);

        match expected_init {
            Init::Randn { mean, stdev } => {
                let expected_stdev = (2.0 / nin as f64).sqrt();
                assert!((mean - 0.0).abs() < 1e-6, "Mean value mismatch");
                assert!(
                    (stdev - expected_stdev).abs() < 1e-6,
                    "Standard deviation mismatch"
                );
            }
            _ => panic!("Expected Randn initialization"),
        }
    }
}
