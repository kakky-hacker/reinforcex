mod misc;

pub mod agents;
pub mod curiosity;
pub mod explorers;
pub mod memory;
pub mod models;
pub mod prob_distributions;
pub mod selector;

/// Best-effort CUDA initialization retained for existing Rust callers.
///
/// Use [`try_load_cuda_dlls`] when the caller needs to handle a loading failure.
pub fn load_cuda_dlls() {
    let _ = try_load_cuda_dlls();
}

/// Load the Windows CUDA DLL configured by `TORCH_CUDA_DLL`.
///
/// A successful load is retained for the lifetime of the process and reused by
/// subsequent calls. Failures are not cached, so configuration can be corrected
/// before retrying. Other platforms and CPU builds need no explicit loading.
pub fn try_load_cuda_dlls() -> Result<(), String> {
    #[cfg(all(feature = "cuda", target_os = "windows"))]
    {
        cuda_dll::try_load()
    }
    #[cfg(not(all(feature = "cuda", target_os = "windows")))]
    {
        Ok(())
    }
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
mod cuda_dll {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use std::sync::Mutex;
    use winapi::um::libloaderapi::LoadLibraryW;

    // Serialize initialization so repeated/concurrent queries do not acquire
    // additional DLL references. Windows retains the successful load until exit.
    static LOADED: Mutex<bool> = Mutex::new(false);

    pub(super) fn try_load() -> Result<(), String> {
        let mut loaded = LOADED
            .lock()
            .map_err(|_| "CUDA DLL initialization lock was poisoned".to_string())?;
        if *loaded {
            return Ok(());
        }

        let path = std::env::var_os("TORCH_CUDA_DLL")
            .ok_or_else(|| "TORCH_CUDA_DLL is not set".to_string())?;
        load_path(&path)?;
        *loaded = true;
        Ok(())
    }

    fn load_path(path: &OsStr) -> Result<(), String> {
        let mut wide_path: Vec<u16> = path.encode_wide().collect();
        if wide_path.is_empty() {
            return Err("TORCH_CUDA_DLL is empty".to_string());
        }
        if wide_path.contains(&0) {
            return Err("TORCH_CUDA_DLL contains a null character".to_string());
        }
        wide_path.push(0);

        // The UTF-16 buffer is NUL-terminated and lives throughout the call.
        let handle = unsafe { LoadLibraryW(wide_path.as_ptr()) };
        if handle.is_null() {
            let error = std::io::Error::last_os_error();
            return Err(format!(
                "Failed to load TORCH_CUDA_DLL '{}': {}",
                path.to_string_lossy(),
                error
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::load_path;
        use std::ffi::OsStr;

        #[test]
        fn rejects_empty_path() {
            assert_eq!(
                load_path(OsStr::new("")),
                Err("TORCH_CUDA_DLL is empty".to_string())
            );
        }

        #[test]
        fn rejects_embedded_null() {
            assert_eq!(
                load_path(OsStr::new("torch\0_cuda.dll")),
                Err("TORCH_CUDA_DLL contains a null character".to_string())
            );
        }

        #[test]
        fn reports_missing_unicode_dll() {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "reinforcex-存在しない-cuda-{}-{}.dll",
                std::process::id(),
                unique
            ));
            let error = load_path(path.as_os_str()).unwrap_err();
            assert!(error.contains("Failed to load TORCH_CUDA_DLL"));
            assert!(error.contains("存在しない"));
        }
    }
}
