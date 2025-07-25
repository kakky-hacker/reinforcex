mod misc;

pub mod agents;
pub mod explorers;
pub mod memory;
pub mod models;
pub mod prob_distributions;

pub fn load_cuda_dlls() {
    #[cfg(all(feature = "cuda", target_os = "windows"))]
    {
        println!("load TORCH_CUDA_DLL.");

        use std::env;
        use std::ffi::CString;
        use std::ptr::null_mut;
        use winapi::um::libloaderapi::LoadLibraryA;

        let path_str = env::var("TORCH_CUDA_DLL").expect("TORCH_CUDA_DLL not set");
        let c_path = CString::new(path_str).expect("Path contains null byte");
        unsafe {
            let handle = LoadLibraryA(c_path.as_ptr());
        }
    }
}
