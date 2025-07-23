fn main() {
    tonic_build::compile_protos("sample_env/env_server.proto").unwrap();
}
