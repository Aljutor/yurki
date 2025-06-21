fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    // Allow Py_GIL_DISABLED cfg check
    println!("cargo::rustc-check-cfg=cfg(py_sys_config, values(\"Py_GIL_DISABLED\"))");

    // Debug: let's see what config we get
    let config = pyo3_build_config::get();
    println!("cargo:warning=Python version: {:?}", config.version);
    println!(
        "cargo:warning=Build flags: {:?}",
        config.build_flags.0.iter().collect::<Vec<_>>()
    );
}
