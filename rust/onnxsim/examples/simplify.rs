//! Simplify an ONNX model from the command line.
//!
//! Usage:
//!     cargo run --example simplify -- <input.onnx> <output.onnx>
//!
//! Requires the native `onnxsim_c` library to be built/linked (see the crate
//! README for build modes).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.onnx> <output.onnx>", args[0]);
        return ExitCode::from(2);
    }
    let input = &args[1];
    let output = &args[2];

    // Simplify in memory (rather than with `simplify_path`) so we still hold the
    // original bytes and can print the before/after diff, mirroring the Python
    // CLI's "here is the difference" output.
    let model_bytes = match std::fs::read(input) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("error: cannot read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };

    match onnxsim::simplify(&model_bytes) {
        Ok(simplified) => {
            if let Err(err) = std::fs::write(output, &simplified) {
                eprintln!("error: cannot write {output}: {err}");
                return ExitCode::FAILURE;
            }
            println!("simplified {input} -> {output}");
            match onnxsim::model_info_diff(&model_bytes, &simplified) {
                Ok(diff) => print!("{diff}"),
                Err(err) => eprintln!("warning: could not compute model diff: {err}"),
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
