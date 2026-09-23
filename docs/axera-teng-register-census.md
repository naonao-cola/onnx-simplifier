# Register census of the ResNet18 step's elementwise ops (in progress)

Status: work in progress. Builds for Relu, Add, Sub, Mul, Div, Sqrt,
Greater->Cast and ReduceSum under engineered calibrations are complete
(`scripts/axera/teng_register_census.py build`); the census/locate analysis
and this write-up are being finished. Offline only -- no device runs.
