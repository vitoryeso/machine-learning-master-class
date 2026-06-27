# Use this target instead of bare 'cargo run' to ensure output goes to the
# project-level output/ directory (not target/release/output/).
# Background: output_dir() in main.rs resolves PROJETO3_OUTPUT first; without
# it, the binary writes next to itself in target/release/output/.

PROJECT_ROOT := $(shell pwd)

.PHONY: run build clean

run: build
	PROJETO3_OUTPUT=$(PROJECT_ROOT)/output ./target/release/projeto-3

build:
	cargo build --release

clean:
	cargo clean
	rm -rf output/*.png output/report.txt
