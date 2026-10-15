// moondream_cli.cpp - standalone binary for the moondream3.1 engine.
//
// The engine itself lives behind moondream_main() so the unified vqa-cli can run
// it with the same code path (see src/cli/vqa_cli.cpp).
#include "moondream_main.hpp"

int main(int argc, char ** argv) { return moondream_main(argc, argv); }
